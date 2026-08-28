import os
import sys


import os
import sys
from pathlib import Path


# 获取项目根目录 (llm106/)
project_root = Path(__file__).resolve().parent.parent

# 将所有需要的子目录添加到 sys.path
paths_to_add = [
    project_root / 'configs',
    project_root / 'ch2',
    project_root / 'ch3',
]
# `python step10_sft.py` 时 sys.path 只有 ch4/。必须把仓库根目录 llm106/
# 加进去，因为 dataset_sft 内部是 `from ch2.dataset_utils import ...`。
# 只 insert configs/ 或 ch2/、以及 `__package__ = "ch4"`，都不够。
current_dir = Path(__file__).resolve().parent  # ch4/
parent_dir = current_dir.parent  # llm106/
for extra in (parent_dir, parent_dir / "ch2", parent_dir / "configs", parent_dir / "ch3"):
    extra = str(extra)
    if extra not in sys.path:
        sys.path.insert(0, extra)


from torch import nn
from transformers import PreTrainedModel, GenerationMixin, AutoTokenizer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from LlmConfig import Llm106Config
from step20_embedding import RopeOperation
import torch, torch.nn.functional as F

from utils import get_model_params, Logger
__package__ = "ch3"  # 设置包名，用于模块导入


def init_model(lm_config, from_weight='pretrain', tokenizer_path='./', save_dir='../out', device='cuda'):
    """
    初始化语言模型和对应的分词器

    参数:
        lm_config: 语言模型的配置对象，包含模型结构参数（如hidden_size, use_moe等）
        from_weight: 权重加载来源，可选值：
            - 'pretrain': 从预训练权重加载（默认）
            - 'none': 不加载任何权重，使用随机初始化
            - 其他字符串: 具体的权重文件路径或权重名称
        tokenizer_path: 分词器模型文件的路径，默认为当前目录
        save_dir: 权重文件保存的目录，默认为 '../out'
        device: 模型运行的设备，默认为 'cuda' (GPU)

    返回:
        model: 初始化并加载好权重的语言模型
        tokenizer: 对应的分词器对象
    """

    # 从指定路径加载预训练的分词器
    # AutoTokenizer会自动根据路径中的配置文件选择合适的分词器类型
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 使用配置创建自定义的LLM模型实例 (Llm106Model)
    model = Llm106Model(lm_config)

    # 检查是否需要加载预训练权重（'none'表示不加载）
    if from_weight != 'none':
        # 判断是否包含'/'，用于区分是权重名称还是完整路径
        if '/' not in from_weight:
            # 如果是权重名称（不包含路径分隔符），则构造标准化的权重文件路径

            # 根据是否使用MoE (Mixture of Experts) 添加对应的后缀
            moe_suffix = '_moe' if lm_config.use_moe else ''

            # 构造权重文件路径: {保存目录}/{权重名称}_{隐藏层大小}{moe后缀}.pth
            # 例如: ../out/pretrain_768_moe.pth 或 ../out/pretrain_768.pth
            weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'

            # 加载权重文件到指定设备
            # map_location指定加载到CPU或GPU，避免设备不匹配问题
            weights = torch.load(weight_path, map_location=device)

            # 将加载的权重加载到模型中
            # strict=False 允许部分权重不匹配（例如可以只加载部分层）
            model.load_state_dict(weights, strict=False)
        else:
            # 如果from_weight包含'/'，则视为完整的文件路径
            # 直接从该路径加载权重文件
            weights = torch.load(from_weight, map_location=device)
            model.load_state_dict(weights, strict=False)

    # 调用辅助函数打印模型参数信息（可能包括总参数量、各层参数等）
    get_model_params(model, lm_config)

    # 创建日志记录器并输出可训练参数的数量（单位：百万）
    # 计算所有 requires_grad=True 的参数的元素总数，并转换为百万单位
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')

    # 将模型移动到指定设备（如GPU）并返回模型和分词器
    return model.to(device), tokenizer

class Llm106Model(PreTrainedModel, GenerationMixin):
    config_class = Llm106Config
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Llm106Config = None):
        self.config = config or Llm106Config()
        super().__init__(self.config)
        self.model = RopeOperation(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0,
                labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache,
                                                              **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values,
                                         hidden_states=hidden_states)

    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50,
                 eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True,
                 repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache,
                                   **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)],
                                       -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]);
                    score = logits[i, seen];
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(
                logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1),
                                                                  next_token.new_full((next_token.shape[0], 1),
                                                                                      eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids






