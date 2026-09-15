import os
import sys


import os
import sys
from pathlib import Path


# ============================================================
# 一、路径处理：保证脚本在任何入口下都能正确导入项目模块
# ============================================================

# 获取项目根目录 (llm106/)
# __file__ 是当前脚本的路径，.resolve() 解析为绝对路径（消除符号链接等）
# .parent 是当前脚本所在目录（ch4/），再 .parent 就是项目根目录 llm106/
project_root = Path(__file__).resolve().parent.parent

# 将所有需要的子目录添加到 sys.path
# 目的：让 Python 在导入时能找到 configs、ch2、ch3 这些包/模块
paths_to_add = [
    project_root / 'configs',
    project_root / 'ch2',
    project_root / 'ch3',
]

# `python step10_sft.py` 时 sys.path 只有 ch4/。必须把仓库根目录 llm106/
# 加进去，因为 dataset_sft 内部是 `from ch2.dataset_utils import ...`。
# 只 insert configs/ 或 ch2/、以及 `__package__ = "ch4"`，都不够。
#
# 说明：
# - 直接运行脚本时，Python 只把脚本所在目录（ch4/）加入 sys.path，
#   因此 `import ch2.xxx` 会失败，除非把 llm106/ 加入 sys.path。
# - 这里对每个候选目录做去重后 insert(0, ...)，保证优先级最高。
current_dir = Path(__file__).resolve().parent  # ch4/
parent_dir = current_dir.parent                # llm106/
for extra in (parent_dir, parent_dir / "ch2", parent_dir / "configs", parent_dir / "ch3"):
    extra = str(extra)
    if extra not in sys.path:
        sys.path.insert(0, extra)


# ============================================================
# 二、依赖导入
# ============================================================

from torch import nn
from transformers import PreTrainedModel, GenerationMixin, AutoTokenizer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from LlmConfig import Llm106Config          # 自定义的模型配置类
from step20_embedding import RopeOperation  # 自定义的 Transformer 主体（含 RoPE）
import torch, torch.nn.functional as F

from utils import get_model_params, Logger  # 自定义工具：打印参数量、日志
__package__ = "ch3"

def init_model(lm_config, from_weight='pretrain', tokenizer_path='./',
               save_dir='../out', device='cuda'):
    # 从指定路径加载预训练的分词器
    # AutoTokenizer 会自动读取 tokenizer_config.json / tokenizer.json 等
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 使用配置创建自定义的 LLM 模型实例 (Llm106Model)
    model = Llm106Model(lm_config)

    # 检查是否需要加载预训练权重（'none' 表示不加载，随机初始化）
    if from_weight != 'none':
        # 判断是否包含 '/'，用于区分是"权重名称"还是"完整路径"
        if '/' not in from_weight:
            # 情况 1：传入的是权重名称，按项目约定拼接出标准权重文件路径

            # 根据是否使用 MoE (Mixture of Experts) 添加对应的后缀
            moe_suffix = '_moe' if lm_config.use_moe else ''

            # 构造权重文件路径: {保存目录}/{权重名称}_{隐藏层大小}{moe后缀}.pth
            # 例如: ../out/pretrain_768_moe.pth 或 ../out/pretrain_768.pth
            weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'

            # 加载权重文件到指定设备
            # map_location 指定加载到 CPU 还是 GPU，避免保存/加载设备不一致
            weights = torch.load(weight_path, map_location=device)

            # 将加载的权重注入模型
            # strict=False：允许部分 key 不匹配（例如只加载部分层、或新增头部）
            model.load_state_dict(weights, strict=False)
        else:
            # 情况 2：传入的是完整路径，直接加载
            weights = torch.load(from_weight, map_location=device)
            model.load_state_dict(weights, strict=False)
    # else:
    #     weights = torch.load(from_weight, map_location=device)
    #     model.load_state_dict(weights, strict=False)

    # 打印模型参数信息（总量、各层等），便于确认模型规模
    get_model_params(model, lm_config)

    # 创建日志记录器并输出可训练参数数量（单位：百万）
    # 只统计 requires_grad=True 的参数，适合观察微调时的可训练规模
    Logger(f'Trainable Params: '
           f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')

    # 将模型移动到指定设备（如 GPU）并返回模型和分词器
    return model.to(device), tokenizer


# ============================================================
# 四、自定义语言模型（继承 HF 的 PreTrainedModel + GenerationMixin）
# ============================================================
class Llm106Model(PreTrainedModel, GenerationMixin):
    # 指定配置类，HF 内部会用它来序列化/反序列化 config
    config_class = Llm106Config

    # 声明需要"绑定权重"的参数：
    # 当调用 tie_weights() 时，会把 lm_head.weight 与 embed_tokens.weight 共享
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Llm106Config = None):
        self.config = config or Llm106Config()
        super().__init__(self.config)
        self.model = RopeOperation(self.config)
        # 语言模型头：把 hidden_size 映射到 vocab_size，bias=False 是常见做法
        self.lm_head = nn.Linear(self.config.hidden_size,
                                 self.config.vocab_size, bias=False)

        # 如果配置要求共享词嵌入权重，则让 lm_head 与 embedding 共享同一份权重
        # 注意：这里直接把 lm_head.weight 指向 embedding.weight（同一个 Parameter 对象）
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

        # 初始化权重（HF 的 post_init 会调用 _init_weights，
        # 以及处理 tied weights 等）
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        """
        前向传播：
          - 主体计算 hidden_states、新的 past_key_values、MoE 辅助损失
          - lm_head 得到 logits
          - 若提供 labels，则计算交叉熵损失（标准因果 LM 损失）
        """
        # 调用主体（RopeOperation）：
        #   hidden_states: [B, T, H]
        #   past_key_values: 新的 KV 缓存（用于自回归生成）
        #   aux_loss: MoE 负载均衡等辅助损失（如未使用 MoE 则为 0/None）
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )

        # logits_to_keep 控制只保留最后 N 个位置的 logits（生成时常用，节省显存）
        # 若为 int，则取最后 N 个；否则视为切片对象
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # 通过 lm_head 得到词表分布 logits: [B, T', V]
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # 标准因果 LM 损失：
            #   logits 预测下一个 token，因此 logits[:-1] 对齐 labels[1:]
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()

            # 展平后计算交叉熵，ignore_index=-100 忽略 padding / 不计损失的位置
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        # 返回 HF 标准输出结构，便于 Trainer / generate 等复用
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )


    @torch.inference_mode()  # 关闭梯度，节省显存，加速推理
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192,
                 temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2,
                 streamer=None, use_cache=True, num_return_sequences=1,
                 do_sample=True, repetition_penalty=1.0, **kwargs):
        # 兼容两种调用方式：inputs=... 或 input_ids=...
        # 并按 num_return_sequences 复制输入，实现多样本生成
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)

        # attention_mask 同样复制（若未提供则为 None）
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None

        # 取出可选的 past_key_values（一般用于续写）
        past_key_values = kwargs.pop("past_key_values", None)

        # finished: 标记每个样本是否已经生成了 EOS，用于提前停止
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        # 若提供 streamer，先把 prompt 推送给它（便于流式展示）
        if streamer:
            streamer.put(input_ids.cpu())

        # 自回归生成循环
        for _ in range(max_new_tokens):
            # 计算已缓存的长度：past_key_values[0][0].shape[1] 是 KV 的序列长度
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0

            # 只把"新 token"喂给模型（配合 KV cache），得到下一步 logits
            outputs = self.forward(
                input_ids[:, past_len:], attention_mask, past_key_values,
                use_cache=use_cache, **kwargs
            )

            # 更新 attention_mask：在末尾追加一列 1（表示新生成的 token 有效）
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)],
                -1
            ) if attention_mask is not None else None

            # 取最后一个位置的 logits，并应用温度
            logits = outputs.logits[:, -1, :] / temperature

            # 重复惩罚（repetition_penalty）：
            #   对已经出现过的 token，若 logit > 0 则除以惩罚系数，否则乘以惩罚系数
            #   这样会降低已出现 token 的概率
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(
                        score > 0,
                        score / repetition_penalty,
                        score * repetition_penalty
                    )

            # top-k 过滤：保留概率最高的 k 个 token，其余置为 -inf
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

            # top-p (nucleus) 过滤：
            #   按概率降序排列，累积概率超过 top_p 的位置被置为 -inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # 累积概率超过 top_p 的 mask
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                # 右移一位，保证第一个超过阈值的 token 仍然保留
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                # 把 mask 对应的位置置为 -inf（scatter 回原索引）
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            # 采样：do_sample=True 时按概率多项式采样；否则贪心取 argmax
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) \
                if do_sample else torch.argmax(logits, dim=-1, keepdim=True)

            # 对于已经 finished 的样本，强制其下一个 token 为 EOS（避免继续生成）
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token
                )

            # 拼接新 token 到输入序列
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # 更新 KV cache（若 use_cache=False 则丢弃，下次重新计算）
            past_key_values = outputs.past_key_values if use_cache else None

            # 流式输出新 token
            if streamer:
                streamer.put(next_token.cpu())

            # 更新 finished 标记：若某样本刚生成了 EOS，则标记为已完成
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                # 所有样本都完成则提前退出
                if finished.all():
                    break

        # 结束流式输出
        if streamer:
            streamer.end()

        # 若调用方需要 KV cache（例如多轮对话续写），则一并返回
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}

        # 默认只返回生成的完整 token 序列
        return input_ids