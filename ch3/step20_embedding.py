import torch
from torch import nn
from transformers import AutoTokenizer
import argparse
from ch3.dataset_pretrain import PretrainDataset
from configs.llm_utils import llm_data_dir, llm_model_dir
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from typing import Tuple, Optional, List, Union
import torch.nn.functional as F


token_path='./'

TOKENIZER_SAVE_PATH = llm_data_dir+"/pretrain_t2t_mini.jsonl"


class Rope:
    rope_base = 10000
    attn_factor = 1.0
    input_max=2048
    def __init__(self, dim: int = 512,
                 end: int = 32 * 1024,
                 rope_scaling:bool=0):
        #(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为1到1e6之间增函数
        #1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为小于1降函数，位置越靠前频率越高，相当于将data调制到了freqs频率(data*cos(freqs*data))
        #一定程度上相当于将序列数据做了正交投影，正交频率由位置决定，
        freqs = 1.0 / (self.rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        #NTK-aware 插值 方法，最初由 Reddit 用户 bloc97 在 2023 年提出
        # YaRN 论文（YaRN: Efficient Context Window Extension of Large Language Models）系统记录和改进
        # if rope_scaling:
        # 外推,频率压缩
        #     # 只有当序列长度超过原始最大长度时才应用缩放
        #     if end / self.input_max > 1.0:
        #         # 计算需要缩放的维度范围：根据beta_fast和beta_slow确定插值的起始和结束维度
        #         inv_dim = lambda b: (dim * math.log(self.input_max / (b * 2 * math.pi))) / (2 * math.log(self.rope_base))
        #         low = max(math.floor(inv_dim(beta_fast)), 0)
        #         high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
        #         # 构建线性斜坡函数ramp，使得ramp[low]=0，ramp[high]=1，其余线性插值
        #         ramp = torch.clamp(
        #             (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
        #             0, 1
        #         )
        #         # 调整频率：freqs_new = freqs * (1 - ramp + ramp / factor)
        #         freqs = freqs * (1 - ramp + ramp / self.attn_factor)
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        # 计算 cos 和 sin，并重复拼接以匹配维度
        self.freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * self.attn_factor
        self.freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * self.attn_factor

    def getRope(self):
        return self.freqs_cos, self.freqs_sin


class Embedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 max_position_embeddings):
        super().__init__()
        self.vocab_size=vocab_size
        self.embed_tokens=nn.Embedding(vocab_size,embedding_dim)
        self.lm_head=nn.Linear(embedding_dim,vocab_size,bias=False)
        self.embed_tokens.weight=self.lm_head.weight    #输入输出权重共享，实现可逆映射

        self.ep=Rope(embedding_dim,max_position_embeddings)
        self.freq_cos,self.freq_sin=self.ep.getRope()
        if torch.cuda.is_available():
            self.cuda()
    # def position_embedding(self,dim:int,
    #                        end:int=32*1024,
    #                        rope_base:float=1e6,
    #                        rope_scaling:Optional[dict]=None):
    #     #(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为1到1e6之间增函数
    #     #1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为小于1降函数，位置越靠前频率越高
    #     freqs=1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))
    #     if rope_scaling is not None:
    #         #外推,频率压缩
    #         pass
    #     t=torch.arange(end,device=freqs.device)

    def forward(self,
                input_ids:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.Tensor]=None,
                labels: Optional[torch.Tensor] = None,
                past_kv:Optional[List[Tuple[torch.Tensor,torch.Tensor]]]=None,
                use_cache:bool=False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args
                ):
        bsize,dat_len=input_ids.shape

        embeded_dat=self.dropout(self.embed_tokens(input_ids))
        start_pose=0
        #position_embedding=(self.freq_cos[start_pose:start_pose+dat_len],self.freq_sin[start_pose:start_pose+dat_len])
        present=[]

        past_kv_=past_kv or [None]*len(self.layers)
        for layer_i,(layer,past_kv_item) in enumerate(zip(self.layers,past_kv_)):
            embeded_dat,cur=layer(input=embeded_dat,past_kv=past_kv_item,use_cache=use_cache,attention_mask=attention_mask)
            present.append(cur)
        #当使用moe时，返回0张量，不影响主损失函数
        # aux_loss=sum([l.feedforward.aux_loss for l in self.layers if isinstance(l.feedforward,Moe)],embeded_dat.new_zeros(1).squeeze)
        aux_loss=0

        # return embeded_dat, present, aux_loss


        #hidden_states[:, slice_indices, :] 取出每个样本中指定位置的 hidden states
        #然后只对这些位置的 hidden states 计算 logits，从而避免了为所有 token 计算 logits 的开销。
        #在自回归生成过程中（例如循环调用模型逐个预测下一个 token），每次前向传播只需要知道当前序列最后一个 token 的 logits，用来采样下一个词。如果不加裁剪，模型会为整个输入序列的所有 token 都计算 logits，而大部分位置的 logits 是多余的，浪费了显存和计算资源。
        #训练阶段：通常需要所有 token 的 logits 来计算交叉熵损失（对比每个位置预测的词与真实词），因此应设置 logits_to_keep = 0 或负数，保留全部 logits。生成阶段：例如调用 model.generate() 时，内部会设置 logits_to_keep = 1，让模型只计算最后一个 token 的 logits，大幅提升生成速度。
        # 如果 logits_to_keep 是整数
        if isinstance(logits_to_keep, int):
            # 若大于0，构造一个从倒数第 logits_to_keep 个元素到末尾的切片
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            # 否则直接使用传入的索引（例如 slice(1,5) 或列表等）
            slice_indices = logits_to_keep
        logits = self.lm_head(embeded_dat[:, slice_indices, :])

        loss = None
        if labels is not None:
            # 计算交叉熵损失，忽略-100的位置
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1),
                                    ignore_index=-100)

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_kv,
            hidden_states=embeded_dat
        )


        # 附加MoE的辅助损失（可以用于日志或梯度）
        output.aux_loss = aux_loss
        return output
        # return embeded_dat,present,aux_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm106-")
    parser.add_argument('--data_path', type=str, default=llm_data_dir+"/pretrain_t2t_mini.jsonl", help='模型保存目录')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(token_path,local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )


    embedding()
