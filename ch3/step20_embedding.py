import math

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
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig


token_path='./'

TOKENIZER_SAVE_PATH = llm_data_dir+"/pretrain_t2t_mini.jsonl"


class Rope:
    rope_base = 1000000
    def __init__(self, dim: int = 512,
                 end: int = 32 * 1024,
                 rope_scaling:bool=0):
        self.rope_theta =  1e6
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        }
        # #(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为1到1e6之间增函数
        # #1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为小于1降函数，位置越靠前频率越高，相当于将data调制到了freqs频率(data*cos(freqs*data))
        # #一定程度上相当于将序列数据做了正交投影，正交频率由位置决定，
        # freqs = 1.0 / (self.rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        # t = torch.arange(end, device=freqs.device)
        # freqs = torch.outer(t, freqs).float()
        # # 计算 cos 和 sin，并重复拼接以匹配维度
        # self.freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * self.attn_factor
        # self.freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * self.attn_factor

    # NTK-aware 插值 方法，最初由 Reddit 用户 bloc97 在 2023 年提出
    # YaRN 论文（YaRN: Efficient Context Window Extension of Large Language Models）
    # 论文：https: // arxiv.org / pdf / 2309.00071.pdf
    # 代码：https: // github.com / jquesnelle / yarn
    def YaRN(self,dim: int, end: int = int(32 * 1024), rope_base: float = 1e6):
        # (rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为1到1e6之间增函数
        # 1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为小于1降函数，位置越靠前频率越高，相当于将data调制到了freqs频率(data*cos(freqs*data))
        freqs, attn_factor = 1.0 / (self.rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
        if self.rope_scaling is not None:  # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            orig_max, factor, beta_fast, beta_slow, attn_factor = (
                self.rope_scaling.get("original_max_position_embeddings", 2048),
                self.rope_scaling.get("factor", 16),
                self.rope_scaling.get("beta_fast", 32.0),
                self.rope_scaling.get("beta_slow", 1.0),
                self.rope_scaling.get("attention_factor", 1.0)
            )
            if end / orig_max > 1.0:
                inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
                low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
                ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                                   0, 1)
                freqs = freqs * (1 - ramp + ramp / factor)
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
        return freqs_cos, freqs_sin

    def getRope(self,embedding_dim,max_context_len):
        # return self.freqs_cos, self.freqs_sin
        return self.YaRN(embedding_dim,max_context_len)





class Embedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 max_position_embeddings,
                 self.config = config
                ):
        super().__init__()
        self.vocab_size=vocab_size
        self.embed_tokens=nn.Embedding(vocab_size,embedding_dim)
        # self.lm_head=nn.Linear(embedding_dim,vocab_size,bias=False)
        # self.embed_tokens.weight=self.lm_head.weight    #输入输出权重共享，实现可逆映射
        # self.ep=Rope(embedding_dim,max_position_embeddings)
        # self.freq_cos,self.freq_sin=self.ep.getRope()
        # if torch.cuda.is_available():
        #     self.cuda()

        freqs_cos, freqs_sin = self.rope_YaRN(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    # NTK-aware 插值 方法，最初由 Reddit 用户 bloc97 在 2023 年提出
    # YaRN 论文（YaRN: Efficient Context Window Extension of Large Language Models）
    # 论文：https: // arxiv.org / pdf / 2309.00071.pdf
    # 代码：https: // github.com / jquesnelle / yarn
    def rope_YaRN(self,dim: int, end: int = int(32 * 1024), rope_base: float = 1e6):
        # (rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为1到1e6之间增函数
        # 1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为小于1降函数，位置越靠前频率越高，相当于将data调制到了freqs频率(data*cos(freqs*data))
        freqs, attn_factor = 1.0 / (self.rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
        if rope_scaling is not None:  # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            orig_max, factor, beta_fast, beta_slow, attn_factor = (
                rope_scaling.get("original_max_position_embeddings", 2048),
                rope_scaling.get("factor", 16),
                rope_scaling.get("beta_fast", 32.0),
                rope_scaling.get("beta_slow", 1.0),
                rope_scaling.get("attention_factor", 1.0)
            )
            if end / orig_max > 1.0:
                inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
                low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
                ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                                   0, 1)
                freqs = freqs * (1 - ramp + ramp / factor)
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
        return freqs_cos, freqs_sin

    def forwardforward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim,
                                                        end=self.config.max_position_embeddings,
                                                        rope_base=self.config.rope_theta,
                                                        rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length],
                               self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
                       hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

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
