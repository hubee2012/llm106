import math

import torch
from torch import nn
from transformers import AutoTokenizer
import argparse
import LlmConfig
from step40_norm import RMSNorm
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
__package__ = "ch3"  # 设置包名，用于模块导入

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # 旋转半圈：把向量的前后两半交换并取负
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 标准 RoPE 公式：x * cos + rotate_half(x) * sin
    # cos/sin 原本形状 (S, D)，unsqueeze 后变成 (S, 1, D) 与 (B, H, S, D) 广播
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 在 head 维度后插入一维并复制 n_rep 次，再展平
    # 效果：每个 KV 头被 n_rep 个 Q 头共享
    return (x[:, :, :, None, :]
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, config: LlmConfig):
        super().__init__()
        # KV 头数：若配置为 None 则退化为 MHA
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads          # Q 头数
        self.n_local_kv_heads = self.num_key_value_heads         # KV 头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # 每个 KV 头被多少 Q 头复用
        self.head_dim = config.head_dim
        self.is_causal = True                                    # 因果掩码

        # 投影层（无 bias，符合 LLaMA 风格）
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # QK-Norm：对每个头做 RMSNorm，稳定训练
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)   # 注意力权重 dropout
        self.resid_dropout = nn.Dropout(config.dropout)  # 输出残差 dropout
        self.dropout = config.dropout
        # 是否启用 PyTorch 内置的 SDPA 加速
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape

        # ---- 1) QKV 投影 ----
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # ---- 2) reshape 成多头形式 ----
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # ---- 3) QK-Norm ----
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # ---- 4) 应用 RoPE ----
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # ---- 5) 拼接历史 KV（推理时缓存复用）----
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # ---- 6) 转置为 (B, H, S, D) 并复制 KV 头 ----
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # ---- 7) 注意力计算 ----
        # 快速路径：SDPA 只在「非因果首帧」或「无 mask」时使用，避免与自定义 mask 冲突
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) \
                and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal)
        else:
            # 慢速路径：手写注意力
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                # 上三角（不含对角线）填 -inf，实现因果掩码
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None:
                # padding mask：0 的位置填 -1e9
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            # softmax 用 float32 保证数值稳定，再转回原 dtype
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        # ---- 8) 合并多头并输出投影 ----
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

# class Attention(nn.Module):
#     def __init__(self, config: LlmConfig):
#         super().__init__()
#         self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
#         self.n_local_heads = config.num_attention_heads
#         self.n_local_kv_heads = self.num_key_value_heads
#         self.n_rep = self.n_local_heads // self.n_local_kv_heads
#         self.head_dim = config.head_dim
#         self.is_causal = True
#         self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
#         self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
#         self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
#         self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
#         self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
#         self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
#         self.attn_dropout = nn.Dropout(config.dropout)
#         self.resid_dropout = nn.Dropout(config.dropout)
#         self.dropout = config.dropout
#         self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
#
#     def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
#         bsz, seq_len, _ = x.shape
#         xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
#         xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
#         xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
#         xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
#         xq, xk = self.q_norm(xq), self.k_norm(xk)
#         cos, sin = position_embeddings
#         xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
#         if past_key_value is not None:
#             xk = torch.cat([past_key_value[0], xk], dim=1)
#             xv = torch.cat([past_key_value[1], xv], dim=1)
#         past_kv = (xk, xv) if use_cache else None
#         xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
#         if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
#             output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
#         else:
#             scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
#             if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
#             if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
#             output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
#         output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
#         output = self.resid_dropout(self.o_proj(output))
#         return output, past_kv