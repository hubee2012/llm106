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
        #首帧：调用方传 past_key_value=None → 跳过，xk/xv 保持本次算出的 (1, 3, 4, 64)。
        #后续帧：调用方传 past_key_value=(past_k, past_v) → 进入拼接。
        if past_key_value is not None:
            #dim=1 是序列维（B, S, H, D 里的 S）。拼接效果：
            #past_k: (1, S_past, 4, 64)
            # xk : (1, 3,     4, 64)
            #new : (1, S_past+3, 4, 64),增加序列元素
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # ---- 6) 转置为 (B, H, S, D) 并复制 KV 头 ----
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # ---- 7) 注意力计算 ----
        # 快速路径：SDPA 只在「非因果首帧」或「无 mask」时使用，避免与自定义 mask 冲突
        #        if (self.flash)  # ① 硬件/配置允许
        #           and (seq_len > 1)  # ② 本次喂入不止 1 个 token
        #           and (not self.is_causal or past_key_value is None)  # ③ 因果性"能对得上"
        #           and (attention_mask is None or torch.all(attention_mask == 1)):  # ④ 没有 padding
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) \
                and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal)
        else:
            # ============================================================
            # 慢速路径：手写注意力
            # ------------------------------------------------------------
            # 触发场景（任一条成立就走这里）：
            #   1. 未启用 SDPA（老版本 PyTorch 或 config.flash_attn=False）
            #   2. seq_len == 1（推理单 token，SDPA 的 is_causal 语义对不上）
            #   3. is_causal=True 且 past_key_value 不为 None（非首帧，Q/K 长度不等）
            #   4. attention_mask 含 padding（SDPA 不支持额外的 padding mask）
            # 慢速路径是通用兜底：对任意 S_q / S_k 组合都正确。
            # ============================================================

            # ---- 1) 计算原始注意力分数 ----
            # xq: (B, H, S_q, D)     本次 query
            # xk: (B, H, S_k, D)     本次 key（已含历史，因为前面 cat 过）
            # xk.transpose(-2, -1): (B, H, D, S_k)
            # xq @ xk^T -> (B, H, S_q, S_k)
            # scores[b,h,i,j] = query 位置 i 对 key 位置 j 的原始相似度
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            #                                ↑ 缩放因子 1/sqrt(D)，防止点积随 D 增大而爆炸，
            #                                  保持 softmax 输入方差稳定，梯度不消失。

            # ---- 2) 因果掩码（causal mask）----
            #is_causal=True 表示：这是一个自回归（autoregressive）的因果语言模型。
            #即：预测 token t 时，只能看到 0..t，不能看到未来 t+1, t+2, ...
            if self.is_causal:
                # scores[:, :, :, -seq_len:] 取「最后 seq_len 列」
                #   - 训练 / 首帧：S_q = S_k = seq_len，取全部列
                #   - 推理单 token：S_q = 1，S_k = S_past+1，取最后 1 列（当前 token 自己的 K）
                #     ★ 前面 S_past 列（历史 KV）不在切片内，因此不会被加 -inf，
                #       正好符合「当前 token 应看到全部历史」的语义。
                #
                # torch.full((seq_len, seq_len), -inf)：生成全 -inf 方阵
                # .triu(1)：保留主对角线「以上」（不含对角线），其余置 0
                #   例如 seq_len=4 得到：
                #     [[ 0, -inf, -inf, -inf],
                #      [ 0,    0, -inf, -inf],
                #      [ 0,    0,    0, -inf],
                #      [ 0,    0,    0,    0]]
                #   含义：query 位置 i 只能看到 key 位置 j ≤ i。
                #
                # += 广播相加：mask 形状 (seq_len, seq_len)
                #             切片形状 (B, H, seq_len, seq_len)
                #             -inf 位置 → scores 变 -inf → softmax 后权重精确为 0
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device).triu(1)

            # ---- 3) Padding 掩码 ----
            if attention_mask is not None:
                # attention_mask: (B, S_k)，1=有效，0=padding
                #   注意长度必须等于当前 KV 总长（历史 + 新增），
                #   否则 unsqueeze 后广播形状对不上。
                #
                # unsqueeze(1).unsqueeze(2) -> (B, 1, 1, S_k)
                #   与 scores (B, H, S_q, S_k) 广播，作用到每个 head、每个 query 位置。
                #
                # (1.0 - mask)：把 1→0，0→1
                #   即：有效位 → 0（不加惩罚）
                #       padding 位 → 1 * -1e9 = -1e9（近似 -inf，softmax 后≈0）
                #
                # ★ 这里用 -1e9 而非 -inf，是为了避免与因果掩码的 -inf 相加时
                #   出现 -inf + inf = nan 的边界情况（虽然本实现里因果掩码是 +=，
                #   但保持用有限大负数更稳妥、兼容性更好）。
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9

            # ---- 4) Softmax + Dropout + 加权求和 ----
            # F.softmax(scores.float(), dim=-1)
            #   - dim=-1：对 key 维（最后一维）做归一化，得到注意力权重
            #   - .float()：先转 fp32 再 softmax，避免 fp16/bf16 下 exp 溢出或精度丢失
            #   - 因果 / padding 位置的 -inf/-1e9 经 exp 后为 0，权重精确为 0
            #
            # .type_as(xq)：把权重转回原 dtype（与 xv 一致，才能做 matmul）
            #
            # self.attn_dropout(...)：训练时随机置零部分注意力权重（正则化）
            #                          推理时 training=False，Dropout 是恒等映射
            #
            # @ xv：加权求和
            #   attn: (B, H, S_q, S_k)
            #   xv  : (B, H, S_k, D)
            #   out : (B, H, S_q, D)  —— 每个 query 位置的注意力输出
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