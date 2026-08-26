import os
import sys

# When this file is imported as a sibling (`from step20_embedding import ...`)
# while running `python step70_pretrain.py` from ch3/, ch3/__init__.py never
# runs. Put the repo root on sys.path so `configs` / `utils` / `ch3` resolve.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import math
import torch
from torch import nn
from transformers import AutoTokenizer
import argparse
from ch3 import LlmConfig
from ch3.dataset_pretrain import PretrainDataset
from ch3.step30_attention import Attention
from ch3.step40_norm import RMSNorm
from ch3.step50_feedforward import FeedForward, MOEFeedForward
from configs.llm_utils import llm_data_dir
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

token_path='./'

TOKENIZER_SAVE_PATH = llm_data_dir+"/pretrain_t2t_mini.jsonl"

class AssembleBlock(nn.Module):
    def __init__(self, layer_id: int, config: LlmConfig.Llm106Config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value




class RopeOperation(nn.Module):
    def __init__(self,
                config: LlmConfig.Llm106Config
                ):
        super().__init__()
        self.vocab_size=config.vocab_size
        self.embedding_dim=config.hidden_size
        self.embed_tokens=nn.Embedding(self.vocab_size,self.embedding_dim)
        freqs_cos, freqs_sin = self.rope_YaRN(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([AssembleBlock(l, config) for l in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)



    # NTK-aware 插值 方法，最初由 Reddit 用户 bloc97 在 2023 年提出
    # YaRN 论文（YaRN: Efficient Context Window Extension of Large Language Models）
    # 论文：https: // arxiv.org / pdf / 2309.00071.pdf
    # 代码：https: // github.com / jquesnelle / yarn
    def rope_YaRN(self,dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
        # (rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为1到1e6之间增函数
        # 1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))为小于1降函数，位置越靠前频率越高，相当于将data调制到了freqs频率(data*cos(freqs*data))
        freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
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

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if self.embed_tokens.weight.device != input_ids.device:
            self.to(input_ids.device)
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = self.rope_YaRN(dim=self.config.head_dim,
                                                        end=self.config.max_position_embeddings,
                                                        rope_base=self.config.rope_theta,
                                                        rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length],self.freqs_sin[start_pos:start_pos + seq_length])
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
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm106-")
    parser.add_argument('--data_path', type=str, default=llm_data_dir + "/pretrain_t2t_mini.jsonl", help='训练数据')
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    # parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    # parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--max_seq_len', default=240, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--num_workers", type=int, default=6, help="数据加载线程数")
    # parser.add_argument("--save_dir", type=str, default="../../../llm_data/llm106_model", help="模型保存目录")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(token_path, local_files_only=True)
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

    # os.makedirs(args.save_dir, exist_ok=True)
    lm_config = LlmConfig.Llm106Config(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                       use_moe=bool(args.use_moe))
    # ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
    #                          save_dir='../../../checkpoints') if args.from_resume == 1 else None
    model = RopeOperation(lm_config)

    start_step = 0
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        hidden_state, past_kv, aux_loss = model(input_ids, labels=labels)
        print(aux_loss)
    # start_time = time.time()

    # last_step = start_step

    # for epoch in range(args.epochs):
    #     iters = len(loader)
    #     for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    #         input_ids = input_ids.to(args.device)
    #         labels = labels.to(args.device)
    #         last_step = step
    #
    #         # Forward pass
    #         hidden_state, past_kv, aux_loss = model(input_ids, labels=labels)
    #         loss = res.loss + res.aux_loss
    #         loss = loss / args.accumulation_steps  # Normalize for gradient accumulation
    #
    #         # Direct backward pass (no mixed precision)
    #         loss.backward()
    #
    #         # Gradient accumulation step
    #         if step % args.accumulation_steps == 0:
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    #             optimizer.step()
    #             optimizer.zero_grad(set_to_none=True)
    #
    #         # Logging
    #         if step % args.log_interval == 0 or step == iters:
    #             spend_time = time.time() - start_time
    #             current_loss = loss.item() * args.accumulation_steps
    #             current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
    #             current_logits_loss = current_loss - current_aux_loss
    #             current_lr = optimizer.param_groups[-1]['lr']
    #             eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
    #             Logger(
    #                 f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
    #             if swanlab:
    #                 swanlab.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss,
    #                              "learning_rate": current_lr, "epoch_time": eta_min})
    #         # Save checkpoint
    #         if (step % args.save_interval == 0 or step == iters) and is_main_process():
    #             model.eval()
    #             moe_suffix = '_moe' if lm_config.use_moe else ''
    #             ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    #             raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    #             raw_model = getattr(raw_model, '_orig_mod', raw_model)
    #             state_dict = raw_model.state_dict()
    #             torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
    #             lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, epoch=epoch,
    #                           step=step, swanlab=swanlab, save_dir='../../../checkpoints')
    #             model.train()
    #             del state_dict
    #
    #     # Handle remaining gradients at the end of epoch
    #     if last_step > start_step and last_step % args.accumulation_steps != 0:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    #         optimizer.step()
    #         optimizer.zero_grad(set_to_none=True)