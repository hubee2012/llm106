
import os
import sys

__package__ = "trainer"  # 设置包名，用于模块导入

# 导入自定义模块
from ch2.dataset_rlhf import RLHFDataset  # RLHF数据集处理
from ch3.LlmConfig import Llm106Config  # 模型配置类
from ch3.step60_llmmodel import Llm106Model, init_model  # 基础模型和初始化函数
from ch4.rollout_engine import create_rollout_engine  # 生成引擎（用于采样）
from trainer.trainer_utils import LMForRewardModel  # 奖励模型包装类
from utils import is_main_process, Logger, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler

# 添加项目根目录到Python路径（解决导入问题）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 标准库和第三方库
import datasets  # noqa: F401  # Windows下pyarrow/torch DLL冲突的临时解决方案
import argparse
import math
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings('ignore')  # 忽略警告信息


# ============================================================================
# 辅助函数
# ============================================================================
def rep_penalty(text, n=3, cap=0.5):
    """
    计算文本的重复惩罚分数，用于鼓励生成的文本内容多样性

    参数:
        text: 输入文本字符串
        n: n-gram的大小，默认为3
        cap: 惩罚分数的上限，默认为0.5

    返回:
        重复惩罚分数，范围[0, cap]
    """
    # 将文本分词为单词和标点符号（转为小写）
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    # 提取所有n-gram
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    # 如果有n-gram，计算重复比例并应用惩罚；否则返回0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


# ============================================================================
# Critic模型：用于估计状态价值函数 V(s)
# ============================================================================
class CriticModel(Llm106Model):
    """
    自定义的Critic模型，继承自Llama模型架构
    用于PPO算法中的价值函数估计
    """

    def __init__(self, params):
        super().__init__(params)
        # 替换原始的lm_head（语言模型头）为价值头（输出标量值）
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        前向传播：计算给定输入序列每个位置的价值估计
        参数:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        返回:
            values: 每个位置的价值估计 [batch_size, seq_len]
        """
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])  # 应用LayerNorm
        # 使用value_head获取价值估计，并去除最后一维
        values = self.value_head(hidden_states).squeeze(-1)
        return values


# ============================================================================
# 奖励计算函数
# ============================================================================
def calculate_rewards(prompts, responses, reward_model):
    """
    计算每个生成响应的综合奖励分数

    奖励组成:
        1. 长度惩罚：响应长度在20-800字符之间给正奖励
        2. 思考内容惩罚（如果包含</think>标签）
        3. 重复惩罚：基于n-gram重复度
        4. 奖励模型分数：使用预训练的奖励模型打分

    参数:
        prompts: 提示列表 [batch_size]
        responses: 生成的响应列表 [batch_size]
        reward_model: 奖励模型实例

    返回:
        rewards: 每个样本的综合奖励 [batch_size]
    """
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():  # 不计算梯度，节省显存
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # 1. 长度惩罚：鼓励适中的响应长度
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

            # 2. 处理思维链（如果包含</think>标签）
            if '</think>' in response:
                thinking_content, answer_content = response.split('</think>', 1)
                # 思考内容长度在20-300字符之间给正奖励
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                # 确保只有一个</think>标签
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_content.strip()
            else:
                answer = response.strip()

            # 3. 重复惩罚：降低重复度高的文本的奖励
            rewards[i] -= rep_penalty(answer)

            # 4. 奖励模型评分：解析对话格式并打分
            # 提取对话中的角色和内容
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            # 获取奖励模型分数
            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        # 将奖励模型分数转为张量并加到总奖励中
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


# ============================================================================
# PPO训练核心函数
# ============================================================================
def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler,
                    critic_scheduler, reward_model, start_step=0, wandb=None):
    """
    执行一个epoch的PPO训练

    PPO算法步骤：
        1. Rollout：使用Actor模型生成响应
        2. 计算奖励
        3. 计算优势函数 (GAE)
        4. 多次更新：使用裁剪的代理目标函数进行策略更新

    参数:
        epoch: 当前epoch编号
        loader: 数据加载器
        iters: 总迭代步数
        rollout_engine: 生成引擎（用于采样）
        ref_model: 参考模型（用于KL散度惩罚）
        actor_scheduler: Actor学习率调度器
        critic_scheduler: Critic学习率调度器
        reward_model: 奖励模型
        start_step: 起始步数（用于续训）
        wandb: wandb日志实例
    """
    # 设置模型为训练模式
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0  # 梯度累积计数器

    for step, batch in enumerate(loader, start=start_step + 1):
        # ====================================================================
        # 第1步：Rollout - 使用Actor模型生成响应
        # ====================================================================
        prompts = batch["prompt"]  # list[str], 长度为B
        # 对prompts进行分词处理（左填充，保证生成时上下文对齐）
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_seq_len, padding_side="left").to(args.device)
        # enc.input_ids: [B, P], enc.attention_mask: [B, P]

        # 使用rollout引擎生成响应
        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,  # 每个prompt生成1个响应
            max_new_tokens=args.max_gen_len,
            temperature=0.8,  # 采样温度
        )
        # 提取生成结果
        gen_out = rollout_result.output_ids  # 完整序列 [B, P+R]
        completion_ids = rollout_result.completion_ids  # 生成的token [B, R]
        prompt_lens = rollout_result.prompt_lens.to(args.device)  # 每个prompt的长度 [B]
        responses_text = rollout_result.completions  # 生成的文本列表 [B]
        old_resp_logp = rollout_result.per_token_logps.to(args.device)  # 旧策略的log概率 [B, R]

        # ====================================================================
        # 第2步：计算奖励
        # ====================================================================
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]

        # ====================================================================
        # 第3步：调试信息输出（仅在debug模式下）
        # ====================================================================
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-' * 100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] prompt_len={prompt_lens[i].item()}, response_len={len(responses_text[i])}")
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger('=' * 100)

        # ====================================================================
        # 第4步：准备用于PPO更新的数据
        # ====================================================================
        # 构建完整序列的掩码（排除padding）
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        labels = gen_out[:, 1:].clone()  # 用于计算log概率的标签 [B, P+R-1]

        B = len(prompts)
        resp_labels = completion_ids  # [B, R]
        # 计算响应部分在完整序列中的位置索引
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)  # [1, R]
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx  # [B, R]

        # 响应部分的掩码
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()  # [B, R]
        resp_lengths = resp_pad_mask.sum(dim=1)  # [B]
        valid_resp = resp_lengths > 0  # 有效响应的mask

        # 处理EOS token（提前终止）
        eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        has_eos = eos_mask.any(dim=1)
        eos_pos = torch.argmax(eos_mask.int(), dim=1)
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)

        # 创建策略和价值函数的掩码（只关注有效token）
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        resp_value_mask = resp_policy_mask.clone()

        # ====================================================================
        # 第5步：计算优势函数（GAE - Generalized Advantage Estimation）
        # ====================================================================
        with torch.no_grad():  # 不计算梯度，节省显存
            # 5.1 获取Critic模型的价值估计
            critic_for_rollout = critic_model.module if isinstance(critic_model,
                                                                   DistributedDataParallel) else critic_model
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask  # [B, R]

            # 5.2 获取参考模型的log概率（用于KL散度惩罚）
            ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1],
                                          dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

            # 5.3 构建token级别的奖励
            token_rewards = torch.zeros_like(old_resp_logp)
            last_idx = resp_lengths - 1  # [B]
            # 只在最后一个token上加上外部奖励
            token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]

            # 5.4 计算GAE优势
            gen_len = old_resp_values.size(1)
            lastgaelam = torch.zeros(B, device=args.device)
            advs_rev = []
            # 从后向前计算GAE
            for t in reversed(range(gen_len)):
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                # TD误差: δ = r_t + γ * V(s_{t+1}) - V(s_t)
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
                # GAE: A_t = δ_t + (γλ)δ_{t+1} + ...
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advs_rev.append(lastgaelam)
            advantages = torch.stack(advs_rev[::-1], dim=1)  # [B, R]
            returns = advantages + old_resp_values  # [B, R] 目标回报

            # 5.5 标准化优势（提高训练稳定性）
            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        # ====================================================================
        # 第6步：PPO策略更新（多次迭代）
        # ====================================================================
        mb_size = max(1, min(args.mini_batch_size, B))  # mini-batch大小
        stop_ppo = False  # 早停标志
        # 用于日志统计的变量
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        kl_sum = 0.0
        kl_ref_sum = 0.0
        clipfrac_sum = 0.0
        aux_loss_sum = 0.0
        log_count = 0

        # 解包模型（处理DDP包装）
        actor_unwrapped = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
        critic_unwrapped = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model

        # PPO多次更新
        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo:
                break
            # 随机打乱样本顺序
            b_inds = torch.randperm(B, device=args.device)

            # Mini-batch更新
            for i in range(0, B, mb_size):
                inds = b_inds[i:i + mb_size]

                # 6.1 获取当前策略的value估计
                mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])

                # 6.2 获取当前策略的log概率（在混合精度上下文中）
                with autocast_ctx:
                    res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    # MoE辅助损失（如果使用MoE）
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                    # 计算响应部分的log概率
                    mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2,
                                                                                    labels[inds].unsqueeze(-1)).squeeze(
                        -1).gather(1, logp_pos[inds])

                # 6.3 计算概率比 r(θ) = π_θ / π_old
                log_ratio = mb_resp_logp - old_resp_logp[inds]

                # 调试：检查log_ratio的量级（确保ratio≈1）
                if args.debug_log_ratio and ppo_epoch == 0 and i == 0 and is_main_process():
                    _lr = log_ratio.detach()
                    _m = resp_policy_mask[inds].bool()
                    if _m.any():
                        _lrv = _lr[_m]
                        Logger(f"[DBG log_ratio] step={step} max|lr|={_lrv.abs().max().item():.6e} "
                               f"mean|lr|={_lrv.abs().mean().item():.6e} "
                               f"ratio_max={torch.exp(_lrv).max().item():.6f} "
                               f"ratio_min={torch.exp(_lrv).min().item():.6f} "
                               f"dropout={getattr(lm_config, 'dropout', None)} "
                               f"training={actor_unwrapped.training}")

                # 6.4 计算近似KL散度（用于早停判断）
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[
                    inds].sum().clamp(min=1)

                # 同步各卡的KL值（防止某卡早停导致死锁）
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)

                # 如果KL过大，提前停止更新
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True

                # 6.5 计算各项损失
                ratio = torch.exp(log_ratio)

                # Clip fraction：被裁剪的比例（用于监控）
                clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                            / resp_policy_mask[inds].sum().clamp(min=1))

                # KL散度惩罚（相对于参考模型）
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) -
                                   (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)

                # 6.5.1 PPO策略损失（带裁剪的代理目标）
                policy_loss = ((torch.max(-advantages[inds] * ratio,
                                          -advantages[inds] * torch.clamp(ratio,
                                                                          1.0 - args.clip_epsilon,
                                                                          1.0 + args.clip_epsilon))
                                * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                               + args.kl_coef * kl_ref_penalty)

                # 6.5.2 价值函数损失（带裁剪）
                value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                                              (torch.clamp(mb_resp_values,
                                                           old_resp_values[inds] - args.cliprange_value,
                                                           old_resp_values[inds] + args.cliprange_value) - returns[
                                                   inds]) ** 2)
                                    * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)

                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 6.6 反向传播
                # 如果早停，仍需要执行反向传播（保证DDP通信正常），但损失乘以0
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                else:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps

                loss.backward()

                # 记录损失值（用于日志）
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1

                # 6.7 梯度累积和优化器步进
                grad_accum_step += 1

                if grad_accum_step % args.accumulation_steps == 0:
                    # 梯度裁剪
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    # 更新参数
                    actor_optimizer.step()
                    critic_optimizer.step()
                    # 更新学习率
                    actor_scheduler.step()
                    critic_scheduler.step()
                    # 清零梯度
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

        # 处理剩余梯度（当grad_accum_step不是accumulation_steps的整数倍时）
        if grad_accum_step % args.accumulation_steps != 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # ====================================================================
        # 第7步：更新rollout引擎的策略
        # ====================================================================
        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(actor_model)

        # ====================================================================
        # 第8步：日志记录
        # ====================================================================
        if is_main_process():
            # 计算平均损失和指标
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr, critic_lr = actor_optimizer.param_groups[0]['lr'], critic_optimizer.param_groups[0]['lr']

            # 记录到wandb
            if wandb is not None:
                wandb.log({
                    "reward": reward_val,
                    "kl_ref": kl_ref_val,
                    "approx_kl": approx_kl_val,
                    "clipfrac": clipfrac_val,
                    "critic_loss": critic_loss_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                })

            # 打印训练日志
            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                   f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        # ====================================================================
        # 第9步：保存检查点
        # ====================================================================
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 保存Actor模型权重
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)

            # 使用lm_checkpoint保存完整状态（包括优化器、调度器等）
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model,
                          optimizer=actor_optimizer, epoch=epoch, step=step,
                          wandb=wandb, save_dir='../checkpoints',
                          scheduler=actor_scheduler, critic_model=critic_model,
                          critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        # ====================================================================
        # 第10步：清理中间变量（节省显存）
        # ====================================================================
        del enc, gen_out, completion_ids, responses_text, rewards, full_mask, values_seq, advantages
        del labels, resp_labels, resp_idx, resp_pad_mask, valid_resp, eos_mask, has_eos, eos_pos
        del resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_resp_logp
        del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values
        del prompt_lens, logp_pos


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == "__main__":
    # ========================================================================
    # 第1步：解析命令行参数
    # ========================================================================
    parser = argparse.ArgumentParser(description="llm106 PPO (Proximal Policy Optimization)")

    # 训练配置
    parser.add_argument("--save_dir", type=str, default="../../../llm_data/llm106_model/ppo", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")

    # 设备和精度
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")

    # 优化相关
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")

    # 模型架构参数
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")

    # PPO超参数
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")

    # 模型路径
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")

    # 日志和调试
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--debug_log_ratio", action="store_true", help="打印log_ratio差异量级")

    # Rollout引擎配置
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"],
                        help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========================================================================
    # 第2步：初始化分布式环境和随机种子
    # ========================================================================
    local_rank = init_distributed_mode()  # 初始化分布式训练
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))  # 设置随机种子

    # ========================================================================
    # 第3步：配置目录和模型参数
    # ========================================================================
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = Llm106Config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    # 检查是否有检查点可以恢复
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========================================================================
    # 第4步：设置混合精度上下文
    # ========================================================================
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========================================================================
    # 第5步：初始化wandb（日志记录）
    # ========================================================================
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========================================================================
    # 第6步：初始化模型和数据
    # ========================================================================
    base_weight = args.from_weight

    # 6.1 Actor模型（策略网络）
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)

    # 6.2 参考模型（用于KL散度约束，冻结参数）
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # 6.3 Critic模型（价值网络）
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)  # 加载权重（strict=False忽略不匹配的层）
    critic_model = critic_model.to(args.device)

    # 6.4 奖励模型
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)

    # 6.5 Rollout引擎（用于生成响应）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )

    # 6.6 数据集和数据加载器
    train_ds = RLHFDataset(args.data_path, tokenizer,
                           max_length=(args.max_seq_len + args.max_gen_len),
                           thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # 6.7 优化器
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)

    # 6.8 计算总步数并设置学习率调度器
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps,
                                        eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps,
                                         eta_min=args.critic_learning_rate / 10)

    # 6.9 恢复训练状态（如果存在检查点）
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========================================================================
    # 第7步：编译和分布式包装
    # ========================================================================
    # 7.1 使用torch.compile加速
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(actor_model)

    # 7.2 分布式数据并行包装
    if dist.is_initialized():
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
    rollout_engine.update_policy(actor_model)

    # ========================================================================
    # 第8步：开始训练
    # ========================================================================
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的epoch（确保shuffle的一致性）
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)

        # 创建数据加载器（支持跳过已训练的step）
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)

        # 执行训练
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model,
                            actor_scheduler, critic_scheduler, reward_model, start_step, wandb)
        else:
            ppo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model,
                            actor_scheduler, critic_scheduler, reward_model, 0, wandb)

    # ========================================================================
    # 第9步：清理分布式环境
    # ========================================================================
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()