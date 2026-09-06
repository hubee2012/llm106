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
for extra in (project_root, parent_dir, parent_dir / "ch2", parent_dir / "configs", parent_dir / "ch3"):
    extra = str(extra)
    if extra not in sys.path:
        sys.path.insert(0, extra)

# 导入自定义模块
from dataset_rlhf import RLHFDataset  # RLHF数据集处理
from LlmConfig import Llm106Config  # 模型配置类
from step60_llmmodel import Llm106Model, init_model  # 基础模型和初始化函数
from rollout_engine import create_rollout_engine  # 生成引擎（用于采样）
from utils import is_main_process, Logger, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler
from llMForRewardModel import LMForRewardModel
from llm_utils import llm_data_dir
from skyworkRewardModel import SkyworkRewardModel, SkyworkRewardModel_Local  # 假设上面的类保存在 skywork_reward.py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import re
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    """
    计算文本的重复惩罚分数（Repetition Penalty）

    Args:
        text (str): 输入文本
        n (int): n-gram的大小，默认为3
        cap (float): 惩罚分数的上限，默认为0.5

    Returns:
        float: 重复惩罚分数，值域为[0, cap]

    原理：计算文本中所有n-gram的重复比例，重复越多惩罚越高
    """
    # 提取所有单词和标点符号，转为小写
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    # 生成所有n-gram元组
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    # 计算重复比例并乘以cap*2，最后限制在cap以内
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    """
    计算每个生成响应的总奖励分数,一个prompt会生成num_generations个response

    Args:
        prompts (list[str]): 输入提示列表，长度为B
        responses (list[str]): 生成的响应列表，长度为B * num_generations
        reward_model: 奖励模型实例

    Returns:
        torch.Tensor: 奖励分数张量，长度为B * num_generations

    奖励组成：
    1. 长度奖励：鼓励长度在合理范围内
    2. 思考内容奖励：如果包含思考标签，奖励合理长度和正确格式
    3. 重复惩罚：惩罚重复内容
    4. 奖励模型评分：使用外部奖励模型打分
    """
    # 初始化奖励张量
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        # 遍历每个prompt及其对应的所有生成结果,prompts与response为一对多关系
        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 解析prompt中的消息格式（system/user/assistant）
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response

                # 1. 长度奖励：鼓励响应长度在20-800字符之间
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

                # 2. 思考内容奖励（如果有thinking标签）
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    # 思考内容长度在20-300之间奖励
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    # 确保只有一个</think>标签
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()

                # 3. 重复惩罚
                rewards[response_idx] -= rep_penalty(answer)

                # 4. 使用奖励模型评分
                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        # 将奖励模型分数转换为张量并加到总奖励中
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None,
                     use_sglang=False):
    """
    GRPO（Group Relative Policy Optimization）训练一个epoch

    Args:
        epoch (int): 当前epoch编号
        loader: 数据加载器
        iters (int): 总迭代步数
        rollout_engine: 采样引擎（用于生成响应）
        ref_model: 参考模型（用于计算KL散度）
        reward_model: 奖励模型
        start_step (int): 起始步数（用于恢复训练）
        wandb: wandb日志记录器
        use_sglang (bool): 是否使用SGLang引擎

    GRPO核心思想：
    1. 对每个prompt生成多个响应
    2. 计算每个响应的奖励
    3. 在同一组内归一化奖励（组相对优势）
    4. 使用PPO风格的目标函数优化策略
    5. 添加KL散度惩罚防止策略偏离参考模型太远
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        # ========== 1. 数据准备 ==========
        prompts = batch['prompt']  # list[str], 长度为B
        # 对prompts进行tokenization，左填充
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        # 如果指定了最大序列长度，截断prompt
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ========== 2. 使用Rollout引擎生成响应 ==========
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,  # 每个prompt生成多少个响应
            max_new_tokens=args.max_gen_len,
            temperature=0.8,  # 采样温度
        )
        outputs = rollout_result.output_ids  # 完整的prompt+response token IDs
        completion_ids = rollout_result.completion_ids  # 只包含response部分的token IDs
        completions = rollout_result.completions  # 解码后的response文本
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()  # 旧策略的每个token log概率
        prompt_lens = rollout_result.prompt_lens.to(args.device)  # 每个prompt的长度
        full_mask = (outputs != tokenizer.pad_token_id).long()  # 非padding mask
        # 计算每个response token在完整序列中的位置索引
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)

        # ========== 3. 计算奖励 ==========
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

        # ========== 4. 计算当前策略的log概率 ==========
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(outputs, attention_mask=full_mask)
            # 如果使用MoE，获取辅助损失
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            # 计算每个response token的log概率
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2,
                                                                                  outputs[:, 1:].unsqueeze(-1)).squeeze(
                -1).gather(1, logp_pos)

        # ========== 5. 计算参考模型的log概率 ==========
        with torch.no_grad():
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :],
                                                dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1,
                                                                                                                   logp_pos)

        # ========== 6. 调试模式：打印样本 ==========
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-' * 100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('=' * 100)

        # ========== 7. 计算优势函数（组相对归一化）==========,GRPO的创新！不是绝对分数，而是组内相对优势
        grouped_rewards = rewards.view(-1, args.num_generations)  # 重塑为 [B, num_generations]，每行是一个prompt的所有响应，按prompt分组
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*num_gen]
        # 在同一组内进行归一化，得到相对优势
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]

        # ========== 8. 构建完成mask（只计算到EOS为止）==========
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # 只计算EOS之前的token，之后的token不计入loss
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0),
                                                                                    -1) <= eos_idx.unsqueeze(
            1)) & completion_pad_mask).int()  # [B*num_gen, R]

        # ========== 9. 计算KL散度和策略损失 ==========
        # KL散度：D_KL(π_ref || π_θ) = exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]

        # 重要性采样比例：π_θ / π_old
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]

        if args.loss_type == "cispo":
            # CISPO：对比例进行截断，只限制上界
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()  #重要性采样比率，截断重要性采样比率的上界：只限制ratio不超过epsilon_high;detach()阻断梯度传播，让clamped_ratio作为常数参与计算
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)# 策略优化目标 - KL散度惩罚
        else:
            # 标准GRPO：使用PPO风格的clip损失,每个token的损失
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            # 目标：最大化优势项 - β * KL散度
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)

        # 计算最终的策略损失（按token平均，然后按序列平均）,序列级别的损失（按有效token平均）
        # 虽然最终policy_loss是标量，但反向传播时，每个响应的梯度贡献不,每个响应的梯度贡献是独立计算的
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()

        # 总损失 = 策略损失 + 辅助损失（MoE的负载均衡损失）
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # 缩放用于梯度累积
        loss.backward()

        # ========== 10. 梯度累积和优化器更新 ==========
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ========== 11. 日志记录 ==========
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(
                completion_mask.sum().item(), 1)
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            # 记录到wandb
            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        # ========== 12. 保存模型检查点 ==========
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # ========== 13. 更新Rollout引擎的策略模型 ==========
        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(model)

        # ========== 14. 清理显存 ==========
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos

    # 处理剩余的梯度累积
    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="llm106 GRPO (Group Relative Policy Optimization)")

    # 训练基本配置
    parser.add_argument("--save_dir", type=str, default="../../../llm_data/llm106_model/grpo", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")

    # 优化器配置
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")

    # 模型架构配置
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")

    # 序列长度配置
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")

    # 数据和采样配置
    parser.add_argument("--data_path", type=str, default=llm_data_dir + "/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")

    # GRPO算法参数
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")

    # 模型权重配置
    parser.add_argument('--from_weight', default='../../../llm_data/llm106_model/pretrain_768_9900k.pth', type=str,
                        help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str,
                        default="/home/hub/llm_data/llm106_model/Skywork-Reward-V2-Llama-3.2-1B", help="Reward模型路径")

    # 训练恢复
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")

    # Wandb日志
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="grpo-GRPO", help="wandb项目名")

    # 性能优化
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")

    # 调试
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")

    # 数据配置
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")

    # Rollout引擎配置
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"],
                        help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练环境
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    # 设置随机种子确保可重复性，不同进程使用不同种子
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 创建模型配置对象
    lm_config = Llm106Config(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                             max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    # 如果要从检查点恢复，加载检查点信息
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 自动混合精度上下文
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配置wandb日志 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # 使用swanlab替代wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"llm106-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight

    # Policy模型（要训练的模型）
    model, tokenizer = init_model(lm_config, from_weight=base_weight, tokenizer_path='../ch3', device=args.device)

    # Reference模型（固定参数，用于计算KL散度）
    ref_model, _ = init_model(lm_config, from_weight=base_weight, tokenizer_path='../ch3', device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)  # 冻结参数

    # Reward模型（外部奖励模型）
    reward_model = SkyworkRewardModel_Local(
        model_path=args.reward_model_path,  # 根据显存选择
        device=args.device,
        use_quantization=False,  # 显存不足可设为True
    )

    # Rollout引擎（负责生成响应，支持torch和sglang两种后端）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )

    # 数据集和优化器
    train_ds = RLHFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len,
                           thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 计算迭代次数
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # 总优化步数 = 总batch数 / 梯度累积步数
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    # 使用余弦退火学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ========== 6. 从检查点恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    # 使用torch.compile加速（PyTorch 2.0+特性）
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)

    # 分布式数据并行
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 更新rollout引擎的policy模型
    rollout_engine.update_policy(model)

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的epoch
        train_sampler and train_sampler.set_epoch(epoch)
        # 设置随机种子
        setup_seed(42 + epoch)

        # 创建随机打乱的索引列表
        indices = torch.randperm(len(train_ds)).tolist()
        # 如果是从中间恢复，计算需要跳过的步数
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 创建跳过指定步数的batch sampler
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

        # 执行训练epoch
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step,
                             wandb, use_sglang=(args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb,
                             use_sglang=(args.rollout_engine == "sglang"))

    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.barrier()  # 同步所有进程
        dist.destroy_process_group()  # 销毁进程组