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
# from dataset_rlhf import RLHFDataset  # RLHF数据集处理
from dataset_dpo import DPODataset  # 导入DPO数据集处理类

from LlmConfig import Llm106Config  # 模型配置类
from step60_llmmodel import Llm106Model, init_model  # 基础模型和初始化函数
from rollout_engine import create_rollout_engine  # 生成引擎（用于采样）
from utils import is_main_process, Logger, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, get_lr
from llMForRewardModel import LMForRewardModel
from llm_utils import llm_data_dir

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """
    将模型的logits输出转换为每个token的对数概率

    参数:
        logits: 模型输出，shape (batch_size, seq_len, vocab_size)
        labels: 真实token标签，shape (batch_size, seq_len)

    返回:
        log_probs_per_token: 每个token的对数概率，shape (batch_size, seq_len)
    """
    # 在vocab维度上计算log_softmax，得到每个token的对数概率
    log_probs = F.log_softmax(logits, dim=2)
    # 根据labels索引收集对应token的对数概率
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算DPO (Direct Preference Optimization) 损失

    DPO的核心思想：直接优化策略模型，使其对chosen样本的偏好概率高于rejected样本

    参数:
        ref_log_probs: 参考模型的对数概率，shape (batch_size, seq_len)
        policy_log_probs: 策略模型的对数概率，shape (batch_size, seq_len)
        mask: 有效token的掩码，shape (batch_size, seq_len)
        beta: 温度参数，控制对偏好的敏感度

    返回:
        loss: DPO损失值（标量）
    """
    # 应用mask，只计算有效token的log概率之和
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将chosen和rejected数据分开（batch的前半部分是chosen，后半部分是rejected）
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]  # 参考模型对chosen的log概率
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]  # 参考模型对rejected的log概率
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]  # 策略模型对chosen的log概率
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]  # 策略模型对rejected的log概率

    # 计算策略模型的偏好差异（chosen - rejected）
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    # 计算参考模型的偏好差异
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    # 计算DPO的logits：策略模型与参考模型的偏好差异
    logits = pi_logratios - ref_logratios
    # 使用sigmoid损失：最大化策略模型对chosen的偏好概率
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """
    训练一个epoch的DPO

    参数:
        epoch: 当前epoch编号（从0开始）
        loader: 数据加载器
        iters: 当前epoch的总迭代步数
        ref_model: 参考模型（冻结，不更新）
        lm_config: 语言模型配置
        start_step: 起始步数（用于续训）
        wandb: wandb日志对象
        beta: DPO的温度参数
    """
    start_time = time.time()
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step

        # 从batch中获取数据并转移到指定设备
        x_chosen = batch['x_chosen'].to(args.device)  # chosen样本的输入
        x_rejected = batch['x_rejected'].to(args.device)  # rejected样本的输入
        y_chosen = batch['y_chosen'].to(args.device)  # chosen样本的目标输出
        y_rejected = batch['y_rejected'].to(args.device)  # rejected样本的目标输出
        mask_chosen = batch['mask_chosen'].to(args.device)  # chosen样本的掩码
        mask_rejected = batch['mask_rejected'].to(args.device)  # rejected样本的掩码

        # 将chosen和rejected拼接在一起，方便批量处理
        x = torch.cat([x_chosen, x_rejected], dim=0)#按batch_size合并，只是放在同一个批次
        y = torch.cat([y_chosen, y_rejected], dim=0)#
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 计算当前步的学习率（使用余弦退火或其他调度策略）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 混合精度上下文
        with autocast_ctx:
            # --- 参考模型前向传播（无梯度） ---
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits #[B, P+R, vocab_size]
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # --- 策略模型前向传播 ---
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            # 计算DPO损失
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            # 总损失 = DPO损失 + 辅助损失（如MoE的负载均衡损失）
            loss = dpo_loss_val + outputs.aux_loss
            # 梯度累积：除以累积步数
            loss = loss / args.accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积达到指定步数时，更新参数
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 反缩放梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新scaler
            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        # 打印训练日志
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算剩余时间（分钟）
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            # 记录wandb日志
            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss,
                                 "learning_rate": current_lr, "epoch_time": eta_min})

        # 保存模型检查点
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 如果是分布式训练，获取原始模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 保存为半精度以节省空间
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存完整检查点（包含优化器状态等）
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 释放显存（有助于避免OOM）
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss

    # 处理最后不足一个accumulation_steps的梯度
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="llm106 DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../../../llm_data/llm106_model/dpo", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default=llm_data_dir +"/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default="../../../llm_data/llm106_model/sft/full_sft_768_9900k.pth", type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="llm106-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化分布式环境和随机种子 ==========
    local_rank = init_distributed_mode()  # 初始化分布式训练
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))  # 设置随机种子确保可复现性

    # ========== 2. 配置目录、模型参数、检查检查点 ==========
    os.makedirs(args.save_dir, exist_ok=True)  # 创建保存目录
    lm_config = Llm106Config(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                             use_moe=bool(args.use_moe))
    # 如果需要续训，加载之前的检查点
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度训练 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配置wandb日志 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # 使用swanlab作为wandb的替代

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"llm106-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化策略模型和参考模型 ==========
    # 策略模型：需要训练的模型
    model, tokenizer = init_model(lm_config, args.from_weight,tokenizer_path='../ch3', device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')

    # 参考模型：用于计算DPO损失的基准，完全冻结
    ref_model, _ = init_model(lm_config, args.from_weight,tokenizer_path='../ch3', device=args.device)
    ref_model.eval()  # 设置为评估模式
    ref_model.requires_grad_(False)  # 冻结所有参数
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    # ========== 6. 准备数据加载器和优化器 ==========
    # 加载DPO训练数据集
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 分布式采样器
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 梯度缩放器（用于float16混合精度）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 优化器：使用AdamW
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 7. 从检查点恢复状态（续训） ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])  # 加载模型权重
        optimizer.load_state_dict(ckp_data['optimizer'])  # 加载优化器状态
        scaler.load_state_dict(ckp_data['scaler'])  # 加载scaler状态
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 8. 模型编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)  # 使用torch.compile加速
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # 使用DistributedDataParallel进行分布式训练
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 9. 开始训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置epoch用于分布式采样器
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)

        # 创建数据索引的随机排列
        indices = torch.randperm(len(train_ds)).tolist()
        # 如果是续训且当前是起始epoch，需要跳过已训练的steps
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 使用自定义的SkipBatchSampler跳过已训练的批次
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

        # 训练当前epoch
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    # ========== 10. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.barrier()  # 等待所有进程完成
        dist.destroy_process_group()  # 销毁进程组