# =============================================================================
# 导入必要的库
# =============================================================================
import os
import sys
from pathlib import Path

# 获取项目根目录 (llm106/)
project_root = Path(__file__).resolve().parent.parent

# 将所有需要的子目录添加到 sys.path，确保能够正确导入自定义模块
paths_to_add = [
    project_root / 'configs',
    project_root / 'ch2',
    project_root / 'ch3',
]
# 当执行 `python step10_sft.py` 时，sys.path 只包含 ch4/ 目录
# 必须把仓库根目录 llm106/ 加进去，因为 dataset_sft 内部使用了 `from ch2.dataset_utils import ...`
# 只添加 configs/ 或 ch2/、以及设置 `__package__ = "ch4"` 都不够
current_dir = Path(__file__).resolve().parent  # ch4/
parent_dir = current_dir.parent  # llm106/
for extra in (project_root, parent_dir, parent_dir / "ch2", parent_dir / "configs", parent_dir / "ch3"):
    extra = str(extra)
    if extra not in sys.path:
        sys.path.insert(0, extra)  # 插入到路径最前面，优先查找

# 导入自定义模块
from dataset_rlhf import RLHFDataset  # RLHF数据集处理
from LlmConfig import Llm106Config  # 模型配置类
from step60_llmmodel import Llm106Model, init_model  # 基础模型和初始化函数
from rollout_engine import create_rollout_engine  # 生成引擎（用于采样）
from utils import is_main_process, Logger, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, get_lr
from llm_utils import llm_data_dir
from skyworkRewardModel import SkyworkRewardModel, SkyworkRewardModel_Local
from dataset_sft import SFTDataset        # 来自 ch2/

# 再次添加父目录到路径（兼容性处理）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL冲突解决方法 (issue #771)
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


# =============================================================================
# 知识蒸馏损失函数
# =============================================================================
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """
    计算知识蒸馏的KL散度损失

    参数:
        student_logits: 学生模型的logits输出
        teacher_logits: 教师模型的logits输出
        temperature: 蒸馏温度参数，控制概率分布的平滑程度
        reduction: 损失聚合方式 ('batchmean', 'sum', 'none')
    返回:
        KL散度损失乘以温度平方
    """
    # 使用教师模型的softmax概率作为目标（不计算梯度）
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 学生模型的对数概率
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # 计算KL散度
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    # 乘以温度平方以补偿温度缩放的影响
    return (temperature ** 2) * kl


# =============================================================================
# 训练一个epoch的函数
# =============================================================================
def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0,
                temperature=1.0):
    """
    执行一个epoch的训练

    参数:
        epoch: 当前epoch编号
        loader: 数据加载器
        iters: 总迭代步数
        teacher_model: 教师模型
        lm_config_student: 学生模型配置
        start_step: 起始步数（用于断点续训）
        wandb: wandb日志记录器
        alpha: CE损失权重（总损失 = alpha*CE + (1-alpha)*KL）
        temperature: 蒸馏温度
    """
    start_time = time.time()
    last_step = start_step

    # 设置教师模型为评估模式，冻结参数
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    # 遍历数据批次
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        last_step = step
        input_ids = input_ids.to(args.device)  # 将数据移到指定设备
        labels = labels.to(args.device)
        loss_mask = (labels[..., 1:] != -100).float()  # 创建损失掩码，忽略填充位置

        # 计算当前学习率（使用余弦退火或线性衰减）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ========== 学生模型前向传播 ==========
        with autocast_ctx:  # 混合精度上下文
            res = model(input_ids)
            student_logits = res.logits[..., :-1, :].contiguous()  # 去掉最后一个token的预测

        # ========== 教师模型前向传播 ==========
        if teacher_model is not None:
            with torch.no_grad():  # 不需要梯度
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                # 如果学生和教师词汇表大小不同，截取教师logits到学生词汇表大小, 确保可以计算损失
                vocab_size_student = student_logits.size(-1)    #获取学生模型的词汇表大小
                teacher_logits = teacher_logits[..., :vocab_size_student]   #截取教师模型logits到学生词汇表大小,... 表示保留所有前面的维度,:vocab_size_student 表示只取前vocab_size_student个词表维度

        # ========== 计算损失 ==========
        # 1) Ground-Truth 交叉熵损失,学生模型的预测分布 与 真实标签（ground truth） 之间的交叉熵损失;让学生模型同时从两个来源学习,真实数据+教师模型;只用CE Loss（传统训练）模型容易过拟合,只用KL Loss（纯蒸馏）失去了从真实数据中学习的机会;CE Loss: 学习新任务,KL Loss: 保持旧知识（通过教师模型回顾）
        shift_labels = labels[..., 1:].contiguous()  # 标签右移一位
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,  # 忽略填充位置
            reduction='none'
        )
        # 对有效token取平均
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        # 如果使用MoE，加上辅助损失（负载均衡损失）
        if lm_config_student.use_moe:
            ce_loss = ce_loss_raw + res.aux_loss
        else:
            ce_loss = ce_loss_raw

        # 2) 蒸馏损失（KL散度）
        if teacher_model is not None:
            # 只在有效token上计算蒸馏损失
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        # 反向传播（使用梯度缩放器处理混合精度）
        scaler.scale(loss).backward()

        # 梯度累积：达到累积步数时更新参数
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 反缩放梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪,在蒸馏训练中，梯度可能特别容易爆炸;原因1：教师和学生模型可能规模差异大;原因2：蒸馏损失和CE损失的梯度方向可能冲突;原因3：温度参数会影响梯度大小
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新梯度缩放器
            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        # ========== 打印日志 ==========
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60  # 预估剩余时间

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            # 记录到wandb
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # ========== 保存模型检查点 ==========
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            # 获取原始模型（处理DDP和compile包装）
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 保存为半精度以节省空间
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存完整检查点（包含优化器状态等，用于续训）
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 释放显存
        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss

    # 处理最后一批次可能的梯度累积
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == "__main__":
    # ========== 1. 解析命令行参数 ==========
    parser = argparse.ArgumentParser(description="llm106 Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../../../llm_data/llm106_model/distillation", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default=llm_data_dir +"/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--student_hidden_size', default=768, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=8, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--student_use_moe', default=0, type=int, choices=[0, 1], help="学生模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--teacher_use_moe', default=1, type=int, choices=[0, 1], help="教师模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--from_student_weight', default="../../../llm_data/llm106_model/sft/full_sft_768_9900k.pth", type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default="../../../llm_data/llm106_model/sft/full_sft_768_9900k.pth", type=str, help="教师模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="llm106-Distillation", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 2. 初始化分布式环境和随机种子 ==========
    local_rank = init_distributed_mode()  # 初始化分布式训练
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"  # 设置当前进程的设备
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))  # 设置随机种子

    # ========== 3. 配置目录、模型参数、检查检查点 ==========
    os.makedirs(args.save_dir, exist_ok=True)  # 创建保存目录

    # 创建学生模型配置
    lm_config_student = Llm106Config(hidden_size=args.student_hidden_size,
                                       num_hidden_layers=args.student_num_layers,
                                       use_moe=bool(args.student_use_moe))
    # 创建教师模型配置
    lm_config_teacher = Llm106Config(hidden_size=args.teacher_hidden_size,
                                       num_hidden_layers=args.teacher_num_layers,
                                       use_moe=bool(args.teacher_use_moe))

    # 如果启用续训，加载检查点信息
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 4. 设置混合精度训练 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU不支持自动混合精度，使用空上下文
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 5. 配置wandb日志 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # 使用swanlab替代wandb

        # 从检查点获取wandb_id以续训
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"llm106-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 6. 定义学生和教师模型 ==========
    # 初始化学生模型
    # model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    model, tokenizer = init_model(lm_config_student, from_weight=args.from_student_weight,tokenizer_path='../ch3',device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')

    # 初始化教师模型（冻结，不训练）
    # teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model, _ = init_model(lm_config_teacher, from_weight=args.from_teacher_weight,tokenizer_path='../ch3',device=args.device)

    teacher_model.eval()  # 设置为评估模式
    teacher_model.requires_grad_(False)  # 冻结参数
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')

    # ========== 7. 准备数据集 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)  # 创建训练数据集
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None  # 分布式采样器

    # ========== 8. 设置优化器和梯度缩放器 ==========
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))  # 仅float16需要梯度缩放
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)  # AdamW优化器

    # ========== 9. 从检查点恢复状态（续训） ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])  # 加载模型权重
        optimizer.load_state_dict(ckp_data['optimizer'])  # 加载优化器状态
        scaler.load_state_dict(ckp_data['scaler'])  # 加载梯度缩放器状态
        start_epoch = ckp_data['epoch']  # 恢复epoch
        start_step = ckp_data.get('step', 0)  # 恢复step

    # ========== 10. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)  # torch.compile加速（PyTorch 2.0+）
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])  # DDP分布式训练

    # ========== 11. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的epoch（确保shuffle一致）
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)  # 每个epoch重置随机种子

        # 打乱数据索引
        indices = torch.randperm(len(train_ds)).tolist()
        # 如果是续训，跳过已训练的steps
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)

        # 创建数据加载器
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)

        # 执行训练
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student,
                        start_step, wandb, args.alpha, args.temperature)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student,
                        0, wandb, args.alpha, args.temperature)

        # 重置start_step（只跳过第一个epoch的开头）
        start_step = 0

    # ========== 12. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.barrier()  # 同步所有进程
        dist.destroy_process_group()  # 销毁进程组