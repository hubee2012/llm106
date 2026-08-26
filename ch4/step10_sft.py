# ============================================================================
# 导入必要的库
# ============================================================================
import os
import sys

__package__ = "trainer"  # 设置包名，用于模块导入

# 导入自定义模块
from ch2.dataset_sft import SFTDataset  # SFT数据集处理类

# 添加项目根目录到Python路径（解决导入问题）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 标准库和第三方库
import datasets  # noqa: F401  # Windows下pyarrow/torch DLL冲突的临时解决方案
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler
from ch3.step60_llmmodel import Llm106Model, init_model  # 基础模型和初始化函数
from ch3.LlmConfig import Llm106Config  # 模型配置类

warnings.filterwarnings('ignore')  # 忽略警告信息


# ============================================================================
# 训练一个epoch的函数
# ============================================================================
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    执行一个epoch的SFT训练

    SFT (Supervised Fine-Tuning) 是监督微调，使用标准的交叉熵损失
    训练模型学习对话/文本生成任务

    参数:
        epoch: 当前epoch编号
        loader: 数据加载器，提供(input_ids, labels)批次
        iters: 当前epoch的总迭代步数
        start_step: 起始步数（用于续训时跳过已训练的步数）
        wandb: wandb日志实例，用于记录训练指标

    训练流程:
        1. 动态调整学习率（使用余弦退火策略）
        2. 前向传播计算损失（包含主损失和辅助损失）
        3. 反向传播（支持梯度累积）
        4. 梯度裁剪防止梯度爆炸
        5. 优化器更新参数
        6. 定期记录日志和保存检查点
    """
    start_time = time.time()  # 记录epoch开始时间
    last_step = start_step  # 记录最后处理的步数

    # 遍历数据加载器
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # ====================================================================
        # 第1步：将数据移动到指定设备
        # ====================================================================
        input_ids = input_ids.to(args.device)  # [batch_size, seq_len]
        labels = labels.to(args.device)  # [batch_size, seq_len]
        last_step = step

        # ====================================================================
        # 第2步：动态调整学习率（余弦退火）
        # ====================================================================
        # 计算当前步数对应的学习率
        # 公式：从初始学习率逐渐降到接近0，遵循余弦曲线
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ====================================================================
        # 第3步：前向传播计算损失
        # ====================================================================
        with autocast_ctx:  # 混合精度上下文（加速训练，减少显存）
            # 模型前向传播
            res = model(input_ids, labels=labels)
            # 总损失 = 交叉熵损失 + 辅助损失（如MoE的负载均衡损失）
            loss = res.loss + res.aux_loss
            # 梯度累积：将损失除以累积步数，使得有效batch_size = batch_size * accumulation_steps
            loss = loss / args.accumulation_steps

        # ====================================================================
        # 第4步：反向传播
        # ====================================================================
        # 使用梯度缩放器进行混合精度的反向传播
        # 对于float16训练，梯度缩放可以防止下溢
        scaler.scale(loss).backward()

        # ====================================================================
        # 第5步：梯度累积达到指定步数时，更新参数
        # ====================================================================
        if step % args.accumulation_steps == 0:
            # 5.1 梯度裁剪（防止梯度爆炸）
            # unscale_ 用于将缩放后的梯度恢复，以便进行梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 5.2 优化器步进
            # step() 会检查梯度是否包含inf/nan，如果包含则跳过更新
            scaler.step(optimizer)
            # 更新梯度缩放器（动态调整缩放因子）
            scaler.update()

            # 5.3 清零梯度（set_to_none=True更高效）
            optimizer.zero_grad(set_to_none=True)

        # ====================================================================
        # 第6步：日志记录
        # ====================================================================
        if step % args.log_interval == 0 or step == iters:
            # 计算已消耗的时间
            spend_time = time.time() - start_time

            # 获取各项损失值
            current_loss = loss.item() * args.accumulation_steps  # 还原真实损失
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss  # 主损失（交叉熵）
            current_lr = optimizer.param_groups[-1]['lr']  # 当前学习率

            # 估算剩余时间（分钟）
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            # 打印训练日志
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, '
                f'epoch_time: {eta_min:.1f}min'
            )

            # 记录到wandb
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # ====================================================================
        # 第7步：保存检查点
        # ====================================================================
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 切换到评估模式（影响Dropout、BatchNorm等层）
            model.eval()

            # 构建保存路径（包含MoE后缀）
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'

            # 解包模型（处理DDP和torch.compile包装）
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)

            # 获取模型状态字典并转为半精度（节省存储空间）
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

            # 保存完整训练状态（包括优化器、梯度缩放器等，用于续训）
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints',
                scaler=scaler
            )

            # 恢复训练模式
            model.train()

            # 释放内存
            del state_dict

        # ====================================================================
        # 第8步：清理中间变量（节省显存）
        # ====================================================================
        del input_ids, labels, res, loss

    # ====================================================================
    # 第9步：处理剩余梯度（当step不是accumulation_steps的整数倍时）
    # ====================================================================
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == "__main__":
    # ========================================================================
    # 第1步：解析命令行参数
    # ========================================================================
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")

    # 训练配置参数
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")

    # 设备和精度配置
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型（bfloat16或float16）")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")

    # 优化相关配置
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")

    # 日志和保存配置
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # 模型架构参数
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")

    # 数据和模型路径
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str,
                        help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1],
                        help="是否自动检测&续训（0=否，1=是）")

    # 实验跟踪
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========================================================================
    # 第2步：初始化分布式环境和随机种子
    # ========================================================================
    # 初始化分布式训练（如果使用多GPU）
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    # 设置随机种子（确保可复现性）
    # 不同进程使用不同的种子（rank+42），保证数据shuffle的多样性
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========================================================================
    # 第3步：配置目录和模型参数
    # ========================================================================
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 创建模型配置对象
    lm_config = Llm106Config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )

    # 检查是否存在检查点（用于续训）
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========================================================================
    # 第4步：设置混合精度上下文
    # ========================================================================
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 如果使用CPU，则使用nullcontext（不进行自动混合精度）
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========================================================================
    # 第5步：初始化wandb（实验跟踪工具）
    # ========================================================================
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # 使用swanlab（国内版wandb替代）

        # 如果有检查点，获取之前的wandb_id以便续训
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None  # 如果存在ID则必须续训
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========================================================================
    # 第6步：初始化模型、数据和优化器
    # ========================================================================
    # 6.1 初始化模型
    # from_weight: 'pretrain'表示从预训练权重开始，'none'表示随机初始化
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    # 6.2 加载数据集
    # SFTDataset将原始文本转换为(input_ids, labels)对
    # labels是input_ids的偏移版本，用于计算交叉熵损失
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # 6.3 创建分布式采样器（用于多GPU训练时数据分片）
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # 6.4 梯度缩放器（用于float16混合精度训练）
    # 如果使用float16，启用梯度缩放以防止下溢
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    # 6.5 优化器
    # AdamW是Adam的变体，正确实现了权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========================================================================
    # 第7步：从检查点恢复训练状态（如果存在）
    # ========================================================================
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型权重
        model.load_state_dict(ckp_data['model'])
        # 恢复优化器状态（包括动量等）
        optimizer.load_state_dict(ckp_data['optimizer'])
        # 恢复梯度缩放器状态
        scaler.load_state_dict(ckp_data['scaler'])
        # 恢复训练进度
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'从检查点恢复：Epoch {start_epoch}, Step {start_step}')

    # ========================================================================
    # 第8步：编译和分布式包装
    # ========================================================================
    # 8.1 使用torch.compile进行图优化（提升训练速度）
    # 注意：torch.compile需要PyTorch 2.0+
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    # 8.2 分布式数据并行包装
    # 将模型复制到多个GPU，每个GPU处理一部分数据
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========================================================================
    # 第9步：开始训练循环
    # ========================================================================
    for epoch in range(start_epoch, args.epochs):
        # 9.1 设置分布式采样器的epoch（确保每个epoch的shuffle不同）
        if train_sampler:
            train_sampler.set_epoch(epoch)

        # 9.2 设置随机种子（每个epoch不同，增加数据多样性）
        setup_seed(42 + epoch)

        # 9.3 创建数据加载器（支持跳过已训练的step）
        # 随机打乱数据索引
        indices = torch.randperm(len(train_ds)).tolist()
        # 计算需要跳过的步数（续训时使用）
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 使用自定义的SkipBatchSampler来跳过已训练的batch
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)

        # 创建数据加载器
        # pin_memory=True可以加速数据从CPU到GPU的传输
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # 9.4 执行训练
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

        # 9.5 epoch结束后重置start_step（后续epoch从0开始）
        start_step = 0

    # ========================================================================
    # 第10步：清理分布式环境
    # ========================================================================
    if dist.is_initialized():
        dist.barrier()  # 等待所有进程完成
        dist.destroy_process_group()  # 销毁进程组

    Logger("训练完成！")