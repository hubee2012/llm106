
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
for extra in (project_root,parent_dir, parent_dir / "ch2", parent_dir / "configs", parent_dir / "ch3"):
    extra = str(extra)
    if extra not in sys.path:
        sys.path.insert(0, extra)
from utils import get_lr, Logger, lm_checkpoint, setup_seed, init_distributed_mode, is_main_process, SkipBatchSampler

# 导入datasets库（被注释忽略的F401警告），用于解决Windows上pyarrow/torch的DLL冲突问题
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse  # 命令行参数解析
import time  # 时间相关操作
import warnings  # 警告控制
import torch  # PyTorch深度学习框架
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器，用于创建空上下文
from torch import optim, nn  # 优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载器和分布式采样器
from dataset_sft import SFTDataset        # 来自 ch2/
from model_lora import save_lora, apply_lora  # LoRA相关功能：保存和应用LoRA
from step60_llmmodel import Llm106Model, init_model  # 基础模型和初始化函数
from LlmConfig import Llm106Config  # 模型配置类
from llm_utils import llm_data_dir          # 来自 configs/
# 忽略所有警告信息，保持输出整洁
warnings.filterwarnings('ignore')


# ==================== 训练一个epoch的函数 ====================
def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    执行一个epoch的训练

    参数:
        epoch: 当前epoch索引（从0开始）
        loader: 数据加载器
        iters: 当前epoch的总迭代步数
        lora_params: LoRA参数列表（用于梯度裁剪）
        start_step: 起始步数（用于续训）
        wandb: wandb日志对象（可选）
    """
    start_time = time.time()  # 记录epoch开始时间
    last_step = start_step  # 记录最后执行的步数

    # 遍历数据加载器，从start_step+1开始计数
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 将数据移动到指定设备（GPU/CPU）
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step  # 更新最后步数

        # ========== 动态学习率调度 ==========
        # 根据当前全局步数计算学习率（使用余弦退火或其他策略）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ========== 前向传播和损失计算 ==========
        with autocast_ctx:  # 混合精度上下文（自动混合精度训练）
            res = model(input_ids, labels=labels)  # 模型前向传播
            # 总损失 = 主损失（logits loss）+ 辅助损失（如MoE的负载均衡损失）
            loss = res.loss + res.aux_loss
            # 梯度累积：损失除以累积步数，使得小batch也能达到大batch的效果
            loss = loss / args.accumulation_steps

        # ========== 反向传播 ==========
        # 使用梯度缩放器进行反向传播（用于float16精度训练）
        scaler.scale(loss).backward()

        # ========== 梯度累积更新 ==========
        # 每accumulation_steps步更新一次参数
        if step % args.accumulation_steps == 0:
            # 梯度反缩放（将缩放后的梯度恢复为原始尺度）
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            # 优化器步进（更新参数）
            scaler.step(optimizer)
            # 更新梯度缩放器（动态调整缩放因子）
            scaler.update()
            # 清零梯度，set_to_none=True比zero_grad()更高效
            optimizer.zero_grad(set_to_none=True)

        # ========== 日志记录 ==========
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time  # 已耗时
            current_loss = loss.item() * args.accumulation_steps  # 恢复真实损失值
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0  # 辅助损失
            current_logits_loss = current_loss - current_aux_loss  # 主损失
            current_lr = optimizer.param_groups[-1]['lr']  # 当前学习率

            # 估算剩余时间（分钟）
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            # 打印训练信息
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')

            # 如果使用wandb，记录训练指标
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # ========== 模型保存 ==========
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()  # 切换到评估模式（影响Dropout等层）

            # 构建MoE后缀（如果使用MoE架构）
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # LoRA权重保存路径
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth'

            # 只保存LoRA权重（轻量级，不保存完整模型）
            save_lora(model, lora_save_path)

            # 保存完整的检查点（用于续训）
            lm_checkpoint(
                lm_config,
                weight=args.lora_name,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints'
            )

            model.train()  # 切回训练模式

        # ========== 清理显存 ==========
        del input_ids, labels, res, loss

    # ========== 处理最后一个不完整批次 ==========
    # 如果最后几步没有达到accumulation_steps，仍需更新参数
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="llm106 LoRA Fine-tuning")

    # 模型保存相关
    parser.add_argument("--save_dir", type=str, default="../../../llm_data/llm106_model/lora", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_medical",
                        help="LoRA权重名称(如lora_identity/lora_medical等)")

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")

    # 硬件相关
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")

    # 优化策略
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # 模型架构参数
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")

    # 数据和权重
    parser.add_argument("--data_path", type=str, default=llm_data_dir +"/sft_t2t_mini.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='../../../llm_data/llm106_model/pretrain_768_9900k.pth', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--tokenizer_path", type=str, default= "../ch3", help="训练数据路径")

    # 实验跟踪
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="llm106_lora", help="wandb项目名")

    # 性能优化
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")

    args = parser.parse_args()

    # ========== 1. 初始化分布式环境和随机种子 ==========
    local_rank = init_distributed_mode()  # 初始化分布式训练模式，返回本地rank
    # 如果分布式已初始化，设置设备为对应的GPU
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    # 设置随机种子以保证可复现性（不同进程使用不同种子避免同步问题）
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查检查点 ==========
    os.makedirs(args.save_dir, exist_ok=True)  # 创建保存目录
    # 创建模型配置对象
    lm_config = Llm106Config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    # 如果需要续训，加载检查点信息
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度训练 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 自动混合精度上下文：CPU上不使用，GPU上根据dtype设置
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配置wandb实验跟踪 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # 使用SwanLab（类似wandb的开源替代）

        # 如果是续训，使用之前的wandb run ID
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"llm106-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config, args.from_weight,tokenizer_path=args.tokenizer_path,  device=args.device)
    # 对模型应用LoRA（在指定层添加低秩适配器）
    apply_lora(model)

    # ========== 统计模型参数 ==========
    total_params = sum(p.numel() for p in model.parameters())  # 总参数量
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)  # LoRA参数量
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

    # ========== 冻结非LoRA参数，收集LoRA参数 ==========
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:  # LoRA参数可训练
            param.requires_grad = True
            lora_params.append(param)
        else:  # 原始模型参数冻结
            param.requires_grad = False

    # ========== 6. 定义数据集和优化器 ==========
    # 创建SFT数据集（监督微调）
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 分布式采样器（用于多卡训练时数据分片）
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 梯度缩放器（用于float16混合精度）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # AdamW优化器（只优化LoRA参数）
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    # ========== 7. 从检查点恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 加载模型、优化器、缩放器的状态
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']  # 从上次保存的epoch继续
        start_step = ckp_data.get('step', 0)  # 从上次保存的step继续

    # ========== 8. 编译和分布式包装 ==========
    if args.use_compile == 1:
        # torch.compile与monkey-patch forward不兼容，自动关闭
        args.use_compile = 0
        Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')

    # 分布式数据并行包装（多卡训练）
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 9. 开始训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的epoch（确保不同epoch的数据shuffle不同）
        train_sampler and train_sampler.set_epoch(epoch)
        # 设置随机种子并生成随机索引（用于数据shuffle）
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()

        # 如果是续训且是起始epoch，计算需要跳过的步数
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 创建跳过指定步数的批次采样器
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)

        # 创建数据加载器
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True  # 加速数据从CPU到GPU的传输
        )

        # 如果有跳过的步数，记录日志并开始训练
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)

    # ========== 10. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.barrier()  # 等待所有进程完成
        dist.destroy_process_group()  # 销毁进程组