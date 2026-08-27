import math
import os

import random
import math
from logging import Logger

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer


def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None,
                  save_dir='../checkpoints', **kwargs):
    """
    语言模型检查点保存与加载函数
    支持分布式训练、混合精度、wandb集成和恢复检查点

    Args:
        lm_config: 语言模型配置对象，包含use_moe, hidden_size等属性
        weight: 权重类型标识，用于命名检查点文件，默认为'full_sft'
        model: 要保存的模型实例，为None时进入加载模式
        optimizer: 优化器实例，保存时会保存其状态
        epoch: 当前训练轮次
        step: 当前训练步数（全局步数）
        wandb: wandb运行实例，用于保存实验ID以便恢复
        save_dir: 检查点保存根目录
        **kwargs: 额外需要保存的组件，如学习率调度器、混合精度scaler等

    Returns:
        加载模式下返回检查点数据字典，保存模式下返回None
    """
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 构建检查点文件名，如果使用MoE（混合专家模型）则添加'_moe'后缀
    moe_path = '_moe' if lm_config.use_moe else ''
    # 主权重文件路径，包含模型参数
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    # 恢复检查点文件路径，包含训练状态（优化器、步数等）
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # ==================== 保存模式 ====================

        # 处理分布式封装：如果模型被DistributedDataParallel包装，提取原始模型
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        # 处理torch.compile：如果模型被编译，提取原始模型
        raw_model = getattr(raw_model, '_orig_mod', raw_model)

        # 获取模型状态字典（所有参数）
        state_dict = raw_model.state_dict()
        # 将参数转换为半精度并移到CPU以减少内存占用
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        # 使用临时文件原子性保存，防止保存过程中断导致文件损坏
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)  # 原子替换

        # 获取wandb运行ID以便恢复时重新连接同一个实验
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 构建恢复检查点数据（包含完整训练状态）
        resume_data = {
            'model': state_dict,  # 模型参数
            'optimizer': optimizer.state_dict(),  # 优化器状态
            'epoch': epoch,  # 当前epoch
            'step': step,  # 当前全局步数
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,  # 分布式训练进程数
            'wandb_id': wandb_id  # wandb运行ID
        }

        # 处理额外的需要保存的组件（如学习率调度器、梯度缩放器等）
        for key, value in kwargs.items():
            if value is not None:
                # 如果对象有state_dict方法，保存其状态
                if hasattr(value, 'state_dict'):
                    # 处理分布式封装和编译
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    # 否则直接保存值（如标量、配置等）
                    resume_data[key] = value

        # 原子性保存恢复检查点
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

        # 清理内存，释放显存
        del state_dict, resume_data
        torch.cuda.empty_cache()

    else:
        # ==================== 加载模式 ====================

        # 检查恢复检查点是否存在
        if os.path.exists(resume_path):
            # 加载检查点到CPU（避免显存溢出）
            ckp_data = torch.load(resume_path, map_location='cpu')

            # 处理分布式训练进程数变化的情况
            saved_ws = ckp_data.get('world_size', 1)  # 保存时的world_size
            current_ws = dist.get_world_size() if dist.is_initialized() else 1  # 当前的world_size

            if saved_ws != current_ws:
                # 调整步数以适应不同的batch大小
                # 步数 = 原步数 * 原进程数 / 当前进程数
                # 例如：从8卡变为4卡，总batch减半，步数应加倍
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')

            return ckp_data  # 返回恢复数据
        return None  # 没有检查点则返回None

def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')



def Logger(content):
    if is_main_process():
        print(content)



class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)

