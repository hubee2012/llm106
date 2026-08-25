import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from pathlib import Path
import os
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))





llm_data_dir='/home/hub/llm_data/llm106_data'
llm_model_dir='/home/hub/llm_data/llm106_model'


def read_wandb_config(config_rel_path="/home/hub/llm_data/config.txt", levels_up=2):
    """
    从当前文件所在目录向上 levels_up 级目录下的 config_rel_path 文件中读取 wandb 配置。
    返回一个字典，包含所有以 'wandb_' 开头的键值对（键名转换为小写）。
    """
    # 获取当前文件所在目录的绝对路径
    current_dir = Path(__file__).resolve().parent
    # 向上 levels_up 级
    target_dir = current_dir
    for _ in range(levels_up):
        target_dir = target_dir.parent
    config_path = target_dir / config_rel_path

    wandb_config = {}
    if not config_path.exists():
        logging.warning(f"配置文件不存在: {config_path}")
        return wandb_config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # 跳过空行和注释（以#开头）
                if not line or line.startswith('#'):
                    continue
                # 尝试用 '=' 或 ':' 分隔
                if '=' in line:
                    key, value = line.split('=', 1)
                elif ':' in line:
                    key, value = line.split(':', 1)
                else:
                    continue  # 不符合格式的行跳过
                key = key.strip().lower()
                value = value.strip()
                # 只保留 wandb 相关的配置
                if key.startswith('wandb_'):
                    wandb_config[key] = value
    except Exception as e:
        logging.error(f"读取配置文件 {config_path} 时出错: {e}")

    return wandb_config

