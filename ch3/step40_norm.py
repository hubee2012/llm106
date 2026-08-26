import os
import sys

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
from configs.llm_utils import llm_data_dir, llm_model_dir
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from typing import Tuple, Optional, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)