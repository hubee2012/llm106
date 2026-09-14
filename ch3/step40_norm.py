
import math
import torch
from torch import nn
from transformers import AutoTokenizer
import argparse
__package__ = "ch3"

class RMSNorm(torch.nn.Module):
    # 初始化方法
    # dim: 归一化的维度大小（通常是特征维度）
    # eps: 防止除零的小常数，默认为 1e-5
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # 保存 eps 到实例属性，后续在 norm 计算中使用
        self.eps = eps
        # 定义可学习的缩放参数 weight，形状为 (dim,)，初始值全为 1
        self.weight = nn.Parameter(torch.ones(dim))

    # 定义归一化计算方法
    def norm(self, x):
        # 计算过程：
        # 1. x.pow(2)：对 x 逐元素求平方
        # 2. .mean(-1, keepdim=True)：在最后一个维度上求均值，并保持维度不变（便于广播）
        # 3. + self.eps：加上一个极小值，防止分母为 0
        # 4. torch.rsqrt(...)：计算平方根的倒数，即 1 / sqrt(...)
        # 5. x * ...：将输入 x 乘以该倒数，实现归一化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 计算过程：
        # 1. self.norm(x.float())：先将 x 转为 float32 类型，再执行归一化（避免低精度溢出）
        # 2. self.weight * ...：用可学习参数 weight 对归一化结果进行逐元素缩放
        # 3. .type_as(x)：将结果转换回输入 x 原本的数据类型（如 float16/bfloat16）
        return (self.weight * self.norm(x.float())).type_as(x)