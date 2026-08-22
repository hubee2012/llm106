import math
import logging
import sys
import traceback
import typing
from datetime import timedelta

import torch

def get_lr(current_step, total_steps, lr):
    #学习率,以cos速度下降，相比1/N,前期下降慢，后期下降快
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))

# def is_main_process():
#     return not dist.is_initialized() or dist.get_rank() == 0