import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 上一级目录
sys.path.insert(0, parent_dir)  # 将父目录添加到Python路径
import logging
from tqdm import tqdm
import gc
from configs.llm_utils import llm_data_dir, llm_model_dir
sys.path.append('../')
