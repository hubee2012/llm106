import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 上一级目录
sys.path.insert(0, parent_dir)  # 将父目录添加到Python路径
sys.path.insert(1, parent_dir/'config')  # 将父目录添加到Python路径
sys.path.insert(2, parent_dir/'ch2')  # 将父目录添加到Python路径
sys.path.insert(3, parent_dir/'ch3')  # 将父目录添加到Python路径
import logging
from tqdm import tqdm
import gc
from configs.llm_utils import llm_data_dir, llm_model_dir
sys.path.append('../')

# from pathlib import Path
#
# # 获取当前文件所在目录 (ch4/)
# current_dir = Path(__file__).resolve().parent
# # 获取上级目录 (llm106/)
# parent_dir = current_dir.parent
# # 拼接 configs 路径
# configs_dir = parent_dir / 'configs'
#
# # 添加到系统路径
# sys.path.insert(0, str(configs_dir))