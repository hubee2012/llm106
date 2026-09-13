# 从 PyTorch 的 utils.data 模块导入 Dataset 基类
# 自定义数据集需要继承这个基类，并实现 __len__ 和 __getitem__ 方法
from torch.utils.data import Dataset

# 导入 PyTorch 主库，用于张量操作（如 torch.tensor、torch.long）
import torch

# 导入操作系统接口模块，用于设置环境变量
import os

# 导入随机数模块（当前代码中未实际使用，可能是预留）
import random

# 从 HuggingFace datasets 库导入 load_dataset 函数
# 用于方便地加载 json/jsonl 格式的数据文件
from datasets import load_dataset

# 设置环境变量 TOKENIZERS_PARALLELISM 为 "false"
# 关闭 tokenizer 的多线程并行，避免在 DataLoader 多进程时出现警告或死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 定义预训练数据集类，继承自 torch.utils.data.Dataset
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        # 调用父类 Dataset 的初始化方法
        super().__init__()
        # 保存传入的 tokenizer（用于将文本编码为 token id 序列）
        self.tokenizer = tokenizer
        # 保存最大序列长度，用于截断和填充
        self.max_length = max_length
        # 使用 datasets.load_dataset 加载 json 格式的数据文件
        # data_files 指定文件路径，split='train' 表示只取训练集部分
        # 加载后 self.samples 是一个 Dataset 对象（支持索引访问和 len）
        self.samples = load_dataset('json', data_files=data_path, split='train')

    # 返回数据集样本总数，DataLoader 需要用到
    def __len__(self):
        return len(self.samples)

    # 根据索引 index 返回一条处理后的样本 (input_ids, labels)
    def __getitem__(self, index):
        # 从加载的数据集中取出第 index 条样本（字典形式，至少包含 'text' 字段）
        sample = self.samples[index]

        # 使用 tokenizer 将样本的文本编码为 token id 列表：
        # - str(sample['text'])：确保文本是字符串类型
        # - add_special_tokens=False：不自动添加 BOS/EOS 等特殊 token（后面手动加）
        # - max_length=self.max_length - 2：预留 2 个位置给 BOS 和 EOS
        # - truncation=True：超出长度时截断
        # 取出 .input_ids 得到编码后的 token id 列表
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True
        ).input_ids

        # 在序列开头加上 BOS（句子起始）token，在结尾加上 EOS（句子结束）token
        # 这样模型可以学习到句子的开始和结束
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # 对序列进行填充（padding）：
        # 用 pad_token_id 填充到 max_length 长度，保证同一 batch 内所有样本等长
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

        # 将 Python 列表转换为 PyTorch 的 LongTensor（int64），dtype=torch.long
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 语言模型训练中，labels 通常与 input_ids 相同（自回归预测下一个 token）
        # 这里先克隆一份 input_ids 作为 labels
        labels = input_ids.clone()

        # 将 padding 位置的 label 设为 -100
        # 因为 CrossEntropyLoss 默认 ignore_index=-100，这样填充部分不计入 loss
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # 返回处理好的 input_ids 和 labels，供模型训练使用
        return input_ids, labels


# from torch.utils.data import Dataset
# import torch
# import os
# import random
# from datasets import load_dataset
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
#
# class PretrainDataset(Dataset):
#     def __init__(self, data_path, tokenizer, max_length=512):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.samples = load_dataset('json', data_files=data_path, split='train')
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, index):
#         sample = self.samples[index]
#         tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
#         tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
#         #token填充
#         input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
#         input_ids = torch.tensor(input_ids, dtype=torch.long)
#         labels = input_ids.clone()
#         labels[input_ids == self.tokenizer.pad_token_id] = -100
#         return input_ids, labels
#
