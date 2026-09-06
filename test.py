import sys
from pathlib import Path

# 获取项目根目录
project_root = Path(__file__).resolve().parent.parent
print(f"项目根目录: {project_root}")

# 添加路径
paths_to_add = [
    project_root / 'configs',
    project_root / 'ch2',
]

for path in paths_to_add:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    print(f"添加路径: {path_str}")

# 打印所有 sys.path
print("\n=== sys.path 内容 ===")
for i, p in enumerate(sys.path):
    print(f"{i}: {p}")

# 检查文件是否存在
ch2_dir = project_root / 'ch2'
dataset_sft_file = ch2_dir / 'dataset_sft.py'
print(f"\n检查文件: {dataset_sft_file}")
print(f"文件存在: {dataset_sft_file.exists()}")

# 尝试导入
try:
    from dataset_sft import *
    print("✅ 导入成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")