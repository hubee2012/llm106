"""
快速将训练好的 .pth 模型转换为 HuggingFace 格式
"""
import os
import sys
import torch
import json

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from LlmConfig import Llm106Config
from step60_llmmodel import init_model
from utils import is_main_process, Logger


def convert_pth_to_hf(pth_path, output_dir):
    """将 .pth 权重文件转换为 HuggingFace 格式"""
    # 1. 加载模型
    lm_config = Llm106Config(hidden_size=768, num_hidden_layers=8, use_moe=False)
    model, tokenizer = init_model(lm_config, pth_path, device="cpu")

    # 2. 保存为 HuggingFace 格式
    os.makedirs(output_dir, exist_ok=True)

    # 保存 config
    config_dict = {
        "model_type": "llm106",
        "hidden_size": lm_config.hidden_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "vocab_size": lm_config.vocab_size,
        "max_position_embeddings": lm_config.max_position_embeddings,
        "use_moe": lm_config.use_moe,
        "torch_dtype": "float32"
    }

    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # 保存权重
    state_dict = model.state_dict()
    torch.save({k: v.float().cpu() for k, v in state_dict.items()},
               f"{output_dir}/pytorch_model.bin")

    # 保存 tokenizer（如果有）
    if tokenizer:
        tokenizer.save_pretrained(output_dir)

    Logger(f"✅ Converted to HuggingFace format: {output_dir}")


if __name__ == "__main__":
    convert_pth_to_hf(
        pth_path="../../../llm_data/llm106_model/pretrain_768_9900k.pth",
        output_dir="../../../llm_data/llm106_model/hf_format"
    )