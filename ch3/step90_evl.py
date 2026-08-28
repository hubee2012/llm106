# # 安装
# git clone https://github.com/EleutherAI/lm-evaluation-harness
# cd lm-evaluation-harness && pip install -e .
# # 启动测试
# # 使用的数据集：ceval-valid/cmmlu/arc_easy/piqa/openbookqa/hellaswag/social_iqa # 查看支持的数据集：lm_eval ls tasks
# # 对于指令模型，评测时需要加上 --apply_chat_template；对于 gpt2 这类纯基座模型，则不需要。

# HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf \
#   --model_args pretrained="/home/hub/llm_data/llm106_model/hf_format",dtype=auto,trust_remote_code=True \
#   --tasks hellaswag \
#   --batch_size 4 \
#   --device cpu \
#   --trust_remote_code \
#   --apply_chat_template

import os
import sys
from pathlib import Path

# 获取项目根目录 (llm106/)
project_root = Path(__file__).resolve().parent.parent

# 将所有需要的子目录添加到 sys.path
paths_to_add = [
    project_root / 'configs',
    project_root / 'ch2',
    project_root / 'ch3',
]
# `python step10_sft.py` 时 sys.path 只有 ch4/。必须把仓库根目录 llm106/
# 加进去，因为 dataset_sft 内部是 `from ch2.dataset_utils import ...`。
# 只 insert configs/ 或 ch2/、以及 `__package__ = "ch4"`，都不够。
current_dir = Path(__file__).resolve().parent  # ch4/
parent_dir = current_dir.parent  # llm106/
for extra in (parent_dir, parent_dir / "ch2", parent_dir / "configs", parent_dir / "ch3"):
    extra = str(extra)
    if extra not in sys.path:
        sys.path.insert(0, extra)

import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from LlmConfig import Llm106Config
from step60_llmmodel import Llm106Model

# Add lm-evaluation-harness to path
sys.path.insert(0, "/home/hub/git_test/lm-evaluation-harness")

# 加载配置
model_path = "/home/hub/llm_data/llm106_model/hf_format"
config_file = os.path.join(model_path, "config.json")
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config_data = json.load(f)
        for key, value in config_data.items():
            if key in Llm106Config.__init__.__code__.co_varnames:
                continue
            setattr(Llm106Config, key, value)

# 注册模型
print("Registering LLM106 model...")
AutoConfig.register("llm106", Llm106Config)
AutoModel.register(Llm106Config, Llm106Model)
AutoModelForCausalLM.register(Llm106Config, Llm106Model)
print("✓ Model registered successfully!")

# ====== 定义要评测的任务列表 ======
# 使用公认的常识推理和知识评测数据集
TASKS = [
    "hellaswag",  # 常识推理
    "piqa",  # 物理常识推理
    "arc_easy",  # ARC 简单集
    "arc_challenge",  # ARC 挑战集（更难的推理）
    "openbookqa",  # 开放书本问答
    "social_iqa",  # 社会常识推理
    "winogrande",  # 指代消解
    "mmlu",  # 多任务语言理解（可选，数据量大）
    "ceval-valid",  # 中文评测（可选）
]

# 定义基准模型分数（需要根据实际基准模型调整）
# 这里以 GPT-2 (124M) 和 GPT-2 (350M) 作为参考
BENCHMARK_SCORES = {
    "hellaswag": {"GPT-2 (124M)": 0.38, "GPT-2 (350M)": 0.42, "GPT-2 (1.5B)": 0.49},
    "piqa": {"GPT-2 (124M)": 0.61, "GPT-2 (350M)": 0.64, "GPT-2 (1.5B)": 0.71},
    "arc_easy": {"GPT-2 (124M)": 0.45, "GPT-2 (350M)": 0.48, "GPT-2 (1.5B)": 0.52},
    "arc_challenge": {"GPT-2 (124M)": 0.22, "GPT-2 (350M)": 0.24, "GPT-2 (1.5B)": 0.28},
    "openbookqa": {"GPT-2 (124M)": 0.32, "GPT-2 (350M)": 0.35, "GPT-2 (1.5B)": 0.40},
    "social_iqa": {"GPT-2 (124M)": 0.45, "GPT-2 (350M)": 0.48, "GPT-2 (1.5B)": 0.52},
    "winogrande": {"GPT-2 (124M)": 0.50, "GPT-2 (350M)": 0.52, "GPT-2 (1.5B)": 0.55},
    "mmlu": {"GPT-2 (124M)": 0.25, "GPT-2 (350M)": 0.27, "GPT-2 (1.5B)": 0.30},
}

from lm_eval import simple_evaluate


def load_cached_results(cache_file="evaluation_cache.json"):
    """加载缓存的评测结果"""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_cached_results(results, cache_file="evaluation_cache.json"):
    """保存评测结果到缓存"""
    with open(cache_file, 'w') as f:
        json.dump(results, f, indent=2)


def run_evaluation(tasks, batch_size=4, device="cpu", limit=None, cache_file="evaluation_cache.json", force_rerun=[]):
    """运行评测并返回结果，支持缓存

    Args:
        tasks: 要评测的任务列表
        batch_size: 批次大小
        device: 设备 (cpu/cuda)
        limit: 每个任务的样本限制
        cache_file: 缓存文件路径
        force_rerun: 强制重新计算的任务列表
    """
    results = load_cached_results(cache_file)
    completed_tasks = set(results.keys())

    # 过滤出需要计算的任务
    tasks_to_run = []
    for task in tasks:
        if task in force_rerun:
            tasks_to_run.append(task)
            print(f"⏳ Force re-running {task} (cached score will be overwritten)")
        elif task in completed_tasks and results.get(task) is not None:
            print(f"⏭️  Using cached result for {task}: {results[task]:.4f}")
        else:
            tasks_to_run.append(task)

    if not tasks_to_run:
        print("✅ All tasks have cached results!")
        return results

    print(f"\n📊 Running evaluation for {len(tasks_to_run)} tasks: {tasks_to_run}")

    # 运行需要评测的任务
    new_results = {}
    for task in tasks_to_run:
        print(f"\n{'=' * 50}")
        print(f"Running evaluation on: {task}")
        print(f"{'=' * 50}")

        try:
            result = simple_evaluate(
                model="hf",
                model_args=f"pretrained={model_path},dtype=auto,trust_remote_code=True",
                tasks=[task],
                batch_size=batch_size,
                device=device,
                limit=limit,  # 可设置限制加快测试
                apply_chat_template=True,  # 指令模型需要
            )

            # 提取准确率
            if task in result.get("results", {}):
                acc = result["results"][task].get("acc,none", None)
                if acc is None:
                    acc = result["results"][task].get("acc", None)
                if acc is None:
                    acc = result["results"][task].get("accuracy", None)

                if acc is not None:
                    new_results[task] = float(acc)
                    print(f"✓ {task}: {acc:.4f}")
                else:
                    print(f"⚠ Could not extract accuracy for {task}")
                    print(f"Available metrics: {list(result['results'][task].keys())}")
                    new_results[task] = None
            else:
                print(f"⚠ No results for {task}")
                new_results[task] = None

        except Exception as e:
            print(f"✗ Error on {task}: {e}")
            new_results[task] = None

    # 合并结果（保留旧结果，新结果覆盖）
    for task, score in new_results.items():
        results[task] = score

    # 保存缓存
    save_cached_results(results, cache_file)
    print(f"\n💾 Results cached to: {cache_file}")

    return results


def plot_radar_chart(model_scores, benchmark_scores, output_path="evaluation_radar.png"):
    """绘制雷达图对比"""

    # 获取所有任务
    tasks = [t for t in model_scores.keys() if model_scores[t] is not None]
    if not tasks:
        print("No valid scores to plot")
        return

    # 准备数据
    num_vars = len(tasks)
    angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

    # 绘制模型分数
    model_values = [model_scores[t] for t in tasks]
    model_values += model_values[:1]
    ax.plot(angles, model_values, 'o-', linewidth=2, label="LLM106 Model", color='blue')
    ax.fill(angles, model_values, alpha=0.25, color='blue')

    # 绘制基准模型
    colors = ['red', 'green', 'orange']
    for idx, (benchmark_name, scores) in enumerate(benchmark_scores.items()):
        if idx >= len(colors):
            break
        values = [scores.get(t, 0) for t in tasks]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=benchmark_name,
                color=colors[idx], linestyle='--', alpha=0.8)

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([t.replace('_', ' ').title() for t in tasks], size=10)

    # 设置Y轴范围
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=8)

    # 添加网格
    ax.grid(True)

    # 添加标题和图例
    plt.title('Model Performance Comparison (Accuracy)', size=16, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # 在雷达图中心添加信息
    ax.text(0, 0, f"n={len(tasks)}", ha='center', va='center', size=10, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Radar chart saved to: {output_path}")


def plot_bar_comparison(model_scores, benchmark_scores, output_path="evaluation_bar.png"):
    """绘制条形图对比（补充）"""

    tasks = [t for t in model_scores.keys() if model_scores[t] is not None]
    if not tasks:
        return

    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))

    # 模型分数
    model_values = [model_scores[t] for t in tasks]
    bars1 = ax.bar(x - width, model_values, width, label='LLM106 Model', color='blue', alpha=0.8)

    # 基准模型（取第一个）
    benchmark_names = list(benchmark_scores.keys())
    if benchmark_names:
        ref_name = benchmark_names[0]
        ref_values = [benchmark_scores[ref_name].get(t, 0) for t in tasks]
        bars2 = ax.bar(x, ref_values, width, label=ref_name, color='red', alpha=0.7)

    # 第二个基准模型
    if len(benchmark_names) > 1:
        ref_name2 = benchmark_names[1]
        ref_values2 = [benchmark_scores[ref_name2].get(t, 0) for t in tasks]
        bars3 = ax.bar(x + width, ref_values2, width, label=ref_name2, color='green', alpha=0.7)

    # 添加标签
    ax.set_xlabel('Tasks')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_', ' ').title() for t in tasks], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)

    # 添加数值标签
    for bars in [bars1]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Bar chart saved to: {output_path}")


def print_evaluation_summary(model_scores, results_file="evaluation_results.json"):
    """打印并保存评测结果汇总"""
    print("\n" + "=" * 60)
    print("Evaluation Results Summary:")
    print("=" * 60)
    print(f"{'Task':<20} {'Accuracy':<12} {'Status':<10}")
    print("-" * 45)

    successful = 0
    failed = 0

    for task, score in model_scores.items():
        if score is not None:
            print(f"{task:<20} {score:.4f}     ✓")
            successful += 1
        else:
            print(f"{task:<20} {'N/A':<12}     ✗")
            failed += 1

    print("-" * 45)
    print(f"Total: {len(model_scores)} tasks, {successful} successful, {failed} failed")

    # 保存结果到文件
    with open(results_file, "w") as f:
        json.dump(model_scores, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")

    return successful, failed


# ====== 主执行流程 ======
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("Starting Model Evaluation with Cache...")
    print("=" * 60)
    print(f"Tasks to evaluate: {len(TASKS)} tasks")
    print(f"Tasks: {TASKS}")
    print("=" * 60)

    # ===== 配置选项 =====
    # 强制重新计算的任务列表（如果某些任务失败或需要更新）
    FORCE_RERUN = []  # 例如: ["mmlu", "ceval-valid"]

    # 是否只使用缓存的分数（不进行任何新计算）
    USE_CACHE_ONLY = False  # 设为 True 则只加载缓存，不运行任何评测

    # 缓存文件路径
    CACHE_FILE = "evaluation_cache.json"

    # 可选：限制每个任务的样本数（用于快速测试）
    LIMIT = None  # 设为数字如 100 来快速测试，None 表示使用完整数据集

    if USE_CACHE_ONLY:
        print("\n📂 Loading cached results only...")
        model_scores = load_cached_results(CACHE_FILE)
        if not model_scores:
            print("⚠ No cached results found!")
            sys.exit(1)
        print(f"✅ Loaded {len(model_scores)} cached results")
    else:
        # 1. 评测模型（支持缓存）
        model_scores = run_evaluation(
            tasks=TASKS,
            batch_size=4,
            device="cpu",
            limit=LIMIT,
            cache_file=CACHE_FILE,
            force_rerun=FORCE_RERUN
        )

    # 2. 打印结果汇总
    print_evaluation_summary(model_scores, "evaluation_results.json")

    # 3. 绘制雷达图（跳过失败的任务）
    print("\n" + "=" * 60)
    print("Generating Radar Chart...")
    print("=" * 60)

    # 只使用成功计算的任务
    valid_scores = {k: v for k, v in model_scores.items() if v is not None}
    if len(valid_scores) >= 3:
        plot_radar_chart(valid_scores, BENCHMARK_SCORES, "evaluation_radar.png")
    else:
        print(f"⚠ Not enough valid scores ({len(valid_scores)}) to generate radar chart (need at least 3)")

    # 4. 绘制条形图（补充）
    if len(valid_scores) >= 1:
        plot_bar_comparison(valid_scores, BENCHMARK_SCORES, "evaluation_bar.png")
    else:
        print("⚠ No valid scores to generate bar chart")

    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print(f"Cache file: {CACHE_FILE}")
    print(f"Results file: evaluation_results.json")
    print("=" * 60)