import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Union, Optional

# 设置国内镜像源（解决网络问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

#下载到本地
#pip install -U huggingface_hub
#export HF_ENDPOINT=https://hf-mirror.com
#hf download Skywork/Skywork-Reward-V2-Llama-3.2-1B --local-dir ./Skywork-Reward-V2-Llama-3.2-1B
#各尺寸模型地址https://hf-mirror.com/Skywork

class SkyworkRewardModel:
    """
    Skywork-Reward-V2 奖励模型封装类
    支持中英文评估，适用于RLHF训练
    """

    def __init__(
            self,
            model_name: str = "Skywork/Skywork-Reward-V2-Qwen3.2-1B",
            device: Optional[str] = None,
            use_quantization: bool = False,
            torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        初始化奖励模型

        Args:
            model_name: 模型名称，可选:
                - Skywork/Skywork-Reward-V2-Qwen3-0.6B  (最轻量，推荐)
                - Skywork/Skywork-Reward-V2-Qwen3-1.7B
                - Skywork/Skywork-Reward-V2-Llama-3.1-8B (性能最强)
                - agentlans/Skywork-Reward-V2-Llama-3.1-8B-8bit (8bit量化版)
            device: 设备 ('cuda', 'cpu')，默认自动检测
            use_quantization: 是否使用8bit量化（节省显存）
            torch_dtype: 数据类型
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        print(f"正在加载奖励模型: {model_name}")
        print(f"使用设备: {self.device}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )

        # 设置padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        if use_quantization:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                num_labels=1,
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                device_map="auto" if self.device == 'cuda' else None,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                num_labels=1,
            )
            if self.device == 'cpu':
                self.model = self.model.to(self.device)

        self.model.eval()
        print("✅ 模型加载完成")

    def get_score(
            self,
            prompts: Union[str, List[str]],
            responses: Union[str, List[str]],
            max_length: int = 2048,
            batch_size: int = 4,
    ) -> List[float]:
        """
        评估prompt-response对的质量分数

        Args:
            prompts: 单个prompt或prompt列表
            responses: 单个response或response列表
            max_length: 最大输入长度
            batch_size: 批处理大小

        Returns:
            scores: 分数列表，分数越高表示质量越好
        """
        # 统一转换为列表
        if isinstance(prompts, str):
            prompts = [prompts]
            responses = [responses]

        assert len(prompts) == len(responses), "prompts和responses数量必须一致"

        all_scores = []

        # 分批处理
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]

            # 构建对话格式（无system prompt）
            messages_list = []
            for prompt, response in zip(batch_prompts, batch_responses):
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                messages_list.append(messages)

            # Tokenize
            inputs = self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().tolist()

            # 如果是单个分数，确保是列表
            if isinstance(scores, float):
                scores = [scores]

            all_scores.extend(scores)

        return all_scores

    def evaluate_pair(
            self,
            prompt: str,
            response_a: str,
            response_b: str,
            max_length: int = 2048,
    ) -> dict:
        """
        评估两个回复的偏好对比

        Args:
            prompt: 提示词
            response_a: 回复A
            response_b: 回复B
            max_length: 最大输入长度

        Returns:
            dict: {
                'score_a': float,
                'score_b': float,
                'preferred': 'A' or 'B',
                'margin': float  # 分数差值
            }
        """
        scores = self.evaluate(
            prompts=[prompt, prompt],
            responses=[response_a, response_b],
            max_length=max_length,
        )

        score_a, score_b = scores[0], scores[1]

        return {
            'score_a': score_a,
            'score_b': score_b,
            'preferred': 'A' if score_a > score_b else 'B',
            'margin': abs(score_a - score_b),
        }

    def __call__(self, prompt: str, response: str, **kwargs) -> float:
        """使类可调用，方便快速评估单个样本"""
        return self.evaluate([prompt], [response], **kwargs)[0]



class SkyworkRewardModel_Local:
    """
    Skywork-Reward-V2 奖励模型封装类
    支持中英文评估，适用于RLHF训练
    """

    def __init__(
            self,
            model_path: str = "Skywork/Skywork-Reward-V2-Qwen3.2-1B",
            device: Optional[str] = None,
            use_quantization: bool = False,
            torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        初始化奖励模型

        Args:
            model_name: 模型名称，可选:
                - Skywork/Skywork-Reward-V2-Qwen3-0.6B  (最轻量，推荐)
                - Skywork/Skywork-Reward-V2-Qwen3-1.7B
                - Skywork/Skywork-Reward-V2-Llama-3.1-8B (性能最强)
                - agentlans/Skywork-Reward-V2-Llama-3.1-8B-8bit (8bit量化版)
            device: 设备 ('cuda', 'cpu')，默认自动检测
            use_quantization: 是否使用8bit量化（节省显存）
            torch_dtype: 数据类型
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path

        print(f"正在加载奖励模型: {model_path}")
        print(f"使用设备: {self.device}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 设置padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.float16,
        )

        self.model.eval()
        print("✅ 模型加载完成")

    def auto_align(self, data1: Union[str, List[str]], data2: Union[str, List[str]]) -> tuple:
        """
        自动对齐两个数据列表，将短的那个复制到和长的一样长

        Args:
            data1: 第一个数据（字符串或列表）
            data2: 第二个数据（字符串或列表）

        Returns:
            (aligned_data1, aligned_data2): 对齐后的两个列表
        """
        # 统一转换为列表
        if isinstance(data1, str):
            data1 = [data1]
        if isinstance(data2, str):
            data2 = [data2]

        len1, len2 = len(data1), len(data2)

        # 如果长度相同，直接返回
        if len1 == len2:
            return data1, data2

        # 将短的那个复制到和长的一样长
        if len1 < len2:
            # data1 较短，复制 data1
            repeat_times = (len2 + len1 - 1) // len1  # 向上取整
            data1 = data1 * repeat_times
            data1 = data1[:len2]  # 截取到和 data2 一样长
            print(f"🔄 自动对齐: 将 {len1} 个 prompts 复制为 {len2} 个")
        else:
            # data2 较短，复制 data2
            repeat_times = (len1 + len2 - 1) // len2  # 向上取整
            data2 = data2 * repeat_times
            data2 = data2[:len1]  # 截取到和 data1 一样长
            print(f"🔄 自动对齐: 将 {len2} 个 responses 复制为 {len1} 个")

        return data1, data2

    def get_score(
            self,
            prompts: Union[str, List[str]],
            responses: Union[str, List[str]],
            max_length: int = 2048,
            batch_size: int = 4,
    ) -> List[float]:
        """
        评估prompt-response对的质量分数

        Args:
            prompts: 单个prompt或prompt列表
            responses: 单个response或response列表
            max_length: 最大输入长度
            batch_size: 批处理大小

        Returns:
            scores: 分数列表，分数越高表示质量越好
        """
        # 统一转换为列表
        if isinstance(prompts, str):
            prompts = [prompts]
            responses = [responses]


        if len(prompts) != len(responses):
            prompts,responses=self.auto_align(prompts,responses)

        assert len(prompts) == len(responses), "prompts和responses数量必须一致"

        all_scores = []

        # 分批处理
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]

            # 构建对话格式（无system prompt）
            messages_list = []
            for prompt, response in zip(batch_prompts, batch_responses):
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                messages_list.append(messages)

            # Tokenize
            inputs = self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().tolist()

            # 如果是单个分数，确保是列表
            if isinstance(scores, float):
                scores = [scores]

            all_scores.extend(scores)
        if len(response)<1 :
            return 0
        return sum(all_scores)/len(response)

    def evaluate_pair(
            self,
            prompt: str,
            response_a: str,
            response_b: str,
            max_length: int = 2048,
    ) -> dict:
        """
        评估两个回复的偏好对比

        Args:
            prompt: 提示词
            response_a: 回复A
            response_b: 回复B
            max_length: 最大输入长度

        Returns:
            dict: {
                'score_a': float,
                'score_b': float,
                'preferred': 'A' or 'B',
                'margin': float  # 分数差值
            }
        """
        scores = self.evaluate(
            prompts=[prompt, prompt],
            responses=[response_a, response_b],
            max_length=max_length,
        )

        score_a, score_b = scores[0], scores[1]

        return {
            'score_a': score_a,
            'score_b': score_b,
            'preferred': 'A' if score_a > score_b else 'B',
            'margin': abs(score_a - score_b),
        }

    def __call__(self, prompt: str, response: str, **kwargs) -> float:
        """使类可调用，方便快速评估单个样本"""
        return self.evaluate([prompt], [response], **kwargs)[0]



# ========== 使用示例 ==========
if __name__ == "__main__":
    # 1. 初始化模型（自动使用镜像源）
    reward_model = SkyworkRewardModel(
        model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B",  # 轻量级，适合测试
        # model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B",  # 高性能
        use_quantization=False,
    )

    # 2. 单个评估
    prompt = "如何学习Python编程？"
    response1 = "你可以通过阅读官方文档、在线课程和练习项目来学习Python。"
    response2 = "学Python很简单，看几个视频就会了。"

    score1 = reward_model(prompt, response1)
    score2 = reward_model(prompt, response2)

    print(f"\n📊 评估结果:")
    print(f"回复1 (详细): {score1:.4f}")
    print(f"回复2 (简单): {score2:.4f}")
    print(f"偏好: {'回复1' if score1 > score2 else '回复2'}")

    # 3. 批量评估
    prompts = [
        "如何学习Python编程？",
        "什么是机器学习？",
        "推荐一本好书。"
    ]
    responses = [
        "通过阅读文档和练习项目来学习。",
        "机器学习是人工智能的一个分支。",
        "推荐《深入理解计算机系统》。"
    ]

    scores = reward_model.evaluate(prompts, responses)
    print(f"\n📊 批量评估分数: {[round(s, 4) for s in scores]}")

    # 4. 偏好对比
    result = reward_model.evaluate_pair(
        prompt="如何学习Python编程？",
        response_a="通过阅读官方文档、在线课程和练习项目。",
        response_b="看视频就学会了。"
    )
    print(f"\n📊 偏好对比:")
    print(f"回复A得分: {result['score_a']:.4f}")
    print(f"回复B得分: {result['score_b']:.4f}")
    print(f"偏好: {result['preferred']}")
    print(f"分数差: {result['margin']:.4f}")