import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer




class OpenAssistantRewardModel:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        # model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
        # model_name = "OpenAssistant/reward-model-deberta-v3-large"
        model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
        try:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=1, torch_dtype=dtype
            ).to(device).eval()
            self.use_model = True
            print(f"✅ 成功加载奖励模型: {model_name}")
        except Exception as e:
            print(f"⚠️ 加载失败: {e}")
            self.use_model = False
            self._init_fallback()

    def _init_fallback(self):
        """备选方案"""
        try:
            model_name = "bert-base-chinese"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=1, torch_dtype=self.dtype
            ).to(self.device).eval()
            self.use_model = True
            print(f"✅ 降级使用: {model_name}")
        except:
            self.use_model = False
            print("⚠️ 使用纯规则评分")

    def get_score(self, messages, answer):
        # 构建对话
        conv = ""
        for msg in messages:
            conv += f"{msg['role']}: {msg['content']}\n"
        conv += f"assistant: {answer}"

        if not self.use_model:
            return self._rule_score(answer)

        try:
            inputs = self.tokenizer(conv, return_tensors="pt",
                                    truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                score = self.model(**inputs).logits.squeeze().item()
            # 归一化到[-1, 1]
            return torch.tensor(score / 10, device=self.device)
        except:
            return self._rule_score(answer)

    def _rule_score(self, answer):
        score = 0.0
        if 50 <= len(answer) <= 500:
            score += 0.5
        if len(set(answer.split())) / max(1, len(answer.split())) > 0.3:
            score += 0.3
        return torch.tensor(score, device=self.device)


# reward_model = OpenAssistantRewardModel(device=args.device, dtype=torch.float16)


