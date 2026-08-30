import torch

# 最终保底方案
class SimpleRuleReward:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def get_score(self, messages, answer):
        score = 0.0
        # 长度
        if 20 <= len(answer) <= 800:
            score += 0.5
        # 多样性
        words = answer.split()
        if words and len(set(words)) / len(words) > 0.4:
            score += 0.3
        # 完整性
        if "。" in answer:
            score += 0.2
        return torch.tensor(score, device=self.device)


# reward_model = SimpleRuleReward(device=args.device, dtype=torch.float16)