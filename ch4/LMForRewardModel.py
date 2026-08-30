import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from internlm2_compat import (
    internlm2_score_forward_kwargs,
    normalize_internlm2_rope_scaling,
    patch_dynamic_cache_legacy_api,
    patch_internlm2_init_rope,
)


class LMForRewardModel:
    def __init__(self, model_path, device="cuda:0", dtype=torch.float16):
        self.device = device
        # Restore DynamicCache.from_legacy_cache before InternLM2 modeling loads.
        patch_dynamic_cache_legacy_api()

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        normalize_internlm2_rope_scaling(config)
        patch_internlm2_init_rope(config, model_path)

        if not hasattr(config, "attn_implementation") or config.attn_implementation is None:
            config.attn_implementation = "eager"
        if getattr(config, "return_dict", None) is None:
            try:
                config.return_dict = True
            except (AttributeError, TypeError):
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device).eval()

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages[:-1]]
        )
        last_query = messages[-1]["content"] if messages else ""
        message_context = (
            f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}"
            if history_text
            else last_query
        )
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response},
        ]
        score = self.model.get_score(
            self.tokenizer,
            eval_messages,
            **internlm2_score_forward_kwargs(),
        )
        return max(min(score, 3.0), -3.0)
