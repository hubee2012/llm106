import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

_ch4_dir = Path(__file__).resolve().parent
if str(_ch4_dir) not in sys.path:
    sys.path.insert(0, str(_ch4_dir))
from internlm2_compat import normalize_internlm2_rope_scaling


def _patch_internlm2_init_rope(config, model_path):
    """Re-normalize rope_scaling at InternLM2 attention init time.

    Newer transformers may re-inject ``{"rope_type": "default"}`` after we
    clean the config. Patching ``_init_rope`` makes loading succeed anyway.
    """
    auto_map = getattr(config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModel")
    if not class_ref:
        return
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        get_class_from_dynamic_module(class_ref, model_path, trust_remote_code=True)
    except Exception:
        return

    for mod in list(sys.modules.values()):
        attn = getattr(mod, "InternLM2Attention", None)
        if attn is None or getattr(attn, "_llm106_rope_patched", False):
            continue
        orig = getattr(attn, "_init_rope", None)
        if orig is None:
            continue

        def _init_rope(self, _orig=orig):
            normalize_internlm2_rope_scaling(self.config)
            return _orig(self)

        attn._init_rope = _init_rope
        attn._llm106_rope_patched = True
        for cls in getattr(mod, "INTERNLM2_ATTENTION_CLASSES", {}).values():
            if getattr(cls, "_init_rope", None) is orig:
                cls._init_rope = _init_rope
                cls._llm106_rope_patched = True


class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        normalize_internlm2_rope_scaling(config)
        _patch_internlm2_init_rope(config, model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
