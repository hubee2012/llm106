"""InternLM2 + transformers 5.x compatibility (no full model download)."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CH4 = REPO_ROOT / "ch4"
if str(CH4) not in sys.path:
    sys.path.insert(0, str(CH4))

from internlm2_compat import (  # noqa: E402
    _from_legacy_cache,
    _to_legacy_cache,
    internlm2_compatible_rope_scaling,
    internlm2_score_forward_kwargs,
    normalize_internlm2_rope_scaling,
    patch_dynamic_cache_legacy_api,
    patch_internlm2_init_rope,
)


class _FakeDynamicCache:
    """Mirrors transformers 5 DynamicCache enough to test the legacy shim."""

    def __init__(self, ddp_cache_data=None):
        self.layers = []
        if ddp_cache_data is not None:
            for key_states, value_states in ddp_cache_data:
                self.layers.append(SimpleNamespace(keys=key_states, values=value_states))

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        while len(self.layers) <= layer_idx:
            self.layers.append(SimpleNamespace(keys=None, values=None))
        self.layers[layer_idx].keys = key_states
        self.layers[layer_idx].values = value_states
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        if layer_idx >= len(self.layers) or self.layers[layer_idx].keys is None:
            return 0
        return 1


def test_reported_crash_from_legacy_cache_missing():
    """Reproduce: AttributeError: type object 'DynamicCache' has no attribute 'from_legacy_cache'."""

    class DynamicCache:
        def __init__(self, ddp_cache_data=None):
            self.layers = []

    past_key_values = None
    with pytest.raises(AttributeError, match="from_legacy_cache"):
        DynamicCache.from_legacy_cache(past_key_values)

    patch_dynamic_cache_legacy_api(DynamicCache)
    cache = DynamicCache.from_legacy_cache(past_key_values)
    assert isinstance(cache, DynamicCache)
    assert cache.to_legacy_cache() == ()


def test_from_legacy_cache_none_matches_internlm2_first_forward():
    """InternLM2 calls from_legacy_cache(None) when use_cache and past is None."""
    cache = _from_legacy_cache(_FakeDynamicCache, None)
    assert isinstance(cache, _FakeDynamicCache)
    assert cache.get_seq_length() == 0


def test_from_legacy_cache_tuple_uses_constructor():
    legacy = (("k0", "v0"), ("k1", "v1"))
    cache = _from_legacy_cache(_FakeDynamicCache, legacy)
    assert [layer.keys for layer in cache.layers] == ["k0", "k1"]
    assert [layer.values for layer in cache.layers] == ["v0", "v1"]


def test_patch_does_not_overwrite_existing_legacy_methods():
    class AlreadyHas:
        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            return "original"

        def to_legacy_cache(self):
            return "original-to"

    patch_dynamic_cache_legacy_api(AlreadyHas)
    assert AlreadyHas.from_legacy_cache(None) == "original"
    assert AlreadyHas().to_legacy_cache() == "original-to"


def test_to_legacy_cache_roundtrip_from_layers():
    cache = _FakeDynamicCache((("k0", "v0"), ("k1", "v1")))
    assert _to_legacy_cache(cache) == (("k0", "v0"), ("k1", "v1"))


def test_internlm2_style_forward_uses_patched_from_legacy_cache():
    """Reproduce modeling_internlm2.py: use_cache and not isinstance(past, Cache)."""
    patch_dynamic_cache_legacy_api(_FakeDynamicCache)
    use_cache = True
    past_key_values = None
    if use_cache and not isinstance(past_key_values, _FakeDynamicCache):
        past_key_values = _FakeDynamicCache.from_legacy_cache(past_key_values)
    assert isinstance(past_key_values, _FakeDynamicCache)
    assert past_key_values.get_seq_length() == 0


def test_score_forward_kwargs_skip_cache_and_deprecated_return_dict():
    kwargs = internlm2_score_forward_kwargs()
    assert kwargs["use_cache"] is False
    assert kwargs["return_dict"] is True


def _class_method(tree, class_name, method_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    return None


def test_lm_for_reward_model_get_score_passes_compat_kwargs():
    source = (CH4 / "LMForRewardModel.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    get_score = _class_method(tree, "LMForRewardModel", "get_score")
    assert get_score is not None
    text = ast.get_source_segment(source, get_score)
    assert "internlm2_score_forward_kwargs" in text
    assert "self.tokenizer" in text


def test_lm_for_reward_model_patches_cache_before_from_pretrained():
    source = (CH4 / "LMForRewardModel.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    init = _class_method(tree, "LMForRewardModel", "__init__")
    assert init is not None
    patch_line = None
    load_line = None
    for node in ast.walk(init):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name == "patch_dynamic_cache_legacy_api":
            patch_line = node.lineno
        if name == "from_pretrained" and load_line is None:
            load_line = node.lineno
    assert patch_line is not None, "must patch DynamicCache before loading InternLM2"
    assert load_line is not None
    assert patch_line < load_line
    assert "AutoTokenizer" in source


def test_normalize_default_rope_scaling_is_none():
    config = SimpleNamespace(rope_scaling={"rope_type": "default"})
    normalize_internlm2_rope_scaling(config)
    assert config.rope_scaling is None


def test_normalize_linear_rope_keeps_type_and_factor():
    config = SimpleNamespace(rope_scaling={"type": "linear", "factor": 2.0})
    normalize_internlm2_rope_scaling(config)
    assert config.rope_scaling == {"type": "linear", "factor": 2.0}


def test_compatible_rope_scaling_missing_factor_defaults():
    assert internlm2_compatible_rope_scaling({"type": "dynamic"}) == {
        "type": "dynamic",
        "factor": 1.0,
    }


def test_patch_internlm2_init_rope_is_noop_without_auto_map():
    patch_internlm2_init_rope(SimpleNamespace(auto_map=None), "/missing")
    patch_internlm2_init_rope(SimpleNamespace(), "/missing")


def test_dummy_reward_model_get_score_forwards_kwargs():
    """LMForRewardModel.get_score must forward use_cache=False, return_dict=True."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    captured = {}

    class DummyHF:
        def get_score(self, tokenizer, messages, **kwargs):
            captured["tokenizer"] = tokenizer
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return 1.25

    import LMForRewardModel as mod

    wrapper = object.__new__(mod.LMForRewardModel)
    wrapper.tokenizer = object()
    wrapper.model = DummyHF()
    score = wrapper.get_score(
        [{"role": "user", "content": "hi"}],
        "hello",
    )
    assert score == 1.25
    assert captured["kwargs"]["use_cache"] is False
    assert captured["kwargs"]["return_dict"] is True
    assert captured["tokenizer"] is wrapper.tokenizer
    assert captured["messages"][0]["role"] == "user"
    assert captured["messages"][1]["content"] == "hello"
