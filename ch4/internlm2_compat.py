"""Compatibility helpers for InternLM2 custom modeling + newer transformers.

InternLM2's trust_remote_code modeling was written against transformers 4.x.
Two failures show up under transformers 5.x:

1. RoPE: InternLM2 ``_init_rope`` reads ``rope_scaling["type"]`` / ``["factor"]``.
   Official internlm2-1_8b-reward ships ``rope_scaling: null``. Transformers
   4.46+ / 5.x rewrites that into ``{"rope_type": "default"}`` (no ``factor``).

2. KV cache: InternLM2 ``forward`` still does::
       past_key_values = DynamicCache.from_legacy_cache(past_key_values)
   Transformers 5 removed ``from_legacy_cache`` / ``to_legacy_cache``, which
   crashes the first ``get_score`` call after weights load.

InternLM2 also reads deprecated ``config.use_return_dict``. Callers should pass
``return_dict=True`` (and ``use_cache=False`` for reward scoring) so that path
is never taken.
"""
import sys


def internlm2_compatible_rope_scaling(rope_scaling):
    """Return InternLM2 ``_init_rope``-compatible rope_scaling.

    None, empty, or transformers' injected ``{"rope_type": "default"}`` become
    None (original InternLM2 RoPE). Linear/dynamic scaling is reduced to the
    two keys InternLM2 validates: ``type`` and ``factor``.
    """
    if not isinstance(rope_scaling, dict) or not rope_scaling:
        return None
    rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
    if rope_type in (None, "default"):
        return None
    factor = rope_scaling.get("factor", 1.0)
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        factor = 1.0
    if factor < 1.0:
        factor = 1.0
    return {"type": str(rope_type), "factor": factor}


def normalize_internlm2_rope_scaling(config):
    """Rewrite ``config.rope_scaling`` so InternLM2 ``_init_rope`` can run."""
    current = getattr(config, "rope_scaling", None)
    if not isinstance(current, dict):
        current = getattr(config, "rope_parameters", None)
    compatible = internlm2_compatible_rope_scaling(
        current if isinstance(current, dict) else None
    )
    try:
        config.rope_scaling = compatible
    except (AttributeError, TypeError):
        object.__setattr__(config, "rope_scaling", compatible)
    # Property aliases (transformers 5.x) may keep serving rope_parameters.
    if getattr(config, "rope_scaling", None) != compatible:
        if hasattr(config, "rope_parameters"):
            try:
                config.rope_parameters = compatible
            except (AttributeError, TypeError):
                object.__setattr__(config, "rope_parameters", compatible)
        object.__setattr__(config, "rope_scaling", compatible)
    return config


def _from_legacy_cache(cls, past_key_values=None):
    """transformers 4.x ``DynamicCache.from_legacy_cache`` replacement."""
    if past_key_values is None:
        return cls()
    if isinstance(past_key_values, cls):
        return past_key_values
    try:
        return cls(past_key_values)
    except TypeError:
        cache = cls()
        for layer_idx, layer_cache in enumerate(past_key_values):
            key_states, value_states = layer_cache[:2]
            cache.update(key_states, value_states, layer_idx)
        return cache


def _to_legacy_cache(self):
    """transformers 4.x ``DynamicCache.to_legacy_cache`` replacement."""
    layers = getattr(self, "layers", None)
    if layers:
        legacy = []
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                continue
            legacy.append((keys, values))
        if legacy:
            return tuple(legacy)
    key_cache = getattr(self, "key_cache", None) or []
    value_cache = getattr(self, "value_cache", None) or []
    return tuple(zip(key_cache, value_cache))


def patch_dynamic_cache_legacy_api(cache_cls=None):
    """Restore ``from_legacy_cache`` / ``to_legacy_cache`` on DynamicCache.

    InternLM2 remote modeling calls these methods. Transformers 5 deleted them;
    ``DynamicCache(past_key_values)`` is the replacement constructor.

    Returns the patched class, or None if transformers is unavailable and no
    ``cache_cls`` was provided. Existing methods are left untouched.
    """
    if cache_cls is None:
        try:
            from transformers.cache_utils import DynamicCache as cache_cls
        except ImportError:
            return None
    if not hasattr(cache_cls, "from_legacy_cache"):
        cache_cls.from_legacy_cache = classmethod(_from_legacy_cache)
    if not hasattr(cache_cls, "to_legacy_cache"):
        cache_cls.to_legacy_cache = _to_legacy_cache
    return cache_cls


def patch_internlm2_init_rope(config, model_path):
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


def internlm2_score_forward_kwargs():
    """Kwargs InternLM2 ``get_score`` should forward into ``forward``.

    ``use_cache=False`` skips ``DynamicCache.from_legacy_cache`` (reward scoring
    does not need a KV cache). ``return_dict=True`` avoids InternLM2 reading
    deprecated ``config.use_return_dict``.
    """
    return {"use_cache": False, "return_dict": True}
