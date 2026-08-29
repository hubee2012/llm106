"""Compatibility helpers for InternLM2 custom modeling + newer transformers.

InternLM2's trust_remote_code modeling reads::

    if self.config.rope_scaling is None:
        # original RoPE
    else:
        scaling_type = self.config.rope_scaling["type"]
        scaling_factor = self.config.rope_scaling["factor"]

Official internlm2-1_8b-reward ships ``rope_scaling: null``. Transformers
4.46+ / 5.x rewrites that into ``{"rope_type": "default"}`` (no ``factor``),
which makes InternLM2 treat scaling as enabled and crash with KeyError.
"""


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
