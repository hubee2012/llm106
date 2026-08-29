"""InternLM2 reward-model RoPE config vs newer transformers."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
CH4 = REPO_ROOT / "ch4"

if str(CH4) not in sys.path:
    sys.path.insert(0, str(CH4))

from internlm2_compat import (  # noqa: E402
    internlm2_compatible_rope_scaling,
    normalize_internlm2_rope_scaling,
)


def _internlm2_init_rope(config):
    """Mirror InternLM2Attention._init_rope's rope_scaling lookups."""
    if config.rope_scaling is None:
        return "default"
    scaling_type = config.rope_scaling["type"]
    scaling_factor = config.rope_scaling["factor"]
    return scaling_type, scaling_factor


def test_injected_default_rope_type_becomes_none():
    assert internlm2_compatible_rope_scaling({"rope_type": "default"}) is None
    assert internlm2_compatible_rope_scaling({"type": "default"}) is None
    assert internlm2_compatible_rope_scaling(None) is None
    assert internlm2_compatible_rope_scaling({}) is None


def test_dynamic_scaling_keeps_type_and_factor():
    assert internlm2_compatible_rope_scaling(
        {"rope_type": "dynamic", "factor": 2.0, "original_max_position_embeddings": 2048}
    ) == {"type": "dynamic", "factor": 2.0}
    assert internlm2_compatible_rope_scaling({"type": "linear", "factor": 1}) == {
        "type": "linear",
        "factor": 1.0,
    }


def test_missing_factor_does_not_keyerror():
    config = SimpleNamespace(rope_scaling={"rope_type": "default"})
    normalize_internlm2_rope_scaling(config)
    assert _internlm2_init_rope(config) == "default"


def test_normalize_rewrites_transformers_default_dict():
    config = SimpleNamespace(rope_scaling={"rope_type": "default", "rope_theta": 1000000.0})
    normalize_internlm2_rope_scaling(config)
    assert config.rope_scaling is None
    assert _internlm2_init_rope(config) == "default"


def test_normalize_aliases_rope_type_and_fills_factor():
    config = SimpleNamespace(rope_scaling={"rope_type": "dynamic"})
    normalize_internlm2_rope_scaling(config)
    assert _internlm2_init_rope(config) == ("dynamic", 1.0)


def test_raw_transformers_default_dict_keyerrors_without_normalize():
    # Transformers 4.46+ injects {"rope_type": "default"} which has neither
    # InternLM2 key. After BC copies rope_type -> type, the crash is KeyError: factor,
    # which matches the internlm2-1_8b-reward traceback.
    config = SimpleNamespace(rope_scaling={"rope_type": "default"})
    try:
        _internlm2_init_rope(config)
        raised = None
    except KeyError as exc:
        raised = exc
    assert raised is not None
    assert raised.args[0] in {"type", "factor"}

    config = SimpleNamespace(rope_scaling={"type": "default", "rope_type": "default"})
    try:
        _internlm2_init_rope(config)
        raised = None
    except KeyError as exc:
        raised = exc
    assert raised is not None
    assert raised.args[0] == "factor"
    normalize_internlm2_rope_scaling(config)
    assert _internlm2_init_rope(config) == "default"


def test_normalize_property_alias_to_rope_parameters():
    class Config:
        def __init__(self):
            self.rope_parameters = {"rope_type": "default", "rope_theta": 1e6}

        @property
        def rope_scaling(self):
            return self.rope_parameters

        @rope_scaling.setter
        def rope_scaling(self, value):
            self.rope_parameters = value

    config = Config()
    normalize_internlm2_rope_scaling(config)
    assert config.rope_scaling is None
    assert _internlm2_init_rope(config) == "default"


def test_step30_imports_local_reward_model_wrapper():
    source = (CH4 / "step30_ppo.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [alias.name for alias in node.names]
            imports.append((node.module, names))
    assert ("LMForRewardModel", ["LMForRewardModel"]) in imports
    assert all(module != "trainer.trainer_utils" for module, _ in imports)
