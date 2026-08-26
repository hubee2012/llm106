"""Regression tests for running step70_pretrain.py from the ch3/ directory."""
import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STEP70 = REPO_ROOT / "ch3" / "step70_pretrain.py"


def _is_sys_path_mutation(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    if not isinstance(func, ast.Attribute) or func.attr not in {"insert", "append"}:
        return False
    value = func.value
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "path"
        and isinstance(value.value, ast.Name)
        and value.value.id == "sys"
    )


def _prelude_before_project_imports(source: str) -> str:
    """Return the top of step70_pretrain.py up to the first project import."""
    tree = ast.parse(source)
    cutoff = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".", 1)[0]
            if top in {"ch3", "utils", "configs"}:
                cutoff = node.lineno
                break
    assert cutoff is not None, "expected a project import in step70_pretrain.py"
    lines = source.splitlines(keepends=True)
    return "".join(lines[: cutoff - 1])


def test_sys_path_is_updated_before_ch3_imports():
    tree = ast.parse(STEP70.read_text(encoding="utf-8"))
    path_setup_line = None
    ch3_import_line = None
    for node in tree.body:
        if path_setup_line is None and _is_sys_path_mutation(node):
            path_setup_line = node.lineno
        if (
            ch3_import_line is None
            and isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "ch3" or node.module.startswith("ch3."))
        ):
            ch3_import_line = node.lineno
    assert path_setup_line is not None, "step70_pretrain.py must add the repo root to sys.path"
    assert ch3_import_line is not None, "expected a from ch3 ... import"
    assert path_setup_line < ch3_import_line, (
        "sys.path must be updated before importing ch3.* so "
        "`python step70_pretrain.py` works from the ch3/ directory"
    )


def test_step70_bootstrap_makes_ch3_discoverable_from_ch3_dir():
    """Reproduce `cd ch3 && python step70_pretrain.py` without running training."""
    prelude = _prelude_before_project_imports(STEP70.read_text(encoding="utf-8"))
    probe = (
        f"__file__ = {str(STEP70)!r}\n"
        + prelude
        + "\nfrom importlib.util import find_spec\n"
        "assert find_spec('ch3') is not None, sys.path\n"
        "assert find_spec('utils') is not None, sys.path\n"
        "assert find_spec('configs.llm_utils') is not None, sys.path\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=STEP70.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "ok"
