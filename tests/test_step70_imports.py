"""Regression tests for running ch3 scripts from the ch3/ directory."""
import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CH3 = REPO_ROOT / "ch3"
STEP20 = CH3 / "step20_embedding.py"
STEP60 = CH3 / "step60_llmmodel.py"
STEP70 = CH3 / "step70_pretrain.py"
PROJECT_PACKAGES = ("ch3", "utils", "configs")


def _is_sys_path_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in {"insert", "append"}:
        return False
    value = func.value
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "path"
        and isinstance(value.value, ast.Name)
        and value.value.id == "sys"
    )


def _contains_sys_path_mutation(node: ast.AST) -> bool:
    return any(_is_sys_path_call(child) for child in ast.walk(node))


def _bootstrap_source(source: str) -> str:
    """Keep os/sys imports, assignments, and sys.path updates; skip torch etc."""
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".", 1)[0]
            if top in PROJECT_PACKAGES:
                break
        if isinstance(node, ast.Import):
            names = {alias.name.split(".", 1)[0] for alias in node.names}
            if names <= {"os", "sys"}:
                keep.append(node)
            continue
        if isinstance(node, ast.ImportFrom):
            continue
        keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    return ast.unparse(module)


def _assert_path_setup_before_project_imports(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    path_setup_line = None
    project_import_line = None
    for node in tree.body:
        if path_setup_line is None and _contains_sys_path_mutation(node):
            path_setup_line = node.lineno
        if (
            project_import_line is None
            and isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".", 1)[0] in PROJECT_PACKAGES
        ):
            project_import_line = node.lineno
    assert path_setup_line is not None, f"{path.name} must add the repo root to sys.path"
    assert project_import_line is not None, f"{path.name} expected a project import"
    assert path_setup_line < project_import_line, (
        f"{path.name}: sys.path must be updated before importing {PROJECT_PACKAGES}"
    )


def _assert_bootstrap_discovers_packages(script: Path, packages) -> None:
    bootstrap = _bootstrap_source(script.read_text(encoding="utf-8"))
    checks = "\n".join(
        f"assert find_spec({name!r}) is not None, sys.path" for name in packages
    )
    probe = (
        f"__file__ = {str(script)!r}\n"
        + bootstrap
        + "\nfrom importlib.util import find_spec\n"
        + checks
        + "\nprint('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=CH3,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_sys_path_is_updated_before_project_imports():
    for path in (STEP20, STEP60, STEP70):
        _assert_path_setup_before_project_imports(path)


def test_step70_bootstrap_makes_ch3_discoverable_from_ch3_dir():
    _assert_bootstrap_discovers_packages(STEP70, ("ch3", "utils", "configs.llm_utils"))


def test_step20_sibling_import_finds_configs_from_ch3_dir():
    """Reproduce: from step20_embedding import ... -> from configs.llm_utils."""
    _assert_bootstrap_discovers_packages(STEP20, ("configs", "configs.llm_utils", "ch3"))


def test_step60_sibling_import_finds_ch3_and_utils_from_ch3_dir():
    _assert_bootstrap_discovers_packages(STEP60, ("ch3", "utils", "configs.llm_utils"))
