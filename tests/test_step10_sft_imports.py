"""Regression tests for running ch4/step10_sft.py from the ch4/ directory."""
import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CH4 = REPO_ROOT / "ch4"
CH2 = REPO_ROOT / "ch2"
STEP10 = CH4 / "step10_sft.py"
PROJECT_PACKAGES = ("ch2", "ch3", "utils", "configs")


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
            if top in PROJECT_PACKAGES or top == "dataset_sft":
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
        if project_import_line is None and isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".", 1)[0]
            if top in PROJECT_PACKAGES or top == "dataset_sft":
                project_import_line = node.lineno
    assert path_setup_line is not None, f"{path.name} must add the repo root to sys.path"
    assert project_import_line is not None, f"{path.name} expected a project import"
    assert path_setup_line < project_import_line, (
        f"{path.name}: sys.path must be updated before importing {PROJECT_PACKAGES}"
    )


def _assert_bootstrap_discovers_packages(script: Path, packages, cwd: Path) -> None:
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
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_dataset_sft_py_exists():
    assert (CH2 / "dataset_sft.py").is_file()
    assert (CH4 / "dataset_sft.py").is_file()
    assert not (CH2 / "dataset_sft.pyi").exists()


def test_sys_path_is_updated_before_project_imports():
    _assert_path_setup_before_project_imports(STEP10)


def test_step10_bootstrap_finds_dataset_sft_from_ch4_dir():
    """Reproduce: cd ch4 && python step10_sft.py -> from dataset_sft import *."""
    _assert_bootstrap_discovers_packages(
        STEP10,
        ("dataset_sft", "ch2.dataset_sft", "ch3", "utils"),
        cwd=CH4,
    )


def test_step10_star_import_resolves_sftdataset_from_ch4_dir():
    """Importing dataset_sft from ch4/ must expose SFTDataset without executing torch."""
    source = STEP10.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found_star_import = False
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "dataset_sft"
            and any(alias.name == "*" for alias in node.names)
        ):
            found_star_import = True
            break
    assert found_star_import, "step10_sft.py should `from dataset_sft import *`"

    spec_probe = r"""
from importlib.util import find_spec
spec = find_spec('dataset_sft')
assert spec is not None, sys.path
assert spec.origin is not None
print(spec.origin)
"""
    bootstrap = _bootstrap_source(source)
    probe = f"__file__ = {str(STEP10)!r}\n" + bootstrap + spec_probe
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=CH4,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    origin = Path(result.stdout.strip().splitlines()[-1]).resolve()
    assert origin in {(CH4 / "dataset_sft.py").resolve(), (CH2 / "dataset_sft.py").resolve()}
