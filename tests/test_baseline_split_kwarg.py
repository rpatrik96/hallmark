"""Every registered baseline must tolerate the dispatch-level ``split`` kwarg.

``registry.run_baseline`` forwards ``split`` to the matching runner so that
self-referential evaluation can be detected (title_oracle on dev_public).
The runners are thin ``**kw`` wrappers, so a runner whose *inner* function
rejects ``split`` raises ``TypeError`` only at call time — which is how
``verify_citations`` and ``doi_presence_heuristic`` shipped broken while the
static signature of the registered runner looked fine.

The `Run Free Baselines` workflow marks its matrix ``continue-on-error: true``,
so that TypeError surfaces there as a green tick. This test is the gate instead.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib

import pytest

from hallmark.baselines import registry as R


def _wrapper_consumes_split(node: ast.FunctionDef) -> bool:
    """True if the wrapper itself removes ``split`` from **kw before delegating."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "pop"
            and sub.args
            and isinstance(sub.args[0], ast.Constant)
            and sub.args[0].value == "split"
        ):
            return True
    return False


def _inner_run_functions() -> dict[str, tuple[str, str, bool]]:
    """Map each ``_run_*`` wrapper in registry.py to the ``run_*`` it delegates to.

    The third element records whether the wrapper consumes ``split`` itself, in
    which case the inner function never sees it.
    """
    tree = ast.parse(pathlib.Path(R.__file__).read_text())
    wrappers: dict[str, tuple[str, str, bool]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("_run_")):
            continue
        consumes = _wrapper_consumes_split(node)
        for sub in ast.walk(node):
            if isinstance(sub, ast.ImportFrom) and sub.module and "baselines" in sub.module:
                for alias in sub.names:
                    if alias.name.startswith("run_"):
                        wrappers[node.name] = (sub.module, alias.name, consumes)
    return wrappers


def _accepts_split(fn: object, preset_kwargs: dict[str, object] | None) -> bool:
    if "split" in (preset_kwargs or {}):
        return True
    sig = inspect.signature(fn)  # type: ignore[arg-type]
    if "split" in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


_WRAPPERS = _inner_run_functions()
_CASES = sorted(
    name for name, info in R._REGISTRY.items() if getattr(info.runner, "__name__", "") in _WRAPPERS
)


def test_registry_is_not_empty() -> None:
    assert _CASES, "no baselines resolved to an inner run_* function — the AST map broke"


@pytest.mark.parametrize("name", _CASES)
def test_baseline_runner_tolerates_split_kwarg(name: str) -> None:
    """The inner run_* function must accept (or absorb) ``split``."""
    info = R._REGISTRY[name]
    module_name, fn_name, wrapper_consumes = _WRAPPERS[info.runner.__name__]
    if wrapper_consumes:
        return  # the wrapper pops 'split'; the inner function never sees it
    inner = getattr(importlib.import_module(module_name), fn_name)

    assert _accepts_split(inner, info.runner_kwargs), (
        f"{name}: {module_name}.{fn_name}{inspect.signature(inner)} rejects the "
        "'split' kwarg that registry.run_baseline forwards. Add '**_kw: object' "
        "to its signature (see run_doi_only) or pop 'split' in the wrapper."
    )
