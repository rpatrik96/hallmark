"""An archived script must not still rewrite the released data.

``scripts/archive/`` holds one-pass data fixes that ``scripts/build_dataset.py``
superseded. Each was left runnable and each still points at the live released
splits by absolute path: ``fix_data_quality_v1_2.py`` writes ``dev_public`` first
and validates afterwards, so an exception partway leaves one split rewritten and
the other two untouched.

The header note names the successor; the guard says the same thing where someone
running the file will actually see it. Importing stays free, because the module
constants and helpers are still worth reading.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_ARCHIVE = Path(__file__).resolve().parent.parent / "scripts" / "archive"

ARCHIVED_SCRIPTS = [
    "fix_data_quality_v1_2.py",
    "fix_subtest_and_dedup.py",
    "patch_data_v1.py",
]


def _load(name: str) -> ModuleType:
    path = _ARCHIVE / name
    spec = importlib.util.spec_from_file_location(f"archived_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ARCHIVED_SCRIPTS)
def test_importing_an_archived_script_does_not_exit(name):
    """The guard belongs in main(), not at module scope: importing one to read
    its pools or its patch rules must keep working."""
    module = _load(name)
    assert callable(module.main)


@pytest.mark.parametrize("name", ARCHIVED_SCRIPTS)
def test_running_an_archived_script_stops_before_it_writes(name):
    module = _load(name)
    with pytest.raises(SystemExit) as excinfo:
        module.main()
    message = str(excinfo.value)
    assert "archived" in message
    assert "scripts/build_dataset.py" in message, "the guard must name the successor"
    assert "copy it out" in message, "the guard must say how to run it deliberately"


@pytest.mark.parametrize("name", ARCHIVED_SCRIPTS)
def test_the_archived_scripts_still_point_at_the_released_splits(name):
    """The reason the guard exists: each names the split directory as a module
    constant rather than taking it as an argument. If one is ever rewritten to
    take its paths from the caller, this test goes with the guard."""
    module = _load(name)
    targets = [
        value
        for value in vars(module).values()
        if isinstance(value, Path) and value.parts[-2:] == ("data", "v1.2")
    ]
    assert targets, f"{name} names no data/v1.2 directory — has it been re-pointed?"
