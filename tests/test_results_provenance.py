"""Guard the provenance of baseline result files.

Two directories once held two different runs under one set of names:
``results/`` and ``data/v1.2/baseline_results/`` shared fifteen filenames and
**every pair disagreed** on detection rate, false-positive rate and F1, three of
them on the number of entries scored as well. Nothing compared them, nothing
recorded which was current, and ``results/`` is the default ``--results-dir``
for the CLI leaderboard and the figure scripts — so the uncanonical set was the
one being read.

A reproducer hits that before anything else, so these tests make the collision
impossible rather than documenting it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _REPO_ROOT / "results"
_SUPERSEDED = _RESULTS / "superseded_pre_relabel"

#: Comparable summary fields. A disagreement on any of these means the two files
#: describe different runs, whatever their names say.
_METRIC_KEYS = ("detection_rate", "false_positive_rate", "f1_hallucination", "num_entries")


def _baseline_result_dirs() -> list[Path]:
    return sorted(p for p in _REPO_ROOT.glob("data/*/baseline_results") if p.is_dir())


def _canonical_names() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in _baseline_result_dirs():
        for p in d.glob("*.json"):
            if p.name != "manifest.json":
                out[p.name] = p
    return out


def test_at_least_one_baseline_results_dir_exists():
    """Otherwise the collision tests below would pass vacuously."""
    assert _baseline_result_dirs(), "no data/*/baseline_results directory found"


def test_no_result_filename_collides_with_a_released_one():
    """A name may live in exactly one place.

    Superseded runs belong under ``results/superseded_pre_relabel/``, which is
    excluded here precisely because it is the archive for the old collisions.
    """
    canonical = _canonical_names()
    collisions = sorted(p.name for p in _RESULTS.glob("*.json") if p.name in canonical)
    assert not collisions, (
        f"{len(collisions)} result file(s) share a name with a released result and will "
        f"be read instead of it by anything defaulting to --results-dir results: "
        f"{collisions}. Move them to results/superseded_pre_relabel/ or delete them."
    )


def test_superseded_files_are_not_also_canonical():
    """The archive must not shadow a live name either."""
    if not _SUPERSEDED.is_dir():
        pytest.skip("no superseded archive present")
    canonical = _canonical_names()
    # Files here are *expected* to share names with canonical ones — that is why
    # they were archived. What must not happen is the reverse: an archived file
    # being the only copy of a name, which would mean a released result was
    # retired by accident.
    orphans = sorted(p.name for p in _SUPERSEDED.glob("*.json") if p.name not in canonical)
    assert not orphans, (
        f"archived result(s) with no canonical counterpart: {orphans}. These were "
        "retired without a replacement; restore them or record why the name is gone."
    )


def test_archive_documents_itself():
    if not _SUPERSEDED.is_dir():
        pytest.skip("no superseded archive present")
    readme = _SUPERSEDED / "README.md"
    assert readme.is_file(), "an archive of superseded results must say what it is"
    assert "canonical" in readme.read_text().lower()


@pytest.mark.parametrize("results_dir", _baseline_result_dirs(), ids=lambda p: p.parent.name)
def test_released_results_are_internally_consistent(results_dir: Path):
    """Every released result must at least parse and report the fields it claims."""
    bad: list[str] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            bad.append(f"{path.name}: unparseable ({exc})")
            continue
        if not isinstance(data, dict):
            bad.append(f"{path.name}: not an object")
            continue
        if data.get("tool_name") is None:
            bad.append(f"{path.name}: no tool_name")
        for key in ("detection_rate", "num_entries"):
            if data.get(key) is None:
                bad.append(f"{path.name}: no {key}")
    assert not bad, "malformed released results: " + "; ".join(bad)
