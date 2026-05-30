"""Tests for the CI results-freshness guard (task #2).

The guard (``scripts/check_results_freshness.py``) fails when an aggregate
result JSON is older than the split it scores, or when its recorded ground-truth
counts disagree with the current split.

The logic is validated against synthetic tmp fixtures (always deterministic and
green). A final test runs the guard over the real repository: the released
result JSONs predate the May-2026 relabel, so the guard is *expected* to report
them as stale until they are regenerated in a later stage. That test is marked
``xfail(strict=False)`` so it documents the known-stale state without breaking
CI for unrelated work, and will start passing automatically once the results
are regenerated.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

# Make scripts/ importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import check_results_freshness as crf  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_RESULTS_DIR = _REPO_ROOT / "data" / "v1.0" / "baseline_results"
_REAL_DATA_DIR = _REPO_ROOT / "data"


# --- Fixtures --------------------------------------------------------------


def _write_split(path: Path, *, n_hall: int, n_valid: int) -> None:
    """Write a minimal JSONL split with the requested label counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n_hall):
        lines.append(
            json.dumps(
                {
                    "bibtex_key": f"h{i}",
                    "bibtex_type": "article",
                    "fields": {"title": f"H{i}", "author": "A", "year": "2024"},
                    "label": "HALLUCINATED",
                    "explanation": "x",
                    "hallucination_type": "fabricated_doi",
                    "difficulty_tier": 1,
                }
            )
        )
    for i in range(n_valid):
        lines.append(
            json.dumps(
                {
                    "bibtex_key": f"v{i}",
                    "bibtex_type": "article",
                    "fields": {"title": f"V{i}", "author": "A", "year": "2024"},
                    "label": "VALID",
                    "explanation": "x",
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _write_result(
    path: Path,
    *,
    tool: str,
    split: str,
    n_entries: int,
    n_hall: int,
    n_valid: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tool_name": tool,
                "split_name": split,
                "num_entries": n_entries,
                "num_hallucinated": n_hall,
                "num_valid": n_valid,
                "detection_rate": 0.8,
                "false_positive_rate": 0.1,
                "f1_hallucination": 0.75,
                "tier_weighted_f1": 0.7,
            }
        )
    )


def _build_env(tmp_path: Path, *, n_hall: int = 10, n_valid: int = 10):
    """Create a fake data dir + results dir mirroring the real layout."""
    data_dir = tmp_path / "data"
    split_file = data_dir / "v1.0" / "dev_public.jsonl"
    _write_split(split_file, n_hall=n_hall, n_valid=n_valid)
    results_dir = data_dir / "v1.0" / "baseline_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, split_file, results_dir


# --- Tests: fresh case -----------------------------------------------------


def test_fresh_result_passes(tmp_path):
    data_dir, _split_file, results_dir = _build_env(tmp_path, n_hall=10, n_valid=10)
    result_file = results_dir / "mytool_dev_public.json"
    _write_result(
        result_file,
        tool="mytool",
        split="dev_public",
        n_entries=20,
        n_hall=10,
        n_valid=10,
    )
    # Make the result newer than the split.
    future = time.time() + 100
    os.utime(result_file, (future, future))

    res = crf.check_freshness(results_dir, version="v1.0", data_dir=data_dir)
    assert res.passed is True
    assert res.stale_files == []


# --- Tests: stale by mtime -------------------------------------------------


def test_stale_by_mtime(tmp_path):
    data_dir, split_file, results_dir = _build_env(tmp_path)
    result_file = results_dir / "mytool_dev_public.json"
    _write_result(
        result_file,
        tool="mytool",
        split="dev_public",
        n_entries=20,
        n_hall=10,
        n_valid=10,
    )
    # Result older than the split.
    old = time.time() - 10_000
    os.utime(result_file, (old, old))
    new = time.time() + 100
    os.utime(split_file, (new, new))

    res = crf.check_freshness(results_dir, version="v1.0", data_dir=data_dir)
    assert res.passed is False
    assert "mytool_dev_public.json" in res.stale_files
    report = next(r for r in res.reports if r.result_file == "mytool_dev_public.json")
    assert any("mtime" in reason for reason in report.reasons)


# --- Tests: stale by count mismatch ----------------------------------------


def test_stale_by_count_mismatch(tmp_path):
    data_dir, _split_file, results_dir = _build_env(tmp_path, n_hall=12, n_valid=8)
    result_file = results_dir / "mytool_dev_public.json"
    # Record stale counts (old labeling) but keep result newer than split.
    _write_result(
        result_file,
        tool="mytool",
        split="dev_public",
        n_entries=20,
        n_hall=10,  # current split has 12
        n_valid=10,  # current split has 8
    )
    future = time.time() + 100
    os.utime(result_file, (future, future))

    res = crf.check_freshness(results_dir, version="v1.0", data_dir=data_dir)
    assert res.passed is False
    report = next(r for r in res.reports if r.result_file == "mytool_dev_public.json")
    assert any("num_hallucinated mismatch" in r for r in report.reasons)
    assert any("num_valid mismatch" in r for r in report.reasons)


# --- Tests: split inference & edge cases -----------------------------------


def test_split_inferred_from_filename_when_field_missing(tmp_path):
    data_dir, _split_file, results_dir = _build_env(tmp_path, n_hall=10, n_valid=10)
    result_file = results_dir / "cascade_db_diagnosis_dev_public.json"
    # No split_name field -> must fall back to filename suffix.
    result_file.write_text(
        json.dumps(
            {
                "tool_name": "cascade_db_diagnosis",
                "num_entries": 20,
                "num_hallucinated": 10,
                "num_valid": 10,
            }
        )
    )
    future = time.time() + 100
    os.utime(result_file, (future, future))

    res = crf.check_freshness(results_dir, version="v1.0", data_dir=data_dir)
    report = next(r for r in res.reports if r.result_file == "cascade_db_diagnosis_dev_public.json")
    assert report.split == "dev_public"
    assert report.is_stale is False


def test_manifest_json_is_ignored(tmp_path):
    data_dir, _split_file, results_dir = _build_env(tmp_path)
    (results_dir / "manifest.json").write_text(json.dumps({"version": "1.0", "files": {}}))
    result_file = results_dir / "mytool_dev_public.json"
    _write_result(
        result_file,
        tool="mytool",
        split="dev_public",
        n_entries=20,
        n_hall=10,
        n_valid=10,
    )
    future = time.time() + 100
    os.utime(result_file, (future, future))

    res = crf.check_freshness(results_dir, version="v1.0", data_dir=data_dir)
    assert "manifest.json" not in [r.result_file for r in res.reports]
    assert res.passed is True


def test_dual_mode_payload_uses_conservative_block(tmp_path):
    data_dir, _split_file, results_dir = _build_env(tmp_path, n_hall=10, n_valid=10)
    result_file = results_dir / "cascade_db_diagnosis_evalmode_dev_public.json"
    inner = {
        "tool_name": "cascade_db_diagnosis",
        "split_name": "dev_public",
        "num_entries": 20,
        "num_hallucinated": 10,
        "num_valid": 10,
    }
    result_file.write_text(json.dumps({"conservative": inner, "aggressive": inner}))
    future = time.time() + 100
    os.utime(result_file, (future, future))

    res = crf.check_freshness(results_dir, version="v1.0", data_dir=data_dir)
    report = next(r for r in res.reports if r.result_file == result_file.name)
    assert report.split == "dev_public"
    assert report.is_stale is False


def test_missing_results_dir_fails():
    res = crf.check_freshness("/nonexistent/dir/xyz", version="v1.0")
    assert res.passed is False
    assert res.errors


def test_unparseable_split_makes_stale(tmp_path):
    data_dir, _split_file, results_dir = _build_env(tmp_path)
    # A result for a split whose file does not exist.
    result_file = results_dir / "mytool_test_public.json"
    _write_result(
        result_file,
        tool="mytool",
        split="test_public",
        n_entries=20,
        n_hall=10,
        n_valid=10,
    )
    res = crf.check_freshness(results_dir, version="v1.0", data_dir=data_dir)
    report = next(r for r in res.reports if r.result_file == result_file.name)
    assert report.is_stale is True
    assert any("split file missing" in r for r in report.reasons)


# --- Real repository guard (documents the known-stale state) ---------------


@pytest.mark.skipif(not _REAL_RESULTS_DIR.is_dir(), reason="real baseline_results dir not present")
@pytest.mark.xfail(
    reason="released result JSONs predate the relabel; regenerate them in a later stage",
    strict=False,
)
def test_real_repo_results_are_fresh():
    res = crf.check_freshness(_REAL_RESULTS_DIR, version="v1.0", data_dir=_REAL_DATA_DIR)
    assert res.passed, f"Stale result artifacts: {res.stale_files}"
