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
_REAL_RESULTS_DIR = _REPO_ROOT / "data" / "v1.2" / "baseline_results"
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
    split_sha256: str | None = None,
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
                **({"split_sha256": split_sha256} if split_sha256 is not None else {}),
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
    split_file = data_dir / "v1.2" / "dev_public.jsonl"
    _write_split(split_file, n_hall=n_hall, n_valid=n_valid)
    results_dir = data_dir / "v1.2" / "baseline_results"
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

    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
    assert res.passed is True
    assert res.stale_files == []


# --- Tests: stale by split hash --------------------------------------------


def test_stale_when_recorded_split_hash_does_not_match(tmp_path):
    """The result names a split revision that is no longer on disk."""
    data_dir, _split_file, results_dir = _build_env(tmp_path)
    result_file = results_dir / "mytool_dev_public.json"
    _write_result(
        result_file,
        tool="mytool",
        split="dev_public",
        n_entries=20,
        n_hall=10,
        n_valid=10,
        split_sha256="0" * 64,
    )

    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
    assert res.passed is False
    assert "mytool_dev_public.json" in res.stale_files
    report = next(r for r in res.reports if r.result_file == "mytool_dev_public.json")
    assert any("scored split" in reason for reason in report.reasons)


def test_fresh_when_recorded_split_hash_matches(tmp_path):
    data_dir, split_file, results_dir = _build_env(tmp_path)
    _write_result(
        results_dir / "mytool_dev_public.json",
        tool="mytool",
        split="dev_public",
        n_entries=20,
        n_hall=10,
        n_valid=10,
        split_sha256=crf.compute_sha256(split_file),
    )

    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
    assert res.passed is True
    assert res.stale_files == []


def test_missing_split_hash_is_unverifiable_not_stale(tmp_path):
    """Results predating the field must not be called stale.

    Treating them as stale is what made the old guard permanently red, which is
    why it ended up behind --warn-only and an xfail.
    """
    data_dir, _split_file, results_dir = _build_env(tmp_path)
    _write_result(
        results_dir / "mytool_dev_public.json",
        tool="mytool",
        split="dev_public",
        n_entries=20,
        n_hall=10,
        n_valid=10,
    )

    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
    assert res.passed is True
    assert res.stale_files == []
    report = next(r for r in res.reports if r.result_file == "mytool_dev_public.json")
    assert report.unverifiable is True


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

    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
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

    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
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

    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
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

    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
    report = next(r for r in res.reports if r.result_file == result_file.name)
    assert report.split == "dev_public"
    assert report.is_stale is False


def test_missing_results_dir_fails():
    res = crf.check_freshness("/nonexistent/dir/xyz", version="v1.2")
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
    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
    report = next(r for r in res.reports if r.result_file == result_file.name)
    assert report.is_stale is True
    assert any("split file missing" in r for r in report.reasons)


# --- Real repository guard --------------------------------------------------
#
# This was xfail-ed, with the reason "released result JSONs predate the relabel;
# regenerate them in a later stage". The deeper problem was that the check
# compared file mtimes, which git does not preserve, so on a fresh clone it read
# checkout order and called every released result stale. It now hashes the split
# file, and the two genuinely stale results are registered in KNOWN_STALE.


@pytest.mark.skipif(not _REAL_RESULTS_DIR.is_dir(), reason="real baseline_results dir not present")
def test_real_repo_results_are_fresh():
    res = crf.check_freshness(_REAL_RESULTS_DIR, version="v1.2", data_dir=_REAL_DATA_DIR)
    unexpected = [f for f in res.stale_files if f not in crf.KNOWN_STALE]
    assert not unexpected, f"Unexpectedly stale result artifacts: {unexpected}"
    assert res.passed, f"Freshness check failed: {res.errors}"


@pytest.mark.skipif(not _REAL_RESULTS_DIR.is_dir(), reason="real baseline_results dir not present")
def test_known_stale_register_only_shrinks():
    """A KNOWN_STALE entry that is no longer stale must be removed.

    Otherwise the register becomes a list of excuses that outlive their cause,
    which is how the xfail above survived long enough to hide a broken check.
    """
    res = crf.check_freshness(_REAL_RESULTS_DIR, version="v1.2", data_dir=_REAL_DATA_DIR)
    stale = set(res.stale_files)
    obsolete = sorted(name for name in crf.KNOWN_STALE if name not in stale)
    assert not obsolete, (
        f"listed in KNOWN_STALE but no longer stale: {obsolete}. Remove them from the register."
    )


# --- Tests: stale by per-type counts ------------------------------------------


def test_stale_when_per_type_counts_sum_past_the_split(tmp_path):
    """Per-type counts that sum to more positives than the split has are a
    different run's per-type block under patched headline counts.

    Two released dev_public results carry per-type counts summing to 633 (the
    pre-relabel split) while declaring num_hallucinated 606. The headline
    counts were patched; the per-type block was not; the count check saw the
    patched numbers and passed them.
    """
    data_dir, _split_file, results_dir = _build_env(tmp_path, n_hall=10, n_valid=10)
    result_file = results_dir / "mytool_dev_public.json"
    _write_result(
        result_file, tool="mytool", split="dev_public", n_entries=20, n_hall=10, n_valid=10
    )
    payload = json.loads(result_file.read_text())
    payload["per_type_metrics"] = {
        "fabricated_doi": {"count": 8, "detection_rate": 0.5},
        "wrong_venue": {"count": 4, "detection_rate": 0.5},
        "valid": {"count": 10},
    }
    result_file.write_text(json.dumps(payload))

    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
    assert res.stale_files == ["mytool_dev_public.json"]
    assert any("per_type" in r for r in res.reports[0].reasons), res.reports[0].reasons


def test_per_type_counts_that_match_are_fresh(tmp_path):
    data_dir, _split_file, results_dir = _build_env(tmp_path, n_hall=10, n_valid=10)
    result_file = results_dir / "mytool_dev_public.json"
    _write_result(
        result_file, tool="mytool", split="dev_public", n_entries=20, n_hall=10, n_valid=10
    )
    payload = json.loads(result_file.read_text())
    payload["per_type_metrics"] = {
        "fabricated_doi": {"count": 6, "detection_rate": 0.5},
        "wrong_venue": {"count": 4, "detection_rate": 0.5},
        "valid": {"count": 10},
    }
    result_file.write_text(json.dumps(payload))
    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
    assert res.passed and res.stale_files == []


@pytest.mark.skipif(not _REAL_RESULTS_DIR.is_dir(), reason="real results dir not present")
def test_the_two_patched_claude_dev_results_are_caught_and_registered():
    """Pins the finding: per-type over 633 positives, headline 606, and they are
    in KNOWN_STALE so the guard stays green without forgetting them."""
    res = crf.check_freshness(_REAL_RESULTS_DIR, version="v1.2", data_dir=_REAL_DATA_DIR)
    by_name = {r.result_file: r for r in res.reports}
    for name in (
        "llm_openrouter_claude_opus_4_7_dev_public.json",
        "llm_openrouter_claude_sonnet_4_6_dev_public.json",
    ):
        assert by_name[name].is_stale, f"{name} not flagged"
        assert any("per_type" in reason for reason in by_name[name].reasons)
        assert name in crf.KNOWN_STALE, f"{name} flagged but not registered"
    assert res.passed, res.errors


# --- Tests: the superseded per-type definition --------------------------------


def _per_type_result(results_dir: Path, per_type: dict, *, split_sha256: str | None = None) -> Path:
    result_file = results_dir / "mytool_dev_public.json"
    _write_result(
        result_file,
        tool="mytool",
        split="dev_public",
        n_entries=20,
        n_hall=10,
        n_valid=10,
        split_sha256=split_sha256,
    )
    payload = json.loads(result_file.read_text())
    payload["per_type_metrics"] = per_type
    result_file.write_text(json.dumps(payload))
    return result_file


def test_per_type_rows_without_num_valid_are_unverifiable(tmp_path):
    """39 of the 42 released results carry the superseded per-type definition.

    Those rows counted false positives inside the type, so every hallucination
    type reports a false-positive rate of 0.0 and an f1 of 2*DR/(1+DR). They are
    not comparable with the current rows and they are not evidence of a moved
    split, so they are reported on the unverifiable contract rather than failed:
    a check that trips on 39 of 42 files at once is a check someone switches off.
    """
    data_dir, _split_file, results_dir = _build_env(tmp_path, n_hall=10, n_valid=10)
    _per_type_result(
        results_dir,
        {
            "fabricated_doi": {
                "detection_rate": 0.5,
                "false_positive_rate": 0.0,
                "f1": 2 / 3,
                "count": 6,
            },
            "wrong_venue": {
                "detection_rate": 0.5,
                "false_positive_rate": 0.0,
                "f1": 2 / 3,
                "count": 4,
            },
        },
    )

    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
    report = next(r for r in res.reports if r.result_file == "mytool_dev_public.json")
    assert report.unverifiable is True
    assert report.superseded_per_type is True
    assert report.is_stale is False, "a superseded definition is not a moved split"
    assert res.passed is True
    assert any("2 per_type_metrics row(s) predate" in reason for reason in report.reasons), (
        report.reasons
    )


def test_per_type_rows_with_num_valid_are_not_flagged(tmp_path):
    """A row under the current definition names the valid pool it scored against."""
    data_dir, split_file, results_dir = _build_env(tmp_path, n_hall=10, n_valid=10)
    _per_type_result(
        results_dir,
        {
            "fabricated_doi": {
                "detection_rate": 0.5,
                "false_positive_rate": 0.1,
                "f1": 0.5,
                "precision": 0.5,
                "count": 6,
                "num_valid": 10,
            },
            "wrong_venue": {
                "detection_rate": 0.5,
                "false_positive_rate": 0.1,
                "f1": 0.5,
                "precision": 0.5,
                "count": 4,
                "num_valid": 10,
            },
        },
        split_sha256=crf.compute_sha256(split_file),
    )

    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
    report = next(r for r in res.reports if r.result_file == "mytool_dev_public.json")
    assert report.superseded_per_type is False
    assert report.unverifiable is False
    assert res.passed is True


@pytest.mark.skipif(not _REAL_RESULTS_DIR.is_dir(), reason="real results dir not present")
def test_the_released_results_report_the_superseded_definition():
    """Pins the count: 39 of the 42 released results, reported and not fatal."""
    res = crf.check_freshness(_REAL_RESULTS_DIR, version="v1.2", data_dir=_REAL_DATA_DIR)
    flagged = [r.result_file for r in res.reports if r.superseded_per_type]
    assert len(flagged) == 39, f"{len(flagged)} of {len(res.reports)} flagged: {flagged[:5]}"
    assert res.passed, res.errors


# --- Tests: results/archive/ is outside the gate ------------------------------


def test_files_under_archive_are_skipped(tmp_path):
    """A run kept for the record scores no current split, and saying so once in
    the directory layout beats saying it once per file in a register."""
    data_dir, _split_file, results_dir = _build_env(tmp_path, n_hall=10, n_valid=10)
    archive = results_dir / "archive"
    archive.mkdir()
    # Stale by every check the gate makes: wrong counts, wrong split hash.
    _write_result(
        archive / "oldtool_dev_public.json",
        tool="oldtool",
        split="dev_public",
        n_entries=99,
        n_hall=99,
        n_valid=99,
        split_sha256="0" * 64,
    )
    _write_result(
        results_dir / "mytool_dev_public.json",
        tool="mytool",
        split="dev_public",
        n_entries=20,
        n_hall=10,
        n_valid=10,
    )

    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
    assert [r.result_file for r in res.reports] == ["mytool_dev_public.json"]
    assert res.passed is True


def test_an_empty_results_dir_passes(tmp_path):
    """The gate over ``results/`` guards a surface that is currently empty; it
    has to stay green until someone drops a result there."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    data_dir, _split_file, _ = _build_env(tmp_path)
    res = crf.check_freshness(results_dir, version="v1.2", data_dir=data_dir)
    assert res.passed is True
    assert res.reports == []
