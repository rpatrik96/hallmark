"""Tests for the derived-table freshness guard.

The results guard checks a result JSON against the split it scored. This one
checks the layer above: ``tables/*.csv`` and ``*.tex`` are arithmetic over those
results, and nothing noticed when ``base_rate_precision.csv`` went on reporting
a doi_only false-positive rate of 0.2788 after its source run was re-run at
0.0417 — a table claiming a tool hands the user roughly seven times more false
accusations than it does.

Two mechanisms, tested separately: recorded input hashes
(:func:`record_table`), which works for a table of any shape, and a direct
comparison against the runs a ``tool`` column names, which works with no
recorded provenance at all and is what makes the guard bite today.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

# Make scripts/ importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import check_results_freshness as crf  # noqa: E402

from hallmark.evaluation.table_provenance import (  # noqa: E402
    PROVENANCE_FILE,
    check_tables,
    read_provenance,
    record_table,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_TABLES_DIR = _REPO_ROOT / "tables"
_REAL_RESULTS_DIR = _REPO_ROOT / "data" / "v1.2" / "baseline_results"


# --- Fixtures --------------------------------------------------------------


def _write_result(results_dir: Path, tool: str, **fields: float) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{tool}.json"
    path.write_text(json.dumps({"split_name": "test_public", **fields}))
    return path


def _write_table(tables_dir: Path, name: str, rows: list[dict[str, str]]) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / name
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A miniature repository with one result and one table derived from it."""
    _write_result(
        tmp_path / "results",
        "doi_only_test_public",
        detection_rate=0.1908,
        false_positive_rate=0.0417,
    )
    _write_table(
        tmp_path / "tables",
        "derived.csv",
        [
            {
                "tool": "doi_only_test_public",
                "split": "test_public",
                "detection_rate": "0.1908",
                "false_positive_rate": "0.0417",
                "precision": "0.8200",
            }
        ],
    )
    return tmp_path


def _report(reports, name):
    return next(r for r in reports if r.table == name)


# --- Value check -----------------------------------------------------------


def test_table_agreeing_with_its_results_is_fresh(repo: Path):
    reports = check_tables(repo / "tables", repo / "results", repo_root=repo)
    report = _report(reports, "derived.csv")
    assert not report.is_stale, report.reasons
    assert report.value_checked
    assert not report.unverifiable


def test_table_is_stale_when_a_result_moves_under_it(repo: Path):
    """The defect this guard exists for: the run is re-run, the table is not."""
    _write_result(
        repo / "results",
        "doi_only_test_public",
        detection_rate=0.3873,
        false_positive_rate=0.2788,
    )
    report = _report(check_tables(repo / "tables", repo / "results", repo_root=repo), "derived.csv")
    assert report.is_stale
    joined = "; ".join(report.reasons)
    assert "detection_rate" in joined and "0.1908" in joined and "0.3873" in joined


def test_comparison_is_at_the_table_s_own_precision(repo: Path):
    """A cell written to four decimals is checked to four, not to float equality."""
    _write_result(
        repo / "results",
        "doi_only_test_public",
        detection_rate=0.19080001,
        false_positive_rate=0.04170002,
    )
    report = _report(check_tables(repo / "tables", repo / "results", repo_root=repo), "derived.csv")
    assert not report.is_stale, report.reasons


def test_column_absent_from_the_result_is_not_compared(repo: Path):
    """``precision`` is computed by the generator, not read from the run."""
    report = _report(check_tables(repo / "tables", repo / "results", repo_root=repo), "derived.csv")
    assert not report.is_stale
    assert not any("precision" in reason for reason in report.reasons)


def test_display_names_are_unverifiable_not_stale(tmp_path: Path):
    """``baseline_cost_latency.csv`` names tools for a reader, not for a lookup.

    Failing those would make the guard red for a table it simply cannot check,
    which is how the previous freshness guard ended up switched off.
    """
    _write_result(tmp_path / "results", "doi_only_test_public", detection_rate=0.19)
    _write_table(
        tmp_path / "tables",
        "costs.csv",
        [{"tool": "DOI-only", "mean_sec_per_entry": "0.4000"}],
    )
    report = _report(
        check_tables(tmp_path / "tables", tmp_path / "results", repo_root=tmp_path), "costs.csv"
    )
    assert not report.is_stale
    assert report.unverifiable
    assert "tool column" in "; ".join(report.reasons)


def test_table_without_tool_column_or_provenance_is_unverifiable(tmp_path: Path):
    _write_table(tmp_path / "tables", "per_tier.csv", [{"model": "x", "tier": "1", "dr": "0.5"}])
    (tmp_path / "results").mkdir()
    report = _report(
        check_tables(tmp_path / "tables", tmp_path / "results", repo_root=tmp_path), "per_tier.csv"
    )
    assert not report.is_stale
    assert report.unverifiable


# --- Provenance check ------------------------------------------------------


def test_record_table_stores_repo_relative_paths_and_hashes(repo: Path):
    result = repo / "results" / "doi_only_test_public.json"
    record_table(
        repo / "tables" / "derived.csv", [result], generator="scripts/x.py", repo_root=repo
    )
    entry = read_provenance(repo / "tables")["derived.csv"]
    assert entry["generator"] == "scripts/x.py"
    assert list(entry["inputs"]) == ["results/doi_only_test_public.json"]
    assert len(next(iter(entry["inputs"].values()))) == 64


def test_recording_unchanged_inputs_does_not_churn_the_file(repo: Path):
    """No timestamp, so regenerating an unchanged table leaves an empty diff."""
    result = repo / "results" / "doi_only_test_public.json"
    args = dict(generator="scripts/x.py", repo_root=repo)
    record_table(repo / "tables" / "derived.csv", [result], **args)
    first = (repo / "tables" / PROVENANCE_FILE).read_bytes()
    record_table(repo / "tables" / "derived.csv", [result], **args)
    assert (repo / "tables" / PROVENANCE_FILE).read_bytes() == first


def test_provenance_catches_a_changed_input_for_a_table_it_cannot_read(tmp_path: Path):
    """A per-tier table has no checkable cells; the recorded hash is all there is."""
    source = tmp_path / "results" / "eval.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"per_tier_metrics": {"1": {"dr": 0.5}}}')
    table = _write_table(tmp_path / "tables", "per_tier.csv", [{"model": "x", "dr": "0.5000"}])
    record_table(table, [source], generator="scripts/agg.py", repo_root=tmp_path)

    fresh = _report(
        check_tables(tmp_path / "tables", tmp_path / "results", repo_root=tmp_path), "per_tier.csv"
    )
    assert not fresh.is_stale and fresh.provenance_checked

    source.write_text('{"per_tier_metrics": {"1": {"dr": 0.9}}}')
    stale = _report(
        check_tables(tmp_path / "tables", tmp_path / "results", repo_root=tmp_path), "per_tier.csv"
    )
    assert stale.is_stale
    assert "eval.json" in "; ".join(stale.reasons)


def test_provenance_catches_a_deleted_input(tmp_path: Path):
    source = tmp_path / "results" / "eval.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    table = _write_table(tmp_path / "tables", "per_tier.csv", [{"model": "x", "dr": "0.5"}])
    record_table(table, [source], generator="scripts/agg.py", repo_root=tmp_path)
    source.unlink()
    report = _report(
        check_tables(tmp_path / "tables", tmp_path / "results", repo_root=tmp_path), "per_tier.csv"
    )
    assert report.is_stale
    assert "no longer exists" in "; ".join(report.reasons)


def test_tex_tables_are_checked_through_provenance(tmp_path: Path):
    source = tmp_path / "results" / "eval.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    tex = tmp_path / "tables" / "per_tier.tex"
    tex.parent.mkdir(parents=True)
    tex.write_text("\\begin{tabular}{l}\\end{tabular}\n")
    record_table(tex, [source], generator="scripts/agg.py", repo_root=tmp_path)
    source.write_text('{"changed": true}')
    report = _report(
        check_tables(tmp_path / "tables", tmp_path / "results", repo_root=tmp_path), "per_tier.tex"
    )
    assert report.is_stale


def test_provenance_file_is_not_itself_a_table(repo: Path):
    record_table(
        repo / "tables" / "derived.csv",
        [repo / "results" / "doi_only_test_public.json"],
        generator="scripts/x.py",
        repo_root=repo,
    )
    reports = check_tables(repo / "tables", repo / "results", repo_root=repo)
    assert PROVENANCE_FILE not in [r.table for r in reports]


# --- Register and CLI wrapper ----------------------------------------------


def test_known_stale_table_does_not_fail_the_run(repo: Path, monkeypatch):
    _write_result(repo / "results", "doi_only_test_public", detection_rate=0.9)
    monkeypatch.setattr(crf, "KNOWN_STALE_TABLES", {"derived.csv": "pending regeneration"})
    res = crf.check_table_freshness(repo / "tables", repo / "results", repo_root=repo)
    assert res.stale_tables == ["derived.csv"]
    assert res.passed


def test_unregistered_stale_table_fails_the_run(repo: Path, monkeypatch):
    _write_result(repo / "results", "doi_only_test_public", detection_rate=0.9)
    monkeypatch.setattr(crf, "KNOWN_STALE_TABLES", {})
    res = crf.check_table_freshness(repo / "tables", repo / "results", repo_root=repo)
    assert not res.passed


def test_register_may_only_shrink(repo: Path, monkeypatch):
    """An exemption whose table is fresh again is a stale excuse, and fails."""
    monkeypatch.setattr(crf, "KNOWN_STALE_TABLES", {"derived.csv": "pending regeneration"})
    res = crf.check_table_freshness(repo / "tables", repo / "results", repo_root=repo)
    assert not res.passed
    assert any("remove it from the register" in e for e in res.errors)


def test_missing_tables_dir_is_an_error(tmp_path: Path):
    res = crf.check_table_freshness(tmp_path / "nope", tmp_path, repo_root=tmp_path)
    assert not res.passed and res.errors


# --- Real repository guard --------------------------------------------------


@pytest.mark.skipif(not _REAL_TABLES_DIR.is_dir(), reason="real tables dir not present")
def test_real_repo_tables_are_fresh():
    res = crf.check_table_freshness(_REAL_TABLES_DIR, _REAL_RESULTS_DIR, repo_root=_REPO_ROOT)
    unexpected = [t for t in res.stale_tables if t not in crf.KNOWN_STALE_TABLES]
    assert not unexpected, f"Unexpectedly stale tables: {unexpected}"
    assert res.passed, f"Table freshness check failed: {res.errors}"


@pytest.mark.skipif(not _REAL_TABLES_DIR.is_dir(), reason="real tables dir not present")
def test_real_known_stale_table_register_only_shrinks():
    res = crf.check_table_freshness(_REAL_TABLES_DIR, _REAL_RESULTS_DIR, repo_root=_REPO_ROOT)
    stale = set(res.stale_tables)
    obsolete = sorted(name for name in crf.KNOWN_STALE_TABLES if name not in stale)
    assert not obsolete, (
        f"listed in KNOWN_STALE_TABLES but no longer stale: {obsolete}. "
        "Remove them from the register."
    )
