"""The tool stamp describes the run it is attached to, and no other.

``scripts/run_all_baselines.py`` evaluates nineteen baselines in one process
with bibtex-updater fourth, and ``--parallel`` runs four of them at once. A
stamp read off process-wide state therefore claimed bibtex-updater's build and
outage report for every baseline that followed it, and under ``--parallel``
recorded whichever worker finished last.
"""

from __future__ import annotations

import json
import subprocess
import threading

import pytest

from hallmark.baselines import bibtexupdater as btu
from hallmark.baselines.registry import run_baseline
from hallmark.dataset.schema import BenchmarkEntry, BlindEntry, EvaluationResult
from hallmark.evaluation.provenance import stamp_provenance

#: Below ``SOURCE_OUTAGE_THRESHOLD``, so the wrapper records the condition and
#: returns instead of refusing the run.
_OUTAGE_A = (
    "WARNING: 4 of 50 entries (8.0%) had at least one source lookup that did "
    "not complete: dblp (4).\n"
)
_OUTAGE_B = (
    "WARNING: 3 of 50 entries (6.0%) had at least one source lookup that did "
    "not complete: openalex (3).\n"
)


def _blind(n: int = 2) -> list[BlindEntry]:
    return [
        BlindEntry(bibtex_key=f"b{i}", bibtex_type="article", fields={"title": f"T{i}"})
        for i in range(n)
    ]


def _benchmark(n: int = 2) -> list[BenchmarkEntry]:
    return [
        BenchmarkEntry(
            bibtex_key=f"e{i}",
            bibtex_type="article",
            fields={"title": f"T{i}", "year": "2020"},
            label="VALID",
        )
        for i in range(n)
    ]


def _result(tool_name: str) -> EvaluationResult:
    return EvaluationResult(
        tool_name=tool_name,
        split_name="dev_public",
        num_entries=2,
        num_hallucinated=1,
        num_valid=1,
        detection_rate=1.0,
        false_positive_rate=0.0,
        f1_hallucination=1.0,
        tier_weighted_f1=1.0,
    )


@pytest.fixture
def stub_bibtex_check(monkeypatch: pytest.MonkeyPatch):
    """Run the wrapper against a canned bibtex-check, with no binary on PATH."""
    monkeypatch.setattr(btu, "resolve_bibtex_check_bin", lambda: "/fake/bibtex-check")
    monkeypatch.setattr(btu, "bibtex_check_version", lambda binary=None: "9.9.9")
    monkeypatch.delenv(btu.ALLOW_OUTAGE_ENV, raising=False)
    monkeypatch.delenv(btu.BIBTEX_CHECK_RATE_ENV, raising=False)
    monkeypatch.delenv(btu.BIBTEX_CHECK_MAILTO_ENV, raising=False)
    monkeypatch.delenv("S2_API_KEY", raising=False)

    def _install(stdout: str = _OUTAGE_A):
        def _run(cmd, **kw):
            jsonl_path = cmd[cmd.index("--jsonl") + 1]
            with open(jsonl_path, "w") as fh:
                fh.write(json.dumps({"key": "b0", "status": "verified"}) + "\n")
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(btu.subprocess, "run", _run)

    return _install


def test_the_next_baseline_is_not_stamped_with_the_previous_one_s_tool(
    stub_bibtex_check,
) -> None:
    """A baseline that never touched bibtex-check carries no bibtex-check stamp.

    The bibtexupdater baseline runs fourth of nineteen in one process; every
    baseline after it used to inherit its version and its outage report.
    """
    stub_bibtex_check()
    btu.run_bibtex_check(_blind(), skip_prescreening=True)
    assert btu.ran_bibtex_check() is True

    run_baseline("doi_presence_heuristic", _benchmark(), split="dev_public")

    result = _result("doi_presence_heuristic")
    stamp_provenance(result, None, None, "v1.2", "doi_presence_heuristic")

    assert result.tool_version is None
    assert result.source_condition is None


def test_run_baseline_clears_the_wrapper_record_before_dispatch(
    stub_bibtex_check,
) -> None:
    """Dispatch starts from a clean record, so a read after it describes it."""
    stub_bibtex_check()
    btu.run_bibtex_check(_blind(), skip_prescreening=True)
    assert btu.last_source_condition() is not None

    run_baseline("doi_presence_heuristic", _benchmark(), split="dev_public")

    assert btu.ran_bibtex_check() is False
    assert btu.last_source_condition() is None


def test_the_stamp_takes_the_run_it_is_given_over_the_module_record(
    stub_bibtex_check,
) -> None:
    """An explicit run wins, so no writer depends on what the process last did."""
    stub_bibtex_check()
    btu.run_bibtex_check(_blind(), skip_prescreening=True)
    tool_run = btu.current_bibtex_check_run()
    assert tool_run is not None and tool_run.ran is True

    # Whatever else the process does afterwards, the run stamped is this one.
    btu.adopt_run_state(
        btu.BibtexCheckRun(predictions=[], raw_records=[], ran=True, condition={"other": 1})
    )
    ran_result = _result("bibtexupdater")
    stamp_provenance(ran_result, None, None, "v1.2", "bibtexupdater", tool_run=tool_run)
    assert ran_result.tool_version == "bibtex-updater 9.9.9"
    assert ran_result.source_condition == tool_run.condition

    idle_result = _result("doi_only")
    stamp_provenance(
        idle_result,
        None,
        None,
        "v1.2",
        "doi_only",
        tool_run=btu.BibtexCheckRun(predictions=[], raw_records=[], ran=False),
    )
    assert idle_result.tool_version is None
    assert idle_result.source_condition is None


def test_two_workers_each_read_their_own_run(stub_bibtex_check, monkeypatch) -> None:
    """Concurrent wrapper calls do not overwrite each other's outcome.

    ``requires_single_worker`` permits several workers per baseline, and each
    shells out on its own. Worker B entering the wrapper used to clear the
    record worker A had just written, so A's completed run read as "the binary
    never started" and A's condition became B's.
    """
    b_in_subprocess = threading.Event()
    a_has_read = threading.Event()

    def _run(cmd, **kw):
        jsonl_path = cmd[cmd.index("--jsonl") + 1]
        with open(jsonl_path, "w") as fh:
            fh.write(json.dumps({"key": "b0", "status": "verified"}) + "\n")
        if threading.current_thread().name == "B":
            # Park B inside its call, after it entered the wrapper, until A has
            # read its own outcome.
            b_in_subprocess.set()
            a_has_read.wait(timeout=10)
            return subprocess.CompletedProcess(cmd, 0, stdout=_OUTAGE_B, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=_OUTAGE_A, stderr="")

    monkeypatch.setattr(btu, "resolve_bibtex_check_bin", lambda: "/fake/bibtex-check")
    monkeypatch.setattr(btu.subprocess, "run", _run)
    monkeypatch.delenv(btu.ALLOW_OUTAGE_ENV, raising=False)
    monkeypatch.delenv(btu.BIBTEX_CHECK_RATE_ENV, raising=False)
    monkeypatch.delenv(btu.BIBTEX_CHECK_MAILTO_ENV, raising=False)
    monkeypatch.delenv("S2_API_KEY", raising=False)

    observed: dict[str, tuple[bool, object]] = {}
    a_finished = threading.Event()

    def _worker_a() -> None:
        btu.run_bibtex_check(_blind(), skip_prescreening=True)
        a_finished.set()
        b_in_subprocess.wait(timeout=10)
        condition = btu.last_source_condition() or {}
        observed["A"] = (btu.ran_bibtex_check(), condition.get("entries_with_incomplete_lookups"))
        a_has_read.set()

    def _worker_b() -> None:
        a_finished.wait(timeout=10)
        btu.run_bibtex_check(_blind(), skip_prescreening=True)
        condition = btu.last_source_condition() or {}
        observed["B"] = (btu.ran_bibtex_check(), condition.get("entries_with_incomplete_lookups"))

    threads = [
        threading.Thread(target=_worker_a, name="A"),
        threading.Thread(target=_worker_b, name="B"),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert observed["A"] == (True, 4)
    assert observed["B"] == (True, 3)
