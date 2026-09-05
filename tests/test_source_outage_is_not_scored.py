"""A run the tool disowned must not become a scored result.

``bibtex-check`` exits 5 when failed lookups touched at least 10% of entries,
and prints alongside it: *"Treat this run as incomplete and discard its
could-not-verify verdicts."* The wrapper logged that at ERROR and then parsed
the output and returned predictions anyway.

It happened for real on 2026-09-04. dblp.org was unreachable — every request
timing out, homepage included — and the ablation's first arm produced
``bibtexupdater_dev_public.json`` at DR 0.8185, FPR 0.0312 and **coverage
1.0000**, from a run where 285 of 1,119 entries (25.5%) never got a complete set
of source lookups. Nothing downstream could tell those abstentions from real
ones, and the result recorded full coverage.

This is the same defect as an API failure written into a prediction file as a
verdict, and as a timed-out batch scoring zero: a tool reported a problem
honestly and the consumer discarded the report. It is the fourth instance found
in one day, which is why the fix records the condition rather than only refusing
— availability moves outcomes, so it belongs beside the numbers.
"""

from __future__ import annotations

import subprocess
from argparse import Namespace

import pytest

from hallmark.baselines import registry as R
from hallmark.baselines.bibtexupdater import (
    ALLOW_OUTAGE_ENV,
    EXIT_SOURCE_OUTAGE,
    SourceOutageError,
    last_source_condition,
    parse_source_condition,
    run_bibtex_check,
)
from hallmark.dataset.schema import BenchmarkEntry, BlindEntry, EvaluationResult

#: bibtex-check's actual output from the 2026-09-04 dblp outage.
REAL_OUTAGE_OUTPUT = """\
INFO: Loaded 1119 entries from dev_public
WARNING: 285 of 1119 entries (25.5%) had at least one source lookup that did not \
complete: dblp (275), openalex (26). Those entries report api_error, not not_found \
-- a source that never answered is not evidence that a reference is absent.
WARNING: Hosts that could not be reached (DNS / connection / TLS / timeout / 5xx): \
dblp.org (275)
ERROR: Source outage: 25.5% of entries could not be checked against a complete set \
of sources (threshold 10%). Treat this run as incomplete and discard its \
could-not-verify verdicts; exiting 5.
"""


class TestSourceConditionParsing:
    def test_reproduces_the_real_outage(self):
        cond = parse_source_condition(REAL_OUTAGE_OUTPUT)
        assert cond == {
            "entries_with_incomplete_lookups": 285,
            "entries_total": 1119,
            "incomplete_fraction": pytest.approx(0.255),
            "per_source_failures": {"dblp": 275, "openalex": 26},
        }

    def test_a_healthy_run_reports_no_condition(self):
        assert parse_source_condition("INFO: Loaded 1119 entries\nINFO: done") is None

    def test_the_final_summary_wins(self):
        """bibtex-check may report progressively; the last line is the total."""
        progressive = (
            "WARNING: 10 of 100 entries (10.0%) had at least one source lookup "
            "that did not complete: dblp (10)\n"
            "WARNING: 40 of 100 entries (40.0%) had at least one source lookup "
            "that did not complete: dblp (35), openalex (5)\n"
        )
        cond = parse_source_condition(progressive)
        assert cond is not None
        assert cond["entries_with_incomplete_lookups"] == 40
        assert cond["per_source_failures"] == {"dblp": 35, "openalex": 5}

    def test_malformed_counts_do_not_raise(self):
        """Provenance parsing must never break an evaluation."""
        cond = parse_source_condition(
            "WARNING: 5 of 50 entries (10.0%) had at least one source lookup "
            "that did not complete: dblp (not-a-number), openalex (3)"
        )
        assert cond is not None
        assert cond["per_source_failures"] == {"openalex": 3}


def test_the_exit_code_is_the_one_the_tool_documents():
    """Pinned because the whole guard keys on it."""
    assert EXIT_SOURCE_OUTAGE == 5


# --- Through the subprocess, not only the regex -----------------------------------


def _entries(n: int = 3) -> list[BlindEntry]:
    return [
        BlindEntry(bibtex_key=f"e{i}", bibtex_type="article", fields={"title": f"T{i}"})
        for i in range(n)
    ]


@pytest.fixture
def fake_bibtex_check(monkeypatch):
    """Drive ``_run_bibtex_check_subprocess`` with a canned exit code and output."""
    from hallmark.baselines import bibtexupdater as btu

    monkeypatch.setattr(btu, "resolve_bibtex_check_bin", lambda: "/fake/bibtex-check")
    monkeypatch.setattr(btu, "bibtex_check_version", lambda binary=None: "1.2.0")
    monkeypatch.delenv(ALLOW_OUTAGE_ENV, raising=False)

    def _install(returncode: int, output: str):
        def _run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, returncode, stdout=output, stderr="")

        monkeypatch.setattr(btu.subprocess, "run", _run)

    return _install


def test_exit_5_raises_through_the_public_runner(fake_bibtex_check):
    fake_bibtex_check(EXIT_SOURCE_OUTAGE, REAL_OUTAGE_OUTPUT)
    with pytest.raises(SourceOutageError, match="285 of 1119"):
        run_bibtex_check(_entries(), skip_prescreening=True)


def test_scoring_an_outage_on_purpose_records_the_condition(fake_bibtex_check, monkeypatch):
    fake_bibtex_check(EXIT_SOURCE_OUTAGE, REAL_OUTAGE_OUTPUT)
    monkeypatch.setenv(ALLOW_OUTAGE_ENV, "1")
    preds = run_bibtex_check(_entries(), skip_prescreening=True)
    assert len(preds) == 3, "the override scores the run"
    cond = last_source_condition()
    assert cond is not None
    assert cond["per_source_failures"] == {"dblp": 275, "openalex": 26}


def test_a_sub_threshold_outage_is_recorded_even_though_the_tool_exits_0(fake_bibtex_check):
    """bibtex-check prints the same summary below its 10% threshold and exits 0."""
    fake_bibtex_check(
        0,
        "WARNING: 5 of 50 entries (10.0%) had at least one source lookup that did not "
        "complete: dblp (5). Those entries report api_error, not not_found.\n",
    )
    run_bibtex_check(_entries(), skip_prescreening=True)
    cond = last_source_condition()
    assert cond is not None and cond["entries_with_incomplete_lookups"] == 5


def test_a_healthy_run_clears_the_condition(fake_bibtex_check):
    fake_bibtex_check(0, "INFO: Loaded 3 entries\nINFO: done\n")
    run_bibtex_check(_entries(), skip_prescreening=True)
    assert last_source_condition() is None


def test_the_condition_lands_on_the_result(fake_bibtex_check, monkeypatch):
    """Where a reader of the JSON will find it: beside the numbers."""
    from hallmark.cli import _stamp_provenance

    fake_bibtex_check(EXIT_SOURCE_OUTAGE, REAL_OUTAGE_OUTPUT)
    monkeypatch.setenv(ALLOW_OUTAGE_ENV, "1")
    run_bibtex_check(_entries(), skip_prescreening=True)
    result = EvaluationResult(
        tool_name="bibtexupdater",
        split_name="dev_public",
        num_entries=3,
        num_hallucinated=1,
        num_valid=2,
        detection_rate=0.0,
        false_positive_rate=0.0,
        f1_hallucination=0.0,
        tier_weighted_f1=0.0,
    )
    _stamp_provenance(result, Namespace(baseline="bibtexupdater", split=None))
    assert result.source_condition == last_source_condition()
    assert result.to_dict()["source_condition"]["entries_total"] == 1119


def test_the_ensemble_does_not_swallow_an_outage(monkeypatch):
    """A component that disowned its run must not be silently dropped.

    Observed live: with DBLP down, ``ensemble`` logged "skipping bibtexupdater"
    and scored ``doi_only`` alone under the ensemble's name.
    """
    monkeypatch.setattr(R, "check_available", lambda name: (True, ""))

    def _outage(entries, **kw):
        raise SourceOutageError("bibtex-check reported a source outage (exit 5)")

    monkeypatch.setattr(R._REGISTRY["doi_only"], "runner", lambda entries, **kw: [])
    monkeypatch.setattr(R._REGISTRY["bibtexupdater"], "runner", _outage)
    entries = [
        BenchmarkEntry(
            bibtex_key=f"e{i}", bibtex_type="article", fields={"title": f"T{i}"}, label="VALID"
        )
        for i in range(3)
    ]
    with pytest.raises(SourceOutageError):
        R.run_baseline("ensemble", entries)
