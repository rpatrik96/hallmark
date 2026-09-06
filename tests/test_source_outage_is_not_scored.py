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

import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from hallmark.baselines import registry as R
from hallmark.baselines.bibtexupdater import (
    ALLOW_OUTAGE_ENV,
    BIBTEX_CHECK_BIN_ENV,
    EXIT_SOURCE_OUTAGE,
    SourceOutageError,
    last_source_condition,
    parse_source_condition,
    run_bibtex_check,
    run_bibtex_check_with_health,
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


def test_the_exit_code_matches_the_pinned_tool():
    """Read the contract from the build the wrapper would actually run."""
    from hallmark.baselines import bibtexupdater as btu

    binary = btu.resolve_bibtex_check_bin()
    if binary is None:
        pytest.skip("bibtex-check is not installed")
    with open(Path(binary).resolve()) as script:
        first_line = script.readline().strip()
    if not first_line.startswith("#!"):
        pytest.skip("bibtex-check console script has no shebang")
    interpreter = first_line.removeprefix("#!").strip().split()[0]
    probe = subprocess.run(
        [
            interpreter,
            "-c",
            "from bibtex_updater.fact_checker import EXIT_SOURCE_OUTAGE; print(EXIT_SOURCE_OUTAGE)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert probe.returncode == 0, probe.stderr
    assert int(probe.stdout.strip()) == EXIT_SOURCE_OUTAGE


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

    def _install(
        returncode: int,
        output: str,
        records: list[dict] | None = None,
        stderr: str = "",
    ):
        calls: list[list[str]] = []

        def _run(cmd, **kw):
            calls.append(cmd)
            if records is not None:
                jsonl_path = Path(cmd[cmd.index("--jsonl") + 1])
                jsonl_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            return subprocess.CompletedProcess(cmd, returncode, stdout=output, stderr=stderr)

        monkeypatch.setattr(btu.subprocess, "run", _run)
        return calls

    return _install


def test_exit_5_raises_through_the_public_runner(fake_bibtex_check):
    fake_bibtex_check(EXIT_SOURCE_OUTAGE, REAL_OUTAGE_OUTPUT)
    with pytest.raises(SourceOutageError, match="285 of 1119"):
        run_bibtex_check(_entries(), skip_prescreening=True)


def test_strict_exit_4_with_an_outage_report_is_refused(fake_bibtex_check):
    fake_bibtex_check(4, REAL_OUTAGE_OUTPUT)
    with pytest.raises(SourceOutageError, match="285 of 1119"):
        run_bibtex_check(_entries(), skip_prescreening=True)


def test_strict_exit_4_below_the_threshold_is_not_an_outage(fake_bibtex_check):
    """bibtex-check prints its source report under the threshold as well, so a
    strict-mode exit 4 beside an 8% report is a strict verdict, not an outage."""
    fake_bibtex_check(
        4,
        "WARNING: 4 of 50 entries (8.0%) had at least one source lookup that did not "
        "complete: dblp (4).\n",
    )
    run_bibtex_check(_entries(), skip_prescreening=True)


def test_threshold_fraction_is_refused_even_when_exit_is_zero(fake_bibtex_check):
    fake_bibtex_check(
        0,
        "WARNING: 5 of 50 entries (10.0%) had at least one source lookup that did not "
        "complete: dblp (5).\n",
    )
    with pytest.raises(SourceOutageError, match="5 of 50"):
        run_bibtex_check(_entries(), skip_prescreening=True)


def test_wrapper_passes_its_outage_threshold_explicitly(fake_bibtex_check):
    calls = fake_bibtex_check(0, "INFO: done\n")
    run_bibtex_check(_entries(), skip_prescreening=True)
    (cmd,) = calls
    threshold_index = cmd.index("--outage-threshold")
    assert float(cmd[threshold_index + 1]) == pytest.approx(0.10)


def test_exit_5_without_a_source_report_logs_a_broken_parser(fake_bibtex_check, caplog):
    fake_bibtex_check(EXIT_SOURCE_OUTAGE, "ERROR: source outage\n")
    with (
        caplog.at_level("ERROR", logger="hallmark.baselines.bibtexupdater"),
        pytest.raises(SourceOutageError),
    ):
        run_bibtex_check(_entries(), skip_prescreening=True)
    assert any("source-condition report" in record.message for record in caplog.records)


def test_a_missing_pinned_binary_never_falls_back_to_path(tmp_path, monkeypatch):
    missing = tmp_path / "missing" / "bibtex-check"
    monkeypatch.setenv(BIBTEX_CHECK_BIN_ENV, str(missing))

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("subprocess.run must not be called for a missing pinned binary")

    monkeypatch.setattr(subprocess, "run", _must_not_run)
    with pytest.raises(RuntimeError, match=f"{BIBTEX_CHECK_BIN_ENV}.*{missing}"):
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
        "WARNING: 4 of 50 entries (8.0%) had at least one source lookup that did not "
        "complete: dblp (4). Those entries report api_error, not not_found.\n",
    )
    run_bibtex_check(_entries(), skip_prescreening=True)
    cond = last_source_condition()
    assert cond is not None and cond["entries_with_incomplete_lookups"] == 4


def test_per_source_failures_come_from_jsonl_records(fake_bibtex_check):
    records = [
        {"key": "e0", "status": "api_error", "sources_failed": ["dblp", "openalex"]},
        {"key": "e1", "status": "api_error", "sources_failed": ["dblp"]},
        {"key": "e2", "status": "verified", "sources_failed": []},
    ]
    fake_bibtex_check(
        0,
        "WARNING: 2 of 50 entries (4.0%) had at least one source lookup that did not "
        "complete: stale-log-value (99).\n",
        records,
    )
    run_bibtex_check(_entries(), skip_prescreening=True)
    cond = last_source_condition()
    assert cond is not None
    assert cond["per_source_failures"] == {"dblp": 2, "openalex": 1}


def test_record_coverage_flags_reach_batch_health(fake_bibtex_check):
    records = [
        {
            "key": f"e{i}",
            "status": "unconfirmed" if i < 20 else "verified",
            "coverage_incomplete": i < 20,
        }
        for i in range(40)
    ]
    fake_bibtex_check(0, "INFO: done\n", records)
    _, _, health = run_bibtex_check_with_health(_entries(40), skip_prescreening=True)
    assert health.coverage_incomplete == 20
    assert health.no_evidence == 20
    assert health.suspected_transport_failure


def test_a_healthy_run_records_its_pace_and_contact_condition(fake_bibtex_check):
    fake_bibtex_check(0, "INFO: Loaded 3 entries\nINFO: done\n")
    run_bibtex_check(_entries(), skip_prescreening=True)
    assert last_source_condition() == {
        "rate_limit": 120,
        "workers": 8,
        "mailto_configured": False,
        "mailto_domain": None,
    }


def test_command_and_condition_record_configured_pace_and_contact(
    fake_bibtex_check, monkeypatch, caplog
):
    mailto = "researcher@example.org"
    monkeypatch.setenv("BIBTEX_CHECK_MAILTO", mailto)
    calls = fake_bibtex_check(0, "INFO: done\n")
    with caplog.at_level("INFO", logger="hallmark.baselines.bibtexupdater"):
        run_bibtex_check(_entries(), skip_prescreening=True)

    (cmd,) = calls
    assert cmd[cmd.index("--rate-limit") + 1] == "120"
    assert cmd[cmd.index("--workers") + 1] == "8"
    assert cmd[cmd.index("--mailto") + 1] == mailto
    assert last_source_condition() == {
        "rate_limit": 120,
        "workers": 8,
        "mailto_configured": True,
        "mailto_domain": "example.org",
    }
    assert "scale 120/45" in caplog.text
    assert mailto not in caplog.text


def test_clean_exit_stderr_is_not_discarded(fake_bibtex_check, caplog):
    warning = "No contact email configured; using placeholder"
    fake_bibtex_check(0, "INFO: done\n", stderr=f"WARNING: {warning}\n")
    with caplog.at_level("WARNING", logger="hallmark.baselines.bibtexupdater"):
        run_bibtex_check(_entries(), skip_prescreening=True)
    assert warning in caplog.text


def test_exit_2_is_logged_but_partial_jsonl_is_retained(fake_bibtex_check, caplog):
    records = [{"key": "e0", "status": "verified", "unconfirmed_fields": []}]
    fake_bibtex_check(2, "", records, stderr="parse failure")
    with caplog.at_level("ERROR", logger="hallmark.baselines.bibtexupdater"):
        preds = run_bibtex_check(_entries(), skip_prescreening=True)
    assert any(pred.bibtex_key == "e0" and pred.reason == "Status: verified" for pred in preds)
    assert "exit 2" in caplog.text


def test_unconfirmed_fields_capability_is_checked_per_run(fake_bibtex_check, caplog):
    legacy = [{"key": "e0", "status": "verified"}]
    fake_bibtex_check(0, "", legacy)
    with caplog.at_level("WARNING", logger="hallmark.baselines.bibtexupdater"):
        run_bibtex_check(_entries(), skip_prescreening=True)
    assert "unconfirmed_fields" in caplog.text

    caplog.clear()
    modern = [{"key": "e0", "status": "verified", "unconfirmed_fields": []}]
    fake_bibtex_check(0, "", modern)
    with caplog.at_level("WARNING", logger="hallmark.baselines.bibtexupdater"):
        run_bibtex_check(_entries(), skip_prescreening=True)
    assert "unconfirmed_fields" not in caplog.text


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
