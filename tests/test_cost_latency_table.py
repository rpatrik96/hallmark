"""Tests for ``scripts/compute_cost_latency_table.py``.

The table answers a reviewer's fairness question about DeepSeek-R1's
chain-of-thought, so the properties that carry the argument are the ones worth
locking down:

* the robust statistics behave as advertised on retry-contaminated data
  (median and p90 ignore a stray multi-hour stall that swamps the mean);
* every zero-shot row really does record one API call per entry;
* the committed CSV and LaTeX outputs are reproducible from the dumps;
* the price table has exactly one definition shared with the legacy script.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import compute_cost_latency_table as cct  # noqa: E402

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None


# --- Robust statistics ------------------------------------------------------


def test_percentile_matches_numpy_linear_interpolation() -> None:
    """p90 must agree with the numpy default so the reported tail is standard."""
    numpy = pytest.importorskip("numpy")
    values = [0.5, 1.0, 1.25, 2.0, 3.5, 4.0, 7.75, 9.0, 12.0, 30.0, 1894.46]
    for q in (50, 90, 95, 99):
        assert cct.percentile(values, q) == pytest.approx(float(numpy.percentile(values, q)))


def test_percentile_handles_degenerate_inputs() -> None:
    assert cct.percentile([4.2], 90) == pytest.approx(4.2)
    assert cct.percentile([1.0, 3.0], 50) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        cct.percentile([], 90)


def test_median_and_p90_survive_a_retry_stall_that_swamps_the_mean() -> None:
    """The measurement caveat in prose, asserted in code.

    ``wall_clock_seconds`` includes client-side backoff, so a single stalled
    entry can move the mean by an order of magnitude. The headline statistics
    must not move with it.
    """
    clean = [19.0] * 100
    stalled = [*clean[:-1], 34321.25]  # the DeepSeek-R1 test_public hang

    assert cct.percentile(clean, 90) == pytest.approx(cct.percentile(stalled, 90))
    from statistics import fmean, median

    assert median(clean) == pytest.approx(median(stalled))
    assert fmean(stalled) > 10 * fmean(clean)


def test_trimmed_mean_is_labelled_and_drops_both_tails() -> None:
    values = [*([10.0] * 98), 0.001, 5000.0]
    assert cct.trimmed_mean(values, frac=0.05) == pytest.approx(10.0)
    # frac=0 must be the plain mean: no silent trimming anywhere.
    from statistics import fmean

    assert cct.trimmed_mean(values, frac=0.0) == pytest.approx(fmean(values))


# --- Cost model -------------------------------------------------------------


def test_usd_per_1k_uses_list_prices() -> None:
    # GPT-5.1 at 600 prompt + 80 completion tokens: (600*1.25 + 80*10)/1e6 * 1000.
    assert cct.usd_per_1k("openai/gpt-5.1", 600, 80) == pytest.approx(1.55)
    # No model or an unpriced tool costs nothing: DOI-only issues no LLM calls.
    assert cct.usd_per_1k(None, 600, 80) == 0.0
    assert cct.usd_per_1k("not/a-model", 600, 80) == 0.0


def test_prompt_share_splits_a_measured_total_consistently() -> None:
    total = ASSUMED = cct.ASSUMED_PROMPT_TOKENS + cct.ASSUMED_COMPLETION_TOKENS
    prompt = total * cct.PROMPT_SHARE
    assert prompt == pytest.approx(cct.ASSUMED_PROMPT_TOKENS)
    assert ASSUMED - prompt == pytest.approx(cct.ASSUMED_COMPLETION_TOKENS)


def test_price_table_is_shared_with_the_legacy_script() -> None:
    """One price table, so the two scripts cannot drift apart."""
    import compute_baseline_costs as cbc

    assert cbc.PRICES is cct.PRICES
    assert cbc.ASSUMED_PROMPT_TOKENS == cct.ASSUMED_PROMPT_TOKENS
    assert cbc.ASSUMED_COMPLETION_TOKENS == cct.ASSUMED_COMPLETION_TOKENS


def test_every_priced_model_is_reachable_from_the_registry() -> None:
    registry_models = {b.model_id for b in cct.BASELINES if b.model_id}
    assert registry_models <= set(cct.PRICES)


def test_measured_total_tokens_parses_the_agentic_reason_field() -> None:
    rows = [
        {"reason": "[Agentic|tool|tools=search_crossref|tokens=2469] The entry claims..."},
        {"reason": "[Agentic|parametric|tools=none|tokens=1335] No lookup needed."},
        {"reason": "no token record here"},
    ]
    assert cct.measured_total_tokens(rows) == [2469, 1335]


# --- Registry and dumps -----------------------------------------------------


def test_registry_groups_match_the_paper_table() -> None:
    assert {b.group for b in cct.BASELINES} == set(cct.GROUP_HEADER)
    # Groups appear as contiguous blocks, in tab:results order.
    seen: list[str] = []
    for b in cct.BASELINES:
        if not seen or seen[-1] != b.group:
            seen.append(b.group)
    assert seen == list(cct.GROUP_HEADER)


def test_every_registry_source_exists() -> None:
    for b in cct.BASELINES:
        for rel in (b.dump, b.aggregate):
            if rel:
                assert (_REPO / rel).is_file(), f"{b.tool}: missing {rel}"


@pytest.mark.parametrize(
    "baseline",
    [b for b in cct.BASELINES if b.group == "Zero-shot LLMs" and b.dump],
    ids=lambda b: b.tool,
)
def test_zero_shot_rows_record_exactly_one_api_call_per_entry(baseline: cct.Baseline) -> None:
    """The direct answer to the fairness charge, including for DeepSeek-R1.

    ``api_calls`` counts harness-issued completion requests, so this asserts
    that chain-of-thought buys DeepSeek-R1 no extra calls, not that the SDK
    never retried underneath (retries are not counted; see the module docstring).
    """
    keys = cct.split_keys(baseline.split)
    rows = [r for r in cct.read_jsonl(_REPO / baseline.dump) if r["bibtex_key"] in keys]
    assert rows
    assert {r["api_calls"] for r in rows} == {1}


def test_agentic_rows_record_more_than_one_call_and_real_token_totals() -> None:
    for b in cct.BASELINES:
        if b.group != "Agentic":
            continue
        keys = cct.split_keys(b.split)
        rows = [r for r in cct.read_jsonl(_REPO / b.dump) if r["bibtex_key"] in keys]
        assert min(r["api_calls"] for r in rows) >= 2, b.tool
        # Token totals are measured on this path, so pricing is not a pure guess.
        assert len(cct.measured_total_tokens(rows)) == len(rows), b.tool


def test_split_key_sets_are_disjoint_and_correctly_sized() -> None:
    dev, test = cct.split_keys("dev_public"), cct.split_keys("test_public")
    assert len(dev) == 1119
    assert len(test) == 831
    assert not dev & test


def test_gpt51_checkpoint_is_split_filtered() -> None:
    """The GPT-5.1 checkpoint holds both splits; the row must use dev only."""
    row = next(b for b in cct.BASELINES if b.tool == "GPT-5.1 (zero-shot)")
    all_rows = cct.read_jsonl(_REPO / row.dump)
    assert len(all_rows) == 1119 + 831
    cct.compute(row)
    assert row.computed["n"] == 1119


# --- End-to-end reproducibility --------------------------------------------


@pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
def test_committed_outputs_are_reproducible() -> None:
    """`--check` must pass against the committed CSV and LaTeX bodies."""
    for b in cct.BASELINES:
        cct.compute(b)
    rows = cct.csv_rows(cct.BASELINES)
    assert (_REPO / "tables" / "baseline_cost_latency.csv").read_text() == cct.render_csv(rows)
    assert (_REPO / "tables" / "baseline_cost_latency.tex").read_text() == cct.tex_body(
        cct.BASELINES
    )


def test_deepseek_r1_latency_is_measured_not_estimated() -> None:
    """The defect this table was written to fix: R1 was a hand-written estimate."""
    row = next(b for b in cct.BASELINES if b.tool == "DeepSeek-R1")
    assert row.provenance == cct.PROV_PER_ENTRY
    assert row.dump is not None
    cct.compute(row)
    assert row.computed["n"] == 1119
    assert row.computed["mean_api_calls"] == pytest.approx(1.0)
    # Median well under the mean: the mean carries a 1,894 s retry stall.
    assert row.computed["median_sec"] < row.computed["mean_sec"]
    assert row.computed["max_sec"] > 1000


def test_rows_without_timing_report_no_latency_rather_than_a_guess() -> None:
    row = next(b for b in cct.BASELINES if b.tool.startswith("bibtex-updater"))
    assert row.provenance == cct.PROV_NONE
    cct.compute(row)
    assert row.computed["median_sec"] is None
    assert row.computed["p90_sec"] is None
    assert row.computed["mean_sec"] is None
    # api_calls is still measured in the pinned aggregate.
    assert row.computed["mean_api_calls"] == pytest.approx(1.9902, abs=1e-3)


def test_tex_body_marks_provenance_and_is_valid_six_column_latex() -> None:
    for b in cct.BASELINES:
        cct.compute(b)
    body = cct.tex_body(cct.BASELINES)
    for line in body.splitlines():
        if not line.startswith("%") and line != r"\midrule":
            assert line.rstrip().endswith(r"\\"), line
            if not line.startswith(r"\multicolumn"):
                assert line.count("&") == 5, line
    # Re-run and aggregate-only rows carry a footnote marker.
    assert r"Claude Sonnet~4.6$^{\ddagger}$" in body
    assert r"\gls{doi}-only$^{\dagger}$" in body
    assert r"\texttt{bibtex-updater} (v1.2.0)$^{\ast}$" in body


def test_crosscheck_flags_the_two_known_anthropic_reruns() -> None:
    """Dumps must reproduce their pinned aggregate, except where we say they don't."""
    differs = set()
    for b in cct.BASELINES:
        cct.compute(b)
        if "DIFFERS" in str(b.computed["crosscheck"]):
            differs.add(b.tool)
    assert differs == {"Claude Sonnet 4.6", "Claude Opus 4.7"}
    for tool in differs:
        row = next(b for b in cct.BASELINES if b.tool == tool)
        assert row.provenance == cct.PROV_PER_ENTRY_RERUN


def test_dumps_carry_complete_timing_coverage() -> None:
    """No silently-missing wall_clock_seconds behind a computed median."""
    for b in cct.BASELINES:
        if not b.dump:
            continue
        keys = cct.split_keys(b.split)
        rows = [r for r in cct.read_jsonl(_REPO / b.dump) if r["bibtex_key"] in keys]
        missing = [r for r in rows if r.get("wall_clock_seconds") is None]
        assert not missing, f"{b.tool}: {len(missing)} entries lack wall_clock_seconds"


def test_price_table_provenance_is_a_dated_list_price() -> None:
    """Guard the claim the caption makes about where the dollars come from."""
    source = (_SCRIPTS_DIR / "compute_cost_latency_table.py").read_text()
    assert "OpenRouter list price as of 2026-05-04" in source


def test_no_network_imports() -> None:
    """The script must be re-runnable offline with no API keys."""
    source = (_SCRIPTS_DIR / "compute_cost_latency_table.py").read_text()
    for banned in ("import requests", "import httpx", "import openai", "urllib.request"):
        assert banned not in source


def test_aggregate_only_rows_recover_the_mean_from_entries_per_second() -> None:
    row = next(b for b in cct.BASELINES if b.tool == "DOI-only")
    assert row.provenance == cct.PROV_AGGREGATE
    cct.compute(row)
    pinned = json.loads((_REPO / row.aggregate).read_text())
    assert row.computed["mean_sec"] == pytest.approx(1.0 / pinned["cost_efficiency"])
    assert row.computed["median_sec"] is None
