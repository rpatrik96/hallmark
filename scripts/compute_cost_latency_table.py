"""Recompute the per-entry cost and latency table from the released timing dumps.

Written for NeurIPS 2026 reviewer 16Cs, who asked whether DeepSeek-R1's
chain-of-thought makes the zero-shot cohort an unfair comparison, and for a
cost- or latency comparison to back the answer. It supersedes the flat-estimate
table produced by ``scripts/compute_baseline_costs.py``: every row that has a
per-entry dump is now recomputed from that dump instead of being guessed.

Why the median and not the mean
-------------------------------
``wall_clock_seconds`` is *not* a clean latency measurement. The timed region in
``hallmark/baselines/llm_verifier.py`` opens before ``call_fn`` and closes after
it, and the OpenAI SDK client is constructed with ``max_retries=5`` and
``timeout=120.0``, so client-side backoff and provider queueing land inside the
recorded interval. The dumps show it plainly: the DeepSeek-R1 ``test_public``
run contains a 34,321-second entry (the 9.5-hour mid-stream network hang
documented in ``scripts/parallel_resume_test_public.py``), and Llama 4 Maverick
and Qwen3-VL-235B each contain a ~4,300-second entry. No single completion takes
hours, so the mean is a network-conditions statistic, not a compute statistic.
We therefore report the **median** as the headline latency and **p90** as the
tail, keep the mean and the maximum in the CSV so the contamination stays
visible, and additionally report a *labelled* 10% trimmed mean. No outlier is
dropped from any headline number.

Why every dollar figure is an estimate
--------------------------------------
The zero-shot path discards ``resp.usage`` (``llm_verifier.py`` returns only
``resp.choices[0].message.content``), so per-call token counts were never
recorded for the zero-shot cohort; their cost is priced from a fixed token
assumption and provider list prices. The *agentic* path is the exception: it
accumulates ``resp.usage.total_tokens`` and embeds the total in the prediction's
``reason`` string (``llm_agentic.py``), so the agentic rows are priced from a
measured token total, split at the same prompt:completion ratio as the
assumption. Either way the dollars are estimates, and the ``token_basis`` column
says which kind.

Offline by construction: reads only local JSONL dumps and pinned aggregate
JSONs. No network access, no API keys, no model inference.

Outputs
-------
``tables/baseline_cost_latency.csv``   full audit table, one row per baseline
``tables/baseline_cost_latency.tex``   LaTeX row bodies for ``tab:cost_latency``

Usage
-----
    uv run python scripts/compute_cost_latency_table.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Price table: (prompt_$/M_tok, completion_$/M_tok)
# Source: OpenRouter list price as of 2026-05-04.
# This dict is the single source of truth for baseline pricing;
# scripts/compute_baseline_costs.py imports it from here.
# ---------------------------------------------------------------------------
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-5.1": (1.25, 10.0),
    "openai/gpt-5.4": (1.25, 10.0),
    "anthropic/claude-sonnet-4.6": (3.0, 15.0),
    "anthropic/claude-opus-4.7": (5.0, 25.0),
    "deepseek/deepseek-r1": (0.55, 2.19),
    "deepseek/deepseek-v3.2": (0.27, 1.10),
    "qwen/qwen3-235b-a22b-2507": (0.20, 0.60),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.20, 0.60),
    "mistralai/mistral-large-2512": (2.0, 6.0),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-2.5-pro": (3.5, 15.0),
    "meta-llama/llama-4-maverick": (0.20, 0.60),
    # DOI-only and bibtex-updater issue no LLM calls: $0 LLM cost.
}

# Token assumption used when usage was not recorded. ~600 prompt tokens for a
# BibTeX entry plus the shared verification prompt; ~80 completion tokens for
# the short JSON verdict (~320 chars at ~4 chars/token).
ASSUMED_PROMPT_TOKENS = 600
ASSUMED_COMPLETION_TOKENS = 80
# Prompt share of the assumed budget, reused to split measured *total* token
# counts (the agentic path records only the combined total).
PROMPT_SHARE = ASSUMED_PROMPT_TOKENS / (ASSUMED_PROMPT_TOKENS + ASSUMED_COMPLETION_TOKENS)

# Fraction trimmed from *each* tail for the labelled trimmed mean.
TRIM_FRACTION = 0.05

# Provenance vocabulary for the latency columns.
PROV_PER_ENTRY = "measured (per-entry dump)"
PROV_PER_ENTRY_RERUN = "measured (per-entry dump; timing re-run, see note)"
PROV_AGGREGATE = "measured (aggregate mean only; no per-entry dump)"
PROV_NONE = "not recorded"

# Token-basis vocabulary for the cost column.
TOK_ASSUMED = f"assumed {ASSUMED_PROMPT_TOKENS}+{ASSUMED_COMPLETION_TOKENS} per LLM call"
TOK_ASSUMED_LOWER_BOUND = (
    f"assumed {ASSUMED_PROMPT_TOKENS}+{ASSUMED_COMPLETION_TOKENS} per LLM call "
    "(lower bound: tool evidence is injected into the prompt)"
)
TOK_MEASURED = "measured total tokens (median), split at the assumed prompt share"
TOK_NO_LLM = "no LLM calls"

_TOKENS_RE = re.compile(r"tokens=(\d+)")


@dataclass
class Baseline:
    """One row of the cost/latency table."""

    tool: str
    group: str
    model_id: str | None
    split: str
    # Per-entry JSONL dump, relative to the repo root.
    dump: str | None = None
    # Pinned aggregate JSON, used for rows without a per-entry dump and as a
    # cross-check for rows that have one.
    aggregate: str | None = None
    provenance: str = PROV_PER_ENTRY
    token_basis: str = TOK_ASSUMED
    # Number of LLM completions charged per entry when tokens are assumed.
    llm_calls_per_entry: float = 1.0
    notes: str = ""
    computed: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry. Order and grouping mirror tab:results in the paper. Each row points
# at the dump whose aggregate reproduces the pinned tab:results run, verified by
# the entries_per_second cross-check in ``crosscheck`` below.
# ---------------------------------------------------------------------------
BASELINES: list[Baseline] = [
    # --- Citation-database tools -------------------------------------------
    Baseline(
        tool="DOI-only",
        group="Citation-database tools",
        model_id=None,
        split="dev_public",
        dump=None,
        aggregate="data/v1.0/baseline_results/doi_only_dev_public.json",
        provenance=PROV_AGGREGATE,
        token_basis=TOK_NO_LLM,
        llm_calls_per_entry=0.0,
        notes="per-entry predictions not retained; mean recovered as 1/entries_per_second",
    ),
    # --- Zero-shot LLMs (tab:results order, FPR ascending) -----------------
    Baseline(
        tool="Gemini 2.5 Pro",
        group="Zero-shot LLMs",
        model_id="google/gemini-2.5-pro",
        split="test_public",
        dump="results/checkpoints/llm_openrouter_gemini_pro/openrouter_google_gemini-2.5-pro.jsonl",
        aggregate="data/v1.0/baseline_results/llm_openrouter_gemini_pro_test_public.json",
        notes="no per-entry dev_public dump; dev aggregate mean is 1/0.0868 = 11.5 s",
    ),
    Baseline(
        tool="Claude Opus 4.7",
        group="Zero-shot LLMs",
        model_id="anthropic/claude-opus-4.7",
        split="dev_public",
        dump="results/llm_openrouter_claude_opus_4_7_dev_public_predictions.jsonl",
        aggregate="data/v1.0/baseline_results/llm_openrouter_claude_opus_4_7_dev_public.json",
        provenance=PROV_PER_ENTRY_RERUN,
        notes="dev_public record behind tab:results is summary-only; timing from a "
        "later per-entry re-run of the same configuration",
    ),
    Baseline(
        tool="Gemini 2.5 Flash",
        group="Zero-shot LLMs",
        model_id="google/gemini-2.5-flash",
        split="dev_public",
        dump="results/llm_openrouter_gemini_flash_dev_public_predictions.jsonl",
        aggregate="data/v1.0/baseline_results/llm_openrouter_gemini_flash_dev_public.json",
    ),
    Baseline(
        tool="Claude Sonnet 4.6",
        group="Zero-shot LLMs",
        model_id="anthropic/claude-sonnet-4.6",
        split="dev_public",
        dump="results/llm_openrouter_claude_sonnet_4_6_dev_public_predictions.jsonl",
        aggregate="data/v1.0/baseline_results/llm_openrouter_claude_sonnet_4_6_dev_public.json",
        provenance=PROV_PER_ENTRY_RERUN,
        notes="dev_public record behind tab:results is summary-only; timing from a "
        "later per-entry re-run of the same configuration",
    ),
    Baseline(
        tool="Llama 4 Maverick",
        group="Zero-shot LLMs",
        model_id="meta-llama/llama-4-maverick",
        split="test_public",
        dump="results/checkpoints/llm_openrouter_llama_4_maverick_test_public/"
        "openrouter_meta-llama_llama-4-maverick.jsonl",
        aggregate="data/v1.0/baseline_results/llm_openrouter_llama_4_maverick_test_public.json",
        notes="no per-entry dev_public dump; contains a ~4,361 s stall, so the mean is "
        "network-dominated",
    ),
    Baseline(
        tool="GPT-5.4 (zero-shot)",
        group="Zero-shot LLMs",
        model_id="openai/gpt-5.4",
        split="dev_public",
        dump="results/checkpoints/llm_openai_gpt54_dev_public_v3/openai_gpt-5.4.jsonl",
        aggregate="data/v1.0/baseline_results/llm_openai_gpt54_dev_public.json",
    ),
    Baseline(
        tool="Mistral Large",
        group="Zero-shot LLMs",
        model_id="mistralai/mistral-large-2512",
        split="dev_public",
        dump="results/llm_openrouter_mistral_dev_public_predictions.jsonl",
        aggregate="data/v1.0/baseline_results/llm_openrouter_mistral_dev_public.json",
    ),
    Baseline(
        tool="GPT-5.1 (zero-shot)",
        group="Zero-shot LLMs",
        model_id="openai/gpt-5.1",
        split="dev_public",
        dump="results/checkpoints/llm_openai/openai_gpt-5.1.jsonl",
        aggregate="data/v1.0/baseline_results/llm_openai_dev_public.json",
        notes="checkpoint holds dev_public and test_public together; filtered by split keys",
    ),
    Baseline(
        tool="Qwen3-235B",
        group="Zero-shot LLMs",
        model_id="qwen/qwen3-235b-a22b-2507",
        split="dev_public",
        dump="results/llm_openrouter_qwen_dev_public_predictions.jsonl",
        aggregate="data/v1.0/baseline_results/llm_openrouter_qwen_dev_public.json",
    ),
    Baseline(
        tool="Qwen3-VL-235B",
        group="Zero-shot LLMs",
        model_id="qwen/qwen3-vl-235b-a22b-instruct",
        split="test_public",
        dump="results/checkpoints/llm_openrouter_qwen_max_test_public/"
        "openrouter_qwen_qwen3-vl-235b-a22b-instruct.jsonl",
        aggregate="data/v1.0/baseline_results/llm_openrouter_qwen_max_test_public.json",
        notes="no per-entry dev_public dump; contains a ~4,336 s stall",
    ),
    Baseline(
        tool="DeepSeek-R1",
        group="Zero-shot LLMs",
        model_id="deepseek/deepseek-r1",
        split="dev_public",
        dump="results/llm_openrouter_deepseek_r1_dev_public_predictions.jsonl",
        aggregate="data/v1.0/baseline_results/llm_openrouter_deepseek_r1_dev_public.json",
        notes="chain-of-thought is the model's own default, not requested by the prompt; "
        "one API call per entry like every other zero-shot row",
    ),
    Baseline(
        tool="DeepSeek-V3.2",
        group="Zero-shot LLMs",
        model_id="deepseek/deepseek-v3.2",
        split="dev_public",
        dump="results/llm_openrouter_deepseek_v3_dev_public_predictions.jsonl",
        aggregate="data/v1.0/baseline_results/llm_openrouter_deepseek_v3_dev_public.json",
    ),
    # --- Agentic ------------------------------------------------------------
    Baseline(
        tool="GPT-5.1 + CrossRef/OpenAlex/arXiv",
        group="Agentic",
        model_id="openai/gpt-5.1",
        split="test_public",
        dump="results/checkpoints/llm_agentic_openai_test_public/agentic_openai_openai_gpt-5.1.jsonl",
        aggregate="data/v1.0/baseline_results/llm_agentic_openai_test_public.json",
        token_basis=TOK_MEASURED,
        notes="no per-entry dev_public dump; mean api_calls on dev is 2.62",
    ),
    Baseline(
        tool="GPT-5.1 + bibtex-updater (agentic)",
        group="Agentic",
        model_id="openai/gpt-5.1",
        split="test_public",
        dump="results/checkpoints/llm_agentic_btu_openai_test_public/"
        "agentic_btu_openai_openai_gpt-5.1.jsonl",
        aggregate="data/v1.0/baseline_results/llm_agentic_btu_openai_test_public.json",
        token_basis=TOK_MEASURED,
        notes="no per-entry dev_public dump; mean api_calls on dev is 2.00",
    ),
    Baseline(
        tool="Sonnet 4.6 + bibtex-updater (agentic)",
        group="Agentic",
        model_id="anthropic/claude-sonnet-4.6",
        split="dev_public",
        dump="results/checkpoints/llm_agentic_btu_sonnet_4_6_dev_public_v2/"
        "agentic_btu_openai_anthropic_claude-sonnet-4.6.jsonl",
        aggregate="data/v1.0/baseline_results/llm_agentic_btu_sonnet_4_6_dev_public.json",
        token_basis=TOK_MEASURED,
    ),
    # --- Co-designed --------------------------------------------------------
    Baseline(
        tool="bibtex-updater (v1.2.0)",
        group="Co-designed",
        model_id=None,
        split="dev_public",
        dump=None,
        aggregate="data/v1.0/baseline_results/bibtexupdater_dev_public.json",
        provenance=PROV_NONE,
        token_basis=TOK_NO_LLM,
        llm_calls_per_entry=0.0,
        notes="entries_per_second is null in the pinned run and the raw per-entry "
        "records carry no timing, so no latency figure is available",
    ),
    Baseline(
        tool="GPT-5.1 + bibtex-updater (always-call)",
        group="Co-designed",
        model_id="openai/gpt-5.1",
        split="dev_public",
        dump="data/v1.0/baseline_results/llm_tool_augmented_dev_public.jsonl",
        aggregate=None,  # only a test_public aggregate exists; nothing to cross-check on dev
        token_basis=TOK_ASSUMED_LOWER_BOUND,
        notes="one LLM call per entry, preceded by an unconditional bibtex-check call; "
        "usage not recorded on this path",
    ),
]


# ---------------------------------------------------------------------------
# Loading and statistics
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping blank lines."""
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_keys(split: str) -> set[str]:
    """Return the bibtex keys belonging to a released split."""
    path = REPO / "data" / "v1.0" / f"{split}.jsonl"
    return {row["bibtex_key"] for row in read_jsonl(path)}


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (the numpy default), q in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def trimmed_mean(values: list[float], frac: float = TRIM_FRACTION) -> float:
    """Mean after dropping ``frac`` of the observations from each tail."""
    ordered = sorted(values)
    k = math.floor(len(ordered) * frac)
    kept = ordered[k : len(ordered) - k] if k and len(ordered) - 2 * k > 0 else ordered
    return statistics.fmean(kept)


def measured_total_tokens(rows: list[dict]) -> list[int]:
    """Extract the ``tokens=N`` totals the agentic path writes into ``reason``."""
    out: list[int] = []
    for row in rows:
        match = _TOKENS_RE.search(row.get("reason") or "")
        if match:
            out.append(int(match.group(1)))
    return out


def usd_per_1k(model_id: str | None, prompt_tok: float, completion_tok: float) -> float:
    """Estimated USD per 1,000 entries at list prices."""
    if model_id is None or model_id not in PRICES:
        return 0.0
    price_in, price_out = PRICES[model_id]
    per_entry = (prompt_tok * price_in + completion_tok * price_out) / 1_000_000
    return per_entry * 1000


def compute(baseline: Baseline) -> None:
    """Fill ``baseline.computed`` from the dump and/or the pinned aggregate."""
    aggregate: dict = {}
    if baseline.aggregate:
        aggregate = json.loads((REPO / baseline.aggregate).read_text())

    result: dict[str, object] = {
        "n": None,
        "median_sec": None,
        "p90_sec": None,
        "mean_sec": None,
        "trimmed_mean_sec": None,
        "max_sec": None,
        "mean_api_calls": None,
        "median_total_tokens": None,
        "crosscheck": "",
    }

    if baseline.dump:
        rows = read_jsonl(REPO / baseline.dump)
        keys = split_keys(baseline.split)
        rows = [r for r in rows if r["bibtex_key"] in keys]
        if not rows:
            raise SystemExit(f"{baseline.tool}: no rows for split {baseline.split}")
        waits = [float(r["wall_clock_seconds"]) for r in rows]
        calls = [float(r["api_calls"]) for r in rows]
        result.update(
            n=len(rows),
            median_sec=statistics.median(waits),
            p90_sec=percentile(waits, 90),
            mean_sec=statistics.fmean(waits),
            trimmed_mean_sec=trimmed_mean(waits),
            max_sec=max(waits),
            mean_api_calls=statistics.fmean(calls),
        )
        tokens = measured_total_tokens(rows)
        if tokens and len(tokens) == len(rows):
            result["median_total_tokens"] = statistics.median(tokens)
        # Cross-check the dump against the pinned aggregate for the same split.
        pinned_eps = aggregate.get("cost_efficiency")
        if pinned_eps and aggregate.get("split_name") == baseline.split:
            dump_eps = len(waits) / sum(waits)
            drift = abs(dump_eps - pinned_eps) / pinned_eps
            result["crosscheck"] = (
                f"entries_per_second {dump_eps:.5f} vs pinned {pinned_eps:.5f} "
                f"({'match' if drift < 0.01 else 'DIFFERS'})"
            )
    else:
        eps = aggregate.get("cost_efficiency")
        if eps:
            result["mean_sec"] = 1.0 / eps
            result["crosscheck"] = f"entries_per_second {eps:.5f} (pinned aggregate)"
        api = aggregate.get("mean_api_calls")
        if api is not None:
            result["mean_api_calls"] = float(api)
        # dev_public / test_public sizes are fixed by the release.
        result["n"] = len(split_keys(baseline.split))

    # Cost estimate.
    if baseline.token_basis == TOK_NO_LLM:
        result["prompt_tok"] = 0.0
        result["completion_tok"] = 0.0
        result["est_usd_per_1k"] = 0.0
    elif baseline.token_basis == TOK_MEASURED:
        total = result["median_total_tokens"]
        if total is None:
            raise SystemExit(f"{baseline.tool}: token_basis is measured but no totals found")
        prompt_tok = float(total) * PROMPT_SHARE
        completion_tok = float(total) - prompt_tok
        result["prompt_tok"] = prompt_tok
        result["completion_tok"] = completion_tok
        result["est_usd_per_1k"] = usd_per_1k(baseline.model_id, prompt_tok, completion_tok)
    else:
        prompt_tok = ASSUMED_PROMPT_TOKENS * baseline.llm_calls_per_entry
        completion_tok = ASSUMED_COMPLETION_TOKENS * baseline.llm_calls_per_entry
        result["prompt_tok"] = prompt_tok
        result["completion_tok"] = completion_tok
        result["est_usd_per_1k"] = usd_per_1k(baseline.model_id, prompt_tok, completion_tok)

    baseline.computed = result


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "tool",
    "group",
    "split",
    "n",
    "median_sec_per_entry",
    "p90_sec_per_entry",
    "mean_sec_per_entry",
    "trimmed_mean_sec_per_entry",
    "max_sec_per_entry",
    "mean_api_calls",
    "est_prompt_tok_per_entry",
    "est_completion_tok_per_entry",
    "est_usd_per_1k_entries",
    "token_basis",
    "latency_provenance",
    "source",
    "crosscheck",
    "note",
]


def _round(value: object, digits: int) -> object:
    return round(float(value), digits) if isinstance(value, (int, float)) else ""


def csv_rows(baselines: list[Baseline]) -> list[dict]:
    out = []
    for b in baselines:
        c = b.computed
        out.append(
            {
                "tool": b.tool,
                "group": b.group,
                "split": b.split,
                "n": c["n"] if c["n"] is not None else "",
                "median_sec_per_entry": _round(c["median_sec"], 2),
                "p90_sec_per_entry": _round(c["p90_sec"], 2),
                "mean_sec_per_entry": _round(c["mean_sec"], 2),
                "trimmed_mean_sec_per_entry": _round(c["trimmed_mean_sec"], 2),
                "max_sec_per_entry": _round(c["max_sec"], 2),
                "mean_api_calls": _round(c["mean_api_calls"], 4),
                "est_prompt_tok_per_entry": _round(c.get("prompt_tok"), 0),
                "est_completion_tok_per_entry": _round(c.get("completion_tok"), 0),
                "est_usd_per_1k_entries": _round(c.get("est_usd_per_1k"), 3),
                "token_basis": b.token_basis,
                "latency_provenance": b.provenance,
                "source": b.dump or b.aggregate or "",
                "crosscheck": c["crosscheck"],
                "note": b.notes,
            }
        )
    return out


def render_csv(rows: list[dict]) -> str:
    """Render the audit rows as CSV text (kept in memory so --check can diff it)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


_TEX_ESCAPES = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}

# Paper-side row labels, where tab:results wording differs from the CSV label.
TEX_LABEL = {
    "DOI-only": r"\gls{doi}-only",
    "Gemini 2.5 Pro": r"Gemini~2.5~Pro",
    "Gemini 2.5 Flash": r"Gemini~2.5~Flash",
    "Claude Opus 4.7": r"Claude Opus~4.7",
    "Claude Sonnet 4.6": r"Claude Sonnet~4.6",
    "Llama 4 Maverick": r"Llama~4~Maverick",
    "GPT-5.1 + CrossRef/OpenAlex/arXiv": r"GPT-5.1 $+$ CrossRef/OpenAlex/arXiv",
    "GPT-5.1 + bibtex-updater (agentic)": r"GPT-5.1 $+$ \texttt{bibtex-updater} (agentic)",
    "Sonnet 4.6 + bibtex-updater (agentic)": (r"Sonnet~4.6 $+$ \texttt{bibtex-updater} (agentic)"),
    "bibtex-updater (v1.2.0)": r"\texttt{bibtex-updater} (v1.2.0)",
    "GPT-5.1 + bibtex-updater (always-call)": (
        r"GPT-5.1 $+$ \texttt{bibtex-updater} (always-call)"
    ),
}

GROUP_HEADER = {
    "Citation-database tools": r"\emph{Citation-database tools}",
    "Zero-shot LLMs": r"\emph{Zero-shot \glspl{llm} (one API call per entry)}",
    "Agentic": r"\emph{Agentic (tool-use; up to 5 tool calls per entry)}",
    "Co-designed": r"\emph{Co-designed (reference upper bound)}",
}

# Rows whose latency provenance needs a marker in the paper table.
PROV_MARKER = {
    PROV_PER_ENTRY: "",
    PROV_PER_ENTRY_RERUN: r"$^{\ddagger}$",
    PROV_AGGREGATE: r"$^{\dagger}$",
    PROV_NONE: r"$^{\ast}$",
}


def tex_escape(text: str) -> str:
    for raw, esc in _TEX_ESCAPES.items():
        text = text.replace(raw, esc)
    return text


def tex_body(baselines: list[Baseline]) -> str:
    """Emit the LaTeX row bodies for tab:cost_latency (6 columns)."""
    lines = [
        "% Generated by scripts/compute_cost_latency_table.py — do not edit by hand.",
        "% Columns: verifier & n & median s/entry & p90 s/entry & mean API calls & est. USD/1k",
    ]
    current = None
    for b in baselines:
        if b.group != current:
            if current is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{6}}{{l}}{{{GROUP_HEADER[b.group]}}} \\")
            current = b.group
        c = b.computed
        label = TEX_LABEL.get(b.tool, tex_escape(b.tool)) + PROV_MARKER[b.provenance]
        n = f"{int(c['n']):,}" if c["n"] is not None else "---"
        median = f"{float(c['median_sec']):.2f}" if c["median_sec"] is not None else "---"
        p90 = f"{float(c['p90_sec']):.2f}" if c["p90_sec"] is not None else "---"
        calls = f"{float(c['mean_api_calls']):.3f}" if c["mean_api_calls"] is not None else "---"
        usd = f"{float(c['est_usd_per_1k']):.2f}"
        lines.append(rf"{label} & {n} & {median} & {p90} & {calls} & {usd} \\")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and diff against the committed outputs without writing",
    )
    args = parser.parse_args()

    for b in BASELINES:
        compute(b)

    out_dir = REPO / "tables"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "baseline_cost_latency.csv"
    tex_path = out_dir / "baseline_cost_latency.tex"

    rows = csv_rows(BASELINES)
    body = tex_body(BASELINES)
    csv_text = render_csv(rows)

    if args.check:
        stale = [
            f"{path} {'missing' if not path.exists() else 'out of date'}"
            for path, expected in ((csv_path, csv_text), (tex_path, body))
            if not path.exists() or path.read_text() != expected
        ]
        if stale:
            print("STALE: " + "; ".join(stale), file=sys.stderr)
            raise SystemExit(1)
        print("outputs are up to date")
        return

    csv_path.write_text(csv_text)
    tex_path.write_text(body)

    # Console report.
    print("Price table (OpenRouter list price, 2026-05-04):")
    for model, (pin, pout) in PRICES.items():
        print(f"  {model}: ${pin}/M prompt + ${pout}/M completion")
    print()
    header = f"{'tool':40s} {'split':12s} {'n':>5s} {'med':>8s} {'p90':>8s} {'mean':>9s} "
    header += f"{'trim':>8s} {'max':>10s} {'api':>7s} {'$/1k':>7s}"
    print(header)
    print("-" * len(header))
    for b in BASELINES:
        c = b.computed

        def f(key: str, width: int, digits: int = 2, c: dict = c) -> str:
            value = c.get(key)
            return f"{value:>{width}.{digits}f}" if isinstance(value, (int, float)) else "-" * 3

        print(
            f"{b.tool:40s} {b.split:12s} {c['n'] or '':>5} {f('median_sec', 8)} "
            f"{f('p90_sec', 8)} {f('mean_sec', 9)} {f('trimmed_mean_sec', 8)} "
            f"{f('max_sec', 10)} {f('mean_api_calls', 7, 3)} {f('est_usd_per_1k', 7)}"
        )
    print()
    for b in BASELINES:
        if b.computed["crosscheck"]:
            flag = "  <-- CHECK" if "DIFFERS" in str(b.computed["crosscheck"]) else ""
            print(f"  {b.tool:40s} {b.computed['crosscheck']}{flag}")
    print(f"\nWrote {csv_path}\nWrote {tex_path}")


if __name__ == "__main__":
    main()
