"""E2: Non-Anthropic / non-OpenAI late-cutoff control experiment.

Research question
-----------------
The HALLMARK paper's failure-mode (iii) shows most LLMs over-flag real
2024-2025 papers (FPR collapse) while the two Anthropic models resist. Recency
and the Anthropic pipeline are confounded. This experiment decorrelates them:
does a NON-Anthropic, NON-OpenAI model with a cutoff comparable-to-or-later-than
the Anthropic pair RESIST the FPR collapse, or collapse like the mid-2025 models?

Control model
-------------
DeepSeek V4-Pro (`deepseek/deepseek-v4-pro` on OpenRouter, served via DeepInfra/
Together/Novita). Knowledge cutoff: 2025 (reported ~May 2025 by the
aiknowledgecutoff.com tracker; corroborated by 36kr reporting that V4-Pro's
cutoff "still remains in 2025"). Released 2026-04-24.

Provider-allowlist constraint: the ideal candidate, xAI Grok 4.3 (Dec-2025
cutoff), is served ONLY by the `xai` provider, which this OpenRouter account's
allowlist excludes (404 "No allowed providers"). Among NON-Anthropic/NON-OpenAI
models reachable through the allowed providers, DeepSeek V4-Pro has the latest
*documented* cutoff: Gemini 3.5 Flash / 3.1 Pro are Jan-2025 (same era as the
already-tested Gemini), and Qwen 3.7-Max is Alibaba-only (blocked). May-2025 is
later than the original DeepSeek V3 (July-2024, a collapse-cluster model) and
comparable to the lower end of the Anthropic pair (Sonnet 4.6 reliable ~Aug-2025,
Opus 4.7 ~Oct-2025), though not as late as Grok would have been. The experiment
is therefore a "comparable-cutoff" control rather than a strictly "later-than"
one; see CAVEATS in SUMMARY.md.

V4-Pro is a thinking model; with the default 1024-token budget it emits empty
JSON (reasoning tokens overrun the cap -> finish_reason='length'). We raise
max_completion_tokens to 4096 for the control, which makes responses parse
reliably (smoke-tested 2026-05-28).

Comparators on the SAME subsample (apples-to-apples, this data version):
  - gpt-5.1 (OpenAI) ........ collapse exemplar
  - anthropic/claude-sonnet-4.6 (OpenRouter mirror) ... resist exemplar

Subsample: stratified N=300 of the 858 temporal set, ~150 VALID + ~150
HALLUCINATED, stratified by (label, publication_year in {2024,2025}), seed=42.

Outputs -> results/reviewer_experiments/e2_latecutoff_control/
Checkpoints -> results/reviewer_experiments/e2_latecutoff_control/checkpoints/
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from hallmark.baselines.llm_verifier import verify_with_openai, verify_with_openrouter
from hallmark.dataset.schema import load_entries
from hallmark.evaluation.metrics import evaluate

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results/temporal_supplement/temporal_supplement_2024_2025.jsonl"
OUT = REPO / "results/reviewer_experiments/e2_latecutoff_control"
CKPT = OUT / "checkpoints"
SEED = 42
N_TARGET = 300

# (label, alias, model_id, provider) -> verify fn chosen by provider below.
CONTROL = (
    "DeepSeek V4-Pro",
    "deepseek-v4-pro",
    "deepseek/deepseek-v4-pro",
    "DeepSeek",
    "~May 2025",
)
COMPARATORS = [
    # (label, alias, model_id, provider, cutoff, runner)
    ("GPT-5.1", "gpt-5.1", "gpt-5.1", "OpenAI", "~Oct 2024", "openai"),
    (
        "Claude Sonnet 4.6",
        "claude-sonnet-4-6",
        "anthropic/claude-sonnet-4.6",
        "Anthropic",
        "reliable ~Aug 2025",
        "openrouter",
    ),
]


def build_subsample() -> list:
    entries = load_entries(str(DATA))

    def ybucket(e: object) -> str:
        return (getattr(e, "publication_date", "") or "")[:4]

    # 4 strata: (label, year). Allocate ~150 per label, split across years
    # proportional to that label's year distribution.
    strata: dict[tuple[str, str], list] = defaultdict(list)
    for e in entries:
        strata[(e.label, ybucket(e))].append(e)

    rng = random.Random(SEED)
    per_label = N_TARGET // 2  # 150

    selected: list = []
    for label in ("VALID", "HALLUCINATED"):
        label_strata = {k: v for k, v in strata.items() if k[0] == label}
        total = sum(len(v) for v in label_strata.values())
        # Proportional allocation by year, largest-remainder rounding.
        raw = {k: per_label * len(v) / total for k, v in label_strata.items()}
        alloc = {k: int(r) for k, r in raw.items()}
        remainder = per_label - sum(alloc.values())
        # distribute leftover to largest fractional parts
        order = sorted(label_strata, key=lambda k: raw[k] - alloc[k], reverse=True)
        for k in order[:remainder]:
            alloc[k] += 1
        for k, pool in label_strata.items():
            pool_sorted = sorted(pool, key=lambda e: e.bibtex_key)  # deterministic
            picks = rng.sample(pool_sorted, alloc[k])
            selected.extend(picks)

    selected.sort(key=lambda e: e.bibtex_key)
    return selected


def run_model(alias: str, model_id: str, runner: str, subsample: list, api_calls_log: dict) -> dict:
    blind = [e.to_blind() for e in subsample]
    log_dir = OUT / "api_logs" / alias
    kwargs: dict = {"checkpoint_dir": CKPT, "log_dir": log_dir}

    if runner == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        # GPT-5 family forced to temperature=1.0 per validated setup.
        preds = verify_with_openai(
            blind, model=model_id, api_key=api_key, temperature=1.0, **kwargs
        )
    elif runner == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")  # KNOWN BUG 2: must pass explicitly
        # DeepSeek V4-Pro is a thinking model: bump the completion budget so
        # reasoning tokens don't truncate the JSON verdict. 4096 still produced
        # ~22% empty responses on the first pass; 8192 to suppress that, and
        # retry_failed so resumed runs re-attempt prior [Error fallback] rows.
        if model_id == "deepseek/deepseek-v4-pro":
            kwargs["max_completion_tokens"] = 8192
            kwargs["retry_failed"] = True
        preds = verify_with_openrouter(blind, model=model_id, api_key=api_key, **kwargs)
    else:
        raise ValueError(runner)

    total_calls = sum(p.api_calls for p in preds)
    api_calls_log[alias] = total_calls

    # Overall metrics.
    res = evaluate(subsample, preds, tool_name=alias, split_name="temporal_2024_2025_n300")

    # Per-year FPR/DR breakdown (recency lens).
    def subset_eval(year: str) -> dict:
        keys = {
            e.bibtex_key
            for e in subsample
            if (getattr(e, "publication_date", "") or "")[:4] == year
        }
        sub_entries = [e for e in subsample if e.bibtex_key in keys]
        sub_preds = [p for p in preds if p.bibtex_key in keys]
        r = evaluate(sub_entries, sub_preds, tool_name=alias, split_name=f"y{year}")
        return {
            "n": len(sub_entries),
            "fpr": r.false_positive_rate,
            "dr": r.detection_rate,
            "f1": r.f1_hallucination,
        }

    label_pred = Counter(p.label for p in preds)
    out = {
        "alias": alias,
        "model_id": model_id,
        "runner": runner,
        "n": len(subsample),
        "api_calls": total_calls,
        "predicted_label_dist": dict(label_pred),
        "overall": res.to_dict(),
        "by_year": {"2024": subset_eval("2024"), "2025": subset_eval("2025")},
    }
    (OUT / f"result_{alias}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only", choices=["deepseek-v4-pro", "gpt-5.1", "claude-sonnet-4-6"], default=None
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    subsample = build_subsample()
    # Persist the fixed subsample (bibtex_keys + strata) for reproducibility.
    manifest = {
        "seed": SEED,
        "n": len(subsample),
        "label_dist": dict(Counter(e.label for e in subsample)),
        "year_dist": dict(
            Counter((getattr(e, "publication_date", "") or "")[:4] for e in subsample)
        ),
        "label_year_dist": {
            f"{lbl}_{yr}": c
            for (lbl, yr), c in sorted(
                Counter(
                    (e.label, (getattr(e, "publication_date", "") or "")[:4]) for e in subsample
                ).items()
            )
        },
        "bibtex_keys": [e.bibtex_key for e in subsample],
    }
    (OUT / "subsample_n300_seed42.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"Subsample: N={len(subsample)} {manifest['label_dist']} years={manifest['year_dist']}")

    jobs = [
        (CONTROL[1], CONTROL[2], "openrouter"),
        (COMPARATORS[0][1], COMPARATORS[0][2], COMPARATORS[0][5]),
        (COMPARATORS[1][1], COMPARATORS[1][2], COMPARATORS[1][5]),
    ]
    if args.only:
        jobs = [j for j in jobs if j[0] == args.only]

    api_calls_log: dict[str, int] = {}
    results: dict[str, dict] = {}
    for alias, model_id, runner in jobs:
        print(f"\n=== Running {alias} ({model_id}) via {runner} ===")
        out = run_model(alias, model_id, runner, subsample, api_calls_log)
        results[alias] = out
        ov = out["overall"]
        print(
            f"  {alias}: FPR={ov['false_positive_rate']}, DR={ov['detection_rate']}, "
            f"F1={ov['f1_hallucination']}, api_calls={out['api_calls']}, "
            f"pred_dist={out['predicted_label_dist']}"
        )

    (OUT / "api_calls_total.json").write_text(
        json.dumps(
            {"per_model": api_calls_log, "total": sum(api_calls_log.values())},
            indent=2,
        )
    )
    print(f"\nTOTAL API CALLS: {sum(api_calls_log.values())}")


if __name__ == "__main__":
    main()
