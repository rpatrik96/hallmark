"""A4 — INPUT-FORMAT / FIELD LEAVE-ONE-OUT ablation (full, n=150).

Dynamic complement to the static W6 format-tell audit. Two questions:

  (a) Format sensitivity: full-BibTeX string input vs a structured labelled
      field-list input — does the *rendering* of identical content change a
      verifier's verdicts (DR / FPR)?
  (b) Field leave-one-out: starting from the structured rendering, drop one
      field at a time {title, authors, venue, year, doi} and measure the
      DR / FPR shift. This localizes *which* field anchors a VALID verdict
      (FPR climbs when the dropped field was the model's main "this is real"
      evidence) vs a HALLUCINATED verdict.

Design:
  * Sample: n=150 drawn deterministically (seed 42) from data/v1.0/dev_public.jsonl,
    stratified to preserve the split's HALL:VALID ratio.
  * Conditions (7): full, structured, loo_{title,author,venue,year,doi}.
  * Models: deepseek/deepseek-v3.2 (cheap anchor) + google/gemini-2.5-flash
    (OpenRouter confirmation model). Both non-thinking, temp=0, seed=42.
  * 150 x 7 x 2 = 2100 calls; checkpointed/resumable per (model, condition).

Reuses the pilot scaffolding (results/ablations/run_ablation4_input_format.py):
same prompt-head/tail split so only the *input rendering* changes between
full and structured, same structured renderer + LOO drop map. Generalized
here to parameterize model, sample, n, and output dir.

ENDPOINT-DRIFT POLICY: every run here is a FRESH dated snapshot (OpenRouter
endpoints drift vs the 2026-05-04 published eval). The summary.json stamps the
run date, endpoint, and provider/model so these numbers are never confused with
the published delta-eval aggregates. This ablation measures *within-run* format/
field deltas, which are robust to the absolute-level drift.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from collections.abc import Callable
from pathlib import Path

from hallmark.baselines.llm_verifier import (
    VERIFICATION_PROMPT,
    _verify_with_openai_compatible,
)
from hallmark.dataset.schema import BenchmarkEntry, BlindEntry
from hallmark.evaluation import evaluate

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEV_PUBLIC = REPO / "data" / "v1.0" / "dev_public.jsonl"
OUT_DIR = HERE / "a4_field_loo"

BASE_URL = "https://openrouter.ai/api/v1"
ENDPOINT = "openrouter.ai/api/v1"
# Anchor (cheap) + confirmation model. Both non-thinking, temp=0.
# Gemini-2.5-Flash chosen as the confirmation model over Sonnet: it is a
# distinct, non-DeepSeek architecture, far cheaper for 1050 calls, and (unlike
# the Anthropic OpenRouter mirror) is not the specific endpoint flagged for the
# published-eval drift. The within-run format/field deltas are the deliverable,
# and they replicate across two unrelated model families if the effect is real.
MODELS: dict[str, str] = {
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
}
ANCHOR = "deepseek-v3.2"

SEED = 42
DEFAULT_N = 150

CONDITIONS = ["full", "structured", "loo_title", "loo_author", "loo_venue", "loo_year", "loo_doi"]
DROPPED_FIELD_LABEL = {
    "loo_title": "title",
    "loo_author": "authors",
    "loo_venue": "venue",
    "loo_year": "year",
    "loo_doi": "doi",
}

VENUE_FIELDS = ("booktitle", "journal")

# Map condition name -> field(s) to drop (() = no drop, structured rendering).
LOO_DROP: dict[str, tuple[str, ...]] = {
    "structured": (),
    "loo_title": ("title",),
    "loo_author": ("author",),
    "loo_venue": VENUE_FIELDS,
    "loo_year": ("year",),
    "loo_doi": ("doi",),
}

# Split the published VERIFICATION_PROMPT around its bibtex block so the
# structured conditions keep IDENTICAL instructions/JSON-schema text — only the
# *input rendering* changes (full bibtex block vs labelled field list).
_PROMPT_HEAD, _PROMPT_TAIL = VERIFICATION_PROMPT.split("```bibtex\n{bibtex}\n```")
_PROMPT_HEAD_STRUCT = _PROMPT_HEAD.replace("BibTeX entry:\n", "")


def _structured_input(entry: BlindEntry, drop: tuple[str, ...]) -> str:
    """Render the entry as a labelled field list, omitting dropped fields."""
    order = ["title", "author", "booktitle", "journal", "year", "doi"]
    lines = [f"Entry type: {entry.bibtex_type}"]
    for k in order:
        if k in drop:
            continue
        v = entry.fields.get(k)
        if v:
            label = {"author": "authors", "booktitle": "venue", "journal": "venue"}.get(k, k)
            lines.append(f"{label}: {v}")
    return "\n".join(lines)


def make_prompt_fn(condition: str) -> Callable[[BlindEntry], str]:
    if condition == "full":

        def _full_fn(entry: BlindEntry) -> str:
            return VERIFICATION_PROMPT.format(bibtex=entry.to_bibtex())

        return _full_fn

    drop = LOO_DROP[condition]

    def _struct_fn(entry: BlindEntry) -> str:
        body = _structured_input(entry, drop)
        return f"{_PROMPT_HEAD_STRUCT}Citation fields:\n{body}\n{_PROMPT_TAIL}"

    return _struct_fn


def load_subset(n: int) -> tuple[list[BenchmarkEntry], list[BlindEntry]]:
    """Deterministic (seed 42) stratified n-sample preserving HALL:VALID ratio."""
    import random

    rows = [json.loads(line) for line in DEV_PUBLIC.read_text().splitlines() if line.strip()]
    entries = [BenchmarkEntry.from_dict(r) for r in rows]
    hall = sorted((e for e in entries if e.label == "HALLUCINATED"), key=lambda e: e.bibtex_key)
    valid = sorted((e for e in entries if e.label == "VALID"), key=lambda e: e.bibtex_key)
    rng = random.Random(SEED)
    rng.shuffle(hall)
    rng.shuffle(valid)
    n_hall = round(n * len(hall) / len(entries))
    n_valid = n - n_hall
    subset = hall[:n_hall] + valid[:n_valid]
    subset.sort(key=lambda e: e.bibtex_key)  # stable order across conditions/models
    blinds = [e.to_blind() for e in subset]
    return subset, blinds


def _n_affected(entries: list[BenchmarkEntry], drop: tuple[str, ...]) -> int:
    """How many sampled entries actually carry (at least one of) the dropped field(s)."""
    return sum(1 for e in entries if any(e.fields.get(f) for f in drop))


def run_model(
    model_key: str,
    model_id: str,
    entries: list[BenchmarkEntry],
    blinds: list[BlindEntry],
    api_key: str,
) -> dict:
    n = len(entries)
    n_hall = sum(1 for e in entries if e.label == "HALLUCINATED")
    safe = model_key.replace("/", "_")
    model_dir = OUT_DIR / safe
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== model={model_key} ({model_id}) | n={n} ({n_hall} HALL / {n - n_hall} VALID) ===")

    summary: dict[str, dict] = {}
    for cond in CONDITIONS:
        ckpt = model_dir / f"ckpt_{cond}"
        preds = _verify_with_openai_compatible(
            blinds,
            model=model_id,
            api_key=api_key,
            base_url=BASE_URL,
            source_prefix=f"a4_{cond}",
            checkpoint_dir=ckpt,
            prompt_fn=make_prompt_fn(cond),
            temperature=0.0,
        )
        res = evaluate(entries, preds, tool_name=f"{model_key}-{cond}", split_name="a4_dev150")
        n_uncertain = sum(1 for p in preds if p.label == "UNCERTAIN")
        n_hall_pred = sum(1 for p in preds if p.label == "HALLUCINATED")
        rec: dict = {
            "detection_rate": res.detection_rate,
            "false_positive_rate": res.false_positive_rate,
            "f1_hallucination": res.f1_hallucination,
            "num_uncertain": n_uncertain,
            "num_pred_hallucinated": n_hall_pred,
            "n": n,
            "n_hall": n_hall,
        }
        if LOO_DROP.get(cond):
            rec["n_drop_affected"] = _n_affected(entries, LOO_DROP[cond])
        summary[cond] = rec
        fpr = res.false_positive_rate
        fpr_s = f"{fpr:.3f}" if fpr is not None else "NA"
        aff = f" aff={rec['n_drop_affected']}" if "n_drop_affected" in rec else ""
        print(
            f"  {cond:12s} DR={res.detection_rate:.3f} FPR={fpr_s} "
            f"F1={res.f1_hallucination:.3f} UNC={n_uncertain}{aff}"
        )
        (model_dir / f"preds_{cond}.jsonl").write_text("\n".join(p.to_json() for p in preds) + "\n")

    base = summary["structured"]
    loo_deltas = {}
    for cond in ["loo_title", "loo_author", "loo_venue", "loo_year", "loo_doi"]:
        s = summary[cond]
        loo_deltas[cond] = {
            "dropped_field": DROPPED_FIELD_LABEL[cond],
            "n_drop_affected": s.get("n_drop_affected"),
            "dDR_vs_structured": s["detection_rate"] - base["detection_rate"],
            "dFPR_vs_structured": (
                (s["false_positive_rate"] or 0.0) - (base["false_positive_rate"] or 0.0)
            ),
        }
    model_out = {
        "model_key": model_key,
        "model_id": model_id,
        "endpoint": ENDPOINT,
        "n": n,
        "n_hall": n_hall,
        "seed": SEED,
        "temperature": 0.0,
        "summary": summary,
        "loo_deltas_vs_structured": loo_deltas,
        "format_delta_full_vs_structured": {
            "dDR": summary["structured"]["detection_rate"] - summary["full"]["detection_rate"],
            "dFPR": (
                (summary["structured"]["false_positive_rate"] or 0.0)
                - (summary["full"]["false_positive_rate"] or 0.0)
            ),
        },
    }
    (model_dir / "results.json").write_text(json.dumps(model_out, indent=2))
    return model_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated model keys")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = os.environ["OPENROUTER_API_KEY"]
    entries, blinds = load_subset(args.n)
    n_hall = sum(1 for e in entries if e.label == "HALLUCINATED")

    # Persist the exact sample so the run is reproducible without re-deriving it.
    (OUT_DIR / "sample_keys.json").write_text(
        json.dumps(
            {
                "source": "data/v1.0/dev_public.jsonl",
                "n": len(entries),
                "n_hall": n_hall,
                "n_valid": len(entries) - n_hall,
                "seed": SEED,
                "keys": [e.bibtex_key for e in entries],
            },
            indent=2,
        )
    )

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    per_model = {}
    for mk in model_keys:
        per_model[mk] = run_model(mk, MODELS[mk], entries, blinds, api_key)

    summary = {
        "experiment": "A4_input_format_field_leave_one_out",
        "description": (
            "Full-BibTeX vs structured-field input, plus leave-one-field-out "
            "(title/authors/venue/year/doi). DR/FPR deltas. Dynamic complement "
            "to the static W6 format-tell audit."
        ),
        "snapshot_date": _dt.date.today().isoformat(),
        "endpoint": ENDPOINT,
        "endpoint_drift_note": (
            "FRESH dated snapshot; NOT a reproduction of the 2026-05-04 published "
            "delta-eval aggregates. Within-run format/field deltas are the deliverable "
            "and are robust to absolute-level endpoint drift."
        ),
        "sample": {
            "source": "data/v1.0/dev_public.jsonl",
            "n": len(entries),
            "n_hall": n_hall,
            "n_valid": len(entries) - n_hall,
            "seed": SEED,
        },
        "conditions": CONDITIONS,
        "temperature": 0.0,
        "anchor_model": ANCHOR,
        "models": {mk: MODELS[mk] for mk in model_keys},
        "per_model": per_model,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
