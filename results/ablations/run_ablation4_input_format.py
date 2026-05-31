"""Ablation 4 — INPUT FORMAT / FIELD ablation (cheap pilot).

Questions:
  (a) Full BibTeX string vs a structured field list as the prompt input —
      does formatting change verdicts?
  (b) Leave-one-field-out (drop title / authors / venue / year / doi) — which
      field drives detection? Report DR/FPR change per dropped field.

Design (cap <= ~180 API calls):
  Fixed stratified subset of n=24 (drawn deterministically from the n=60 pilot
  sample) x 7 conditions = 168 calls. One cheap OpenRouter model
  (deepseek/deepseek-v3.2, non-thinking), temp=0, seed=42 — the question is
  prompt/format sensitivity, which generalizes across models.

Conditions:
  full        : default VERIFICATION_PROMPT (full BibTeX block)
  structured  : same instructions, input rendered as a structured field list
  loo_title   : structured minus title
  loo_author  : structured minus author
  loo_venue   : structured minus booktitle/journal
  loo_year    : structured minus year
  loo_doi     : structured minus doi

Reuses repo verify functions (_verify_with_openai_compatible) with a custom
prompt_fn rather than reimplementing the call/parse loop. Scores with
hallmark.evaluation.evaluate against the sample's ground-truth labels.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from hallmark.baselines.llm_verifier import (
    VERIFICATION_PROMPT,
    _verify_with_openai_compatible,
)
from hallmark.dataset.schema import BenchmarkEntry, BlindEntry
from hallmark.evaluation import evaluate

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "pilot_sample_dev60.jsonl"
OUT_DIR = HERE / "ablation4_input_format"
MODEL = "deepseek/deepseek-v3.2"
SUBSET_N = 24
SEED = 42

VENUE_FIELDS = ("booktitle", "journal")

# Map condition name -> field(s) to drop (None = no drop, structured rendering).
LOO_DROP: dict[str, tuple[str, ...]] = {
    "structured": (),
    "loo_title": ("title",),
    "loo_author": ("author",),
    "loo_venue": VENUE_FIELDS,
    "loo_year": ("year",),
    "loo_doi": ("doi",),
}

# Structured-field-list body. We keep the SAME instruction/JSON-schema text as
# VERIFICATION_PROMPT so only the *input rendering* changes between full and
# structured (clean format ablation), then strip the bibtex block.
_PROMPT_HEAD, _PROMPT_TAIL = VERIFICATION_PROMPT.split("```bibtex\n{bibtex}\n```")
# Drop the dangling "BibTeX entry:" line for the structured conditions so the
# input-rendering contrast (full bibtex block vs labelled field list) is clean.
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


def make_prompt_fn(condition: str):
    if condition == "full":

        def _fn(entry: BlindEntry) -> str:
            return VERIFICATION_PROMPT.format(bibtex=entry.to_bibtex())

        return _fn

    drop = LOO_DROP[condition]

    def _fn(entry: BlindEntry) -> str:
        body = _structured_input(entry, drop)
        return f"{_PROMPT_HEAD_STRUCT}Citation fields:\n{body}\n{_PROMPT_TAIL}"

    return _fn


def load_subset() -> tuple[list[BenchmarkEntry], list[BlindEntry]]:
    rows = [json.loads(line) for line in SAMPLE.read_text().splitlines() if line.strip()]
    entries = [BenchmarkEntry.from_dict(r) for r in rows]
    # Stratified deterministic subset: keep HALL:VALID ratio of the full sample.
    hall = [e for e in entries if e.label == "HALLUCINATED"]
    valid = [e for e in entries if e.label == "VALID"]
    rng = random.Random(SEED)
    rng.shuffle(hall)
    rng.shuffle(valid)
    n_hall = round(SUBSET_N * len(hall) / len(entries))
    n_valid = SUBSET_N - n_hall
    subset = hall[:n_hall] + valid[:n_valid]
    subset.sort(key=lambda e: e.bibtex_key)  # stable order across conditions
    blinds = [e.to_blind() for e in subset]
    return subset, blinds


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = os.environ["OPENROUTER_API_KEY"]
    entries, blinds = load_subset()
    n_hall = sum(1 for e in entries if e.label == "HALLUCINATED")
    print(f"Subset n={len(entries)} ({n_hall} HALL / {len(entries) - n_hall} VALID), model={MODEL}")

    conditions = [
        "full",
        "structured",
        "loo_title",
        "loo_author",
        "loo_venue",
        "loo_year",
        "loo_doi",
    ]
    summary: dict[str, dict] = {}

    for cond in conditions:
        ckpt = OUT_DIR / f"ckpt_{cond}"
        preds = _verify_with_openai_compatible(
            blinds,
            model=MODEL,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            source_prefix=f"or_{cond}",
            checkpoint_dir=ckpt,
            prompt_fn=make_prompt_fn(cond),
            temperature=0.0,
        )
        res = evaluate(entries, preds, tool_name=f"ds-v3.2-{cond}", split_name="pilot24")
        n_uncertain = sum(1 for p in preds if p.label == "UNCERTAIN")
        n_hall_pred = sum(1 for p in preds if p.label == "HALLUCINATED")
        summary[cond] = {
            "detection_rate": res.detection_rate,
            "false_positive_rate": res.false_positive_rate,
            "f1_hallucination": res.f1_hallucination,
            "num_uncertain": n_uncertain,
            "num_pred_hallucinated": n_hall_pred,
            "n": len(entries),
            "n_hall": n_hall,
        }
        fpr = res.false_positive_rate
        fpr_s = f"{fpr:.3f}" if fpr is not None else "NA"
        print(
            f"  {cond:12s} DR={res.detection_rate:.3f} FPR={fpr_s} "
            f"F1={res.f1_hallucination:.3f} UNC={n_uncertain}"
        )
        # Persist per-entry predictions for inspection.
        (OUT_DIR / f"preds_{cond}.jsonl").write_text("\n".join(p.to_json() for p in preds) + "\n")

    # Deltas vs full and vs structured baseline.
    base = summary["structured"]
    deltas = {}
    for cond in ["loo_title", "loo_author", "loo_venue", "loo_year", "loo_doi"]:
        s = summary[cond]
        deltas[cond] = {
            "dDR_vs_structured": s["detection_rate"] - base["detection_rate"],
            "dFPR_vs_structured": (
                (s["false_positive_rate"] or 0.0) - (base["false_positive_rate"] or 0.0)
            ),
        }
    out = {
        "model": MODEL,
        "subset_n": len(entries),
        "n_hall": n_hall,
        "seed": SEED,
        "temperature": 0.0,
        "summary": summary,
        "loo_deltas_vs_structured": deltas,
        "format_delta_full_vs_structured": {
            "dDR": summary["structured"]["detection_rate"] - summary["full"]["detection_rate"],
            "dFPR": (
                (summary["structured"]["false_positive_rate"] or 0.0)
                - (summary["full"]["false_positive_rate"] or 0.0)
            ),
        },
        "subset_keys": [e.bibtex_key for e in entries],
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
