"""Ablation 2 — Decoding / Self-consistency pilot (n=60 dev60).

Two cheap pilots, run independently (each <=180 API calls):

  (a) TEMPERATURE: one OpenRouter model that accepts both temps
      (deepseek/deepseek-v3.2) at temperature 0.0 vs 1.0.
      -> verdict-flip rate temp0 vs temp1, DR/FPR/F1 shift.
      60 + 60 = 120 calls.

  (b) SELF-CONSISTENCY: GPT-5.1 at temp=1 (API-forced, the E3 regime),
      three independent draws via the repo verify path (seed=42 fixed,
      exactly mirroring the E3 protocol). Reports per-draw DR/FPR/F1,
      k=3 majority-vote DR/FPR/F1, and across-draw verdict-flip rate.
      60 x 3 = 180 calls.

Reuses hallmark.baselines.llm_verifier verify functions (default
VERIFICATION_PROMPT) and scores with hallmark.evaluation.evaluate against
the sample ground-truth labels. Outputs JSON under results/ablations/.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from hallmark.baselines.llm_verifier import verify_with_openai, verify_with_openrouter
from hallmark.dataset.schema import BlindEntry, Prediction, load_entries
from hallmark.evaluation.metrics import evaluate

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
SAMPLE = REPO / "results/ablations/pilot_sample_dev60.jsonl"
OUTDIR = REPO / "results/ablations/e_decoding"


def _entries_and_blind() -> tuple[list, list[BlindEntry]]:
    entries = load_entries(SAMPLE)
    blind = [
        BlindEntry(
            bibtex_key=e.bibtex_key,
            bibtex_type=e.bibtex_type,
            fields=dict(e.fields),
            raw_bibtex=e.raw_bibtex,
        )
        for e in entries
    ]
    return entries, blind


def _score(entries: list, preds: list[Prediction], tool: str) -> dict:
    res = evaluate(entries, preds, tool_name=tool, split_name="dev60")
    return {
        "detection_rate": res.detection_rate,
        "false_positive_rate": res.false_positive_rate,
        "f1_hallucination": res.f1_hallucination,
        "num_uncertain": res.num_uncertain,
        "coverage": res.coverage,
    }


def _verdicts(preds: list[Prediction]) -> dict[str, str]:
    return {p.bibtex_key: p.label for p in preds}


def _flip_rate(a: dict[str, str], b: dict[str, str]) -> tuple[int, int, float]:
    keys = sorted(set(a) & set(b))
    flips = sum(1 for k in keys if a[k] != b[k])
    return flips, len(keys), (flips / len(keys) if keys else 0.0)


def run_temperature(entries: list, blind: list[BlindEntry]) -> dict:
    model = "deepseek/deepseek-v3.2"
    out: dict[str, object] = {}
    verdicts: dict[str, dict[str, str]] = {}
    for temp in (0.0, 1.0):
        tag = f"t{temp:g}"
        ckpt = OUTDIR / "temp" / tag
        preds = verify_with_openrouter(
            blind,
            model=model,
            api_key=os.environ["OPENROUTER_API_KEY"],
            checkpoint_dir=ckpt,
            temperature=temp,
        )
        out[tag] = _score(entries, preds, f"deepseek-v3.2@{tag}")
        verdicts[tag] = _verdicts(preds)
    flips, ncmp, rate = _flip_rate(verdicts["t0"], verdicts["t1"])
    out["flip_temp0_vs_temp1"] = {"flips": flips, "n_compared": ncmp, "rate": rate}
    out["model"] = model
    return out


def run_self_consistency(
    entries: list,
    blind: list[BlindEntry],
    k: int = 3,
    provider: str = "openai",
    model: str | None = None,
    temperature: float | None = None,
) -> dict:
    """Run k independent draws and report per-draw + majority-vote metrics.

    provider="openai" -> GPT-5.1 (temp forced to 1.0 by API);
    provider="openrouter" -> the given model at the given temperature
    (used as a cheap higher-variance cross-check when the OpenAI quota is
    unavailable).
    """
    if model is None:
        model = "gpt-5.1" if provider == "openai" else "deepseek/deepseek-v3.2"
    tag = model.replace("/", "_")
    per_draw_metrics = []
    per_draw_verdicts: list[dict[str, str]] = []
    for run in range(1, k + 1):
        ckpt = OUTDIR / "sc" / provider / f"run{run}"
        if provider == "openai":
            preds = verify_with_openai(
                blind, model=model, api_key=os.environ["OPENAI_API_KEY"], checkpoint_dir=ckpt
            )
        else:
            preds = verify_with_openrouter(
                blind,
                model=model,
                api_key=os.environ["OPENROUTER_API_KEY"],
                checkpoint_dir=ckpt,
                temperature=temperature if temperature is not None else 1.0,
            )
        per_draw_metrics.append(_score(entries, preds, f"{tag}@run{run}"))
        per_draw_verdicts.append(_verdicts(preds))

    # across-draw verdict-flip rate: fraction of entries whose label is not
    # identical across all k draws.
    keys = sorted(set.intersection(*[set(v) for v in per_draw_verdicts]))
    unstable = sum(1 for key in keys if len({v[key] for v in per_draw_verdicts}) > 1)
    flip_rate = unstable / len(keys) if keys else 0.0

    # k=3 majority vote per entry (ties -> first-most-common; with k=3 a tie is
    # impossible unless all three differ, in which case Counter.most_common
    # picks insertion order — rare). UNCERTAIN is a valid vote.
    from typing import Literal, cast

    mv_preds: list[Prediction] = []
    for key in keys:
        votes = [v[key] for v in per_draw_verdicts]
        label = cast(
            "Literal['VALID', 'HALLUCINATED', 'UNCERTAIN']",
            Counter(votes).most_common(1)[0][0],
        )
        mv_preds.append(
            Prediction(bibtex_key=key, label=label, confidence=0.7, reason="k=3 majority")
        )
    mv_metrics = _score(entries, mv_preds, f"{tag}@maj3")

    return {
        "model": model,
        "k": k,
        "per_draw": per_draw_metrics,
        "majority_vote": mv_metrics,
        "across_draw_flip": {
            "unstable_entries": unstable,
            "n_compared": len(keys),
            "rate": flip_rate,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["temp", "sc", "sc-or", "both"], default="both")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    entries, blind = _entries_and_blind()
    result: dict = {"n": len(entries), "sample": str(SAMPLE)}

    if args.part in ("temp", "both"):
        result["temperature"] = run_temperature(entries, blind)
        (OUTDIR / "result_temperature.json").write_text(json.dumps(result["temperature"], indent=2))
    if args.part in ("sc", "both"):
        result["self_consistency"] = run_self_consistency(entries, blind, provider="openai")
        (OUTDIR / "result_self_consistency.json").write_text(
            json.dumps(result["self_consistency"], indent=2)
        )
    if args.part == "sc-or":
        # Cheap higher-variance cross-check: deepseek-v3.2 @ temp=1, 3 draws.
        result["self_consistency_or"] = run_self_consistency(
            entries, blind, provider="openrouter", temperature=1.0
        )
        (OUTDIR / "result_self_consistency_openrouter.json").write_text(
            json.dumps(result["self_consistency_or"], indent=2)
        )

    (OUTDIR / "result_combined.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
