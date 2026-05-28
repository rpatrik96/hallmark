"""E1 Pass A -- verify.

Run a model's standard HALLMARK verification on the blinded E1 sample (all 150
entries are truly VALID). FPR = fraction predicted HALLUCINATED. Uses the
hallmark verifier with checkpoint_dir for resume.

Usage:
    python verify_pass.py --model gpt-5.1                       # OpenAI
    python verify_pass.py --model anthropic/claude-sonnet-4.6   # OpenRouter
    python verify_pass.py --model anthropic/claude-opus-4.7     # OpenRouter
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_common as c

from hallmark.baselines.llm_verifier import verify_with_openai, verify_with_openrouter


def run(model: str) -> dict:
    sample = c.load_sample()
    blind = [e.to_blind() for e in sample]
    ckpt_dir = c.OUT_DIR / "verify_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if model.startswith("anthropic/"):
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY not set")
        preds = verify_with_openrouter(blind, model=model, api_key=key, checkpoint_dir=ckpt_dir)
    else:
        # verify_with_openai reads OPENAI_API_KEY from env; pass it explicitly anyway.
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY not set")
        preds = verify_with_openai(blind, model=model, api_key=key, checkpoint_dir=ckpt_dir)

    by_key = {p.bibtex_key: p for p in preds}
    # Reorder to sample order.
    ordered = [by_key[e.bibtex_key] for e in sample if e.bibtex_key in by_key]

    n = len(ordered)
    n_halluc = sum(1 for p in ordered if p.label == "HALLUCINATED")
    n_valid = sum(1 for p in ordered if p.label == "VALID")
    n_uncertain = sum(1 for p in ordered if p.label == "UNCERTAIN")
    total_api = sum(p.api_calls for p in ordered)

    # Persist per-entry verify results for the analysis step.
    safe = model.replace("/", "_")
    out_path = c.OUT_DIR / f"verify_{safe}.jsonl"
    with open(out_path, "w") as f:
        for p in ordered:
            f.write(
                json.dumps(
                    {
                        "bibtex_key": p.bibtex_key,
                        "label": p.label,
                        "confidence": p.confidence,
                        "api_calls": p.api_calls,
                        "reason": p.reason[:300],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "model": model,
        "n": n,
        "n_predicted_HALLUCINATED": n_halluc,
        "n_predicted_VALID": n_valid,
        "n_predicted_UNCERTAIN": n_uncertain,
        "fpr": round(n_halluc / n, 4) if n else None,  # all truly VALID
        "total_api_calls": total_api,
        "per_entry_path": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    run(args.model)
