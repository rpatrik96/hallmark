"""Parallel worker: fill ONE (model, condition) checkpoint for the A4 ablation.

The serial runner (run_a4_field_loo.py) walks conditions sequentially; deepseek
at ~9s/call makes 7x150 serial conditions ~2.5h. Each condition has its own
checkpoint dir, so they are independent and safe to fill in parallel. This worker
populates a single condition's checkpoint using the EXACT same sample, prompt_fn,
and checkpoint path as the runner (imported from it), so the serial runner later
loads them from cache and skips — then writes the authoritative summary.json.

Usage: a4_fill_condition.py --model-key deepseek-v3.2 --condition loo_author
"""

from __future__ import annotations

import argparse
import os

import run_a4_field_loo as r

from hallmark.baselines.llm_verifier import _verify_with_openai_compatible


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=list(r.MODELS))
    ap.add_argument("--condition", required=True, choices=r.CONDITIONS)
    ap.add_argument("--n", type=int, default=r.DEFAULT_N)
    args = ap.parse_args()

    api_key = os.environ["OPENROUTER_API_KEY"]
    _entries, blinds = r.load_subset(args.n)
    model_id = r.MODELS[args.model_key]
    safe = args.model_key.replace("/", "_")
    ckpt = r.OUT_DIR / safe / f"ckpt_{args.condition}"

    preds = _verify_with_openai_compatible(
        blinds,
        model=model_id,
        api_key=api_key,
        base_url=r.BASE_URL,
        source_prefix=f"a4_{args.condition}",
        checkpoint_dir=ckpt,
        prompt_fn=r.make_prompt_fn(args.condition),
        temperature=0.0,
    )
    done = sum(1 for p in preds if p.label != "UNCERTAIN" or "[Error" not in (p.reason or ""))
    print(f"{args.model_key}/{args.condition}: {len(preds)} preds ({done} non-error)")


if __name__ == "__main__":
    main()
