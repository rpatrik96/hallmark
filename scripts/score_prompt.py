#!/usr/bin/env python3
"""Score a single prompt file on a HALLMARK split; report accuracy, detection, FPR.

Reuses the LM wrapper, evaluator and sampling from gepa_optimize_prompt.py so
the numbers are directly comparable to a GEPA run's val_score.

    uv run python scripts/score_prompt.py results/gepa_haiku/clean_best_prompt.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "gopt", REPO_ROOT / "scripts" / "gepa_optimize_prompt.py"
)
assert _spec and _spec.loader
gopt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gopt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt_file", type=Path)
    ap.add_argument(
        "--split",
        default=None,
        help="score an entire split (e.g. test_public) instead of the dev sample",
    )
    ap.add_argument(
        "--train-size", type=int, default=50, help="only to reproduce the same val sample"
    )
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--model", default="anthropic/claude-haiku-4.5")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--log-dir", type=Path, default=REPO_ROOT / "results" / "gepa_haiku" / "guarded" / "logs"
    )
    args = ap.parse_args()

    gopt._load_dotenv(REPO_ROOT)
    prompt = args.prompt_file.read_text()
    assert gopt.BIBTEX_MARKER in prompt, f"prompt lacks {gopt.BIBTEX_MARKER}"

    if args.split:
        from hallmark.dataset.loader import load_split

        valset = load_split(args.split)
    else:
        _, valset = gopt.sample_entries(args.train_size, args.val_size, args.seed)
    stem = args.prompt_file.stem + (f"_{args.split}" if args.split else "")
    calls = gopt.JsonlWriter(args.log_dir / f"score_{stem}_calls.jsonl")
    evals = gopt.JsonlWriter(args.log_dir / f"score_{stem}_evals.jsonl")
    lm = gopt.make_openrouter_lm(
        args.model, temperature=0.0, max_completion_tokens=1024, role="task", sink=calls
    )
    evaluate = gopt.make_evaluator(lm, sink=evals)

    print(f"scoring {args.prompt_file.name} on {len(valset)} entries ({args.model})")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda e: evaluate(prompt, e), valset))

    scores = [s for s, _ in results]
    hall = [(s, e) for (s, _), e in zip(results, valset, strict=True) if e.label == "HALLUCINATED"]
    valid = [(s, e) for (s, _), e in zip(results, valset, strict=True) if e.label == "VALID"]
    acc = sum(scores) / len(scores)
    det = sum(s for s, _ in hall) / len(hall)
    fpr = 1 - sum(s for s, _ in valid) / len(valid)
    balanced = (det + (1 - fpr)) / 2
    print(f"\naccuracy  = {acc:.3f}  ({int(sum(scores))}/{len(scores)})")
    print(f"balanced  = {balanced:.3f}")
    print(f"detection = {det:.3f}  ({int(sum(s for s, _ in hall))}/{len(hall)})")
    print(
        f"FPR       = {fpr:.3f}  ({int(len(valid) - sum(s for s, _ in valid))}/{len(valid)} flagged)"
    )
    print(f"\nlogs: {evals.path}")


if __name__ == "__main__":
    main()
