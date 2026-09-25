"""Pairwise agreement (Cohen's kappa) between zero-shot LLM baselines on dev_public.

Backs tab:llm_agreement in the paper: all C(12,2)=66 pairs of the twelve zero-shot
baselines of tab:results, each with stored per-entry dev_public predictions (the same
twelve files the a3 noisy-voter ensemble uses). Predictions are filtered to dev_public
keys, since the GPT-5.1 checkpoint holds dev_public and test_public together.

UNCERTAIN is scored as committed-VALID so every pair is compared on all 1,119 entries.
This differs from tab:results, which excludes an LLM's UNCERTAIN verdicts from
DR/FPR/F1; it exactly reproduces the previously published DeepSeek-V3.2 vs Qwen3-235B
cell (79.9% agreement, kappa=0.454).

Caveat: the two Anthropic prediction files come from the later OpenRouter
snapshot and carry its drift caveat (app:coverage); they do not reproduce their
tab:results rows exactly (Opus 4.7: .909/.162/.889 vs .906/.154/.890).

Usage:
    python scripts/compute_pairwise_kappa.py            # prints matrix + LaTeX rows
    python scripts/compute_pairwise_kappa.py --json OUT # also dump JSON
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILES = {
    "Opus 4.7": REPO / "results/llm_openrouter_claude_opus_4_7_dev_public_predictions.jsonl",
    "Sonnet 4.6": REPO / "results/llm_openrouter_claude_sonnet_4_6_dev_public_predictions.jsonl",
    "DeepSeek-R1": REPO / "results/llm_openrouter_deepseek_r1_dev_public_predictions.jsonl",
    "DeepSeek-V3.2": REPO / "results/llm_openrouter_deepseek_v3_dev_public_predictions.jsonl",
    "Gemini 2.5 Flash": REPO / "results/llm_openrouter_gemini_flash_dev_public_predictions.jsonl",
    "Mistral Large": REPO / "results/llm_openrouter_mistral_dev_public_predictions.jsonl",
    "Qwen3-235B": REPO / "results/llm_openrouter_qwen_dev_public_predictions.jsonl",
    "GPT-5.4": REPO / "results/checkpoints/llm_openai_gpt54_dev_public_v3/openai_gpt-5.4.jsonl",
    "GPT-5.1": REPO / "results/checkpoints/llm_openai/openai_gpt-5.1.jsonl",
    "Llama 4 Maverick": REPO / "results/new_models/llama4_maverick.jsonl",
    "Qwen3-VL-235B": REPO / "results/new_models/qwen_max.jsonl",
    "Gemini 2.5 Pro": REPO / "results/new_models/gemini_pro.jsonl",
}
DEV = REPO / "data/v1.2/dev_public.jsonl"


def load(path: Path, keys: set[str]) -> dict[str, str]:
    preds: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["bibtex_key"] in keys:
                preds[r["bibtex_key"]] = r["label"]
    return preds


def cohen_kappa(a: dict[str, str], b: dict[str, str]) -> tuple[float, float, int]:
    """Percent agreement, Cohen's kappa, n. UNCERTAIN scored as committed-VALID."""
    keys = sorted(set(a) & set(b))
    la = ["VALID" if a[k] == "UNCERTAIN" else a[k] for k in keys]
    lb = ["VALID" if b[k] == "UNCERTAIN" else b[k] for k in keys]
    n = len(keys)
    po = sum(x == y for x, y in zip(la, lb, strict=True)) / n
    pa = la.count("HALLUCINATED") / n
    pb = lb.count("HALLUCINATED") / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    with DEV.open() as f:
        dev_keys = {json.loads(line)["bibtex_key"] for line in f}
    preds = {name: load(path, dev_keys) for name, path in FILES.items()}
    names = list(FILES)
    out = {}
    kappas: dict[tuple[str, str], float] = {}
    for a, b in itertools.combinations(names, 2):
        po, kap, n = cohen_kappa(preds[a], preds[b])
        kappas[(a, b)] = kap
        out[f"{a} vs {b}"] = {"agreement": round(po, 4), "kappa": round(kap, 4), "n": n}
        print(f"{a:18s} vs {b:18s} agree={po * 100:5.1f}%  kappa={kap:.3f}  n={n}")

    print("\nLaTeX lower-triangle rows (kappa, leading zero dropped):")
    for i, row in enumerate(names[1:], start=1):
        cells = []
        for col in names[:-1]:
            j = names.index(col)
            if j < i:
                kap = kappas[(col, row)]  # unrounded: rounding the 4-dp JSON value again drifts
                cells.append(f"{kap:.3f}".lstrip("0"))
            else:
                cells.append("")
        print(f"{row} & " + " & ".join(cells) + r" \\")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
