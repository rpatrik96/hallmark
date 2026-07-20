"""Pairwise agreement (Cohen's kappa) between zero-shot LLM baselines on dev_public.

Backs tab:llm_agreement in the paper: all C(8,2)=28 pairs of the eight zero-shot
baselines with stored per-entry dev_public predictions (the same eight files the
a3 noisy-voter ensemble uses; the other four baselines persisted only aggregate
metrics). UNCERTAIN is scored as committed-VALID, matching the paper's scoring
convention (sec:main_results) -- this convention exactly reproduces the
previously published DeepSeek-V3.2 vs Qwen3-235B cell (79.9% agreement,
kappa=0.454).

Caveat: the two Anthropic prediction files come from the later OpenRouter
snapshot and carry its drift caveat (app:coverage).

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
}


def load(path: Path) -> dict[str, str]:
    preds: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
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

    preds = {name: load(path) for name, path in FILES.items()}
    names = list(FILES)
    out = {}
    for a, b in itertools.combinations(names, 2):
        po, kap, n = cohen_kappa(preds[a], preds[b])
        out[f"{a} vs {b}"] = {"agreement": round(po, 4), "kappa": round(kap, 4), "n": n}
        print(f"{a:18s} vs {b:18s} agree={po * 100:5.1f}%  kappa={kap:.3f}  n={n}")

    print("\nLaTeX lower-triangle rows (kappa, leading zero dropped):")
    for i, row in enumerate(names[1:], start=1):
        cells = []
        for col in names[:7]:
            j = names.index(col)
            if j < i:
                kap = out[f"{col} vs {row}"]["kappa"]
                cells.append(f"{kap:.3f}".lstrip("0"))
            else:
                cells.append("")
        print(f"{row} & " + " & ".join(cells) + r" \\")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
