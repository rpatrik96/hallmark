"""Summarize LLM baseline results on test_public split.

Reads result JSONs from results/ and produces a markdown table at
results/test_public_llm_summary.md.

Usage:
    uv run python scripts/summarize_test_public_llms.py
"""

import json
import pathlib

RESULTS_DIR = pathlib.Path("results")
OUTPUT_FILE = RESULTS_DIR / "test_public_llm_summary.md"

# Models to summarize (in display order)
MODELS = [
    ("llm_openai", "GPT-5.1 (OpenAI)"),
    ("llm_openrouter_claude_sonnet_4_6", "Claude Sonnet 4.6 (OpenRouter)"),
    ("llm_openrouter_deepseek_v3", "DeepSeek-V3 (OpenRouter)"),
    ("llm_openrouter_gemini_pro", "Gemini 2.5 Pro (OpenRouter)"),
]


def fmt(val: float | None, pct: bool = True) -> str:
    if val is None:
        return "N/A"
    if pct:
        return f"{val * 100:.1f}"
    return f"{val:.4f}"


def load_result(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}_test_public.json"
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    rows = []
    for baseline_key, display_name in MODELS:
        d = load_result(baseline_key)
        if d is None:
            rows.append((display_name, "MISSING", "—", "—", "—", "—", "—"))
            continue

        dr = d.get("detection_rate")
        fpr = d.get("false_positive_rate")
        f1 = d.get("f1_hallucination")
        tw_f1 = d.get("tier_weighted_f1")
        ece = d.get("ece")
        n_entries = d.get("num_entries", "?")

        rows.append(
            (
                display_name,
                str(n_entries),
                f"{fmt(dr)}%",
                f"{fmt(fpr)}%",
                f"{fmt(f1)}%",
                f"{fmt(tw_f1)}%",
                fmt(ece, pct=False),
            )
        )

    header = "| Model | N | DR | FPR | F1-Hall | TW-F1 | ECE |"
    sep = "|-------|---|----|----|---------|-------|-----|"
    lines = [
        "# LLM Baselines — test_public Results",
        "",
        "Split: `test_public` (831 entries)",
        "",
        header,
        sep,
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "**Metrics**: DR = Detection Rate (hallucinated recall), FPR = False Positive Rate,",
        "F1-Hall = F1 on hallucinated class, TW-F1 = Tier-weighted F1, ECE = Expected Calibration Error.",
        "",
        "All values as percentages except ECE (lower is better).",
    ]

    table = "\n".join(lines)
    print(table)
    OUTPUT_FILE.write_text(table + "\n")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
