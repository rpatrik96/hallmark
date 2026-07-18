"""Run the two-stage cascade (bibtex-updater Stage 1 + Sonnet 4.6 Stage 2) on the
recency-matched cross-domain split.

The cascade is the paper's high-recall, low-FPR configuration: Stage 1
(bibtex-updater) decides the entries it can confirm/refute against databases and
DEFERS its could-not-verify bucket to Stage 2 (an agentic Sonnet 4.6 diagnoser,
up to 5 tool calls). This is exactly the configuration that matters out of
domain, where Stage 1 abstains on ~70% of biomedical citations (its ML-tuned
databases can't confirm them): the question is whether Stage 2 recovers that
coverage without wrecking precision.

We report the standard two stances:
  * conservative: residual UNCERTAIN kept as abstention (committed-VALID scoring)
  * aggressive: residual UNCERTAIN forced to HALLUCINATED@0.55

Stage 2 is checkpointed (resumable) and runs sequentially (one agentic chain at
a time), so it adds little concurrent OpenRouter load.

Usage:
    source /tmp/.or_env; source /tmp/.s2_env
    python scripts/run_cascade_crossdomain.py
"""

from __future__ import annotations

import json
from pathlib import Path

from hallmark.baselines.cascade import run_cascade
from hallmark.dataset.schema import Prediction, load_entries
from hallmark.evaluation.metrics import evaluate

REPO = Path(__file__).resolve().parents[1]
SPLIT = REPO / "data/v1.1_crossdomain_matched/test_crossdomain_matched.jsonl"
OUT = REPO / "results/crossdomain_matched_llms"
CKPT = OUT / "cascade_stage2_checkpoints"
STAGE2 = "llm_agentic_openrouter_claude_sonnet_4_6"


def biomed(e) -> bool:
    return getattr(e, "source", None) in ("pubmed", "biorxiv", "medrxiv")


def score(entries, preds, tag: str) -> dict:
    res = evaluate(
        entries, preds, tool_name=f"cascade_{tag}", split_name="test_crossdomain_matched"
    )
    bio = [e for e in entries if biomed(e)]
    bio_keys = {e.bibtex_key for e in bio}
    bio_p = [p for p in preds if p.bibtex_key in bio_keys]
    rbio = evaluate(bio, bio_p, tool_name=f"cascade_{tag}_biomed", split_name="biomed")
    d = res.to_dict()
    d["biomed"] = {
        "fpr": rbio.false_positive_rate,
        "dr": rbio.detection_rate,
        "f1": rbio.f1_hallucination,
        "coverage": rbio.coverage,
        "n_valid": rbio.num_valid,
    }
    return d


def aggressive_rescore(preds: list[Prediction]) -> list[Prediction]:
    """Force residual UNCERTAIN -> HALLUCINATED@0.55 (the aggressive stance)."""
    out = []
    for p in preds:
        if p.label == "UNCERTAIN":
            out.append(
                Prediction(
                    bibtex_key=p.bibtex_key,
                    label="HALLUCINATED",
                    confidence=0.55,
                    reason=f"[aggressive] {p.reason}",
                    predicted_hallucination_type="plausible_fabrication",
                    source=p.source,
                )
            )
        else:
            out.append(p)
    return out


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    entries = load_entries(str(SPLIT))
    blind = [e.to_blind() for e in entries]
    n_valid = sum(1 for e in entries if e.label == "VALID")
    print(f"Loaded {len(entries)} ({n_valid} valid / {len(entries) - n_valid} hallucinated)")

    # conservative cascade: Stage-1 btu decides; deferred bucket -> Stage-2 Sonnet
    cons = run_cascade(
        blind,
        stage2_baseline=STAGE2,
        aggressive=False,
        stage2_kwargs={"checkpoint_dir": CKPT},
    )
    agg = aggressive_rescore(cons)

    out = {
        "split": "test_crossdomain_matched",
        "n": len(entries),
        "stage2_baseline": STAGE2,
        "conservative": score(entries, cons, "cons"),
        "aggressive": score(entries, agg, "agg"),
    }
    (OUT / "cascade_matched_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    with (OUT / "cascade_matched_per_entry.jsonl").open("w") as f:
        for p in cons:
            f.write(
                json.dumps(
                    {
                        "bibtex_key": p.bibtex_key,
                        "label": p.label,
                        "confidence": p.confidence,
                        "reason": p.reason,
                    },
                    default=str,
                )
                + "\n"
            )

    c, a = out["conservative"], out["aggressive"]
    print("\n=== CASCADE on matched cross-domain (152 valid biomed, 2021-23) ===")
    for tag, m in (("conservative", c), ("aggressive", a)):
        print(
            f"  {tag:12s} DR={m['detection_rate']:.3f} FPR={m['false_positive_rate']:.3f} "
            f"F1={m['f1_hallucination']:.3f} cov={m['coverage']:.3f} | "
            f"biomed FPR={m['biomed']['fpr']:.3f} cov={m['biomed']['coverage']:.3f}"
        )
    print("\nCompare Stage-1-only (btu) biomed: coverage 0.29, headline FPR 0.112")


if __name__ == "__main__":
    main()
