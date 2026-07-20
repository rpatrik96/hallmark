"""Fast two-stage cascade on the recency-matched cross-domain split.

The live ``run_cascade`` re-invokes ``bibtex-check`` from scratch, and its arXiv
DOI HEAD prescreening on the fabricated future-dated DOIs (the 300 hallucinated
entries) times out repeatedly, hanging the run for hours at ~0% CPU. This mirrors
``regen_btu_v1_2_0_cascade_fast.py``: it reuses the already-persisted btu Stage-1
per-entry verdicts (``bibtexupdater_matched_per_entry.jsonl``) and runs Stage-2
(agentic Sonnet 4.6) ONLY on btu's deferred (abstained) bucket, then scores the two
stances. Stage-1 (DB/metadata) is not LLM-drift-prone and is served from cache;
Stage-2 (Sonnet via OpenRouter) is drift-prone and is a dated snapshot.

Deferred bucket on this split: 189 entries (108 valid biomedical + 81 hallucinated)
that btu could not confirm; btu decides the other 263.

We report the standard two stances:
  * conservative: residual UNCERTAIN kept as abstention (committed-VALID scoring)
  * aggressive:   residual UNCERTAIN forced to HALLUCINATED@0.55

Usage:
    source /tmp/.or_env; source /tmp/.s2_env
    python scripts/run_cascade_crossdomain_fast.py
"""

from __future__ import annotations

import json
from pathlib import Path

from hallmark.baselines.concurrency import parallel_run_baseline
from hallmark.dataset.schema import Prediction, load_entries
from hallmark.evaluation.metrics import evaluate

REPO = Path(__file__).resolve().parents[1]
SPLIT = REPO / "data/v1.1_crossdomain_matched/test_crossdomain_matched.jsonl"
BTU_PE = REPO / "results/crossdomain_matched_llms/bibtexupdater_matched_per_entry.jsonl"
OUT = REPO / "results/crossdomain_matched_llms"
CKPT = OUT / "cascade_stage2_checkpoints"
STAGE2 = "llm_agentic_openrouter_claude_sonnet_4_6"
STAGE2_WORKERS = 3  # low fan-out to avoid self-inflicted 429s on the shared IP


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
    blind = {e.bibtex_key: e.to_blind() for e in entries}
    n_valid = sum(1 for e in entries if e.label == "VALID")
    print(f"Loaded {len(entries)} ({n_valid} valid / {len(entries) - n_valid} hallucinated)")

    # Stage 1: reuse persisted btu per-entry verdicts (no live re-run -> no hang)
    btu = {}
    for line in BTU_PE.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            btu[r["bibtex_key"]] = r

    decided: dict[str, Prediction] = {}
    deferred = []
    for e in entries:
        r = btu.get(e.bibtex_key)
        if r is None or r.get("abstained"):
            deferred.append(blind[e.bibtex_key])
        else:
            decided[e.bibtex_key] = Prediction(
                bibtex_key=e.bibtex_key,
                label=r["pred_label"],
                confidence=0.9,
                reason=f"[Stage 1: btu {r.get('btu_status')}]",
                source="tool",
                cascade_stage="stage1_db",
            )
    print(
        f"Stage-1 decided={len(decided)}  deferred-to-Stage-2={len(deferred)} (of {len(entries)})"
    )

    # Stage 2: agentic Sonnet on the deferred bucket only (checkpointed, resumable)
    stage2 = parallel_run_baseline(
        STAGE2,
        deferred,
        workers=STAGE2_WORKERS,
        checkpoint_dir=CKPT,
        split="test_crossdomain_matched",
    )
    s2_by_key = {p.bibtex_key: p for p in stage2}

    cons: list[Prediction] = []
    for e in entries:
        if e.bibtex_key in decided:
            cons.append(decided[e.bibtex_key])
            continue
        p = s2_by_key.get(e.bibtex_key)
        if p is None:
            cons.append(
                Prediction(
                    bibtex_key=e.bibtex_key,
                    label="UNCERTAIN",
                    confidence=0.5,
                    reason="[Stage 2: no verdict]",
                    source="tool",
                    cascade_stage="stage2_diagnosis",
                )
            )
        else:
            cons.append(
                Prediction(
                    bibtex_key=e.bibtex_key,
                    label=p.label,
                    confidence=p.confidence,
                    reason=f"[Stage 2: {STAGE2}] {p.reason}",
                    source=p.source or "tool",
                    predicted_hallucination_type=p.predicted_hallucination_type,
                    cascade_stage="stage2_diagnosis",
                )
            )
    agg = aggressive_rescore(cons)

    out = {
        "split": "test_crossdomain_matched",
        "n": len(entries),
        "stage1_source": "cached_btu_per_entry",
        "stage2_baseline": STAGE2,
        "stage1_decided": len(decided),
        "stage2_deferred": len(deferred),
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
                        "cascade_stage": getattr(p, "cascade_stage", None),
                        "reason": p.reason,
                    },
                    default=str,
                )
                + "\n"
            )

    c, a = out["conservative"], out["aggressive"]
    print("\n=== FAST CASCADE on matched cross-domain (152 valid biomed, 2021-23) ===")
    for tag, m in (("conservative", c), ("aggressive", a)):
        print(
            f"  {tag:12s} DR={m['detection_rate']:.3f} FPR={m['false_positive_rate']:.3f} "
            f"F1={m['f1_hallucination']:.3f} cov={m['coverage']:.3f} | "
            f"biomed FPR={m['biomed']['fpr']:.3f} cov={m['biomed']['coverage']:.3f}"
        )
    print("\nStage-1-only (btu) biomed: headline FPR 0.112, coverage 0.29 on valid")


if __name__ == "__main__":
    main()
