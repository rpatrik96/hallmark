#!/usr/bin/env python3
"""Rescore the released baseline results on the v1.2.5 labels, with no model calls.

v1.2.5 flips seven labels (dev_public 3, test_public 2, stress_test 2) and drops
six year-only test_public entries. Every released aggregate in
``data/v1.2/baseline_results/`` that scores one of those splits is recomputed
from its per-entry predictions:

1. the prediction file is first scored on the v1.2.4 labels (``OLD_REV``) and
   must reproduce the released DR / FPR / F1 / ECE exactly, otherwise the
   script refuses to write;
2. it is then scored on the current (v1.2.5) split, and the recomputed values
   overwrite only the keys the released JSON already has
   (``results/relabel_delta/_writer.merge_into_published``);
3. ``split_sha256`` and ``run_timestamp`` are set to the current split hash and
   the rescore time, as the v1.2.4 rescore did.

Scope: labels and membership only. The v1.2.5 DOI-suffix fix (``eacc9af``)
changes the *inputs* of 138 public entries; the predictions here were made on
the v1.2.4 inputs and are reused unchanged. That is recorded per file in the log.

bibtex-updater follows ``results/relabel_delta/release_btu_v120.py``: the triple
scores ``pred_label`` (abstentions as committed VALID), and the released
``coverage`` / ``num_uncertain`` are kept. doi_only stores no per-entry file in
the release; its predictions were rebuilt by re-running it on the v1.2.4 inputs
and are used only if they reproduce the release (see ``DOI_ONLY_PREDS``).

HaRC (``harc_with_s2key_dev_public``) stays in ``KNOWN_STALE``: it is scored
against the pre-relabel ground truth and has no per-entry predictions.

Usage:
    uv run python scripts/rescore_v1_2_5.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "results/relabel_delta"))
from _writer import merge_into_published  # noqa: E402

from hallmark.dataset.schema import BenchmarkEntry, Prediction  # noqa: E402
from hallmark.evaluation.metrics import evaluate  # noqa: E402
from hallmark.evaluation.validate import compute_sha256  # noqa: E402

OLD_REV = "c62d296"  # main at the v1.2.4 release (#99)
RESULTS = REPO / "data/v1.2/baseline_results"
LOG = REPO / "results/relabel_recheck/v1_2_5_rescore_log.json"
CHECK = ("detection_rate", "false_positive_rate", "f1_hallucination", "ece")

DOI_ONLY_PREDS = "results/relabel_recheck/doi_only_{split}_v1_2_4_inputs_predictions.jsonl"
BTU_PREDS = "results/relabel_delta/btu_v1_2_0_prescreen_fix/bibtexupdater_{split}_per_entry.jsonl"
CASCADE = "results/cascade_sonnet46_shared/sonnet46_{split}_preds.jsonl"

# result file stem -> (prediction file, eval_mode, loader)
SOURCES: dict[str, tuple[str, str, str]] = {
    **{
        f"bibtexupdater_{s}": (BTU_PREDS.format(split=s), "conservative", "btu")
        for s in ("dev_public", "test_public")
    },
    **{
        f"doi_only_{s}": (DOI_ONLY_PREDS.format(split=s), "conservative", "std")
        for s in ("dev_public", "test_public")
    },
    **{
        f"cascade_db_diagnosis_{s}": (CASCADE.format(split=s), "conservative", "std")
        for s in ("dev_public", "test_public", "stress_test")
    },
    **{
        f"cascade_db_diagnosis_aggressive_{s}": (CASCADE.format(split=s), "aggressive", "std")
        for s in ("dev_public", "test_public", "stress_test")
    },
    "llm_agentic_btu_openai_dev_public": (
        "results/temporal_checkpoints/agentic_btu_openai_gpt-5.1.jsonl",
        "conservative",
        "std",
    ),
    "llm_agentic_btu_openai_test_public": (
        "results/llm_agentic_btu_openai_test_public_predictions.jsonl",
        "conservative",
        "std",
    ),
    "llm_agentic_btu_sonnet_4_6_dev_public": (
        "results/checkpoints/llm_agentic_btu_sonnet_4_6_dev_public_v2/agentic_btu_openai_anthropic_claude-sonnet-4.6.jsonl",
        "conservative",
        "std",
    ),
    "llm_agentic_btu_sonnet_4_6_test_public": (
        "results/llm_agentic_btu_sonnet_4_6_test_public_predictions.jsonl",
        "conservative",
        "std",
    ),
    "llm_agentic_openai_dev_public": (
        "results/temporal_checkpoints/agentic_openai_gpt-5.1.jsonl",
        "conservative",
        "std",
    ),
    "llm_agentic_openai_test_public": (
        "results/llm_agentic_openai_test_public_predictions.jsonl",
        "conservative",
        "std",
    ),
    "llm_openai_dev_public": (
        "results/checkpoints/llm_openai/openai_gpt-5.1.jsonl",
        "conservative",
        "std",
    ),
    "llm_openai_test_public": (
        "results/checkpoints/llm_openai/openai_gpt-5.1.jsonl",
        "conservative",
        "std",
    ),
    "llm_openai_gpt54_dev_public": (
        "results/checkpoints/llm_openai_gpt54_dev_public_v3/openai_gpt-5.4.jsonl",
        "conservative",
        "std",
    ),
    "llm_openai_gpt54_test_public": (
        "results/checkpoints/llm_openai_gpt54_test_public/openai_gpt-5.4.jsonl",
        "conservative",
        "std",
    ),
    **{
        f"llm_openrouter_{m}_{s}": (
            f"results/checkpoints/llm_openrouter_{m}_{s}/openrouter_anthropic_{mid}.jsonl",
            "conservative",
            "std",
        )
        for m, mid in (
            ("claude_opus_4_7", "claude-opus-4.7"),
            ("claude_sonnet_4_6", "claude-sonnet-4.6"),
        )
        for s in ("dev_public", "test_public")
    },
    **{
        f"llm_openrouter_{m}_{s}": (
            f"results/llm_openrouter_{m}_{s}_predictions.jsonl",
            "conservative",
            "std",
        )
        for m in ("deepseek_r1", "deepseek_v3", "gemini_flash", "mistral", "qwen")
        for s in ("dev_public", "test_public")
    },
    "llm_openrouter_gemini_pro_dev_public": (
        "results/new_models/gemini_pro.jsonl",
        "conservative",
        "std",
    ),
    "llm_openrouter_gemini_pro_test_public": (
        "results/llm_openrouter_gemini_pro_test_public_predictions.jsonl",
        "conservative",
        "std",
    ),
    "llm_openrouter_llama_4_maverick_dev_public": (
        "results/new_models/llama4_maverick.jsonl",
        "conservative",
        "std",
    ),
    "llm_openrouter_llama_4_maverick_test_public": (
        "results/llm_openrouter_llama_4_maverick_test_public_predictions.jsonl",
        "conservative",
        "std",
    ),
    "llm_openrouter_qwen_max_dev_public": (
        "results/new_models/qwen_max.jsonl",
        "conservative",
        "std",
    ),
    "llm_openrouter_qwen_max_test_public": (
        "results/llm_openrouter_qwen_max_test_public_predictions.jsonl",
        "conservative",
        "std",
    ),
    "llm_tool_augmented_dev_public": (
        "data/v1.2/baseline_results/llm_tool_augmented_dev_public.jsonl",
        "conservative",
        "std",
    ),
    "llm_tool_augmented_test_public": (
        "results/llm_tool_augmented_test_public_predictions.jsonl",
        "conservative",
        "std",
    ),
}


def entries_at(rev: str | None, split: str) -> list[BenchmarkEntry]:
    rel = f"data/v1.2/{split}.jsonl"
    if rev is None:
        text = (REPO / rel).read_text()
    else:
        text = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{rev}:{rel}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    return [BenchmarkEntry.from_json(line) for line in text.splitlines() if line.strip()]


def load_predictions(path: Path, loader: str) -> dict[str, Prediction]:
    preds: dict[str, Prediction] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if loader == "btu":
            preds[d["bibtex_key"]] = Prediction(
                bibtex_key=d["bibtex_key"],
                label=d["pred_label"],
                confidence=float(d.get("pred_confidence") or 0.5),
                reason=d.get("reason", ""),
                source=d.get("source"),
                predicted_hallucination_type=d.get("pred_type"),
                cascade_stage=d.get("cascade_stage"),
            )
        else:
            preds[d["bibtex_key"]] = Prediction.from_dict(d)  # resume duplicates: last write wins
    return preds


def aligned(entries: list[BenchmarkEntry], preds: dict[str, Prediction]) -> list[Prediction]:
    # Missing -> VALID@0.5, as the released runs and results/relabel_delta/_rescore.py do.
    return [
        preds.get(e.bibtex_key)
        or Prediction(bibtex_key=e.bibtex_key, label="VALID", confidence=0.5)
        for e in entries
    ]


def same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) < 1e-9


def fmt(x: float | None) -> str:
    return "None" if x is None else f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="verify and report, write nothing")
    args = ap.parse_args()

    splits = ("dev_public", "test_public", "stress_test")
    old = {s: entries_at(OLD_REV, s) for s in splits}
    new = {s: entries_at(None, s) for s in splits}
    sha = {s: compute_sha256(REPO / f"data/v1.2/{s}.jsonl") for s in splits}
    now = datetime.now(timezone.utc).isoformat()

    log: dict[str, dict] = {}
    failures: list[str] = []
    summary = (*CHECK, "num_entries", "num_hallucinated", "num_valid")

    def rescore(
        name: str,
        published: dict,
        preds: dict[str, Prediction],
        pred_rel: str,
        mode: str,
        loader: str,
    ):
        split = published["split_name"]
        tool = published["tool_name"]
        r_old = evaluate(old[split], aligned(old[split], preds), tool, split, eval_mode=mode)
        bad = [k for k in CHECK if k in published and not same(getattr(r_old, k), published[k])]
        if bad:
            failures.append(f"{name}: {pred_rel} does not reproduce the release on {bad}")
            return None
        r_new = evaluate(new[split], aligned(new[split], preds), tool, split, eval_mode=mode)
        merged = merge_into_published(published, r_new)
        if loader == "btu":
            merged["coverage"] = published["coverage"]
            merged["num_uncertain"] = published["num_uncertain"]
            merged["coverage_adjusted_f1"] = merged["f1_hallucination"] * published["coverage"]
        if "split_sha256" in merged:
            merged["split_sha256"] = sha[split]
        if "run_timestamp" in merged:
            merged["run_timestamp"] = now
        log[name] = {
            "split": split,
            "predictions": pred_rel,
            "eval_mode": mode,
            "inputs": "v1.2.4 (predictions reused; DOI-suffix fix eacc9af not re-run)",
            "reproduced_v1_2_4": True,
            "v1_2_4": {k: published.get(k) for k in summary},
            "v1_2_5": {k: merged.get(k) for k in summary},
        }
        print(
            f"{name:52s} DR {fmt(published['detection_rate'])}->{fmt(merged['detection_rate'])} "
            f"FPR {fmt(published['false_positive_rate'])}->{fmt(merged['false_positive_rate'])} "
            f"n {published['num_entries']}->{merged['num_entries']}"
        )
        return merged

    # 1. released aggregates
    for stem, (pred_rel, mode, loader) in sorted(SOURCES.items()):
        out_path = RESULTS / f"{stem}.json"
        published = json.loads(out_path.read_text())
        if not (REPO / pred_rel).exists():
            failures.append(f"{stem}: prediction file missing ({pred_rel})")
            continue
        merged = rescore(
            stem, published, load_predictions(REPO / pred_rel, loader), pred_rel, mode, loader
        )
        if merged is not None:
            merged["split_sha256"] = sha[published["split_name"]]
            if not args.dry_run:
                out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")

    # 2. results/cascade_* (one file per split, both eval modes, compact JSON)
    for out_path in sorted(REPO.glob("results/cascade_*/*_*.json")):
        stem = out_path.stem
        split = next((s for s in splits if stem.endswith(s)), None)
        pred_path = out_path.with_name(f"{stem}_preds.jsonl")
        if split is None or not pred_path.exists():
            continue
        doc = json.loads(out_path.read_text())
        preds = load_predictions(pred_path, "std")
        pred_rel = str(pred_path.relative_to(REPO))
        ok = True
        for mode in ("conservative", "aggressive"):
            merged = rescore(
                f"{out_path.parent.name}/{stem}:{mode}", doc[mode], preds, pred_rel, mode, "std"
            )
            if merged is None:
                ok = False
                break
            doc[mode] = merged
        if ok and not args.dry_run:
            out_path.write_text(json.dumps(doc, ensure_ascii=False))

    if failures:
        print("\nREFUSED:", *failures, sep="\n  ")
    if not args.dry_run:
        LOG.write_text(
            json.dumps(
                {"old_rev": OLD_REV, "rescored_at": now, "results": log, "refused": failures},
                indent=2,
            )
            + "\n"
        )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
