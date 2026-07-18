"""Evaluate the zero-shot LLM cohort on the 500-entry cross-domain split.

Closes the paper's "LLM verifiers remain unevaluated on it" gap
(sections/limitations.tex): the co-designed bibtex-updater has cross-domain
numbers (FPR 0.092 -> 0.375, app:codesign), but none of the twelve zero-shot
LLM baselines from tab:results do.

Split: data/v1.0/test_crossdomain.jsonl -- 500 entries (200 valid / 300
hallucinated); domains via `source`: pubmed (156) + biorxiv (143) = biomedical,
dblp_cs_non_ml (201) = non-ML CS venues.

Setup mirrors the dev_public zero-shot runs: same default prompt, same
verify_with_openai/verify_with_openrouter defaults (no cutoff_aware, default
temperature), sequential per model. Checkpoints live in a split-specific dir
(results/crossdomain_llms/checkpoints) because checkpoint filenames are keyed
by provider+model only and would collide with dev_public checkpoint files.

Usage:
    python scripts/evaluate_crossdomain_llms.py --list
    python scripts/evaluate_crossdomain_llms.py --only llm_openrouter_gemini_flash --limit 3
    python scripts/evaluate_crossdomain_llms.py --only llm_openai

Outputs -> results/crossdomain_llms/result_<key>.json (+ predictions jsonl).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from hallmark.baselines.llm_verifier import verify_with_openai, verify_with_openrouter
from hallmark.dataset.schema import load_entries
from hallmark.evaluation.metrics import evaluate

REPO = Path(__file__).resolve().parents[1]

# Named splits. Each gets its OWN output+checkpoint dir: checkpoint filenames are
# keyed by provider+model only, so sharing a dir across splits (same models)
# would make one split read the other's cached verdicts.
SPLITS: dict[str, Path] = {
    # confounded original: valid biomed all 2026 (post-cutoff), CS 2023-25
    "v2026": REPO / "data/v1.0/test_crossdomain.jsonl",
    # recency-matched: all valid entries 2021-2023 (pre-cutoff) -> isolates domain
    "matched": REPO / "data/v1.1_crossdomain_matched/test_crossdomain_matched.jsonl",
}
OUT_DIRS: dict[str, Path] = {
    "v2026": REPO / "results/crossdomain_llms",
    "matched": REPO / "results/crossdomain_matched_llms",
}

# key -> (provider, model_id, api_key_env, display); mirrors the registry in
# evaluate_temporal_supplement.py plus the GPT-5.4 control (gpt54 probe setup).
MODELS: dict[str, tuple[str, str, str, str]] = {
    "llm_openai": ("openai", "gpt-5.1", "OPENAI_API_KEY", "GPT-5.1"),
    "llm_openai_gpt54": ("openai", "gpt-5.4", "OPENAI_API_KEY", "GPT-5.4"),
    "llm_openrouter_qwen": (
        "openrouter",
        "qwen/qwen3-235b-a22b-2507",
        "OPENROUTER_API_KEY",
        "Qwen3-235B",
    ),
    "llm_openrouter_deepseek_v3": (
        "openrouter",
        "deepseek/deepseek-v3.2",
        "OPENROUTER_API_KEY",
        "DeepSeek-V3.2",
    ),
    "llm_openrouter_deepseek_r1": (
        "openrouter",
        "deepseek/deepseek-r1",
        "OPENROUTER_API_KEY",
        "DeepSeek-R1",
    ),
    "llm_openrouter_mistral": (
        "openrouter",
        "mistralai/mistral-large-2512",
        "OPENROUTER_API_KEY",
        "Mistral Large",
    ),
    "llm_openrouter_gemini_flash": (
        "openrouter",
        "google/gemini-2.5-flash",
        "OPENROUTER_API_KEY",
        "Gemini 2.5 Flash",
    ),
    "llm_openrouter_gemini_pro": (
        "openrouter",
        "google/gemini-2.5-pro",
        "OPENROUTER_API_KEY",
        "Gemini 2.5 Pro",
    ),
    "llm_openrouter_llama_4_maverick": (
        "openrouter",
        "meta-llama/llama-4-maverick",
        "OPENROUTER_API_KEY",
        "Llama 4 Maverick",
    ),
    "llm_openrouter_claude_sonnet_4_6": (
        "openrouter",
        "anthropic/claude-sonnet-4.6",
        "OPENROUTER_API_KEY",
        "Claude Sonnet 4.6",
    ),
    "llm_openrouter_claude_opus_4_7": (
        "openrouter",
        "anthropic/claude-opus-4.7",
        "OPENROUTER_API_KEY",
        "Claude Opus 4.7",
    ),
    "llm_openrouter_qwen_max": (
        "openrouter",
        "qwen/qwen3-vl-235b-a22b-instruct",
        "OPENROUTER_API_KEY",
        "Qwen3-VL-235B",
    ),
}


def run_model(
    key: str,
    entries: list,
    limit: int | None,
    out_dir: Path,
    ckpt_dir: Path,
    retry_failed: bool = False,
) -> dict:
    provider, model_id, key_env, display = MODELS[key]
    api_key = os.getenv(key_env)
    if not api_key:
        raise SystemExit(f"{key_env} not set (source the env files in /tmp first)")

    subset = entries[:limit] if limit else entries
    blind = [e.to_blind() for e in subset]
    kwargs: dict = {
        "checkpoint_dir": ckpt_dir,
        "log_dir": out_dir / "api_logs" / key,
        "api_key": api_key,
    }
    if retry_failed:
        # On a resumed run, re-attempt entries whose prior verdict was an
        # [Error fallback] (e.g. an API call cut off by a disconnect), instead
        # of freezing that error into the final result.
        kwargs["retry_failed"] = True
    if provider == "openai":
        preds = verify_with_openai(blind, model=model_id, **kwargs)
    else:
        preds = verify_with_openrouter(blind, model=model_id, **kwargs)

    res = evaluate(subset, preds, tool_name=key, split_name="test_crossdomain")

    def subset_eval(sources: set[str]) -> dict:
        keys = {e.bibtex_key for e in subset if getattr(e, "source", None) in sources}
        sub_e = [e for e in subset if e.bibtex_key in keys]
        sub_p = [p for p in preds if p.bibtex_key in keys]
        if not sub_e:
            return {"n": 0}
        r = evaluate(sub_e, sub_p, tool_name=key, split_name="xd_subset")
        return {
            "n": len(sub_e),
            "fpr": r.false_positive_rate,
            "dr": r.detection_rate,
            "f1": r.f1_hallucination,
            "num_uncertain": r.num_uncertain,
        }

    out = {
        "key": key,
        "display": display,
        "model_id": model_id,
        "provider": provider,
        "n": len(subset),
        "api_calls": sum(p.api_calls for p in preds),
        "predicted_label_dist": dict(Counter(p.label for p in preds)),
        "overall": res.to_dict(),
        "by_domain": {
            "biomed": subset_eval({"pubmed", "biorxiv"}),
            "cs_non_ml": subset_eval({"dblp_cs_non_ml"}),
        },
    }
    suffix = f"_limit{limit}" if limit else ""
    (out_dir / f"result_{key}{suffix}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    with (out_dir / f"{key}_test_crossdomain_predictions{suffix}.jsonl").open("w") as f:
        for p in preds:
            f.write(json.dumps(p.to_dict() if hasattr(p, "to_dict") else vars(p), default=str))
            f.write("\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(MODELS), default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on first N entries")
    ap.add_argument("--list", action="store_true")
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="on resume, re-attempt entries whose prior verdict was an [Error fallback]",
    )
    ap.add_argument(
        "--split",
        choices=sorted(SPLITS),
        default="v2026",
        help="which cross-domain split to evaluate (v2026=confounded original, "
        "matched=recency-matched 2021-23). Each writes to its own dir.",
    )
    args = ap.parse_args()

    if args.list:
        for k, (prov, mid, _, disp) in MODELS.items():
            print(f"{k:36s} {prov:10s} {mid:40s} {disp}")
        return

    data = SPLITS[args.split]
    out_dir = OUT_DIRS[args.split]
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    entries = load_entries(str(data))
    n_valid = sum(1 for e in entries if e.label == "VALID")
    print(f"[{args.split}] {data}")
    print(
        f"Loaded {len(entries)} entries ({n_valid} valid / {len(entries) - n_valid} hallucinated)"
    )

    jobs = [args.only] if args.only else sorted(MODELS)
    for key in jobs:
        print(f"\n=== {key} ({MODELS[key][1]}) ===", flush=True)
        out = run_model(key, entries, args.limit, out_dir, ckpt_dir, retry_failed=args.retry_failed)
        ov = out["overall"]
        print(
            f"  FPR={ov['false_positive_rate']} DR={ov['detection_rate']} "
            f"F1={ov['f1_hallucination']} uncertain={ov.get('num_uncertain')} "
            f"pred={out['predicted_label_dist']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
