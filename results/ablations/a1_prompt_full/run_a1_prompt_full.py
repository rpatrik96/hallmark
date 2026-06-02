"""A1 prompt-sensitivity (full): 4 prompt variants x 3 models x n=150.

Scales the e_prompt_pilot design (n=60, deepseek only) to the full A1 ablation:
the stratified n=150 dev_public sample (seed 42) crossed with three OpenRouter
models and the four prompt variants. GPT-5.1 is EXCLUDED (OpenAI quota exhausted
2026-05-30); the prompt effect is model-agnostic and the three OpenRouter models
already span a reasoning model (deepseek-v3.2), a frontier instruct model
(Sonnet-4.6) and a fast model (Gemini-2.5-Flash).

ENDPOINT-DRIFT POLICY: every run here is a FRESH dated snapshot against the
OpenRouter endpoint, NOT a reproduction of the 2026-05-04 published delta-eval.
Snapshot metadata (date, endpoint, model IDs) is written into summary.json.

Deliverables (summary.json):
  - per (model, variant): DR, FPR, UNCERTAIN-rate, F1, ECE, coverage
  - verdict-flip rate: fraction of entries whose VALID/HALLUCINATED verdict
    changes vs the `default` variant, per model and pooled
  - Spearman tool-RANKING stability across prompts: rank the three models by
    a chosen metric under each prompt variant, then correlate each variant's
    ranking against the default-variant ranking. The reassurance claim is that
    the model ranking is prompt-invariant even when absolute FPR is not.
  - FPR decomposition: how much of the default ~0.89 FPR is prompt-induced
    (default FPR vs the min-FPR variant, per model and pooled).

temp=0, seed=42 fixed inside the shared verify helper. Checkpointed per
(model, variant) so a re-run resumes instead of re-calling the API.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from scipy.stats import spearmanr

from hallmark.baselines.llm_verifier import (
    VERIFICATION_PROMPT,
    _verify_with_openai_compatible,
)
from hallmark.dataset.schema import BenchmarkEntry, BlindEntry
from hallmark.evaluation import evaluate

ROOT = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
OUTDIR = ROOT / "results/ablations/a1_prompt_full"
SAMPLE = OUTDIR / "sample_150.jsonl"
ENDPOINT = "https://openrouter.ai/api/v1"

# Three OpenRouter models + GPT-5.1 via OpenAI direct. IDs from
# OPENROUTER_MODELS / OPENAI_MODELS in hallmark/baselines/llm_verifier.py.
# gpt-5.1 was originally excluded (OpenAI quota exhausted 2026-05-30); it is now
# included via the OpenAI-direct endpoint to match the rest of the paper's
# gpt-5.1 numbers. The three OpenRouter models are fully checkpointed, so a
# re-run resumes them from cache and only gpt-5.1 actually re-calls the API.
MODELS: dict[str, str] = {
    "sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gpt-5.1": "gpt-5.1",
}

# Models routed through the OpenAI-direct endpoint (OPENAI_API_KEY) instead of
# OpenRouter. Everything else goes through OpenRouter.
OPENAI_DIRECT_MODELS: set[str] = {"gpt-5.1"}

# --- Prompt variants (verbatim from the validated e_prompt_pilot design) ----

# default: VERIFICATION_PROMPT (taxonomy-in-prompt, code default).

# notaxo: the prompt as PRINTED in app:prompt-template — taxonomy block removed.
NOTAXO_PROMPT = """\
You are a citation verification expert. Analyze the following BibTeX entry \
and determine if it is a VALID real publication or a HALLUCINATED (fabricated) citation.

BibTeX entry:
```bibtex
{bibtex}
```

Consider:
1. Is the title plausible and does it match known work by these authors?
2. Are the authors real researchers in this field?
3. Is the venue (journal/conference) real?
4. Does the year make sense?
5. If a DOI is present, does it look properly formatted?

Respond with JSON only:
{{
    "label": "VALID" or "HALLUCINATED" or "UNCERTAIN",
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation"
}}"""

# uncertain: NOTAXO + explicit instruction that UNCERTAIN is allowed/preferred.
UNCERTAIN_PROMPT = (
    NOTAXO_PROMPT + "\n\nIf you cannot determine with confidence whether the entry is real or "
    "fabricated, respond with UNCERTAIN rather than guessing VALID or HALLUCINATED."
)

# terse: minimal one-line prompt.
TERSE_PROMPT = """\
Is this BibTeX citation a real publication or fabricated?
```bibtex
{bibtex}
```
Reply with JSON only: {{"label": "VALID" or "HALLUCINATED" or "UNCERTAIN", \
"confidence": 0.0-1.0, "reason": "..."}}"""

VARIANTS: dict[str, str] = {
    "default": VERIFICATION_PROMPT,
    "notaxo": NOTAXO_PROMPT,
    "uncertain": UNCERTAIN_PROMPT,
    "terse": TERSE_PROMPT,
}


def load_entries() -> list[BenchmarkEntry]:
    return [
        BenchmarkEntry.from_dict(json.loads(line))
        for line in SAMPLE.read_text().splitlines()
        if line.strip()
    ]


def make_prompt_fn(template: str):
    def _fn(entry: BlindEntry) -> str:
        return template.format(bibtex=entry.to_bibtex())

    return _fn


def rank_from_scores(scores: dict[str, float]) -> dict[str, float]:
    """Rank models by score (descending). Average ranks for ties. Rank 1 = best."""
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ranks: dict[str, float] = {}
    i = 0
    vals = [v for _, v in items]
    names = [k for k, _ in items]
    while i < len(items):
        j = i
        while j < len(items) and vals[j] == vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # positions i+1..j (1-indexed), averaged
        for k in range(i, j):
            ranks[names[k]] = avg_rank
        i = j
    return ranks


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    entries = load_entries()
    keys = [e.bibtex_key for e in entries]
    blind = [e.to_blind() for e in entries]
    n_hall = sum(1 for e in entries if e.label == "HALLUCINATED")
    n_valid = sum(1 for e in entries if e.label == "VALID")
    print(f"Loaded {len(entries)} entries: {n_hall} HALL / {n_valid} VALID", flush=True)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set; source /tmp/.or_env first")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if OPENAI_DIRECT_MODELS & set(MODELS) and not openai_key:
        raise SystemExit("OPENAI_API_KEY not set; source /tmp/.openai_env first")

    # results[model][variant] = metric record; preds_by[model][variant] = {key: label}
    results: dict[str, dict[str, dict]] = {}
    preds_by: dict[str, dict[str, dict[str, str]]] = {}

    for mname, mid in MODELS.items():
        results[mname] = {}
        preds_by[mname] = {}
        for vname, template in VARIANTS.items():
            print(f"\n=== model={mname} variant={vname} ===", flush=True)
            ckpt = OUTDIR / "checkpoints" / mname / vname
            is_openai = mname in OPENAI_DIRECT_MODELS
            preds = _verify_with_openai_compatible(
                blind,
                model=mid,
                api_key=openai_key if is_openai else api_key,
                base_url=None if is_openai else ENDPOINT,
                source_prefix="openai" if is_openai else "openrouter",
                checkpoint_dir=ckpt,
                prompt_fn=make_prompt_fn(template),
                temperature=0.0,
            )
            res = evaluate(entries, preds, tool_name=f"{mname}-{vname}", split_name="a1_dev150")
            n_unc = sum(1 for p in preds if p.label == "UNCERTAIN")
            n_err = sum(1 for p in preds if "[Error fallback]" in (p.reason or ""))
            rec = {
                "model": mname,
                "model_id": mid,
                "variant": vname,
                "n": len(preds),
                "detection_rate": res.detection_rate,
                "false_positive_rate": res.false_positive_rate,
                "f1_hallucination": res.f1_hallucination,
                "ece": res.ece,
                "num_uncertain": n_unc,
                "uncertain_rate": n_unc / len(preds) if preds else 0.0,
                "num_error_fallback": n_err,
                "coverage": res.coverage,
            }
            results[mname][vname] = rec
            preds_by[mname][vname] = {p.bibtex_key: p.label for p in preds}
            print(json.dumps(rec, indent=2), flush=True)

    # --- Verdict-flip rate vs default (per model + pooled) -------------------
    # A flip = the binary VALID/HALLUCINATED verdict differs from `default`.
    # UNCERTAIN is treated as its own verdict so flips into/out of UNCERTAIN
    # also count; we report both the all-label flip and the strict
    # VALID<->HALLUCINATED flip (excluding entries that are UNCERTAIN in either).
    flip_report: dict[str, dict] = {}
    pooled_flips_all = 0
    pooled_flips_strict = 0
    pooled_total_all = 0
    pooled_total_strict = 0
    for mname in MODELS:
        base = preds_by[mname]["default"]
        per_variant = {}
        for vname in VARIANTS:
            if vname == "default":
                continue
            cur = preds_by[mname][vname]
            n_flip_all = 0
            n_flip_strict = 0
            n_strict_eligible = 0
            for k in keys:
                b, c = base.get(k), cur.get(k)
                if b != c:
                    n_flip_all += 1
                if b in ("VALID", "HALLUCINATED") and c in ("VALID", "HALLUCINATED"):
                    n_strict_eligible += 1
                    if b != c:
                        n_flip_strict += 1
            per_variant[vname] = {
                "flip_rate_all": n_flip_all / len(keys),
                "n_flip_all": n_flip_all,
                "flip_rate_strict": (
                    n_flip_strict / n_strict_eligible if n_strict_eligible else 0.0
                ),
                "n_flip_strict": n_flip_strict,
                "n_strict_eligible": n_strict_eligible,
            }
            pooled_flips_all += n_flip_all
            pooled_total_all += len(keys)
            pooled_flips_strict += n_flip_strict
            pooled_total_strict += n_strict_eligible
        # mean flip rate over the 3 non-default variants
        mean_all = sum(v["flip_rate_all"] for v in per_variant.values()) / len(per_variant)
        flip_report[mname] = {
            "per_variant_vs_default": per_variant,
            "mean_flip_rate_all": mean_all,
        }
    flip_report["_pooled"] = {
        "flip_rate_all": pooled_flips_all / pooled_total_all,
        "flip_rate_strict": (
            pooled_flips_strict / pooled_total_strict if pooled_total_strict else 0.0
        ),
        "note": "pooled over all models x all non-default variants vs default",
    }

    # --- Spearman ranking stability across prompts ---------------------------
    # For each prompt variant, rank the 3 models by a metric, then Spearman-
    # correlate that ranking against the default-variant ranking. Done for both
    # F1-hallucination (composite capability) and DR (pure detection capability).
    ranking_stability: dict[str, dict] = {}
    for metric in ("f1_hallucination", "detection_rate", "false_positive_rate"):
        # For FPR, lower is better -> rank ascending; handle by negating.
        per_variant_ranks: dict[str, dict[str, float]] = {}
        for vname in VARIANTS:
            scores = {}
            for mname in MODELS:
                val = results[mname][vname][metric]
                if val is None:
                    val = 0.0
                # rank_from_scores ranks descending (higher=better). For FPR we
                # want lower=better, so negate.
                scores[mname] = -val if metric == "false_positive_rate" else val
            per_variant_ranks[vname] = rank_from_scores(scores)
        base_rank = per_variant_ranks["default"]
        model_order = list(MODELS.keys())
        base_vec = [base_rank[m] for m in model_order]
        per_variant_rho = {}
        for vname in VARIANTS:
            if vname == "default":
                continue
            vec = [per_variant_ranks[vname][m] for m in model_order]
            # With only 3 items, Spearman is well-defined but coarse (values in
            # {-1, -0.5, 0.5, 1}); report rho and the raw rank vectors.
            rho, _ = spearmanr(base_vec, vec)
            per_variant_rho[vname] = None if rho != rho else float(rho)  # NaN guard
        # Pairwise mean over all C(4,2) variant pairs for a more stable summary.
        variant_list = list(VARIANTS.keys())
        pair_rhos = []
        for i in range(len(variant_list)):
            for j in range(i + 1, len(variant_list)):
                vi = [per_variant_ranks[variant_list[i]][m] for m in model_order]
                vj = [per_variant_ranks[variant_list[j]][m] for m in model_order]
                r, _ = spearmanr(vi, vj)
                if r == r:
                    pair_rhos.append(float(r))
        ranking_stability[metric] = {
            "per_variant_ranks": {
                v: {m: per_variant_ranks[v][m] for m in model_order} for v in VARIANTS
            },
            "rho_vs_default": per_variant_rho,
            "mean_pairwise_rho": sum(pair_rhos) / len(pair_rhos) if pair_rhos else None,
            "n_models": len(MODELS),
        }

    # --- FPR decomposition: how much of default FPR is prompt-induced --------
    fpr_decomp: dict[str, dict] = {}
    for mname in MODELS:
        fprs = {
            v: results[mname][v]["false_positive_rate"]
            for v in VARIANTS
            if results[mname][v]["false_positive_rate"] is not None
        }
        if not fprs:
            continue
        default_fpr = results[mname]["default"]["false_positive_rate"]
        min_v = min(fprs, key=lambda v: fprs[v])
        fpr_decomp[mname] = {
            "default_fpr": default_fpr,
            "min_fpr": fprs[min_v],
            "min_fpr_variant": min_v,
            "prompt_induced_fpr_drop": (
                default_fpr - fprs[min_v] if default_fpr is not None else None
            ),
            "all_variant_fpr": fprs,
        }

    # --- Assemble summary ----------------------------------------------------
    summary = {
        "experiment": "A1_prompt_sensitivity_full",
        "snapshot": {
            "date": date.today().isoformat(),
            "endpoint": ENDPOINT,
            "policy": (
                "FRESH dated snapshot vs OpenRouter endpoint; NOT a reproduction "
                "of the 2026-05-04 published delta-eval. Endpoint may drift."
            ),
            "temperature": 0.0,
            "seed": 42,
            "models_excluded": {},
            "endpoints": {
                "openrouter": "sonnet-4.6, deepseek-v3.2, gemini-2.5-flash",
                "openai_direct": "gpt-5.1 (OPENAI_API_KEY; matches the paper's "
                "other gpt-5.1 numbers, which use the OpenAI endpoint, not OpenRouter)",
            },
        },
        "sample": {
            "path": "results/ablations/a1_prompt_full/sample_150.jsonl",
            "n": len(entries),
            "n_hallucinated": n_hall,
            "n_valid": n_valid,
            "stratified_by": ["label", "difficulty_tier"],
            "seed": 42,
        },
        "models": MODELS,
        "variants": list(VARIANTS.keys()),
        "per_model_per_variant": results,
        "verdict_flip_rate": flip_report,
        "ranking_stability_spearman": ranking_stability,
        "fpr_decomposition": fpr_decomp,
        "framing": {
            "DR": "stable capability — should move little across wording",
            "FPR": "wording-sensitive calibration artifact — moves most",
            "key_reassurance": (
                "model RANKING is prompt-invariant (Spearman) even when absolute "
                "FPR is not; quantifies how much of the ~0.89 default FPR is "
                "prompt-induced rather than a real false-alarm rate"
            ),
        },
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # --- Console deltas ------------------------------------------------------
    print("\n=== DR / FPR / UNCERTAIN-rate per (model, variant) ===", flush=True)
    hdr = f"{'model':<18}{'variant':<11}{'DR':>7}{'FPR':>8}{'UNC%':>7}{'F1':>7}"
    print(hdr, flush=True)
    for mname in MODELS:
        for vname in VARIANTS:
            r = results[mname][vname]
            fpr = r["false_positive_rate"]
            print(
                f"{mname:<18}{vname:<11}"
                f"{r['detection_rate']:>7.3f}"
                f"{(fpr if fpr is not None else float('nan')):>8.3f}"
                f"{r['uncertain_rate'] * 100:>7.1f}"
                f"{r['f1_hallucination']:>7.3f}",
                flush=True,
            )
    print("\n=== Verdict-flip rate (mean over non-default variants) ===", flush=True)
    for mname in MODELS:
        print(f"{mname:<18}{flip_report[mname]['mean_flip_rate_all']:.3f}", flush=True)
    print(f"{'POOLED(all)':<18}{flip_report['_pooled']['flip_rate_all']:.3f}", flush=True)
    print(
        f"{'POOLED(strict)':<18}{flip_report['_pooled']['flip_rate_strict']:.3f}",
        flush=True,
    )
    print("\n=== Spearman ranking stability (mean pairwise rho) ===", flush=True)
    for metric, rs in ranking_stability.items():
        print(f"{metric:<22}{rs['mean_pairwise_rho']}", flush=True)
    print("\nWrote", OUTDIR / "summary.json", flush=True)


if __name__ == "__main__":
    main()
