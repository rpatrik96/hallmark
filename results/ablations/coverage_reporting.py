"""Coverage / abstention-aware reporting (selective prediction) — OFFLINE.

Implements the approved design in ``coverage_reporting_design.md``. NO API calls:
this re-scores stored per-entry predictions against the v1.1.1 corrected labels
(dev_public = 513 VALID / 606 HALL; test_public = 312 / 519) and reshapes the
already-computed A3 threshold ablation into a selective-prediction view.

It produces three things, written to ``coverage_reporting.{md,json}``:

1. Per-tool **Coverage** + **dual scoring** (Table-2 addition):
   - Coverage = (#committed VALID/HALL verdicts) / N = 1 - UNCERTAIN-rate.
   - CONSERVATIVE DR/FPR/F1 (UNCERTAIN excluded; the repo ``build_confusion_matrix``
     protocol via ``evaluate(eval_mode="conservative")``).
   - AGGRESSIVE DR/FPR/F1 (UNCERTAIN + missing -> HALLUCINATED@0.55, via
     ``evaluate(eval_mode="aggressive")``).
   Reuses the same per-entry files as ``results/relabel_delta/regenerate_offline.py``
   and the same ``hallmark.evaluation.evaluate`` so the conservative numbers
   reproduce the regenerated aggregates in ``data/v1.0/baseline_results/``.

2. Risk-coverage / selective-prediction view (appendix figure):
   For each tool with graded confidences, the FPR-vs-coverage curve as an
   abstention band around the decision boundary widens (abstain on the
   least-confident entries first), plus a single **FPR@90%-coverage** number.
   The point confidences are the same ones the A3 ablation consumed.

bibtex-updater (v1.2.0) is scored from its raw status JSONL, mapping statuses to
VALID/HALL/UNCERTAIN via the cascade module's ``STAGE1_VERIFIED`` /
``STATUS_TO_TYPE`` / ``ROUTE_TO_STAGE2`` sets (else -> UNCERTAIN).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
sys.path.insert(0, str(REPO))

from hallmark.baselines.cascade import (  # noqa: E402
    STAGE1_VERIFIED,
    STATUS_TO_TYPE,
)
from hallmark.dataset.schema import (  # noqa: E402
    BenchmarkEntry,
    Prediction,
    is_canary_entry,
)
from hallmark.evaluation.metrics import evaluate  # noqa: E402

ABL = REPO / "results" / "ablations"
A3 = ABL / "a3_threshold_full" / "a3_full_result.json"
DATA = {
    "dev_public": REPO / "data/v1.0/dev_public.jsonl",
    "test_public": REPO / "data/v1.0/test_public.jsonl",
}

# ---------------------------------------------------------------------------
# Tool -> per-entry prediction file, per split. Mirrors regenerate_offline.py.
# (UNCERTAIN-bearing, full-coverage offline files only.) The drift-affected
# Anthropic dev files are handled separately (DRIFT_DEV_POINTS) per the design.
# ---------------------------------------------------------------------------
TOOLS: dict[str, dict[str, str]] = {
    "llm_openrouter_deepseek_r1": {
        "dev_public": "results/llm_openrouter_deepseek_r1_dev_public_predictions.jsonl",
        "test_public": "results/checkpoints/llm_openrouter_deepseek_r1_test_public/"
        "openrouter_deepseek_deepseek-r1.jsonl",
    },
    "llm_openrouter_deepseek_v3": {
        "dev_public": "results/llm_openrouter_deepseek_v3_dev_public_predictions.jsonl",
        "test_public": "results/checkpoints/llm_openrouter_deepseek_v3/"
        "openrouter_deepseek_deepseek-v3.2.jsonl",
    },
    "llm_openrouter_gemini_flash": {
        "dev_public": "results/llm_openrouter_gemini_flash_dev_public_predictions.jsonl",
        "test_public": "results/checkpoints/llm_openrouter_gemini_flash_test_public/"
        "openrouter_google_gemini-2.5-flash.jsonl",
    },
    "llm_openrouter_mistral": {
        "dev_public": "results/llm_openrouter_mistral_dev_public_predictions.jsonl",
        "test_public": "results/checkpoints/llm_openrouter_mistral_test_public/"
        "openrouter_mistralai_mistral-large-2512.jsonl",
    },
    "llm_openrouter_qwen": {
        "dev_public": "results/llm_openrouter_qwen_dev_public_predictions.jsonl",
        "test_public": "results/checkpoints/llm_openrouter_qwen_test_public/"
        "openrouter_qwen_qwen3-235b-a22b-2507.jsonl",
    },
    "llm_openrouter_gemini_pro": {
        "dev_public": "results/new_models/gemini_pro.jsonl",
        "test_public": "results/checkpoints/llm_openrouter_gemini_pro/"
        "openrouter_google_gemini-2.5-pro.jsonl",
    },
    "llm_openrouter_qwen_max": {
        "dev_public": "results/new_models/qwen_max.jsonl",
        "test_public": "results/checkpoints/llm_openrouter_qwen_max_test_public/"
        "openrouter_qwen_qwen3-vl-235b-a22b-instruct.jsonl",
    },
    "llm_openrouter_llama_4_maverick": {
        "dev_public": "results/new_models/llama4_maverick.jsonl",
        "test_public": "results/checkpoints/llm_openrouter_llama_4_maverick_test_public/"
        "openrouter_meta-llama_llama-4-maverick.jsonl",
    },
    "llm_openrouter_claude_opus_4_7": {
        # dev is drift-affected (see DRIFT_DEV_POINTS); test is clean offline.
        "test_public": "results/checkpoints/llm_openrouter_claude_opus_4_7_test_public/"
        "openrouter_anthropic_claude-opus-4.7.jsonl",
    },
    "llm_openrouter_claude_sonnet_4_6": {
        "test_public": "results/checkpoints/llm_openrouter_claude_sonnet_4_6/"
        "openrouter_anthropic_claude-sonnet-4.6.jsonl",
    },
    "llm_openai": {
        "dev_public": "results/checkpoints/llm_openai/openai_gpt-5.1.jsonl",
        "test_public": "results/checkpoints/llm_openai/openai_gpt-5.1.jsonl",
    },
    "llm_openai_gpt_5_4": {
        "dev_public": "results/checkpoints/llm_openai_gpt54_dev_public_v3/openai_gpt-5.4.jsonl",
        "test_public": "results/checkpoints/llm_openai_gpt54_test_public/openai_gpt-5.4.jsonl",
    },
    "llm_agentic_openai": {
        "dev_public": "results/temporal_checkpoints/agentic_openai_gpt-5.1.jsonl",
        "test_public": "results/checkpoints/llm_agentic_openai_test_public/"
        "agentic_openai_openai_gpt-5.1.jsonl",
    },
    "llm_agentic_btu_openai": {
        "dev_public": "results/temporal_checkpoints/agentic_btu_openai_gpt-5.1.jsonl",
        "test_public": "results/checkpoints/llm_agentic_btu_openai_test_public/"
        "agentic_btu_openai_openai_gpt-5.1.jsonl",
    },
    "llm_agentic_btu_sonnet_4_6": {
        "dev_public": "results/checkpoints/llm_agentic_btu_sonnet_4_6_dev_public_v2/"
        "agentic_btu_openai_anthropic_claude-sonnet-4.6.jsonl",
        "test_public": "results/checkpoints/llm_agentic_btu_sonnet_4_6_test_public/"
        "agentic_btu_openai_anthropic_claude-sonnet-4.6.jsonl",
    },
    "llm_tool_augmented": {
        # test only — no dev per-entry file exists (matches regenerate_offline.py).
        "test_public": "results/checkpoints/llm_tool_augmented_test_public/"
        "openai_tool_augmented_openai_gpt-5.1.jsonl",
    },
}

# bibtex-updater v1.2.0 raw status JSONL per split (cascade-style status mapping).
BTU_RAW = {
    "dev_public": "results/relabel_delta/btu_v1_2_0/dev_public/btu_raw.jsonl",
    "test_public": "results/relabel_delta/btu_v1_2_0/test_public/btu_raw.jsonl",
}

# Anthropic dev: per-entry files exist but the OpenRouter Anthropic endpoint
# drifted since the published run (todo_sonnet_opus_dev.json). The design says:
# report coverage "not available (summary-only / drift)" and use the published
# delta-eval point estimates for the conservative cells; aggressive = n/a.
DRIFT_DEV_POINTS = {
    "llm_openrouter_claude_sonnet_4_6": {
        "detection_rate": 0.781,
        "false_positive_rate": 0.127,
        "f1_hallucination": 0.827,
    },
    "llm_openrouter_claude_opus_4_7": {
        "detection_rate": 0.752,
        "false_positive_rate": 0.072,
        "f1_hallucination": 0.830,
    },
}

# A3 threshold-sweep tool key -> our canonical tool name (dev split).
A3_TOOL_TO_NAME = {
    "claude_opus_4_7": "llm_openrouter_claude_opus_4_7",
    "claude_sonnet_4_6": "llm_openrouter_claude_sonnet_4_6",
    "deepseek_r1": "llm_openrouter_deepseek_r1",
    "deepseek_v3": "llm_openrouter_deepseek_v3",
    "gemini_flash": "llm_openrouter_gemini_flash",
    "mistral": "llm_openrouter_mistral",
    "qwen": "llm_openrouter_qwen",
    "gpt_5_4": "llm_openai_gpt_5_4",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_entries(split: str) -> list[BenchmarkEntry]:
    ents = [
        BenchmarkEntry.from_json(ln) for ln in DATA[split].read_text().splitlines() if ln.strip()
    ]
    return [e for e in ents if not is_canary_entry(e)]


def load_pred_map(path: Path) -> dict[str, Prediction]:
    """Load per-entry predictions; last write wins on resume duplicates."""
    pm: dict[str, Prediction] = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        if "label" in d and "bibtex_key" in d:
            pm[d["bibtex_key"]] = Prediction.from_dict(d)
    return pm


def btu_status_to_label(status: str) -> str:
    """Map a bibtex-updater status to VALID / HALLUCINATED / UNCERTAIN.

    Uses the cascade module's canonical sets so the mapping matches the
    DB-first cascade Stage-1 routing exactly:
      - STAGE1_VERIFIED -> VALID
      - STATUS_TO_TYPE  -> HALLUCINATED (positive problem evidence)
      - everything else (ROUTE_TO_STAGE2 + unknown) -> UNCERTAIN (abstain)
    """
    if status in STAGE1_VERIFIED:
        return "VALID"
    if status in STATUS_TO_TYPE:
        return "HALLUCINATED"
    # ROUTE_TO_STAGE2 (not_found / partial_match / unconfirmed / api_error / ...)
    # and any unmapped status are abstentions for a standalone BTU tool.
    return "UNCERTAIN"


def load_btu_pred_map(path: Path) -> dict[str, Prediction]:
    pm: dict[str, Prediction] = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        label = btu_status_to_label(d["status"])
        pm[d["key"]] = Prediction(
            bibtex_key=d["key"],
            label=label,  # type: ignore[arg-type]
            confidence=float(d.get("confidence", 0.5)),
        )
    return pm


# ---------------------------------------------------------------------------
# Dual scoring
# ---------------------------------------------------------------------------
@dataclass
class DualScore:
    tool: str
    split: str
    n: int
    n_committed: int
    n_uncertain: int
    coverage: float
    cons_dr: float
    cons_fpr: float
    cons_f1: float
    agg_dr: float | None = None
    agg_fpr: float | None = None
    agg_f1: float | None = None
    note: str = ""


def aligned_predictions(
    entries: list[BenchmarkEntry], pm: dict[str, Prediction]
) -> list[Prediction]:
    """Align to entry order; missing -> conservative VALID@0.5 (repo default)."""
    out: list[Prediction] = []
    for e in entries:
        p = pm.get(e.bibtex_key)
        if p is None:
            p = Prediction(bibtex_key=e.bibtex_key, label="VALID", confidence=0.5)
        out.append(p)
    return out


def score_dual(
    tool: str, split: str, pm: dict[str, Prediction], entries: list[BenchmarkEntry]
) -> DualScore:
    entry_keys = {e.bibtex_key for e in entries}
    # UNCERTAIN among matched entries (the abstentions that count against coverage).
    n_uncertain = sum(
        1 for k in entry_keys if (p := pm.get(k)) is not None and p.label == "UNCERTAIN"
    )
    n_committed = len(entries) - n_uncertain  # missing keys treated as committed-VALID
    coverage = n_committed / len(entries)

    preds = aligned_predictions(entries, pm)
    cons = evaluate(entries, preds, tool, split, eval_mode="conservative")
    agg = evaluate(entries, preds, tool, split, eval_mode="aggressive")
    return DualScore(
        tool=tool,
        split=split,
        n=len(entries),
        n_committed=n_committed,
        n_uncertain=n_uncertain,
        coverage=coverage,
        cons_dr=cons.detection_rate,
        cons_fpr=cons.false_positive_rate or 0.0,
        cons_f1=cons.f1_hallucination,
        agg_dr=agg.detection_rate,
        agg_fpr=agg.false_positive_rate or 0.0,
        agg_f1=agg.f1_hallucination,
    )


# ---------------------------------------------------------------------------
# Risk-coverage (selective prediction)
# ---------------------------------------------------------------------------
@dataclass
class RiskCoverage:
    tool: str
    auroc: float | None
    curve: list[dict[str, float]] = field(default_factory=list)
    fpr_at_90: float | None = None
    dr_at_90: float | None = None
    f1_at_90: float | None = None
    note: str = ""


def risk_coverage_curve(pm: dict[str, Prediction], entries: list[BenchmarkEntry]) -> RiskCoverage:
    """Sweep an abstention band around the decision boundary.

    Confidence -> P(hallucinated) using the hallmark AUROC convention
    (score = conf if HALL else 1-conf). The selective predictor abstains on the
    entries whose |p_hall - 0.5| is smallest (least confident) first; coverage is
    the fraction it still commits on. At each coverage level we score FPR/DR/F1 on
    the committed entries at the model's native 0.5 decision. UNCERTAIN-labelled
    rows have no reliable score and are always abstained (they never re-enter).
    """
    truth = {e.bibtex_key: e.label for e in entries}
    scored: list[tuple[str, float, str]] = []  # (key, margin, decision)
    for k, t in truth.items():  # noqa: B007  (t unused; kept for symmetry)
        p = pm.get(k)
        if p is None or p.label == "UNCERTAIN":
            continue
        conf = float(p.confidence)
        p_hall = conf if p.label == "HALLUCINATED" else 1.0 - conf
        margin = abs(p_hall - 0.5)
        decision = "HALLUCINATED" if p_hall >= 0.5 else "VALID"
        scored.append((k, margin, decision))

    n_total = len(entries)
    # Sort by margin DESC: most confident first; widen the abstain band by dropping
    # the tail (smallest margin) as coverage decreases.
    scored.sort(key=lambda x: x[1], reverse=True)

    def cm_on(subset: list[tuple[str, float, str]]) -> dict[str, float]:
        tp = fp = tn = fn = 0
        for k, _m, dec in subset:
            t = truth[k]
            if t == "HALLUCINATED":
                tp += dec == "HALLUCINATED"
                fn += dec != "HALLUCINATED"
            else:
                fp += dec == "HALLUCINATED"
                tn += dec != "HALLUCINATED"
        dr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * dr / (prec + dr) if (prec + dr) else 0.0
        return {"DR": dr, "FPR": fpr, "F1": f1}

    curve: list[dict[str, float]] = []
    # Coverage grid relative to the FULL split (so abstaining UNCERTAIN counts).
    fpr_at_90 = dr_at_90 = f1_at_90 = None
    max_cov = len(scored) / n_total if n_total else 0.0
    for pct in range(100, 4, -5):
        cov_target = pct / 100.0
        if cov_target > max_cov + 1e-9:
            continue
        keep = max(1, round(cov_target * n_total))
        keep = min(keep, len(scored))
        sub = scored[:keep]
        cm = cm_on(sub)
        actual_cov = keep / n_total
        curve.append(
            {
                "coverage": round(actual_cov, 4),
                "DR": round(cm["DR"], 4),
                "FPR": round(cm["FPR"], 4),
                "F1": round(cm["F1"], 4),
            }
        )

    # Exact 90%-coverage point (interpolated to the closest achievable keep).
    if max_cov >= 0.90:
        keep90 = min(len(scored), max(1, round(0.90 * n_total)))
        cm90 = cm_on(scored[:keep90])
        fpr_at_90 = round(cm90["FPR"], 4)
        dr_at_90 = round(cm90["DR"], 4)
        f1_at_90 = round(cm90["F1"], 4)

    return RiskCoverage(
        tool="",
        auroc=None,
        curve=curve,
        fpr_at_90=fpr_at_90,
        dr_at_90=dr_at_90,
        f1_at_90=f1_at_90,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    entries = {sp: load_entries(sp) for sp in DATA}

    # ---- 1. Per-tool coverage + dual scoring ----
    dual: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []

    for tool, splits in TOOLS.items():
        dual[tool] = {}
        for sp in ("dev_public", "test_public"):
            rel = splits.get(sp)
            if rel is None:
                # Anthropic dev handled via DRIFT_DEV_POINTS; tool_augmented dev absent.
                if sp == "dev_public" and tool in DRIFT_DEV_POINTS:
                    pt = DRIFT_DEV_POINTS[tool]
                    dual[tool][sp] = {
                        "coverage": None,
                        "coverage_note": "not available (summary-only / endpoint drift)",
                        "conservative": {
                            "DR": pt["detection_rate"],
                            "FPR": pt["false_positive_rate"],
                            "F1": pt["f1_hallucination"],
                            "source": "published delta-eval point estimate "
                            "(results_manifest sec1 / todo_sonnet_opus_dev.json)",
                        },
                        "aggressive": None,
                        "aggressive_note": "n/a (no usable per-entry UNCERTAIN data; drift)",
                    }
                else:
                    skipped.append(
                        {
                            "tool": tool,
                            "split": sp,
                            "reason": "no per-entry prediction file for this split",
                        }
                    )
                continue
            path = REPO / rel
            if not path.exists():
                skipped.append({"tool": tool, "split": sp, "reason": f"file missing: {rel}"})
                continue
            pm = load_pred_map(path)
            ds = score_dual(tool, sp, pm, entries[sp])
            dual[tool][sp] = {
                "n": ds.n,
                "n_committed": ds.n_committed,
                "n_uncertain": ds.n_uncertain,
                "coverage": round(ds.coverage, 4),
                "conservative": {
                    "DR": round(ds.cons_dr, 4),
                    "FPR": round(ds.cons_fpr, 4),
                    "F1": round(ds.cons_f1, 4),
                },
                "aggressive": {
                    "DR": round(ds.agg_dr or 0.0, 4),
                    "FPR": round(ds.agg_fpr or 0.0, 4),
                    "F1": round(ds.agg_f1 or 0.0, 4),
                },
            }

    # bibtex-updater (v1.2.0) from raw status JSONL.
    dual["bibtexupdater_v1_2_0"] = {}
    for sp in ("dev_public", "test_public"):
        raw_path = REPO / BTU_RAW[sp]
        if not raw_path.exists():
            skipped.append(
                {
                    "tool": "bibtexupdater_v1_2_0",
                    "split": sp,
                    "reason": f"raw status file missing: {BTU_RAW[sp]}",
                }
            )
            continue
        pm = load_btu_pred_map(raw_path)
        # Completeness: how many split entries have a raw BTU row at all?
        entry_keys = {e.bibtex_key for e in entries[sp]}
        n_scored = len(entry_keys & set(pm.keys()))
        n_missing_raw = len(entry_keys) - n_scored
        ds = score_dual("bibtexupdater_v1_2_0", sp, pm, entries[sp])
        row: dict[str, Any] = {
            "n": ds.n,
            "n_scored_by_raw": n_scored,
            "n_missing_from_raw": n_missing_raw,
            "n_committed": ds.n_committed,
            "n_uncertain": ds.n_uncertain,
            "coverage": round(ds.coverage, 4),
            "conservative": {
                "DR": round(ds.cons_dr, 4),
                "FPR": round(ds.cons_fpr, 4),
                "F1": round(ds.cons_f1, 4),
            },
            "aggressive": {
                "DR": round(ds.agg_dr or 0.0, 4),
                "FPR": round(ds.agg_fpr or 0.0, 4),
                "F1": round(ds.agg_f1 or 0.0, 4),
            },
        }
        # The BTU test raw is still being produced by GEN workflow wkp97jqbb;
        # only ~492/831 entries are scored. Missing rows are silently treated as
        # committed-VALID, which deflates DR/coverage. Flag rather than ship.
        if n_missing_raw > 0.05 * len(entry_keys):
            row["INCOMPLETE"] = (
                f"raw BTU scored only {n_scored}/{len(entry_keys)} entries "
                f"({n_missing_raw} missing). Coverage/DR/FPR are NOT reliable: "
                f"missing entries default to committed-VALID, deflating DR and "
                f"inflating apparent coverage. PENDING GEN workflow wkp97jqbb."
            )
            skipped.append(
                {
                    "tool": "bibtexupdater_v1_2_0",
                    "split": sp,
                    "reason": f"raw incomplete ({n_scored}/{len(entry_keys)} "
                    f"entries scored); pending GEN workflow wkp97jqbb",
                }
            )
        dual["bibtexupdater_v1_2_0"][sp] = row

    # ---- 2. Risk-coverage (dev_public; uses the same confidences as A3) ----
    # Drift-immune for the curve/AUROC math (deterministic re-score of stored
    # confidences); the design and A3's snapshot note both rely on this. The
    # two Anthropic dev files are drift-affected at the *operating point* (so
    # they stay out of the Table-2 coverage cells) but their risk-coverage
    # *shape* is reported in the appendix with an explicit caveat.
    a3 = json.loads(A3.read_text())
    a3_sweep = a3["threshold_sweep"]
    rc_out: dict[str, Any] = {}
    ANTHROPIC_DEV_FILES = {
        "llm_openrouter_claude_opus_4_7": "results/"
        "llm_openrouter_claude_opus_4_7_dev_public_predictions.jsonl",
        "llm_openrouter_claude_sonnet_4_6": "results/"
        "llm_openrouter_claude_sonnet_4_6_dev_public_predictions.jsonl",
    }
    rc_sources = {tool: splits.get("dev_public") for tool, splits in TOOLS.items()}
    rc_sources.update(ANTHROPIC_DEV_FILES)
    for tool, rel in rc_sources.items():
        if rel is None:
            continue
        path = REPO / rel
        if not path.exists():
            continue
        pm = load_pred_map(path)
        # Only meaningful for tools with graded confidence (>2 distinct levels).
        distinct = len({round(p.confidence, 3) for p in pm.values() if p.label != "UNCERTAIN"})
        if distinct <= 2:
            rc_out[tool] = {
                "skipped": "degenerate confidence (<=2 distinct levels); "
                "risk-coverage uninformative",
                "n_confidence_levels": distinct,
            }
            continue
        rc = risk_coverage_curve(pm, entries["dev_public"])
        # Pull AUROC from A3 where available (same per-entry inputs).
        auroc_val = None
        for a3key, name in A3_TOOL_TO_NAME.items():
            if name == tool and a3key in a3_sweep:
                auroc_val = a3_sweep[a3key]["auroc"]
                break
        rc_out[tool] = {
            "auroc": auroc_val,
            "n_confidence_levels": distinct,
            "fpr_at_90_coverage": rc.fpr_at_90,
            "dr_at_90_coverage": rc.dr_at_90,
            "f1_at_90_coverage": rc.f1_at_90,
            "curve": rc.curve,
        }
        if tool in ANTHROPIC_DEV_FILES:
            rc_out[tool]["caveat"] = (
                "drift-affected operating point (OpenRouter Anthropic endpoint "
                "drifted vs the published run); curve SHAPE/AUROC are drift-immune "
                "(deterministic re-score of stored confidences). Appendix-only; "
                "the Table-2 conservative cells use the published delta-eval point "
                "estimates instead."
            )

    out = {
        "meta": {
            "title": "Coverage / abstention-aware reporting (selective prediction)",
            "design": "results/ablations/coverage_reporting_design.md",
            "method": "OFFLINE re-scoring of stored per-entry predictions; NO API calls",
            "label_version": "v1.1.1",
            "split_counts": {
                "dev_public": {"VALID": 513, "HALLUCINATED": 606, "N": 1119},
                "test_public": {"VALID": 312, "HALLUCINATED": 519, "N": 831},
            },
            "coverage_def": "Coverage = #committed(VALID|HALL) / N = 1 - UNCERTAIN-rate",
            "conservative_def": "UNCERTAIN excluded (build_confusion_matrix protocol)",
            "aggressive_def": "UNCERTAIN + missing -> HALLUCINATED@0.55",
            "btu_mapping": "cascade STAGE1_VERIFIED->VALID, STATUS_TO_TYPE->HALL, "
            "ROUTE_TO_STAGE2/unmapped->UNCERTAIN (bibtex-updater v1.2.0)",
            "risk_coverage_def": "abstention band around the 0.5 decision boundary; "
            "abstain least-confident first; FPR/DR/F1 on committed entries at native decision",
        },
        "per_tool_coverage_dual": dual,
        "risk_coverage": rc_out,
        "could_not_compute": skipped,
    }
    (ABL / "coverage_reporting.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"WROTE {ABL / 'coverage_reporting.json'}")
    return None


if __name__ == "__main__":
    main()
