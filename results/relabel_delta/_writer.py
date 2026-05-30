"""Write regenerated aggregate JSONs, preserving each published JSON's key set.

For offline-rescorable tools we recompute a fresh EvaluationResult on the NEW
labels via the repo `evaluate()`, then overwrite ONLY the keys already present
in the published JSON. per_tier_metrics / per_type_metrics / detect_at_k /
union_recall_at_k are replaced wholesale; *_ci fields are preserved as-is
(published JSONs carry null CIs — we do not recompute bootstrap CIs here).
"""

from __future__ import annotations

import json
from pathlib import Path

# Scalar metric fields we overwrite from the recomputed EvaluationResult.
SCALAR_FIELDS = [
    "num_entries",
    "num_hallucinated",
    "num_valid",
    "detection_rate",
    "false_positive_rate",
    "f1_hallucination",
    "tier_weighted_f1",
    "mcc",
    "macro_f1",
    "temporal_robustness",
    "cost_efficiency",
    "mean_api_calls",
    "ece",
    "auroc",
    "auprc",
    "num_uncertain",
    "tier3_f1",
    "coverage",
    "coverage_adjusted_f1",
]
# Dict/structured fields replaced wholesale when present.
DICT_FIELDS = [
    "per_tier_metrics",
    "per_type_metrics",
    "union_recall_at_k",
    "detect_at_k",
]
# CI fields are preserved (not recomputed).
CI_FIELDS = {
    "detection_rate_ci",
    "f1_hallucination_ci",
    "tier_weighted_f1_ci",
    "fpr_ci",
    "ece_ci",
    "mcc_ci",
    "tier3_f1_ci",
}


def _result_value(result, key):
    """Map a published JSON key to the recomputed EvaluationResult value."""
    if key == "detect_at_k":
        # Older bibtexupdater schema; the recomputed result exposes union_recall_at_k.
        return getattr(result, "union_recall_at_k", {})
    return getattr(result, key, None)


def merge_into_published(published: dict, result) -> dict:
    """Return a new dict = published with recomputed values for present keys only."""
    out = dict(published)  # preserve key order
    rd = json.loads(result.to_json())
    for k in list(out.keys()):
        if k in CI_FIELDS:
            continue  # keep published CI (null)
        if k in SCALAR_FIELDS:
            out[k] = rd.get(k, _result_value(result, k))
        elif k in DICT_FIELDS:
            out[k] = _result_value(result, k)
        # everything else (tool_name, split_name, type_accuracy, type_confusion,
        # cascade_breakdown_stats, etc.) is left untouched unless it is a scalar
        # metric we recompute below.
    # Special-case keys that may not be in SCALAR/DICT lists but exist in result:
    for k in ("type_accuracy", "type_confusion", "cascade_breakdown_stats"):
        if k in out and k in rd:
            out[k] = rd[k]
    return out


def write_aggregate(out_path: Path, published: dict, result) -> dict:
    merged = merge_into_published(published, result)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    return merged
