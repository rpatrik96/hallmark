#!/usr/bin/env python3
"""Stratified analysis of the GPT-5.4 post-freeze probe on the temporal supplement.

GPT-5.4's training cutoff is Aug 31, 2025 (OpenAI disclosure). On the
2024-2025 temporal supplement, this shifts the pre/post line forward by
roughly 11 months vs GPT-5.1's Sep 2024 cutoff.

The natural stratification is:
  - ``pre_cutoff_54``: entries whose year is 2023 or 2024 (all months pre-Aug 2025)
  - ``pre_cutoff_54_likely``: entries from 2025 with venues that publish
    before Aug 2025 (ICLR May, ICML Jul, AAAI Feb, CVPR Jun, ACL Jul)
  - ``post_cutoff_54``: entries from 2025 NeurIPS/Dec venues + 2026+

For GPT-5.1 (Sep 2024), effectively all supplement entries are post-cutoff.

The H1 claim is: FPR drops sharply on entries that are pre-cutoff for
GPT-5.4 but post-cutoff for GPT-5.1. If this replicates, H1 is validated
across a generation of OpenAI models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark.dataset.schema import load_entries, load_predictions
from hallmark.evaluation.metrics import evaluate

SUPPLEMENT = Path("results/temporal_supplement/temporal_supplement_2024_2025.jsonl")
GPT51_PREDS = Path("results/temporal_supplement/llm_openai_temporal_predictions.jsonl")
GPT54_PREDS = Path("results/gpt54_probe/llm_openai_gpt54_default_temporal_predictions.jsonl")

# Venues published before Aug 31, 2025 (GPT-5.4 cutoff).
PRE_AUG_2025_VENUES = {"ICLR", "ICML", "AAAI", "CVPR", "CVPR Workshops", "ACL"}
# NeurIPS 2025 proceedings appear Dec 2025 -> post-cutoff for 5.4.
POST_AUG_2025_VENUES = {"NeurIPS"}


def stratum(entry_year: str, venue: str) -> str:
    """Return pre/post cutoff stratum for GPT-5.4 (Aug 2025 cutoff)."""
    try:
        year = int(entry_year)
    except (ValueError, TypeError):
        return "unknown"
    if year < 2025:
        return "pre_54"  # pre-Aug 2025
    if year > 2025:
        return "post_54"  # future entries (mostly hallucinated by construction)
    # year == 2025
    if venue in PRE_AUG_2025_VENUES:
        return "pre_54"
    if venue in POST_AUG_2025_VENUES:
        return "post_54"
    return "unknown_2025"


def main() -> None:
    if not GPT54_PREDS.exists():
        print(f"MISSING: {GPT54_PREDS}", file=sys.stderr)
        sys.exit(1)

    entries = load_entries(SUPPLEMENT)
    gpt51 = {p.bibtex_key: p for p in load_predictions(GPT51_PREDS)}
    gpt54 = {p.bibtex_key: p for p in load_predictions(GPT54_PREDS)}

    # Bucket entries by stratum.
    buckets: dict[str, list] = {"pre_54": [], "post_54": [], "unknown": [], "unknown_2025": []}
    for e in entries:
        y = e.fields.get("year", "?")
        v = e.fields.get("booktitle") or e.fields.get("journal") or "?"
        buckets[stratum(y, v)].append(e)

    print("=== Stratum sizes ===")
    for k, v in buckets.items():
        n_val = sum(1 for e in v if e.label == "VALID")
        n_hal = sum(1 for e in v if e.label == "HALLUCINATED")
        print(f"  {k:15s} n={len(v):4d}  VALID={n_val}  HAL={n_hal}")

    print("\n=== FPR and UNCERTAIN rate by model x stratum ===")
    print(
        f"  {'stratum':15s}  {'n':>4s}  "
        f"{'GPT-5.1 FPR':>12s}  {'GPT-5.4 FPR':>12s}  "
        f"{'ΔFPR':>7s}  {'5.1 UNC':>8s}  {'5.4 UNC':>8s}"
    )

    summary_rows = []
    for name, bucket in [("pre_54", buckets["pre_54"]), ("post_54", buckets["post_54"])]:
        if not bucket:
            continue
        bucket_keys = {e.bibtex_key for e in bucket}

        p51 = [gpt51[k] for k in bucket_keys if k in gpt51]
        p54 = [gpt54[k] for k in bucket_keys if k in gpt54]
        r51 = evaluate(bucket, p51, tool_name="gpt51", split_name=f"{name}")
        r54 = evaluate(bucket, p54, tool_name="gpt54", split_name=f"{name}")
        d51 = r51.to_dict() if hasattr(r51, "to_dict") else r51.__dict__
        d54 = r54.to_dict() if hasattr(r54, "to_dict") else r54.__dict__

        unc_51 = d51.get("num_uncertain", 0) / max(d51.get("num_entries", 1), 1)
        unc_54 = d54.get("num_uncertain", 0) / max(d54.get("num_entries", 1), 1)
        fpr_51_raw = d51.get("false_positive_rate")
        fpr_54_raw = d54.get("false_positive_rate")
        fpr_51 = (fpr_51_raw or 0.0) * 100
        fpr_54 = (fpr_54_raw or 0.0) * 100

        fpr_note = (
            " (FPR n/a: 0 valid entries)" if fpr_51_raw is None and fpr_54_raw is None else ""
        )
        print(
            f"  {name:15s}  {len(bucket):4d}  "
            f"{fpr_51:12.1f}  {fpr_54:12.1f}  "
            f"{fpr_54 - fpr_51:+7.1f}  {unc_51 * 100:7.1f}%  {unc_54 * 100:7.1f}%{fpr_note}"
        )
        summary_rows.append(
            {
                "stratum": name,
                "n": len(bucket),
                "gpt51_fpr": fpr_51_raw,
                "gpt54_fpr": fpr_54_raw,
                "gpt51_dr": d51.get("detection_rate"),
                "gpt54_dr": d54.get("detection_rate"),
                "gpt51_unc_rate": unc_51,
                "gpt54_unc_rate": unc_54,
            }
        )

    # Aggregate whole-supplement numbers for context.
    r51_all = evaluate(entries, list(gpt51.values()), tool_name="gpt51", split_name="all")
    r54_all = evaluate(entries, list(gpt54.values()), tool_name="gpt54", split_name="all")
    d51_all = r51_all.to_dict() if hasattr(r51_all, "to_dict") else r51_all.__dict__
    d54_all = r54_all.to_dict() if hasattr(r54_all, "to_dict") else r54_all.__dict__
    print(
        f"\n  {'ALL':15s}  {len(entries):4d}  "
        f"{d51_all['false_positive_rate'] * 100:12.1f}  "
        f"{d54_all['false_positive_rate'] * 100:12.1f}  "
        f"{(d54_all['false_positive_rate'] - d51_all['false_positive_rate']) * 100:+7.1f}  "
        f"{d51_all.get('num_uncertain', 0) / max(d51_all.get('num_entries', 1), 1) * 100:7.1f}%  "
        f"{d54_all.get('num_uncertain', 0) / max(d54_all.get('num_entries', 1), 1) * 100:7.1f}%"
    )

    out = Path("figures/gpt54_probe_results.json")
    out.write_text(
        json.dumps(
            {
                "cutoff_gpt51": "2024-09-01",
                "cutoff_gpt54": "2025-08-31",
                "supplement": str(SUPPLEMENT),
                "supplement_size": len(entries),
                "strata": summary_rows,
                "aggregate": {
                    "gpt51_fpr": d51_all["false_positive_rate"],
                    "gpt54_fpr": d54_all["false_positive_rate"],
                    "gpt51_dr": d51_all["detection_rate"],
                    "gpt54_dr": d54_all["detection_rate"],
                    "gpt51_unc": d51_all.get("num_uncertain"),
                    "gpt54_unc": d54_all.get("num_uncertain"),
                    "n": len(entries),
                },
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
