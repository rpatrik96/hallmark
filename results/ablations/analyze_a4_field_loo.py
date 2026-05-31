"""Aggregate + render the A4 field-LOO results across models.

Reads results/ablations/a4_field_loo/summary.json (written by the runner) and
prints two tables for the paper supplement:

  1. Format sensitivity: full-BibTeX vs structured-field input — DR/FPR delta
     per model (structured minus full).
  2. Field leave-one-out: per dropped field {title,authors,venue,year,doi},
     the DR/FPR delta vs the structured baseline, per model. n_drop_affected
     flags how many of the 150 entries actually carried the dropped field
     (doi coverage is partial, so its delta is diluted).

This is read-only reporting; it neither calls the API nor mutates checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "a4_field_loo" / "summary.json"

LOO_ORDER = ["loo_title", "loo_author", "loo_venue", "loo_year", "loo_doi"]


def _fmt(x: float | None) -> str:
    return f"{x:+.3f}" if x is not None else "  NA  "


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    models = list(data["per_model"])
    print(f"A4 INPUT-FORMAT / FIELD LEAVE-ONE-OUT  ({data['snapshot_date']}, {data['endpoint']})")
    s = data["sample"]
    print(f"sample: n={s['n']} ({s['n_hall']} HALL / {s['n_valid']} VALID), seed={s['seed']}")
    print(f"models: {', '.join(models)} | temp={data['temperature']}\n")

    print("=== (1) FORMAT: full vs structured (delta = structured - full) ===")
    print(f"{'model':18s} {'dDR':>8s} {'dFPR':>8s}   (full -> structured)")
    for mk in models:
        m = data["per_model"][mk]
        fd = m["format_delta_full_vs_structured"]
        full = m["summary"]["full"]
        struct = m["summary"]["structured"]
        print(
            f"{mk:18s} {_fmt(fd['dDR']):>8s} {_fmt(fd['dFPR']):>8s}   "
            f"DR {full['detection_rate']:.3f}->{struct['detection_rate']:.3f}  "
            f"FPR {(full['false_positive_rate'] or 0):.3f}->{(struct['false_positive_rate'] or 0):.3f}"
        )

    print("\n=== (2) FIELD LEAVE-ONE-OUT (delta vs structured baseline) ===")
    for mk in models:
        m = data["per_model"][mk]
        base = m["summary"]["structured"]
        print(
            f"\n  [{mk}]  structured baseline: "
            f"DR={base['detection_rate']:.3f} FPR={(base['false_positive_rate'] or 0):.3f}"
        )
        print(f"  {'dropped':10s} {'aff':>5s} {'dDR':>8s} {'dFPR':>8s} {'-> FPR':>9s}")
        for cond in LOO_ORDER:
            d = m["loo_deltas_vs_structured"][cond]
            cell = m["summary"][cond]
            print(
                f"  {d['dropped_field']:10s} {d.get('n_drop_affected') or '-'!s:>5} "
                f"{_fmt(d['dDR_vs_structured']):>8s} {_fmt(d['dFPR_vs_structured']):>8s} "
                f"{(cell['false_positive_rate'] or 0):>9.3f}"
            )


if __name__ == "__main__":
    main()
