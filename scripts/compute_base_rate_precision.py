"""Precision as a function of assumed hallucination prevalence.  [evaluation]

HALLMARK's public splits are hallucination-heavy by construction: `test_public`
is 62.5% HALLUCINATED. Precision is the only metric a deployed user experiences
directly -- it is the share of flags handed to them that are real -- and it
depends on prevalence, while detection rate and false-positive rate do not. So a
tool's headline numbers can be honest and still say almost nothing about what
using it feels like on a real bibliography.

This script holds each tool's measured DR and FPR fixed and sweeps the assumed
prevalence, reporting

    precision(p) = DR*p / (DR*p + FPR*(1-p))

and its reciprocal, the number of flags a user reads per true finding. It adds
no new measurement: it is arithmetic over numbers already in the released
results, which is the point -- the table can be produced today and it addresses
the deployment question directly.

The anchor at the low end is not hypothetical. Running HALLMARK's own
`cascade_db_diagnosis` over 5,043 references from 267 real workshop submissions
produced zero fabricated works: every reference Stage 1 ruled on was located,
and no accusation surviving re-adjudication and hand audit was a work that does
not exist. See ``WILD_CORPUS_NOTE`` below.

Output: prints the sweep and saves tables/base_rate_precision.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hallmark.evaluation.table_provenance import record_table

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_ROOT / "tables"
OUTPUT_CSV = TABLES_DIR / "base_rate_precision.csv"

#: Prevalence points to report. 0.625 is `test_public`'s own base rate, included
#: so the benchmark's operating point is visible on the same axis as realistic
#: ones rather than being the only point anyone sees.
DEFAULT_PREVALENCES: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.20, 0.625)

#: Measured on 5,043 deduplicated references from 267 NeurIPS-workshop
#: submissions, using HALLMARK's own cascade with an agentic Stage 2. Stage 1
#: located a matching record for every reference it ruled on. After
#: re-adjudication and hand audit, 11 accusations stood and none was a work that
#: does not exist: 3 corrupt OpenAlex index records (correct DOI, correct author
#: list, wrong title), 3 real papers whose DOIs were never registered, 2 real
#: works in humanities venues the indexes do not cover, 2 unverifiable by
#: construction (author "Anonymous", under review), and 1 whose existence was
#: proved by the rebuttal naming it in its own title, read as absence.
#: Stage-1 figures are final at bibtex-updater 1.10.3 and will not be refreshed.
WILD_CORPUS_NOTE = (
    "wild corpus: 5,043 real references, zero fabricated works found; "
    "observed prevalence indistinguishable from zero"
)


def precision_at_prevalence(dr: float, fpr: float, prevalence: float) -> float:
    """Positive predictive value at an assumed prevalence.

    ``dr`` and ``fpr`` are properties of the detector and do not move with
    prevalence; precision does, which is the whole point of the table.
    """
    tp = dr * prevalence
    fp = fpr * (1.0 - prevalence)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def load_results(results_dir: Path) -> dict[str, dict[str, float]]:
    """Read DR and FPR out of every result JSON that reports both."""
    out: dict[str, dict[str, float]] = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        dr = data.get("detection_rate")
        fpr = data.get("false_positive_rate")
        # FPR is None when a split has no valid entries (stress_test); such a
        # run cannot support a precision claim at any prevalence, so skip it
        # rather than substituting zero and inventing perfect precision.
        if dr is None or fpr is None:
            continue
        out[path.stem] = {
            "detection_rate": float(dr),
            "false_positive_rate": float(fpr),
            "coverage": float(data.get("coverage") or 0.0),
            "num_uncertain": float(data.get("num_uncertain") or 0.0),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "data" / "v1.2" / "baseline_results",
        help="directory of reference result JSONs",
    )
    parser.add_argument(
        "--split",
        default="test_public",
        help="only report runs whose filename ends with this split",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        parser.error(f"results dir not found: {args.results_dir}")

    results = {
        name: vals
        for name, vals in load_results(args.results_dir).items()
        if name.endswith(args.split)
    }
    if not results:
        parser.error(f"no results for split {args.split!r} in {args.results_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, vals in sorted(results.items()):
        dr, fpr = vals["detection_rate"], vals["false_positive_rate"]
        for p in DEFAULT_PREVALENCES:
            prec = precision_at_prevalence(dr, fpr, p)
            rows.append(
                {
                    "tool": name,
                    "split": args.split,
                    "detection_rate": f"{dr:.4f}",
                    "false_positive_rate": f"{fpr:.4f}",
                    "prevalence": f"{p:.4f}",
                    "precision": f"{prec:.4f}",
                    "flags_per_true_finding": f"{(1.0 / prec):.1f}" if prec > 0 else "inf",
                }
            )

    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Record which runs this table was computed from, so the freshness guard can
    # tell later that it still describes them. Without it a re-run leaves the
    # table saying a tool flags seven times more than it does, and nothing says so.
    record_table(
        args.output,
        [args.results_dir / f"{name}.json" for name in sorted(results)],
        generator="scripts/compute_base_rate_precision.py",
        repo_root=REPO_ROOT,
    )

    header = "  ".join(f"{p * 100:>7.1f}%" for p in DEFAULT_PREVALENCES)
    print(f"Precision at assumed prevalence ({args.split})")
    print(f"# {WILD_CORPUS_NOTE}\n")
    print(f"{'tool':<44} {'DR':>6} {'FPR':>6}   {header}")
    for name, vals in sorted(results.items()):
        dr, fpr = vals["detection_rate"], vals["false_positive_rate"]
        cells = "  ".join(
            f"{precision_at_prevalence(dr, fpr, p) * 100:>7.1f}%" for p in DEFAULT_PREVALENCES
        )
        print(f"{name[:44]:<44} {dr:6.3f} {fpr:6.3f}   {cells}")

    print("\nFlags read per true finding, same runs\n")
    print(f"{'tool':<44} {'DR':>6} {'FPR':>6}   {header}")
    for name, vals in sorted(results.items()):
        dr, fpr = vals["detection_rate"], vals["false_positive_rate"]
        cells = []
        for p in DEFAULT_PREVALENCES:
            prec = precision_at_prevalence(dr, fpr, p)
            cells.append(f"{1.0 / prec:>8.1f}" if prec > 0 else f"{'inf':>8}")
        print(f"{name[:44]:<44} {dr:6.3f} {fpr:6.3f}   {'  '.join(cells)}")

    print(f"\nWrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
