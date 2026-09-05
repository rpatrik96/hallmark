#!/usr/bin/env python3
"""Correct ``coverage`` and ``num_uncertain`` on a released ``bibtex-updater`` result.

The released aggregates report ``coverage: 1.0`` and ``num_uncertain: 0`` for runs
whose raw output carries 147 (``dev_public``) and 101 (``test_public``)
abstentions. The wrapper wrote every abstention as a committed VALID, so the
field that was supposed to record them recorded nothing. The paper's Coverage
column says otherwise, and the artifact a reader can recompute was the wrong one.

**Only those two fields change.** The DR/FPR/F1 triple stays exactly as released,
which is the paper's documented convention: abstentions score as committed-VALID
in the triple, and Coverage reports the fraction the tool committed to. Two
reasons not to re-score the triple here:

1. It would not be comparable. Under the conservative protocol abstentions leave
   the confusion matrix, so a re-scored ``bibtex-updater`` would be the one tool
   in the cohort scored selectively while the other twenty are scored committed.
   Re-scoring lifted its ``test_public`` MCC from 0.750 to 0.918 and made it the
   leader under all three taxonomy-fold scorings -- an artifact of dropping the
   101 entries it abstained on, not a result.
2. It would not reproduce anyway. The released runs go through
   ``run_with_prescreening``, whose overrides are not in the raw JSONL, so a
   raw-only parse gives DR 0.820 / FPR 0.051 against the released 0.865 / 0.092.

**The raw file must be the one behind the aggregate.**
``data/v1.2/baseline_results/bibtexupdater_raw_dev_public.jsonl`` is *not* -- it
is a 0.10.0-era run with no ``unconfirmed`` records at all -- so this checks the
status histogram against the aggregate's own ``_btu_status_histogram`` and
refuses to write when they disagree.

    python scripts/rescore_btu_from_raw.py --split dev_public \
        --raw results/relabel_delta/btu_v1_2_0/dev_public/btu_raw.jsonl --apply
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def status_histogram(raw_path: Path) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for line in raw_path.read_text().splitlines():
        if line.strip():
            counts[json.loads(line).get("status", "")] += 1
    return dict(counts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)
    ap.add_argument("--raw", type=Path, required=True, help="raw bibtex-check JSONL")
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "data/v1.2/baseline_results")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--version", default="v1.2")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = ap.parse_args()

    from hallmark.baselines.bibtexupdater import _parse_jsonl_output
    from hallmark.dataset.loader import SPLIT_PATHS
    from hallmark.dataset.schema import load_entries

    target = args.results_dir / f"bibtexupdater_{args.split}.json"
    if not target.is_file():
        print(f"no released result at {target}", file=sys.stderr)
        return 2
    released = json.loads(target.read_text())

    observed = status_histogram(args.raw)
    recorded = released.get("_btu_status_histogram")
    if recorded and observed != recorded:
        print(
            f"{args.raw} does not match the aggregate it would annotate.\n"
            f"  raw file : {observed}\n"
            f"  aggregate: {recorded}\n"
            "Counting abstentions from a different run than the one published would "
            "put a coverage figure on a result it does not describe.",
            file=sys.stderr,
        )
        return 1

    split_file = args.data_dir / args.version / SPLIT_PATHS[args.split]
    entries = load_entries(split_file)
    predictions = _parse_jsonl_output(args.raw, 0.0, len(entries))
    abstentions = sum(1 for p in predictions if p.label == "UNCERTAIN")
    answered = sum(1 for p in predictions if p.label != "UNCERTAIN")
    coverage = answered / len(entries)

    print(f"{args.split}: {len(entries)} entries, {len(predictions)} records")
    print(f"  coverage        {released['coverage']}  ->  {coverage:.4f}")
    print(f"  num_uncertain   {released['num_uncertain']}  ->  {abstentions}")
    print("  DR/FPR/F1       unchanged (committed-VALID convention, as released)")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return 0

    released["coverage"] = round(coverage, 4)
    released["num_uncertain"] = abstentions
    released.setdefault("_provenance", {})["coverage_corrected"] = (
        f"coverage and num_uncertain recounted from {args.raw.as_posix()} under "
        "ABSTENTION_STATUSES; the DR/FPR/F1 triple is unchanged and still scores "
        "abstentions as committed-VALID, which is the convention the paper reports. "
        "See scripts/rescore_btu_from_raw.py."
    )
    target.write_text(json.dumps(released, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
