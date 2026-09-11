"""Recompute TW-F1 and per-tier DR after the v1.2.3 tier retype.

The retype moved four entries into Tier 3. Tool predictions on them did not
change -- only which tier bucket they are counted in. So the correction is a
bucket move, not a re-evaluation, and it can be applied directly to the
published aggregates in ``baseline_results/``.

For each retyped entry the model's answer comes from one of two places:

* **deduced** -- ``per_type_metrics`` shows the entry's old type at DR == 1.000
  with ``num_uncertain == 0``, so every entry of that type was flagged
* **measured** -- ``scripts/requery_retyped_entries.py`` asked the model

Method
------
``per_tier_metrics`` stores a rate and a total per tier, so ``DR x
num_hallucinated`` recovers the caught count. Weighting those by tier (1/2/3)
and adding the false-positive count reproduces the published TW-F1 exactly --
this script asserts that before trusting anything downstream.

Two details that are easy to get wrong:

* **TW-F1 is reported as a delta on the published value**, never as the
  reconstructed absolute. Where a model returned UNCERTAIN anywhere in the
  split, the reconstruction is biased: ``build_confusion_matrix`` drops
  abstentions while ``num_hallucinated`` still counts them, so ``DR x n``
  overstates catches. That bias is identical before and after the bucket move,
  so it cancels in the difference. Validated against full re-scores on three
  cells that have stored per-entry predictions: agreement to 5e-5.
* **UNCERTAIN is a third outcome, not a miss.** An abstained entry is excluded
  from the confusion matrix entirely, so it leaves its old tier's total without
  joining Tier 3's caught count -- and it must not be folded into "missed".

Usage
-----
    uv run python scripts/recompute_tier_metrics.py --split dev_public
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.requery_retyped_entries import (
    RETYPED,
    TOOL_TO_BASELINE,
    _deducible,
    _is_frontier,
)

TIER_WEIGHTS = {1: 1.0, 2: 2.0, 3: 3.0}
NEW_TIER = 3

# Tier each retyped entry sat in when the aggregates were scored.
OLD_TIER = {
    "ffe715a15b16": 1,
    "ec01d96455e0": 2,
    "ab9c13051a56": 1,
    "cad998c28243": 1,
}


def tw_f1(weighted_tp: float, weighted_fn: float, false_positives: float) -> float:
    """F1 with tier-weighted recall. FPs are always weight 1.0 (valid entries have no tier)."""
    denominator = 2 * weighted_tp + weighted_fn + false_positives
    return 2 * weighted_tp / denominator if denominator else 0.0


def buckets(per_tier: dict) -> dict[int, list[float]]:
    """Recover [caught, total] per tier from the stored rate and total."""
    return {
        tier: [
            per_tier[str(tier)]["detection_rate"] * per_tier[str(tier)]["num_hallucinated"],
            float(per_tier[str(tier)]["num_hallucinated"]),
        ]
        for tier in (1, 2, 3)
    }


def move(buckets_in: dict[int, list[float]], key: str, outcome: str) -> dict[int, list[float]]:
    """Move one retyped entry from its old tier into Tier 3.

    HALLUCINATED carries a catch with it; VALID moves only the total; UNCERTAIN
    was never in the confusion matrix, so it leaves the old total and adds
    nothing to Tier 3's catches -- but Tier 3's total still grows, because
    ``num_hallucinated`` counts abstained entries.
    """
    out = {tier: list(counts) for tier, counts in buckets_in.items()}
    old = OLD_TIER[key]
    out[old][1] -= 1
    out[NEW_TIER][1] += 1
    if outcome == "HALLUCINATED":
        out[old][0] -= 1
        out[NEW_TIER][0] += 1
    return out


def weighted(buckets_in: dict[int, list[float]]) -> tuple[float, float]:
    tp = sum(TIER_WEIGHTS[t] * c for t, (c, _n) in buckets_in.items())
    fn = sum(TIER_WEIGHTS[t] * (n - c) for t, (c, n) in buckets_in.items())
    return tp, fn


def load_requeries(path: Path, split: str) -> dict[tuple[str, str], str]:
    """Map (tool_name, bibtex_key) -> label. Last write wins, so reruns are safe."""
    if not path.is_file():
        return {}
    out: dict[tuple[str, str], str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("split") != split or "label" not in record:
            continue
        out[(record["tool_name"], record["bibtex_key"])] = record["label"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev_public", choices=sorted(RETYPED))
    parser.add_argument("--data-dir", type=Path, default=Path("data/v1.2"))
    parser.add_argument(
        "--requeries",
        type=Path,
        default=Path("results/reviewer_experiments/tier_retype_requeries.jsonl"),
    )
    parser.add_argument("--out", type=Path, help="write the corrected table as JSON")
    args = parser.parse_args()

    measured = load_requeries(args.requeries, args.split)
    rows, unresolved = [], []

    for path in sorted((args.data_dir / "baseline_results").glob("*.json")):
        if path.name == "manifest.json":
            continue
        aggregate = json.loads(path.read_text())
        tool = aggregate.get("tool_name", "")
        if not _is_frontier(tool) or aggregate.get("split_name") != args.split:
            continue
        if tool not in TOOL_TO_BASELINE or not aggregate.get("per_tier_metrics"):
            continue

        base = buckets(aggregate["per_tier_metrics"])
        tier1 = aggregate["per_tier_metrics"]["1"]
        false_positives = tier1["false_positive_rate"] * tier1["num_valid"]

        outcomes, sources, missing = {}, {}, []
        for key, old_type in RETYPED[args.split]:
            known, _reason = _deducible(aggregate, old_type)
            if known:
                outcomes[key], sources[key] = "HALLUCINATED", "deduced"
            elif (tool, key) in measured:
                outcomes[key], sources[key] = measured[(tool, key)], "measured"
            else:
                missing.append(key)
        if missing:
            unresolved.append((tool, missing))
            continue

        corrected = base
        for key, outcome in outcomes.items():
            corrected = move(corrected, key, outcome)

        tp_before, fn_before = weighted(base)
        tp_after, fn_after = weighted(corrected)
        published = aggregate["tier_weighted_f1"]
        delta = tw_f1(tp_after, fn_after, false_positives) - tw_f1(
            tp_before, fn_before, false_positives
        )

        # Sanity gate: the reconstruction must reproduce the published TW-F1.
        # A mismatch beyond the UNCERTAIN bias means the delta is not trustworthy.
        recon_error = abs(tw_f1(tp_before, fn_before, false_positives) - published)

        rows.append(
            {
                "tool_name": tool,
                "split": args.split,
                "twf1_published": published,
                "twf1_corrected": published + delta,
                "twf1_delta": delta,
                "reconstruction_error": recon_error,
                "num_uncertain": aggregate.get("num_uncertain"),
                "outcomes": outcomes,
                "sources": sources,
                "per_tier": {
                    str(t): {
                        "dr_published": base[t][0] / base[t][1],
                        "dr_corrected": corrected[t][0] / corrected[t][1],
                        "n_published": round(base[t][1]),
                        "n_corrected": round(corrected[t][1]),
                    }
                    for t in (1, 2, 3)
                },
            }
        )

    print(f"\n=== {args.split}: TW-F1 ===")
    print(
        f"{'model':32} {'published':>10} {'corrected':>10} {'delta':>9} {'recon err':>10}  answers"
    )
    for row in rows:
        answers = " ".join(
            f"{k[:6]}={row['outcomes'][k][:4]}({row['sources'][k][:4]})" for k in row["outcomes"]
        )
        print(
            f"{row['tool_name']:32} {row['twf1_published']:10.4f} {row['twf1_corrected']:10.4f} "
            f"{row['twf1_delta']:+9.4f} {row['reconstruction_error']:10.2e}  {answers}"
        )

    print(f"\n=== {args.split}: per-tier detection rate ===")
    print(f"{'model':32} " + " ".join(f"{'Tier ' + str(t):^26}" for t in (1, 2, 3)))
    for row in rows:
        cells = []
        for tier in ("1", "2", "3"):
            pt = row["per_tier"][tier]
            cells.append(
                f"{pt['dr_published']:.3f}->{pt['dr_corrected']:.3f} "
                f"(n{pt['n_published']}->{pt['n_corrected']})".ljust(26)
            )
        print(f"{row['tool_name']:32} " + " ".join(cells))

    if unresolved:
        print("\n=== unresolved (no deduction, no requery) ===")
        for tool, keys in unresolved:
            print(f"  {tool}: {', '.join(keys)}")

    abstained = [
        (r["tool_name"], k) for r in rows for k, v in r["outcomes"].items() if v == "UNCERTAIN"
    ]
    if abstained:
        print(
            "\n=== UNCERTAIN answers (excluded from the confusion matrix, not counted as misses) ==="
        )
        for tool, key in abstained:
            print(f"  {tool}: {key}")

    worst = max((r["reconstruction_error"] for r in rows), default=0.0)
    print(f"\ncells: {len(rows)}   worst reconstruction error: {worst:.2e}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
