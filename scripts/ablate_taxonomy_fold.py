"""What the ranking does if miscitation stops counting as fabrication.  [evaluation]

Issue #36: four of the fourteen hallucination modes are conditions on papers
that exist -- ``wrong_venue``, ``preprint_as_published``, ``partial_author_list``
and ``arxiv_version_mismatch``, each defined in the prompt as "real paper
but ..." -- and they sit under a label the same prompt equates with fabrication.
They are 26.1% of the positives on ``dev_public`` and 26.8% on ``test_public``.

Before anyone relabels a split, the question worth answering is whether the
taxonomy is doing work the detectors should be doing. This scores every tool
three ways and compares the rankings:

``as_shipped``
    All fourteen modes are positives. What the leaderboard reports today.

``folded_out``
    The four real-paper modes leave the scored set entirely. A fabrication
    benchmark scoring fabrication: the entries are neither fabrications to find
    nor valid citations to leave alone, so they are a different task.

``as_false_positives``
    The four modes become negatives, so flagging one is a false accusation.
    This is what a user experiences who reads HALLUCINATED as "fabricated" --
    the failure the issue is about -- and it is the pessimistic bound.

Reconstructed from ``per_type_metrics`` in the released result JSONs rather than
from per-entry predictions, which do not exist for every tool. The
reconstruction is validated against each result's own reported figures before
anything is reported, and a tool whose reconstruction does not reproduce its
published MCC is dropped with a reason rather than quietly scored.

    python scripts/ablate_taxonomy_fold.py --split test_public

Writes ``tables/taxonomy_fold_ablation.csv``. Changes no labels and no results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from hallmark.evaluation.table_provenance import record_table

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_ROOT / "tables"

#: The modes the issue is about. Each is defined in the agentic system prompt as
#: a condition on a paper that exists.
REAL_PAPER_MODES = (
    "wrong_venue",
    "preprint_as_published",
    "partial_author_list",
    "arxiv_version_mismatch",
)

#: How far a reconstructed figure may sit from the published one before the tool
#: is dropped. per_type_metrics stores rates, so counts come back through a
#: rounding step and cannot be expected to match exactly.
_TOLERANCE = 0.02


@dataclass
class Confusion:
    tp: float
    fp: float
    fn: float
    tn: float

    @property
    def detection_rate(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def false_positive_rate(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denom if denom else 0.0

    @property
    def mcc(self) -> float:
        num = self.tp * self.tn - self.fp * self.fn
        den = math.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
        )
        return num / den if den else 0.0


def _confusions(payload: dict) -> dict[str, Confusion] | None:
    """Rebuild the three confusion matrices from a result's per-type rates."""
    per_type = payload.get("per_type_metrics")
    num_valid = payload.get("num_valid")
    fpr = payload.get("false_positive_rate")
    if not isinstance(per_type, dict) or num_valid is None or fpr is None:
        return None

    fp = round(float(fpr) * float(num_valid))
    tn = float(num_valid) - fp

    kept = {"tp": 0.0, "fn": 0.0}
    real = {"tp": 0.0, "fn": 0.0}
    for mode, m in per_type.items():
        if mode == "valid" or not isinstance(m, dict):
            continue
        count = float(m.get("count") or 0.0)
        detected = float(m.get("detection_rate") or 0.0) * count
        bucket = real if mode in REAL_PAPER_MODES else kept
        bucket["tp"] += detected
        bucket["fn"] += count - detected

    detection_on_real_modes = (
        real["tp"] / (real["tp"] + real["fn"]) if (real["tp"] + real["fn"]) else 0.0
    )
    return {
        "_real_mode_dr": detection_on_real_modes,
        "as_shipped": Confusion(
            tp=kept["tp"] + real["tp"], fp=fp, fn=kept["fn"] + real["fn"], tn=tn
        ),
        "folded_out": Confusion(tp=kept["tp"], fp=fp, fn=kept["fn"], tn=tn),
        # A flag on a real-paper mode becomes a false accusation; one left alone
        # becomes a correct rejection.
        "as_false_positives": Confusion(
            tp=kept["tp"], fp=fp + real["tp"], fn=kept["fn"], tn=tn + real["fn"]
        ),
    }


def _reconstruction_error(payload: dict, shipped: Confusion) -> list[str]:
    """Where the rebuilt matrix disagrees with the result's own published figures."""
    problems = []
    for field, rebuilt in (
        ("detection_rate", shipped.detection_rate),
        ("false_positive_rate", shipped.false_positive_rate),
        ("mcc", shipped.mcc),
        ("f1_hallucination", shipped.f1),
    ):
        published = payload.get(field)
        if published is None:
            continue
        if abs(float(published) - rebuilt) > _TOLERANCE:
            problems.append(f"{field} rebuilt {rebuilt:.4f} vs published {float(published):.4f}")
    return problems


def _kendall_tau(a: list[str], b: list[str]) -> float:
    """Rank correlation between two orderings of the same tools."""
    pos_a = {name: i for i, name in enumerate(a)}
    pos_b = {name: i for i, name in enumerate(b)}
    names = [n for n in a if n in pos_b]
    concordant = discordant = 0
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            sign = (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test_public")
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "data" / "v1.2" / "baseline_results"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = TABLES_DIR / f"taxonomy_fold_ablation_{args.split}.csv"

    scored: dict[str, dict[str, Confusion]] = {}
    real_mode_drs: dict[str, float] = {}
    dropped: dict[str, str] = {}
    used_inputs: list[Path] = []

    for path in sorted(args.results_dir.glob(f"*_{args.split}.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            dropped[path.stem] = f"unreadable ({exc})"
            continue
        if not isinstance(payload, dict):
            dropped[path.stem] = "unexpected JSON shape"
            continue
        rebuilt = _confusions(payload)
        if rebuilt is None:
            dropped[path.stem] = "no per_type_metrics to reconstruct from"
            continue
        real_mode_dr = float(rebuilt.pop("_real_mode_dr"))
        matrices = {k: v for k, v in rebuilt.items() if isinstance(v, Confusion)}
        real_mode_drs[path.stem] = real_mode_dr
        problems = _reconstruction_error(payload, matrices["as_shipped"])
        if problems:
            dropped[path.stem] = (
                "reconstruction does not reproduce published figures: " + "; ".join(problems)
            )
            continue
        scored[path.stem] = matrices
        used_inputs.append(path)

    if not scored:
        parser.error(f"no reconstructable results for split {args.split!r}")

    rows = []
    for tool, matrices in sorted(scored.items()):
        row = {
            "tool": tool,
            "split": args.split,
            # The mechanism behind any movement: a tool that never flags the
            # real-paper modes gains when they stop counting, and one that
            # flags them all loses.
            "real_mode_dr": f"{real_mode_drs[tool]:.4f}",
        }
        for scoring, cm in matrices.items():
            row[f"{scoring}_dr"] = f"{cm.detection_rate:.4f}"
            row[f"{scoring}_fpr"] = f"{cm.false_positive_rate:.4f}"
            row[f"{scoring}_f1"] = f"{cm.f1:.4f}"
            row[f"{scoring}_mcc"] = f"{cm.mcc:.4f}"
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    record_table(
        args.output,
        used_inputs,
        generator="scripts/ablate_taxonomy_fold.py",
        repo_root=REPO_ROOT,
    )

    order = {
        scoring: [t for t, _ in sorted(scored.items(), key=lambda kv: -kv[1][scoring].mcc)]
        for scoring in ("as_shipped", "folded_out", "as_false_positives")
    }

    print(f"Taxonomy fold ablation, {args.split}: {len(scored)} tools reconstructed")
    if dropped:
        print(f"{len(dropped)} dropped:")
        for name, why in sorted(dropped.items()):
            print(f"  - {name}: {why}")

    print(
        f"\n{'tool':<44} {'MCC shipped':>11} {'folded':>9} {'as FP':>9}  "
        f"{'rank move':>9}  {'DR on the 4':>11}"
    )
    for tool in order["as_shipped"]:
        m = scored[tool]
        move = order["as_shipped"].index(tool) - order["folded_out"].index(tool)
        print(
            f"{tool[:44]:<44} {m['as_shipped'].mcc:>11.4f} {m['folded_out'].mcc:>9.4f} "
            f"{m['as_false_positives'].mcc:>9.4f}  {move:>+9d}  {real_mode_drs[tool]:>11.4f}"
        )

    print("\nRanking agreement with as_shipped (Kendall tau over MCC):")
    for scoring in ("folded_out", "as_false_positives"):
        tau = _kendall_tau(order["as_shipped"], order[scoring])
        moved = sum(
            1
            for t in order["as_shipped"]
            if order["as_shipped"].index(t) != order[scoring].index(t)
        )
        print(f"  {scoring:<20} tau={tau:+.4f}  tools changing position: {moved}/{len(scored)}")
        print(f"    top 5: {', '.join(order[scoring][:5])}")

    print(f"\nWrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
