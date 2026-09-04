"""A derived table must not outlive the results it was computed from.

``scripts/check_results_freshness.py`` verifies result JSONs against the split
revision they scored. Nothing checked the layer above: ``tables/`` holds CSVs
computed *from* those results, and a table regenerates only when someone
remembers to re-run its script.

That gap was live. ``tables/base_rate_precision.csv`` was committed carrying
``doi_only_test_public`` at DR 0.3873 / FPR 0.2788. The baseline was then re-run
against current labels with the transient-HTTP-status fix and now records 0.1908
/ 0.0417, so all six of its rows were wrong -- and the precision and
flags-per-true-finding columns derive from exactly those two numbers, which put
the headline "720 flags per true finding" out by more than a factor of three.
Caught by a peer session reading the committed table against the committed
result, which is precisely the comparison nothing automated was making.

The check is cheap because the table carries its inputs: every row records the
``detection_rate`` and ``false_positive_rate`` it was computed from, so they can
be compared to the result JSON of the same name.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TABLE = _REPO_ROOT / "tables" / "base_rate_precision.csv"
_RESULTS = _REPO_ROOT / "data" / "v1.2" / "baseline_results"

#: The table stores four decimal places.
_TOLERANCE = 5e-5


def _rows() -> list[dict[str, str]]:
    with _TABLE.open() as fh:
        return list(csv.DictReader(fh))


@pytest.mark.skipif(not _TABLE.is_file(), reason="base-rate table not generated")
def test_every_row_matches_the_result_it_was_computed_from():
    stale: list[str] = []
    checked = 0
    for row in _rows():
        result_path = _RESULTS / f"{row['tool']}.json"
        if not result_path.is_file():
            stale.append(f"{row['tool']}: no result JSON to check against")
            continue
        data = json.loads(result_path.read_text())
        for column, key in (
            ("detection_rate", "detection_rate"),
            ("false_positive_rate", "false_positive_rate"),
        ):
            current = data.get(key)
            if current is None:
                continue
            checked += 1
            if abs(float(row[column]) - float(current)) > _TOLERANCE:
                stale.append(
                    f"{row['tool']} {column}: table says {row[column]}, result says {current:.4f}"
                )
    assert checked > 0, "no rows were actually compared — the guard would pass vacuously"
    assert not stale, (
        "derived table is stale relative to its inputs:\n  "
        + "\n  ".join(sorted(set(stale)))
        + "\nRegenerate with `python scripts/compute_base_rate_precision.py`."
    )


@pytest.mark.skipif(not _TABLE.is_file(), reason="base-rate table not generated")
def test_precision_column_is_consistent_with_its_own_inputs():
    """Bayes' rule, recomputed. Catches an edit to one column but not the others."""
    bad: list[str] = []
    for row in _rows():
        dr = float(row["detection_rate"])
        fpr = float(row["false_positive_rate"])
        p = float(row["prevalence"])
        tp, fp = dr * p, fpr * (1 - p)
        expected = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if abs(float(row["precision"]) - expected) > 1e-3:
            bad.append(f"{row['tool']} @ {p}: precision {row['precision']} != {expected:.4f}")
    assert not bad, "precision column disagrees with its own DR/FPR:\n  " + "\n  ".join(bad)
