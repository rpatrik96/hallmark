"""A released result must not claim full coverage on a run full of abstentions.

``bibtexupdater_dev_public.json`` shipped ``coverage: 1.0`` and
``num_uncertain: 0`` for a run whose raw output carries 147 ``unconfirmed``
records; ``test_public`` shipped the same pair against 101. The paper's Coverage
column said otherwise, so the two artifacts disagreed and the wrong one was the
one a reader can recompute.

The DR/FPR/F1 triple is deliberately *not* covered here. Those still score
abstentions as committed-VALID, which is the convention the paper reports, and
re-scoring them selectively would make ``bibtex-updater`` the one tool in the
cohort scored on a different basis from the other twenty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "data/v1.2/baseline_results"

#: Splits whose released result was corrected, and the abstention count in the
#: raw output behind it. Pinned rather than recomputed: a test that re-derives
#: its own expectation from the same file cannot catch the file changing.
EXPECTED_ABSTENTIONS = {"dev_public": 147, "test_public": 101}


@pytest.mark.parametrize("split", sorted(EXPECTED_ABSTENTIONS))
def test_abstentions_are_recorded(split: str):
    path = RESULTS / f"bibtexupdater_{split}.json"
    if not path.is_file():
        pytest.skip(f"{path.name} not present")
    result = json.loads(path.read_text())
    assert result["num_uncertain"] == EXPECTED_ABSTENTIONS[split], (
        f"{path.name} records {result['num_uncertain']} abstentions; the raw output "
        f"behind it has {EXPECTED_ABSTENTIONS[split]}. Zero is what the pre-fix wrapper "
        "wrote, by mapping every abstention to a committed VALID."
    )


@pytest.mark.parametrize("split", sorted(EXPECTED_ABSTENTIONS))
def test_coverage_agrees_with_the_abstention_count(split: str):
    """Coverage and ``num_uncertain`` must tell the same story.

    They disagreed by construction before: coverage came from the label
    distribution, which no longer held any UNCERTAIN.
    """
    path = RESULTS / f"bibtexupdater_{split}.json"
    if not path.is_file():
        pytest.skip(f"{path.name} not present")
    result = json.loads(path.read_text())
    n = result["num_entries"]
    implied = 1.0 - result["num_uncertain"] / n
    assert result["coverage"] < 1.0, (
        f"{path.name} claims full coverage while recording {result['num_uncertain']} abstentions"
    )
    # Missing records lower coverage further, so the recorded value is at most
    # the abstention-implied one; a gap wider than 1pp means something else is
    # unanswered and should be explained rather than absorbed.
    assert implied - result["coverage"] <= 0.01, (
        f"{path.name}: coverage {result['coverage']} against {implied:.4f} implied by "
        f"{result['num_uncertain']} abstentions over {n} entries"
    )


def test_the_triple_still_uses_the_committed_convention():
    """Pins what this correction deliberately did not touch.

    If these move, the released result has been re-scored selectively and no
    longer sits on the same basis as the rest of the cohort.
    """
    path = RESULTS / "bibtexupdater_dev_public.json"
    if not path.is_file():
        pytest.skip("released result not present")
    result = json.loads(path.read_text())
    assert result["detection_rate"] == pytest.approx(0.8647, abs=5e-4)
    assert result["false_positive_rate"] == pytest.approx(0.0916, abs=5e-4)
    assert result["f1_hallucination"] == pytest.approx(0.8904, abs=5e-4)


@pytest.mark.parametrize("split", sorted(EXPECTED_ABSTENTIONS))
def test_coverage_adjusted_f1_matches_its_definition(split: str):
    """``coverage_adjusted_f1`` is ``f1_hallucination * coverage`` (metrics.py).

    The coverage correction rewrote ``coverage`` and left this field at the old
    ``f1 * 1.0``, so the file contradicted its own definition and the leaderboard
    showed bibtex-updater un-penalised on the one column whose purpose is to
    penalise abstention. Under the committed-VALID triple this double-counts
    abstention; that is documented in the provenance string, not hidden in the
    number.
    """
    path = RESULTS / f"bibtexupdater_{split}.json"
    if not path.is_file():
        pytest.skip(f"{path.name} not present")
    result = json.loads(path.read_text())
    expected = result["f1_hallucination"] * result["coverage"]
    assert result["coverage_adjusted_f1"] == pytest.approx(expected, abs=5e-4), (
        f"{path.name}: coverage_adjusted_f1 {result['coverage_adjusted_f1']:.4f} but "
        f"f1 {result['f1_hallucination']:.4f} x coverage {result['coverage']} = {expected:.4f}"
    )
