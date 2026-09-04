"""Regressions for four metric functions that were silently wrong.

Each test here pins a property that the full suite passed without: the bugs were
in the direction that flatters results, so nothing downstream complained. The
docstrings say what the broken behaviour was, because the fix is only obvious
once you know what it replaced.
"""

from __future__ import annotations

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.metrics import (
    _metric_f1,
    evaluate,
    paired_bootstrap_test,
    per_tier_metrics,
    per_tier_rankings,
    per_type_metrics,
)
from hallmark.evaluation.ranking_stability import ranking_sensitivity_analysis

# --- Helpers ---


def _entry(key: str, label: str, tier: int | None = None, h_type: str | None = None):
    kwargs: dict = {
        "bibtex_key": key,
        "bibtex_type": "article",
        "fields": {"title": f"Paper {key}", "author": "Author", "year": "2024"},
        "label": label,
        "explanation": "test",
    }
    if label == "HALLUCINATED":
        kwargs["hallucination_type"] = h_type or "fabricated_doi"
        kwargs["difficulty_tier"] = tier or 1
    return BenchmarkEntry(**kwargs)


def _pred(key: str, label: str, confidence: float = 0.9):
    return Prediction(bibtex_key=key, label=label, confidence=confidence)


def _split(n_hall: int, n_valid: int, tier: int = 1, h_type: str = "fabricated_doi"):
    entries = [_entry(f"h{i}", "HALLUCINATED", tier, h_type) for i in range(n_hall)]
    entries += [_entry(f"v{i}", "VALID") for i in range(n_valid)]
    return entries


# --- 1. Two-sided p-value must be direction-symmetric ---


def test_two_sided_p_value_is_symmetric_under_argument_order():
    """The p-value must not depend on which tool is named first.

    The old form was ``min(1, 2 * P(diff <= 0))``. When A is worse than B every
    bootstrap difference is <= 0, so it returned 1.0 however large the gap.
    Callers enumerate pairs in sorted-name order, so about half of every
    pairwise leaderboard comparison was non-significant by construction.
    """
    entries = _split(n_hall=100, n_valid=100)
    # ``strong`` catches every hallucination; ``weak`` calls everything VALID,
    # so its F1 on the HALLUCINATED class is 0.
    strong = [_pred(e.bibtex_key, e.label) for e in entries]
    weak = [_pred(e.bibtex_key, "VALID") for e in entries]

    diff_ab, p_ab, _ = paired_bootstrap_test(
        entries, strong, weak, _metric_f1, n_bootstrap=300, seed=0
    )
    diff_ba, p_ba, _ = paired_bootstrap_test(
        entries, weak, strong, _metric_f1, n_bootstrap=300, seed=0
    )

    assert diff_ab > 0 > diff_ba, "the two orderings must report opposite signs"
    assert p_ab == p_ba, f"p-value is order-dependent: {p_ab} vs {p_ba}"
    assert p_ba < 0.05, (
        f"a tool separated by a large margin must be significant when named second, not p={p_ba}"
    )


def test_two_sided_p_value_is_high_for_identical_tools():
    """Sanity guard on the other side: identical predictions are not significant."""
    entries = _split(n_hall=60, n_valid=60)
    preds = [_pred(e.bibtex_key, e.label) for e in entries]
    _, p, _ = paired_bootstrap_test(
        entries, preds, list(preds), _metric_f1, n_bootstrap=300, seed=0
    )
    assert p > 0.05


# --- 2. Per-type metrics need a real FPR denominator ---


def test_per_type_fpr_is_not_structurally_zero():
    """Per-type FPR must reflect real false positives.

    Grouping strictly by ``hallucination_type`` puts only hallucinated entries
    in a type's group, forcing precision to 1.0, FPR to 0.0 and F1 to the
    deterministic ``2*DR/(1+DR)``. ``per_tier_metrics`` was fixed for exactly
    this; ``per_type_metrics`` was not.
    """
    entries = _split(n_hall=10, n_valid=10, h_type="chimeric_title")
    # Detect every hallucination AND flag every valid entry: FPR must be 1.0.
    preds = {e.bibtex_key: _pred(e.bibtex_key, "HALLUCINATED") for e in entries}

    pt = per_type_metrics(entries, preds)
    row = pt["chimeric_title"]

    assert row["detection_rate"] == 1.0
    assert row["false_positive_rate"] == 1.0, (
        "every valid entry was flagged, so per-type FPR cannot be 0.0"
    )
    assert row["precision"] == 0.5
    assert row["count"] == 10, "count must stay the hallucinated count for the type"
    assert row["num_valid"] == 10


def test_per_type_f1_is_not_a_function_of_detection_rate_alone():
    """F1 must move when false positives change and DR does not."""
    entries = _split(n_hall=10, n_valid=10, h_type="near_miss_title")

    clean = {e.bibtex_key: _pred(e.bibtex_key, e.label) for e in entries}
    noisy = {e.bibtex_key: _pred(e.bibtex_key, "HALLUCINATED") for e in entries}

    f1_clean = per_type_metrics(entries, clean)["near_miss_title"]["f1"]
    f1_noisy = per_type_metrics(entries, noisy)["near_miss_title"]["f1"]

    assert per_type_metrics(entries, clean)["near_miss_title"]["detection_rate"] == 1.0
    assert per_type_metrics(entries, noisy)["near_miss_title"]["detection_rate"] == 1.0
    assert f1_clean > f1_noisy, "identical DR, more FPs, F1 must drop"


def test_per_type_valid_group_keeps_its_own_semantics():
    """The ``valid`` pseudo-type must not have the valid entries added twice."""
    entries = _split(n_hall=4, n_valid=6)
    preds = {e.bibtex_key: _pred(e.bibtex_key, e.label) for e in entries}
    row = per_type_metrics(entries, preds)["valid"]
    assert row["count"] == 6
    assert row["num_valid"] == 0


# --- 3. The tier-weight sweep must see false positives ---


def test_tier_weight_sweep_distinguishes_tools_differing_only_in_false_positives():
    """The sweep read ``tm["fpr"]`` while per_tier_metrics emits
    ``false_positive_rate``, so the default fired every time, total_fp was
    always 0, and the swept quantity was weighted recall. Two tools identical
    except for their false positives came out perfectly concordant.
    """
    entries = _split(n_hall=30, n_valid=30)

    # Both detect everything; only ``sloppy`` also flags every valid entry.
    careful = [_pred(e.bibtex_key, e.label) for e in entries]
    sloppy = [_pred(e.bibtex_key, "HALLUCINATED") for e in entries]

    out = ranking_sensitivity_analysis(
        entries, {"careful": careful, "sloppy": sloppy}, n_samples=25, seed=0
    )

    ranges = out.per_tool_range
    careful_min = min(ranges["careful"])
    sloppy_max = max(ranges["sloppy"])
    assert careful_min > sloppy_max, (
        "a tool that flags every valid entry must score below one that does not; "
        f"got careful={ranges['careful']} sloppy={ranges['sloppy']}"
    )


def test_per_tier_metrics_emits_false_positive_rate_not_fpr():
    """Pins the key name the sweep and the rankings depend on."""
    entries = _split(n_hall=5, n_valid=5, tier=2)
    preds = {e.bibtex_key: _pred(e.bibtex_key, e.label) for e in entries}
    tier_data = per_tier_metrics(entries, preds)[2]
    assert "false_positive_rate" in tier_data
    assert "fpr" not in tier_data


def test_per_tier_rankings_accepts_fpr_alias_and_orders_ascending():
    """``fpr`` is documented as a valid argument, so it must not silently
    return 0.0 for every tool — and for FPR, best-first means lowest."""
    entries = _split(n_hall=10, n_valid=10, tier=1)
    careful = [_pred(e.bibtex_key, e.label) for e in entries]
    sloppy = [_pred(e.bibtex_key, "HALLUCINATED") for e in entries]

    ranked = per_tier_rankings(entries, {"careful": careful, "sloppy": sloppy}, metric="fpr")[1]

    assert dict(ranked)["sloppy"] == 1.0, "alias must resolve to false_positive_rate"
    assert ranked[0][0] == "careful", "lowest FPR ranks first"


# --- 4. Abstention must cost coverage ---


def test_coverage_excludes_uncertain_predictions():
    """UNCERTAIN is dropped from the confusion matrix, ECE, AUROC and AUPRC, so
    counting it as covered let a tool report its easy-subset metrics at full
    coverage. One committed run answered 68 of 500 entries at coverage 1.0.
    """
    entries = _split(n_hall=10, n_valid=10)
    preds = []
    for i, e in enumerate(entries):
        # Answer only the first five of each class; abstain on the rest.
        label = e.label if i % 10 < 5 else "UNCERTAIN"
        preds.append(_pred(e.bibtex_key, label))

    result = evaluate(entries, preds)

    assert result.num_uncertain == 10
    assert result.coverage == 0.5, f"answered 10 of 20 entries, got {result.coverage}"
    assert result.response_coverage == 1.0, "a record was returned for every entry"
    assert result.coverage_adjusted_f1 < result.f1_hallucination


def test_full_abstention_scores_zero_coverage_not_perfect_coverage():
    entries = _split(n_hall=5, n_valid=5)
    preds = [_pred(e.bibtex_key, "UNCERTAIN") for e in entries]

    result = evaluate(entries, preds)

    assert result.coverage == 0.0
    assert result.response_coverage == 1.0
    assert result.coverage_adjusted_f1 == 0.0


def test_strict_mode_still_keys_on_missing_predictions_not_abstention():
    """Strict mode checks that the tool responded at all. An abstention is a
    different failure from a missing prediction and must not trip it.
    """
    entries = _split(n_hall=5, n_valid=5)
    preds = [_pred(e.bibtex_key, "UNCERTAIN") for e in entries]

    result = evaluate(entries, preds, strict=True)  # must not raise
    assert result.coverage == 0.0
