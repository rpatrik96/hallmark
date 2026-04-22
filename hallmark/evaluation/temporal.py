"""Temporal segmentation analysis for HALLMARK (LiveCodeBench-inspired).

Enables contamination detection by comparing tool performance on entries
from different time periods (pre-cutoff vs post-cutoff).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.metrics import build_confusion_matrix


@dataclass
class TemporalSegment:
    """A named temporal segment with date boundaries."""

    name: str
    start: date
    end: date  # inclusive

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end


def default_segments() -> list[TemporalSegment]:
    """Return temporal segments computed relative to the current date.

    - historical: 2015-01-01 through end of (current_year - 2)
    - recent:     start of (current_year - 1) through end of current_year
    - future:     start of (current_year + 1) through end of (current_year + 5)

    Prefer this over the module-level DEFAULT_SEGMENTS constant, which is
    frozen at import time and becomes stale as years pass.
    """
    today = date.today()
    return [
        TemporalSegment("historical", date(2015, 1, 1), date(today.year - 2, 12, 31)),
        TemporalSegment("recent", date(today.year - 1, 1, 1), date(today.year, 12, 31)),
        TemporalSegment("future", date(today.year + 1, 1, 1), date(today.year + 5, 12, 31)),
    ]


@dataclass
class TemporalAnalysis:
    """Results of temporal segmentation analysis."""

    segment_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    contamination_score: float | None = None
    robustness_delta: float | None = None  # historical DR - future DR


def parse_date(date_str: str) -> date | None:
    """Parse ISO date string to date object."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        try:
            return datetime.strptime(date_str[:4], "%Y").date()
        except ValueError:
            return None


def segment_entries(
    entries: list[BenchmarkEntry],
    segments: list[TemporalSegment] | None = None,
) -> dict[str, list[BenchmarkEntry]]:
    """Split entries into temporal segments based on publication_date."""
    if segments is None:
        segments = default_segments()

    result: dict[str, list[BenchmarkEntry]] = {s.name: [] for s in segments}
    result["unknown"] = []

    for entry in entries:
        d = parse_date(entry.publication_date)
        if d is None:
            result["unknown"].append(entry)
            continue

        placed = False
        for segment in segments:
            if segment.contains(d):
                result[segment.name].append(entry)
                placed = True
                break
        if not placed:
            result["unknown"].append(entry)

    return result


def temporal_analysis(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
    segments: list[TemporalSegment] | None = None,
) -> TemporalAnalysis:
    """Analyze tool performance across temporal segments.

    Contamination detection: if accuracy on historical entries >> future entries,
    the tool may rely on memorization rather than verification logic.
    """
    segmented = segment_entries(entries, segments)
    segment_metrics: dict[str, dict[str, float]] = {}

    for seg_name, seg_entries in segmented.items():
        if not seg_entries:
            continue

        cm = build_confusion_matrix(seg_entries, predictions)
        num_hall = sum(1 for e in seg_entries if e.label == "HALLUCINATED")
        num_valid = sum(1 for e in seg_entries if e.label == "VALID")

        segment_metrics[seg_name] = {
            "detection_rate": cm.detection_rate,
            "false_positive_rate": cm.false_positive_rate,
            "f1": cm.f1,
            "num_entries": len(seg_entries),
            "num_hallucinated": num_hall,
            "num_valid": num_valid,
        }

    # Compute contamination score
    hist = segment_metrics.get("historical", {})
    future = segment_metrics.get("future", {})

    contamination_score = None
    robustness_delta = None

    if hist and future:
        hist_dr = hist.get("detection_rate", 0.0)
        future_dr = future.get("detection_rate", 0.0)
        robustness_delta = hist_dr - future_dr

        # Contamination score: large positive delta suggests memorization
        # Normalized to [0, 1] range
        contamination_score = max(0.0, robustness_delta)

    return TemporalAnalysis(
        segment_metrics=segment_metrics,
        contamination_score=contamination_score,
        robustness_delta=robustness_delta,
    )


# ---------------------------------------------------------------------------
# Cutoff-aware prompt ablation: compare default vs cutoff-aware variants
# ---------------------------------------------------------------------------


def _split_pre_post_cutoff(
    entries: list[BenchmarkEntry],
    cutoff: date,
) -> tuple[list[BenchmarkEntry], list[BenchmarkEntry]]:
    """Split entries into pre-cutoff and post-cutoff lists using publication_date.

    Entries whose publication_date cannot be parsed are dropped from both
    lists (they contribute nothing to a temporal comparison that is keyed on
    ``date <= cutoff`` vs ``date > cutoff``).
    """
    pre: list[BenchmarkEntry] = []
    post: list[BenchmarkEntry] = []
    for e in entries:
        d = parse_date(e.publication_date)
        if d is None:
            continue
        if d <= cutoff:
            pre.append(e)
        else:
            post.append(e)
    return pre, post


def _segment_metrics(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
) -> dict[str, float]:
    """Compute DR, FPR, UNCERTAIN rate, Coverage for one (model, segment).

    - ``detection_rate`` and ``false_positive_rate`` are taken from the
      standard confusion matrix, which excludes UNCERTAIN predictions from
      both numerator and denominator by construction.
    - ``uncertain_rate``: fraction of entries whose prediction label is
      UNCERTAIN (or whose key is missing from ``predictions``).
    - ``coverage``: fraction of entries with a non-UNCERTAIN prediction.
    """
    if not entries:
        return {
            "detection_rate": 0.0,
            "false_positive_rate": 0.0,
            "uncertain_rate": 0.0,
            "coverage": 0.0,
            "num_entries": 0,
        }

    cm = build_confusion_matrix(entries, predictions)
    n = len(entries)
    n_uncertain = sum(
        1
        for e in entries
        if e.bibtex_key not in predictions or predictions[e.bibtex_key].label == "UNCERTAIN"
    )
    uncertain_rate = n_uncertain / n if n else 0.0
    coverage = 1.0 - uncertain_rate
    return {
        "detection_rate": cm.detection_rate,
        "false_positive_rate": cm.false_positive_rate,
        "uncertain_rate": uncertain_rate,
        "coverage": coverage,
        "num_entries": float(n),
    }


def compare_prompt_variants(
    default_preds: dict[str, dict[str, Prediction]],
    cutoff_aware_preds: dict[str, dict[str, Prediction]],
    entries: list[BenchmarkEntry],
    cutoffs: dict[str, date | str],
) -> dict[str, Any]:
    """Stratified comparison of default vs cutoff-aware prompts.

    The ablation tests H2: when explicitly reminded of the training cutoff,
    do LLMs route post-cutoff citations to UNCERTAIN rather than over-flag
    them as HALLUCINATED?  This function is fully computable from cached
    predictions — no live API calls.

    Args:
        default_preds: ``{model_name: {bibtex_key: Prediction}}`` under the
            default prompt.
        cutoff_aware_preds: ``{model_name: {bibtex_key: Prediction}}`` under
            the cutoff-aware prompt.  Models absent from this mapping are
            reported under ``default`` only and ``cutoff_aware`` metrics are
            reported as ``None``.
        entries: All benchmark entries to stratify.  Entries with an
            unparseable ``publication_date`` are excluded from the metric
            computation (they cannot be placed pre- or post-cutoff).
        cutoffs: Per-model training-cutoff dates.  Values may be ``date``
            objects or ISO-formatted strings.

    Returns:
        A dict with shape::

            {
                "rows": [
                    {
                        "model": str,
                        "cutoff": str,            # ISO date
                        "default": {
                            "pre_cutoff": {...segment metrics...},
                            "post_cutoff": {...},
                        },
                        "cutoff_aware": {
                            "pre_cutoff": {...} | None,
                            "post_cutoff": {...} | None,
                        },
                    },
                    ...
                ],
                "columns": ["detection_rate", "false_positive_rate",
                            "uncertain_rate", "coverage"],
            }

        When ``cutoff_aware_preds`` is empty, the ``cutoff_aware`` block is
        populated with ``None`` values — the caller can then report
        default-only numbers.
    """
    rows: list[dict[str, Any]] = []

    for model, preds in default_preds.items():
        cutoff_value = cutoffs.get(model)
        if cutoff_value is None:
            # No cutoff known: skip stratification for this model
            continue
        cutoff = (
            cutoff_value
            if isinstance(cutoff_value, date)
            else datetime.strptime(str(cutoff_value)[:10], "%Y-%m-%d").date()
        )

        pre, post = _split_pre_post_cutoff(entries, cutoff)

        default_block = {
            "pre_cutoff": _segment_metrics(pre, preds),
            "post_cutoff": _segment_metrics(post, preds),
        }

        ca_preds = cutoff_aware_preds.get(model)
        if ca_preds is None:
            ca_block: dict[str, Any] = {"pre_cutoff": None, "post_cutoff": None}
        else:
            ca_block = {
                "pre_cutoff": _segment_metrics(pre, ca_preds),
                "post_cutoff": _segment_metrics(post, ca_preds),
            }

        rows.append(
            {
                "model": model,
                "cutoff": cutoff.isoformat(),
                "default": default_block,
                "cutoff_aware": ca_block,
            }
        )

    return {
        "rows": rows,
        "columns": ["detection_rate", "false_positive_rate", "uncertain_rate", "coverage"],
    }
