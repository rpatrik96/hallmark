"""Temporal segmentation analysis for HALLMARK (LiveCodeBench-inspired).

Enables contamination detection by comparing tool performance on entries
from different time periods (pre-cutoff vs post-cutoff).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

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


# Default temporal segments
DEFAULT_SEGMENTS = [
    TemporalSegment("historical", date(2015, 1, 1), date(2023, 12, 31)),
    TemporalSegment("recent", date(2024, 1, 1), date(2025, 12, 31)),
    TemporalSegment("future", date(2026, 1, 1), date(2030, 12, 31)),
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
        segments = DEFAULT_SEGMENTS

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
