"""Tests for hallmark.evaluation.temporal."""

from datetime import date

import pytest

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.temporal import (
    TemporalSegment,
    default_segments,
    parse_date,
    segment_entries,
    temporal_analysis,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(key: str, pub_date: str | None = None, label: str = "VALID") -> BenchmarkEntry:
    kwargs: dict = {
        "bibtex_key": key,
        "bibtex_type": "article",
        "fields": {"title": f"Paper {key}", "author": "Author", "year": "2024"},
        "label": label,
    }
    if label == "HALLUCINATED":
        kwargs["hallucination_type"] = "fabricated_doi"
        kwargs["difficulty_tier"] = 1
    if pub_date is not None:
        kwargs["publication_date"] = pub_date
    return BenchmarkEntry(**kwargs)


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------


class TestParseDate:
    def test_iso_date(self):
        d = parse_date("2024-01-15")
        assert d == date(2024, 1, 15)

    def test_year_only(self):
        d = parse_date("2024")
        assert d == date(2024, 1, 1)

    def test_empty_string(self):
        assert parse_date("") is None

    def test_malformed(self):
        assert parse_date("not-a-date") is None

    def test_partial_iso_with_extra(self):
        # Only first 10 chars are parsed for ISO format
        d = parse_date("2023-06-30T12:00:00")
        assert d == date(2023, 6, 30)

    def test_none_equivalent_empty(self):
        # Empty string → None
        result = parse_date("")
        assert result is None


# ---------------------------------------------------------------------------
# segment_entries
# ---------------------------------------------------------------------------


class TestSegmentEntries:
    def _segments(self):
        return [
            TemporalSegment("historical", date(2015, 1, 1), date(2022, 12, 31)),
            TemporalSegment("recent", date(2023, 1, 1), date(2024, 12, 31)),
            TemporalSegment("future", date(2025, 1, 1), date(2030, 12, 31)),
        ]

    def test_entry_in_historical_segment(self):
        entry = _make_entry("old", pub_date="2020-06-15")
        result = segment_entries([entry], self._segments())
        assert entry in result["historical"]
        assert entry not in result["recent"]
        assert entry not in result["unknown"]

    def test_entry_in_recent_segment(self):
        entry = _make_entry("rec", pub_date="2023-09-01")
        result = segment_entries([entry], self._segments())
        assert entry in result["recent"]

    def test_entry_at_boundary_start(self):
        # Exactly on the start boundary of 'recent'
        entry = _make_entry("boundary", pub_date="2023-01-01")
        result = segment_entries([entry], self._segments())
        assert entry in result["recent"]

    def test_entry_at_boundary_end(self):
        # Exactly on the end boundary of 'historical'
        entry = _make_entry("boundary_end", pub_date="2022-12-31")
        result = segment_entries([entry], self._segments())
        assert entry in result["historical"]

    def test_entry_outside_all_segments(self):
        # 2031 is outside all defined segments
        entry = _make_entry("far_future", pub_date="2031-01-01")
        result = segment_entries([entry], self._segments())
        assert entry in result["unknown"]

    def test_entry_with_no_date(self):
        # No publication_date set → unknown
        entry = _make_entry("nodatez")
        result = segment_entries([entry], self._segments())
        assert entry in result["unknown"]

    def test_entry_with_empty_date(self):
        entry = _make_entry("emptydate", pub_date="")
        result = segment_entries([entry], self._segments())
        assert entry in result["unknown"]

    def test_entry_with_malformed_date(self):
        entry = _make_entry("baddate", pub_date="not-a-date")
        result = segment_entries([entry], self._segments())
        assert entry in result["unknown"]

    def test_multiple_entries_split_correctly(self):
        entries = [
            _make_entry("e1", pub_date="2019-01-01"),
            _make_entry("e2", pub_date="2023-06-15"),
            _make_entry("e3", pub_date="2026-03-01"),
            _make_entry("e4"),  # no date
        ]
        result = segment_entries(entries, self._segments())
        assert len(result["historical"]) == 1
        assert len(result["recent"]) == 1
        assert len(result["future"]) == 1
        assert len(result["unknown"]) == 1

    def test_default_segments_used_when_none(self):
        # Should not raise; uses default_segments() internally
        entry = _make_entry("e1", pub_date="2020-01-01")
        result = segment_entries([entry])
        assert "unknown" in result

    def test_all_segment_keys_present_in_result(self):
        segments = self._segments()
        result = segment_entries([], segments)
        for seg in segments:
            assert seg.name in result
        assert "unknown" in result


# ---------------------------------------------------------------------------
# default_segments
# ---------------------------------------------------------------------------


class TestDefaultSegments:
    def test_returns_three_segments(self):
        segs = default_segments()
        assert len(segs) == 3

    def test_segment_names(self):
        segs = default_segments()
        names = [s.name for s in segs]
        assert "historical" in names
        assert "recent" in names
        assert "future" in names

    def test_dates_are_date_objects(self):
        for seg in default_segments():
            assert isinstance(seg.start, date)
            assert isinstance(seg.end, date)

    def test_segments_non_overlapping(self):
        segs = default_segments()
        for i in range(len(segs) - 1):
            assert segs[i].end < segs[i + 1].start

    def test_historical_starts_at_2015(self):
        segs = default_segments()
        historical = next(s for s in segs if s.name == "historical")
        assert historical.start == date(2015, 1, 1)

    def test_future_is_after_today(self):
        today = date.today()
        segs = default_segments()
        future = next(s for s in segs if s.name == "future")
        assert future.start > today


# ---------------------------------------------------------------------------
# TemporalSegment.contains
# ---------------------------------------------------------------------------


class TestTemporalSegmentContains:
    def test_contains_within(self):
        seg = TemporalSegment("test", date(2020, 1, 1), date(2020, 12, 31))
        assert seg.contains(date(2020, 6, 15))

    def test_contains_start(self):
        seg = TemporalSegment("test", date(2020, 1, 1), date(2020, 12, 31))
        assert seg.contains(date(2020, 1, 1))

    def test_contains_end(self):
        seg = TemporalSegment("test", date(2020, 1, 1), date(2020, 12, 31))
        assert seg.contains(date(2020, 12, 31))

    def test_not_contains_before(self):
        seg = TemporalSegment("test", date(2020, 1, 1), date(2020, 12, 31))
        assert not seg.contains(date(2019, 12, 31))

    def test_not_contains_after(self):
        seg = TemporalSegment("test", date(2020, 1, 1), date(2020, 12, 31))
        assert not seg.contains(date(2021, 1, 1))


# ---------------------------------------------------------------------------
# temporal_analysis — integration tests
# ---------------------------------------------------------------------------


def _entry_dated(key: str, label: str, pub_date: str) -> BenchmarkEntry:
    kwargs: dict = {
        "bibtex_key": key,
        "bibtex_type": "article",
        "fields": {"title": f"Paper {key}", "author": "Author", "year": "2020"},
        "label": label,
        "explanation": "test",
        "publication_date": pub_date,
    }
    if label == "HALLUCINATED":
        kwargs["hallucination_type"] = "fabricated_doi"
        kwargs["difficulty_tier"] = 1
    return BenchmarkEntry(**kwargs)


def _pred(key: str, label: str, confidence: float = 0.9) -> Prediction:
    return Prediction(bibtex_key=key, label=label, confidence=confidence)


class TestTemporalBasicSegments:
    def test_segment_metrics_keys_present(self):
        """temporal_analysis returns segment_metrics with expected segment name keys."""
        segments = [
            TemporalSegment("historical", date(2015, 1, 1), date(2019, 12, 31)),
            TemporalSegment("future", date(2025, 1, 1), date(2030, 12, 31)),
        ]
        entries = [
            _entry_dated("h1", "HALLUCINATED", "2017-06-01"),
            _entry_dated("h2", "HALLUCINATED", "2018-03-15"),
            _entry_dated("v1", "VALID", "2026-01-10"),
            _entry_dated("v2", "VALID", "2027-05-20"),
        ]
        pred_map = {
            "h1": _pred("h1", "HALLUCINATED"),
            "h2": _pred("h2", "HALLUCINATED"),
            "v1": _pred("v1", "VALID"),
            "v2": _pred("v2", "VALID"),
        }
        result = temporal_analysis(entries, pred_map, segments=segments)

        assert "historical" in result.segment_metrics
        assert "future" in result.segment_metrics
        hist = result.segment_metrics["historical"]
        assert "detection_rate" in hist
        assert "false_positive_rate" in hist
        assert "f1" in hist
        assert "num_entries" in hist


class TestTemporalContaminationHigh:
    def test_contamination_score_positive_when_historical_dr_higher(self):
        """contamination_score > 0 when historical DR >> future DR."""
        segments = [
            TemporalSegment("historical", date(2015, 1, 1), date(2019, 12, 31)),
            TemporalSegment("future", date(2025, 1, 1), date(2030, 12, 31)),
        ]
        # Historical: all hallucinations detected (DR=1.0)
        hist_entries = [_entry_dated(f"h_hist_{i}", "HALLUCINATED", "2017-01-01") for i in range(5)]
        # Future: no hallucinations detected (DR=0.0)
        fut_entries = [_entry_dated(f"h_fut_{i}", "HALLUCINATED", "2026-01-01") for i in range(5)]
        entries = hist_entries + fut_entries

        pred_map: dict[str, Prediction] = {}
        for e in hist_entries:
            pred_map[e.bibtex_key] = _pred(e.bibtex_key, "HALLUCINATED")
        for e in fut_entries:
            pred_map[e.bibtex_key] = _pred(e.bibtex_key, "VALID")

        result = temporal_analysis(entries, pred_map, segments=segments)

        assert result.contamination_score is not None
        assert result.contamination_score > 0.0
        assert result.segment_metrics["historical"]["detection_rate"] == pytest.approx(1.0)
        assert result.segment_metrics["future"]["detection_rate"] == pytest.approx(0.0)


class TestTemporalNoFutureEntries:
    def test_contamination_score_none_without_future_entries(self):
        """contamination_score is None when future segment has no entries."""
        segments = [
            TemporalSegment("historical", date(2015, 1, 1), date(2019, 12, 31)),
            TemporalSegment("future", date(2025, 1, 1), date(2030, 12, 31)),
        ]
        # All entries fall in the historical segment only
        entries = [
            _entry_dated("h1", "HALLUCINATED", "2016-04-01"),
            _entry_dated("h2", "HALLUCINATED", "2018-09-15"),
            _entry_dated("v1", "VALID", "2017-07-01"),
        ]
        pred_map = {
            "h1": _pred("h1", "HALLUCINATED"),
            "h2": _pred("h2", "VALID"),
            "v1": _pred("v1", "VALID"),
        }
        result = temporal_analysis(entries, pred_map, segments=segments)

        # future segment has no entries → contamination_score cannot be computed
        assert result.contamination_score is None
        assert result.robustness_delta is None


class TestTemporalRobustnessDelta:
    def test_robustness_delta_equals_historical_minus_future_dr(self):
        """robustness_delta = historical_DR - future_DR (sign convention check)."""
        segments = [
            TemporalSegment("historical", date(2015, 1, 1), date(2019, 12, 31)),
            TemporalSegment("future", date(2025, 1, 1), date(2030, 12, 31)),
        ]
        # Historical: 3/4 detected (DR=0.75)
        hist_entries = [_entry_dated(f"h_hist_{i}", "HALLUCINATED", "2018-01-01") for i in range(4)]
        # Future: 1/4 detected (DR=0.25)
        fut_entries = [_entry_dated(f"h_fut_{i}", "HALLUCINATED", "2026-01-01") for i in range(4)]
        entries = hist_entries + fut_entries

        pred_map: dict[str, Prediction] = {}
        for i, e in enumerate(hist_entries):
            label = "HALLUCINATED" if i < 3 else "VALID"  # 3 detected, 1 missed
            pred_map[e.bibtex_key] = _pred(e.bibtex_key, label)
        for i, e in enumerate(fut_entries):
            label = "HALLUCINATED" if i < 1 else "VALID"  # 1 detected, 3 missed
            pred_map[e.bibtex_key] = _pred(e.bibtex_key, label)

        result = temporal_analysis(entries, pred_map, segments=segments)

        assert result.robustness_delta is not None
        hist_dr = result.segment_metrics["historical"]["detection_rate"]
        fut_dr = result.segment_metrics["future"]["detection_rate"]
        # Verify the sign convention: robustness_delta == historical_DR - future_DR
        assert result.robustness_delta == pytest.approx(hist_dr - fut_dr)
        # 0.75 - 0.25 = 0.50 → positive delta (historical performs better)
        assert result.robustness_delta > 0.0
