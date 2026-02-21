"""Tests for hallmark.evaluation.temporal."""

from datetime import date

from hallmark.dataset.schema import BenchmarkEntry
from hallmark.evaluation.temporal import (
    TemporalSegment,
    default_segments,
    parse_date,
    segment_entries,
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
