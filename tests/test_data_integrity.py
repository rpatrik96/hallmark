"""Tests for dataset integrity: cross-split contamination, key uniqueness,
type balance, generator quality, and surface diversity.

These tests validate the fixes from the devil's advocate review.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from hallmark.dataset.generator import (
    generate_near_miss_title,
    generate_version_confusion,
)
from hallmark.dataset.schema import (
    BenchmarkEntry,
    HallucinationType,
    load_entries,
)

DATA_DIR = Path("data/v1.0")
DEV_PATH = DATA_DIR / "dev_public.jsonl"
TEST_PATH = DATA_DIR / "test_public.jsonl"
MIN_PER_TYPE = 10


@pytest.fixture(scope="module")
def dev_entries() -> list[BenchmarkEntry]:
    return load_entries(DEV_PATH)


@pytest.fixture(scope="module")
def test_entries() -> list[BenchmarkEntry]:
    return load_entries(TEST_PATH)


# ── Key uniqueness ──────────────────────────────────────────────────────────


class TestKeyUniqueness:
    def test_dev_keys_unique(self, dev_entries: list[BenchmarkEntry]) -> None:
        keys = [e.bibtex_key for e in dev_entries]
        dupes = {k: c for k, c in Counter(keys).items() if c > 1}
        assert not dupes, f"Duplicate bibtex_keys in dev: {dupes}"

    def test_test_keys_unique(self, test_entries: list[BenchmarkEntry]) -> None:
        keys = [e.bibtex_key for e in test_entries]
        dupes = {k: c for k, c in Counter(keys).items() if c > 1}
        assert not dupes, f"Duplicate bibtex_keys in test: {dupes}"


# ── Cross-split contamination ──────────────────────────────────────────────


class TestCrossSplitContamination:
    def test_no_retracted_doi_overlap(
        self, dev_entries: list[BenchmarkEntry], test_entries: list[BenchmarkEntry]
    ) -> None:
        """Retracted paper DOIs must be disjoint across splits."""
        dev_dois = {
            e.fields.get("doi", "")
            for e in dev_entries
            if e.hallucination_type == "retracted_paper"
        }
        test_dois = {
            e.fields.get("doi", "")
            for e in test_entries
            if e.hallucination_type == "retracted_paper"
        }
        overlap = dev_dois & test_dois
        assert not overlap, f"Retracted DOIs in both splits: {overlap}"

    def test_no_version_arxiv_overlap(
        self, dev_entries: list[BenchmarkEntry], test_entries: list[BenchmarkEntry]
    ) -> None:
        """Version confusion arXiv IDs must be disjoint across splits."""
        dev_arxiv = {
            e.fields.get("eprint", "")
            for e in dev_entries
            if e.hallucination_type == "version_confusion"
        }
        test_arxiv = {
            e.fields.get("eprint", "")
            for e in test_entries
            if e.hallucination_type == "version_confusion"
        }
        overlap = dev_arxiv & test_arxiv
        assert not overlap, f"Version confusion arXiv IDs in both splits: {overlap}"


# ── Type balance ────────────────────────────────────────────────────────────


class TestTypeBalance:
    @pytest.mark.parametrize("split_name", ["dev", "test"])
    def test_min_per_type(
        self,
        split_name: str,
        dev_entries: list[BenchmarkEntry],
        test_entries: list[BenchmarkEntry],
    ) -> None:
        entries = dev_entries if split_name == "dev" else test_entries
        type_counts: Counter[str] = Counter()
        for e in entries:
            if e.label == "HALLUCINATED" and e.hallucination_type:
                type_counts[e.hallucination_type] += 1

        for ht in HallucinationType:
            count = type_counts.get(ht.value, 0)
            assert count >= MIN_PER_TYPE, (
                f"{split_name}: {ht.value} has {count} entries, need >= {MIN_PER_TYPE}"
            )

    def test_all_13_types_present_dev(self, dev_entries: list[BenchmarkEntry]) -> None:
        types = {e.hallucination_type for e in dev_entries if e.label == "HALLUCINATED"}
        expected = {ht.value for ht in HallucinationType}
        assert types == expected, f"Missing types in dev: {expected - types}"

    def test_all_13_types_present_test(self, test_entries: list[BenchmarkEntry]) -> None:
        types = {e.hallucination_type for e in test_entries if e.label == "HALLUCINATED"}
        expected = {ht.value for ht in HallucinationType}
        assert types == expected, f"Missing types in test: {expected - types}"


# ── Generator quality: near_miss_title ──────────────────────────────────────


class TestNearMissTitleQuality:
    """Verify near_miss_title mutations are subtle, not generic insertions."""

    def _make_source(self) -> BenchmarkEntry:
        return BenchmarkEntry(
            bibtex_key="test_source",
            bibtex_type="inproceedings",
            fields={
                "title": "Attention Is All You Need",
                "author": "Ashish Vaswani",
                "year": "2017",
                "booktitle": "NeurIPS",
            },
            label="VALID",
        )

    def test_no_generic_insertions(self) -> None:
        """Generated titles should not contain generic words like Novel/Enhanced."""
        generic_words = {"Improved", "Enhanced", "Novel", "Adaptive", "Scalable"}
        source = self._make_source()
        rng = random.Random(42)

        generic_count = 0
        n_trials = 50
        for _ in range(n_trials):
            entry = generate_near_miss_title(source, rng)
            title_words = set(entry.fields["title"].split())
            if title_words & generic_words:
                generic_count += 1

        # Allow at most 10% generic (some may come from synonym fallback)
        assert generic_count <= n_trials * 0.1, (
            f"{generic_count}/{n_trials} near-miss titles contain generic words"
        )

    def test_title_differs_from_original(self) -> None:
        source = self._make_source()
        rng = random.Random(42)
        entry = generate_near_miss_title(source, rng)
        assert entry.fields["title"] != source.fields["title"]

    def test_title_is_close_to_original(self) -> None:
        """Modified title should differ by at most a few words."""
        source = self._make_source()
        rng = random.Random(42)
        entry = generate_near_miss_title(source, rng)

        original_words = source.fields["title"].lower().split()
        modified_words = entry.fields["title"].lower().split()

        # Allow at most 3 word differences (insertions, deletions, substitutions)
        # Using simple set difference as proxy
        added = set(modified_words) - set(original_words)
        removed = set(original_words) - set(modified_words)
        total_diff = len(added) + len(removed)
        assert total_diff <= 4, (
            f"Too many differences ({total_diff}): added={added}, removed={removed}"
        )


# ── Generator quality: version_confusion ────────────────────────────────────


class TestVersionConfusionQuality:
    """Verify version_confusion creates genuine metadata mixing."""

    def _make_source(self) -> BenchmarkEntry:
        return BenchmarkEntry(
            bibtex_key="test_source",
            bibtex_type="inproceedings",
            fields={
                "title": "Attention Is All You Need",
                "author": "Ashish Vaswani",
                "year": "2017",
                "booktitle": "NeurIPS",
                "doi": "10.5555/3295222.3295349",
            },
            label="VALID",
        )

    def test_keeps_original_title(self) -> None:
        source = self._make_source()
        entry = generate_version_confusion(source, "1706.03762", "ICML", "2018")
        assert entry.fields["title"] == source.fields["title"]

    def test_sets_arxiv_eprint(self) -> None:
        source = self._make_source()
        entry = generate_version_confusion(source, "1706.03762", "ICML", "2018")
        assert entry.fields["eprint"] == "1706.03762"
        assert entry.fields["archiveprefix"] == "arXiv"

    def test_sets_conference_venue(self) -> None:
        source = self._make_source()
        entry = generate_version_confusion(source, "1706.03762", "ICML", "2018")
        assert entry.fields["booktitle"] == "ICML"
        assert entry.fields["year"] == "2018"

    def test_removes_doi(self) -> None:
        source = self._make_source()
        entry = generate_version_confusion(source, "1706.03762", "ICML", "2018")
        assert "doi" not in entry.fields


# ── Surface diversity ───────────────────────────────────────────────────────


class TestSurfaceDiversity:
    """Verify no type has too many identical surface patterns."""

    @pytest.mark.parametrize("split_name", ["dev", "test"])
    def test_chimeric_title_diversity(
        self,
        split_name: str,
        dev_entries: list[BenchmarkEntry],
        test_entries: list[BenchmarkEntry],
    ) -> None:
        """Chimeric title entries should have unique titles."""
        entries = dev_entries if split_name == "dev" else test_entries
        titles = [
            e.fields.get("title", "") for e in entries if e.hallucination_type == "chimeric_title"
        ]
        unique_titles = set(titles)
        # At least 80% unique
        assert len(unique_titles) >= len(titles) * 0.8, (
            f"{split_name}: only {len(unique_titles)}/{len(titles)} unique chimeric titles"
        )

    @pytest.mark.parametrize("split_name", ["dev", "test"])
    def test_retracted_doi_diversity(
        self,
        split_name: str,
        dev_entries: list[BenchmarkEntry],
        test_entries: list[BenchmarkEntry],
    ) -> None:
        """Retracted paper entries should use diverse DOIs within a split."""
        entries = dev_entries if split_name == "dev" else test_entries
        dois = [
            e.fields.get("doi", "") for e in entries if e.hallucination_type == "retracted_paper"
        ]
        unique_dois = set(dois)
        # At least 50% unique (some reuse is OK within a split)
        assert len(unique_dois) >= len(dois) * 0.5, (
            f"{split_name}: only {len(unique_dois)}/{len(dois)} unique retracted DOIs"
        )


# ── Schema validation ──────────────────────────────────────────────────────


class TestSchemaValidation:
    def test_save_entries_rejects_duplicates(self, tmp_path: Path) -> None:
        """save_entries should raise on duplicate bibtex_keys."""
        from hallmark.dataset.schema import save_entries

        entries = [
            BenchmarkEntry(
                bibtex_key="same_key",
                bibtex_type="article",
                fields={"title": "Test", "author": "A", "year": "2024"},
                label="VALID",
            ),
            BenchmarkEntry(
                bibtex_key="same_key",
                bibtex_type="article",
                fields={"title": "Test 2", "author": "B", "year": "2024"},
                label="VALID",
            ),
        ]
        with pytest.raises(ValueError, match="Duplicate bibtex_keys"):
            save_entries(entries, tmp_path / "test.jsonl")

    @pytest.mark.parametrize("split_name", ["dev", "test"])
    def test_all_hallucinated_have_type_and_tier(
        self,
        split_name: str,
        dev_entries: list[BenchmarkEntry],
        test_entries: list[BenchmarkEntry],
    ) -> None:
        entries = dev_entries if split_name == "dev" else test_entries
        for e in entries:
            if e.label == "HALLUCINATED":
                assert e.hallucination_type is not None, (
                    f"{e.bibtex_key}: missing hallucination_type"
                )
                assert e.difficulty_tier is not None, f"{e.bibtex_key}: missing difficulty_tier"

    @pytest.mark.parametrize("split_name", ["dev", "test"])
    def test_all_entries_have_subtests(
        self,
        split_name: str,
        dev_entries: list[BenchmarkEntry],
        test_entries: list[BenchmarkEntry],
    ) -> None:
        entries = dev_entries if split_name == "dev" else test_entries
        for e in entries:
            if e.label == "HALLUCINATED":
                assert len(e.subtests) == 6, (
                    f"{e.bibtex_key}: has {len(e.subtests)} subtests, expected 6"
                )
