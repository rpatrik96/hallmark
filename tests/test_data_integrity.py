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
    generate_arxiv_version_mismatch,
    generate_near_miss_title,
)
from hallmark.dataset.schema import (
    BenchmarkEntry,
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
    def test_no_version_arxiv_overlap(
        self, dev_entries: list[BenchmarkEntry], test_entries: list[BenchmarkEntry]
    ) -> None:
        """Version confusion arXiv IDs must be disjoint across splits."""
        dev_arxiv = {
            e.fields.get("eprint", "")
            for e in dev_entries
            if e.hallucination_type == "arxiv_version_mismatch"
        } - {""}
        test_arxiv = {
            e.fields.get("eprint", "")
            for e in test_entries
            if e.hallucination_type == "arxiv_version_mismatch"
        } - {""}
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
        """Main splits must have >= MIN_PER_TYPE for each main taxonomy type."""
        from hallmark.dataset.schema import MAIN_TYPES

        entries = dev_entries if split_name == "dev" else test_entries
        type_counts: Counter[str] = Counter()
        for e in entries:
            if e.label == "HALLUCINATED" and e.hallucination_type:
                type_counts[e.hallucination_type] += 1

        for ht in MAIN_TYPES:
            count = type_counts.get(ht.value, 0)
            assert count >= MIN_PER_TYPE, (
                f"{split_name}: {ht.value} has {count} entries, need >= {MIN_PER_TYPE}"
            )

    def test_all_main_types_present_dev(self, dev_entries: list[BenchmarkEntry]) -> None:
        from hallmark.dataset.schema import MAIN_TYPES

        types = {e.hallucination_type for e in dev_entries if e.label == "HALLUCINATED"}
        expected = {ht.value for ht in MAIN_TYPES}
        assert expected <= types, f"Missing main types in dev: {expected - types}"

    def test_all_main_types_present_test(self, test_entries: list[BenchmarkEntry]) -> None:
        from hallmark.dataset.schema import MAIN_TYPES

        types = {e.hallucination_type for e in test_entries if e.label == "HALLUCINATED"}
        expected = {ht.value for ht in MAIN_TYPES}
        assert expected <= types, f"Missing main types in test: {expected - types}"

    def test_stress_types_present_in_stress_split(self) -> None:
        """Stress-test types should appear in the stress_test split."""
        from hallmark.dataset.schema import STRESS_TEST_TYPES, load_entries

        stress_values = {ht.value for ht in STRESS_TEST_TYPES}
        try:
            entries = load_entries("stress_test")
        except FileNotFoundError:
            pytest.skip("stress_test split not found")
        found = {e.hallucination_type for e in entries if e.label == "HALLUCINATED"}
        assert stress_values <= found, (
            f"Missing stress types in stress_test: {stress_values - found}"
        )


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


# ── Generator quality: arxiv_version_mismatch ────────────────────────────────────


class TestVersionConfusionQuality:
    """Verify arxiv_version_mismatch creates genuine metadata mixing."""

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
        entry = generate_arxiv_version_mismatch(source, "ICML")
        assert entry.fields["title"] == source.fields["title"]

    def test_no_arxiv_eprint(self) -> None:
        """Per P0.3, eprint/archiveprefix are hallucination-only fields and must not be set."""
        source = self._make_source()
        entry = generate_arxiv_version_mismatch(source, "ICML")
        assert "eprint" not in entry.fields
        assert "archiveprefix" not in entry.fields

    def test_sets_wrong_venue(self) -> None:
        source = self._make_source()
        entry = generate_arxiv_version_mismatch(source, "ICML")
        assert entry.fields["booktitle"] == "ICML"
        # Year should be shifted by ±1 from original
        original_year = int(source.fields["year"])
        result_year = int(entry.fields["year"])
        assert abs(result_year - original_year) == 1

    def test_preserves_doi(self) -> None:
        source = self._make_source()
        entry = generate_arxiv_version_mismatch(source, "ICML")
        assert entry.fields.get("doi") == source.fields["doi"]
        assert entry.subtests["doi_resolves"] is True
        assert entry.subtests["cross_db_agreement"] is False


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
    def test_no_all_true_subtests_hallucinated(
        self,
        split_name: str,
        dev_entries: list[BenchmarkEntry],
        test_entries: list[BenchmarkEntry],
    ) -> None:
        """Hallucinated entries must not have all subtests True (except arxiv_version_mismatch)."""
        entries = dev_entries if split_name == "dev" else test_entries
        violators = []
        for e in entries:
            if (
                e.label == "HALLUCINATED"
                and e.subtests
                and all(v is True for v in e.subtests.values())
                and e.hallucination_type != "arxiv_version_mismatch"
            ):
                violators.append(e.bibtex_key)
        assert not violators, (
            f"{split_name}: {len(violators)} hallucinated entries have all subtests=True: "
            f"{violators[:5]}"
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
