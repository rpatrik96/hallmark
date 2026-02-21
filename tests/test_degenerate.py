"""Tests for degenerate (statistical reference) baselines."""

from __future__ import annotations

from hallmark.baselines.degenerate import (
    VALID_VENUE_SET,
    always_hallucinated_baseline,
    always_valid_baseline,
    random_baseline,
    venue_oracle_baseline,
)
from hallmark.dataset.schema import BlindEntry

# --- Fixtures ---


def _blind_entry(key: str) -> BlindEntry:
    return BlindEntry(
        bibtex_key=key,
        bibtex_type="article",
        fields={"title": f"Paper {key}", "author": "Author A", "year": "2024"},
    )


ENTRIES = [_blind_entry(k) for k in ("a", "b", "c", "d", "e")]


# --- random_baseline ---


def test_random_baseline_count():
    preds = random_baseline(ENTRIES, seed=42)
    assert len(preds) == len(ENTRIES)


def test_random_baseline_labels():
    preds = random_baseline(ENTRIES, seed=0)
    valid_labels = {"HALLUCINATED", "VALID"}
    for p in preds:
        assert p.label in valid_labels


def test_random_baseline_determinism():
    preds1 = random_baseline(ENTRIES, seed=7)
    preds2 = random_baseline(ENTRIES, seed=7)
    assert [p.label for p in preds1] == [p.label for p in preds2]
    assert [p.confidence for p in preds1] == [p.confidence for p in preds2]


def test_random_baseline_different_seeds():
    # With enough entries and two different seeds, results differ (probabilistically certain).
    entries = [_blind_entry(str(i)) for i in range(50)]
    preds1 = random_baseline(entries, seed=1)
    preds2 = random_baseline(entries, seed=999)
    labels1 = [p.label for p in preds1]
    labels2 = [p.label for p in preds2]
    assert labels1 != labels2


def test_random_baseline_prevalence_one():
    preds = random_baseline(ENTRIES, seed=42, prevalence=1.0)
    for p in preds:
        assert p.label == "HALLUCINATED"
        assert p.confidence == 1.0


def test_random_baseline_prevalence_zero():
    preds = random_baseline(ENTRIES, seed=42, prevalence=0.0)
    for p in preds:
        assert p.label == "VALID"
        assert p.confidence == 1.0


def test_random_baseline_bibtex_keys():
    preds = random_baseline(ENTRIES, seed=42)
    assert [p.bibtex_key for p in preds] == [e.bibtex_key for e in ENTRIES]


# --- always_hallucinated_baseline ---


def test_always_hallucinated_baseline():
    preds = always_hallucinated_baseline(ENTRIES)
    assert len(preds) == len(ENTRIES)
    for p in preds:
        assert p.label == "HALLUCINATED"
        assert p.confidence == 1.0


def test_always_hallucinated_bibtex_keys():
    preds = always_hallucinated_baseline(ENTRIES)
    assert [p.bibtex_key for p in preds] == [e.bibtex_key for e in ENTRIES]


# --- always_valid_baseline ---


def test_always_valid_baseline():
    preds = always_valid_baseline(ENTRIES)
    assert len(preds) == len(ENTRIES)
    for p in preds:
        assert p.label == "VALID"
        assert p.confidence == 1.0


def test_always_valid_bibtex_keys():
    preds = always_valid_baseline(ENTRIES)
    assert [p.bibtex_key for p in preds] == [e.bibtex_key for e in ENTRIES]


def test_always_valid_empty():
    preds = always_valid_baseline([])
    assert preds == []


def test_always_hallucinated_empty():
    preds = always_hallucinated_baseline([])
    assert preds == []


# --- venue_oracle_baseline ---


def _blind_entry_with_venue(
    key: str, booktitle: str | None = None, journal: str | None = None
) -> BlindEntry:
    fields: dict[str, str] = {"title": f"Paper {key}", "author": "Author A", "year": "2024"}
    if booktitle is not None:
        fields["booktitle"] = booktitle
    if journal is not None:
        fields["journal"] = journal
    return BlindEntry(bibtex_key=key, bibtex_type="article", fields=fields)


def test_venue_oracle_known_venue_valid():
    entries = [_blind_entry_with_venue("a", booktitle="NeurIPS")]
    preds = venue_oracle_baseline(entries)
    assert preds[0].label == "VALID"


def test_venue_oracle_unknown_venue_hallucinated():
    entries = [_blind_entry_with_venue("a", booktitle="Fake Conference on AI")]
    preds = venue_oracle_baseline(entries)
    assert preds[0].label == "HALLUCINATED"
    assert preds[0].confidence == 1.0


def test_venue_oracle_journal_venue():
    entries = [_blind_entry_with_venue("a", journal="J. Mach. Learn. Res.")]
    preds = venue_oracle_baseline(entries)
    assert preds[0].label == "VALID"


def test_venue_oracle_no_venue_defaults_valid():
    entries = [_blind_entry_with_venue("a")]  # no booktitle or journal
    preds = venue_oracle_baseline(entries)
    assert preds[0].label == "VALID"


def test_venue_oracle_count():
    entries = [_blind_entry_with_venue(str(i), booktitle="NeurIPS") for i in range(5)]
    preds = venue_oracle_baseline(entries)
    assert len(preds) == 5


def test_venue_oracle_bibtex_keys():
    entries = [
        _blind_entry_with_venue("x", booktitle="ICML"),
        _blind_entry_with_venue("y", booktitle="Fake"),
    ]
    preds = venue_oracle_baseline(entries)
    assert [p.bibtex_key for p in preds] == ["x", "y"]


def test_venue_oracle_empty():
    preds = venue_oracle_baseline([])
    assert preds == []


def test_venue_oracle_custom_venues():
    custom = frozenset({"CustomVenue"})
    entries = [
        _blind_entry_with_venue("a", booktitle="CustomVenue"),
        _blind_entry_with_venue("b", booktitle="NeurIPS"),
    ]
    preds = venue_oracle_baseline(entries, valid_venues=custom)
    assert preds[0].label == "VALID"  # CustomVenue in custom set
    assert preds[1].label == "HALLUCINATED"  # NeurIPS NOT in custom set


def test_venue_oracle_all_valid_venues():
    """Every venue in VALID_VENUE_SET should be predicted VALID."""
    entries = [
        _blind_entry_with_venue(f"e{i}", booktitle=v) for i, v in enumerate(sorted(VALID_VENUE_SET))
    ]
    preds = venue_oracle_baseline(entries)
    for p in preds:
        assert p.label == "VALID", f"Venue oracle incorrectly flagged {p.bibtex_key}"
