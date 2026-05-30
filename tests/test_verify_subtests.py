"""Tests for scripts/verify_subtests.py — the static sub-test QA pass.

These tests do two things:

1. Unit-test the per-entry consistency logic against the schema truth table
   (covers all six sub-tests, including the three the original script ignored:
   ``title_exists``, ``authors_match``, ``cross_db_agreement``).
2. **CI regression gate**: scan the shipped ``data/*.jsonl`` splits and assert
   the count of truth-table mismatches does not exceed a frozen baseline. The
   data files are final, so this gate locks in the current state and fails only
   if a future change makes sub-test consistency *worse*. When a data pass
   fixes entries, lower ``MAX_MISMATCHES`` to ratchet the bound down.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# verify_subtests lives in scripts/ (not an installed package); load it by path.
# It must be registered in sys.modules before exec so its @dataclass defs (which
# resolve cls.__module__ via sys.modules) initialize correctly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "verify_subtests.py"
_spec = importlib.util.spec_from_file_location("verify_subtests", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vs = importlib.util.module_from_spec(_spec)
sys.modules["verify_subtests"] = vs
_spec.loader.exec_module(vs)


# Frozen baseline: current strict mismatch count across all shipped splits.
# 234/15295 sub-test checks (98.5% agreement). These are documented design
# tensions (uniform cross_db_agreement=False convention, per-entry
# fields_complete / doi_resolves), NOT label errors. Ratchet DOWN after any
# data pass that resolves them; never raise it.
MAX_MISMATCHES = 234


class TestVerifyEntrySubtests:
    """Per-entry logic exercised on synthetic entries (no data files)."""

    def test_chimeric_title_consistent(self):
        # chimeric_title expects title_exists=False; matching assignment is OK.
        entry = {
            "bibtex_key": "k",
            "label": "HALLUCINATED",
            "hallucination_type": "chimeric_title",
            "subtests": {
                "doi_resolves": None,
                "title_exists": False,
                "authors_match": True,
                "venue_correct": True,
                "fields_complete": True,
                "cross_db_agreement": False,
            },
        }
        assert vs.verify_entry_subtests(entry) == []

    def test_title_exists_mismatch_flagged(self):
        # chimeric_title with title_exists=True contradicts the truth table.
        entry = {
            "bibtex_key": "k",
            "label": "HALLUCINATED",
            "hallucination_type": "chimeric_title",
            "subtests": {
                "title_exists": True,
                "authors_match": True,
                "venue_correct": True,
                "fields_complete": True,
                "cross_db_agreement": False,
            },
        }
        m = vs.verify_entry_subtests(entry)
        assert [x.subtest for x in m] == ["title_exists"]
        assert m[0].assigned is True and m[0].expected is False

    def test_authors_match_mismatch_flagged(self):
        # placeholder_authors expects authors_match=False.
        entry = {
            "bibtex_key": "k",
            "label": "HALLUCINATED",
            "hallucination_type": "placeholder_authors",
            "subtests": {
                "title_exists": True,
                "authors_match": True,  # wrong: should be False
                "venue_correct": True,
                "fields_complete": True,
                "cross_db_agreement": False,
            },
        }
        m = vs.verify_entry_subtests(entry)
        assert any(x.subtest == "authors_match" for x in m)

    def test_cross_db_agreement_mismatch_flagged(self):
        # All hallucination types expect cross_db_agreement=False.
        entry = {
            "bibtex_key": "k",
            "label": "HALLUCINATED",
            "hallucination_type": "fabricated_doi",
            "subtests": {
                "doi_resolves": False,
                "title_exists": True,
                "authors_match": True,
                "venue_correct": True,
                "fields_complete": True,
                "cross_db_agreement": True,  # wrong: should be False
            },
        }
        m = vs.verify_entry_subtests(entry)
        assert any(x.subtest == "cross_db_agreement" for x in m)

    def test_none_is_not_a_mismatch(self):
        # None on either side ("depends on source entry") is skipped.
        entry = {
            "bibtex_key": "k",
            "label": "HALLUCINATED",
            "hallucination_type": "nonexistent_venue",  # doi_resolves expected None
            "subtests": {
                "doi_resolves": True,  # concrete vs expected None -> skipped
                "title_exists": True,
                "authors_match": True,
                "venue_correct": False,
                "fields_complete": True,
                "cross_db_agreement": False,
            },
        }
        assert vs.verify_entry_subtests(entry) == []

    def test_unknown_type_skipped(self):
        entry = {
            "bibtex_key": "k",
            "label": "HALLUCINATED",
            "hallucination_type": "not_a_real_type",
            "subtests": {"title_exists": False},
        }
        assert vs.verify_entry_subtests(entry) == []

    def test_valid_entry_against_valid_subtests(self):
        # VALID entry with a sub-test that is concretely False where the
        # truth table expects True is flagged.
        entry = {
            "bibtex_key": "k",
            "label": "VALID",
            "hallucination_type": None,
            "subtests": {
                "doi_resolves": None,  # skipped
                "title_exists": False,  # flagged
                "authors_match": True,
                "venue_correct": True,
                "fields_complete": True,
                "cross_db_agreement": None,  # skipped
            },
        }
        m = vs.verify_entry_subtests(entry)
        assert [x.subtest for x in m] == ["title_exists"]


class TestSubtestConsistencyGate:
    """CI regression gate over the shipped data splits."""

    @pytest.fixture(scope="class")
    def report(self):
        splits = {name: _REPO_ROOT / rel for name, rel in vs.DEFAULT_SPLITS.items()}
        return vs.scan_splits(splits)

    def test_data_files_present(self, report):
        # At least the public splits must have been scanned; otherwise the
        # gate would pass vacuously.
        assert report.total_entries > 0, "no data entries scanned — wrong cwd?"
        assert report.total_checks > 0

    def test_no_unknown_hallucination_types(self, report):
        assert dict(report.unknown_types) == {}, (
            f"entries with hallucination_type outside the taxonomy: {dict(report.unknown_types)}"
        )

    def test_mismatches_within_baseline(self, report):
        assert report.num_mismatches <= MAX_MISMATCHES, (
            f"sub-test consistency regressed: {report.num_mismatches} mismatches "
            f"> baseline {MAX_MISMATCHES}. Inspect with "
            f"`python scripts/verify_subtests.py`. If a data pass intentionally "
            f"changed labels, update MAX_MISMATCHES (ratchet DOWN only)."
        )
