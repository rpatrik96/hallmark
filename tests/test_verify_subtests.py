"""Tests for scripts/verify_subtests.py — the static sub-test QA pass.

These tests do two things:

1. Unit-test the per-entry consistency logic against the schema truth table
   (covers all six sub-tests, including the three the original script ignored:
   ``title_exists``, ``authors_match``, ``cross_db_agreement``).
2. **CI regression gate** over the shipped ``data/*.jsonl`` splits, in two
   parts. Mismatches where an entry *contradicts its own hallucination type*
   are bounded at zero; the remaining per-entry disagreements are ratcheted
   against a frozen baseline, counted separately for the public splits and the
   hidden one because ``data/hidden/`` is gitignored and CI therefore scans a
   different population than a full-dataset checkout does. Lower a baseline
   after a data pass; never raise one.
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


# Frozen baselines for per-entry disagreements with the type-level default in
# EXPECTED_SUBTESTS -- the uniform cross_db_agreement=False convention, and
# fields_complete values that legitimately differ per entry. These are design
# tensions, NOT label errors; contradictions are bounded separately below.
#
# doi_resolves is not among them: v1.2.3 corrected the entries recording a
# failed resolution while carrying no DOI, and the 2026-09-04 pass extended
# that fix to the hidden split, which the original run had omitted. The class
# is reported separately by verify_entry_structural() in
# scripts/verify_subtests.py and does not count against these bounds.
#
# Ratcheted per population, because CI and a full-dataset checkout scan
# different splits. Public: 99 over 2,572 entries. Hidden: 55 over 454, after
# the 2026-09-04 pass repaired 43 doi_resolves not-applicables and 43 sub-tests
# that contradicted their own type. Ratchet DOWN only; never raise either.
MAX_PUBLIC_MISMATCHES = 99
MAX_HIDDEN_MISMATCHES = 55

# A single total hid two different defects, because it was tuned to exactly the
# splits CI can see. ``data/hidden/`` is gitignored, so a contributor's run and
# a CI run scan different populations and neither number says which.
#
# The two directions are not the same kind of finding, so they get separate
# assertions:
#
# ``False -> True`` is a per-entry disagreement with the type-level default --
# a real design tension, ratcheted by the two baselines above.
#
# ``True -> False`` is an entry asserting something its own hallucination type
# forbids: a preprint cited as published claiming cross-database agreement, when
# the databases necessarily report the venue it actually appeared in. That is a
# contradiction, not a tension, and the public splits carry zero of them, so
# zero is the achievable bound.
#
# The one exemption is ``future_date``/``fields_complete``, and the taxonomy is
# what is wrong there rather than the data. ``check_fields_complete`` tests for
# missing required fields plus a 4-digit year and a well-formed DOI; a
# future-dated entry has every field with a perfectly well-formed year, so the
# checker returns True. Run over every ``future_date`` entry it returns True for
# 14 of 15 in the hidden split (which assign True, and are therefore right) and
# for 29 of 30 in dev_public (which assign False, and are therefore wrong).
# Correcting it means changing EXPECTED_SUBTESTS and re-labelling released
# public entries, which is a data decision rather than a test decision.
MAX_TYPE_CONTRADICTIONS = 0

#: (hallucination_type, subtest) pairs exempt from the zero bound above, with
#: the reason they are exempt. Remove an entry here only by fixing the cause.
CONTRADICTION_EXEMPTIONS: dict[tuple[str, str], str] = {
    ("future_date", "fields_complete"): (
        "EXPECTED_SUBTESTS says False but check_fields_complete returns True — "
        "the taxonomy contradicts the checker, so the data is not at fault"
    ),
}


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

    def test_no_subtest_contradicts_its_own_type(self, report):
        """A sub-test may not assert what its entry's hallucination type forbids.

        This is the direction a single total hid. The public splits carry zero,
        so zero is achievable; the hidden split carried 43 until they were
        repaired. Unlike the ratcheted count below, this bound does not move.
        """
        offenders = [
            m
            for m in report.mismatches
            if m.assigned is True
            and m.expected is False
            and (str(m.hallucination_type), str(m.subtest)) not in CONTRADICTION_EXEMPTIONS
        ]
        assert len(offenders) <= MAX_TYPE_CONTRADICTIONS, (
            f"{len(offenders)} sub-test(s) contradict their entry's own hallucination "
            f"type, e.g. {[(m.split, m.bibtex_key, m.hallucination_type, m.subtest) for m in offenders[:5]]}. "
            "EXPECTED_SUBTESTS fixes these by definition of the type; an entry asserting "
            "the opposite is a contradiction, not a tolerated tension. Repair with "
            "`python scripts/fix_subtest_type_contradictions.py --apply`, or add an "
            "exemption to CONTRADICTION_EXEMPTIONS if the taxonomy is what is wrong."
        )

    def test_exemptions_are_still_needed(self, report):
        """An exemption that no longer fires is a stale excuse — drop it.

        Skipped without the hidden split, and that is the point rather than a
        convenience: the only entries currently exercising the
        ``future_date``/``fields_complete`` exemption are in ``test_hidden``,
        because the public splits assign ``False`` there and so agree with the
        (wrong) taxonomy. Asserting staleness over a population that cannot
        contain the cause would report a live exemption as dead — the same
        population-dependent mistake this gate was split apart to stop making.
        """
        if not (_REPO_ROOT / vs.DEFAULT_SPLITS["test_hidden"]).exists():
            pytest.skip("hidden split not present — exemption staleness not checkable")

        seen = {
            (str(m.hallucination_type), str(m.subtest))
            for m in report.mismatches
            if m.assigned is True and m.expected is False
        }
        stale = set(CONTRADICTION_EXEMPTIONS) - seen
        assert not stale, (
            f"CONTRADICTION_EXEMPTIONS entries no longer occur and should be removed: {stale}"
        )

    def test_mismatches_within_baseline(self, report):
        """Ratchet the public splits and the hidden split against their own bounds.

        One combined total cannot work: ``data/hidden/`` is gitignored, so CI
        scans 2,572 entries and a contributor with the full dataset scans 3,026,
        and a single number is either tuned to CI (and blind to the hidden
        split) or unreachable in CI. Counting them separately means each bound
        is checked by whoever can actually see that data.
        """
        public = sum(1 for m in report.mismatches if m.split != "test_hidden")
        hidden = sum(1 for m in report.mismatches if m.split == "test_hidden")

        assert public <= MAX_PUBLIC_MISMATCHES, (
            f"sub-test consistency regressed on the public splits: {public} mismatches "
            f"> baseline {MAX_PUBLIC_MISMATCHES}. Inspect with "
            f"`python scripts/verify_subtests.py`. If a data pass intentionally "
            f"changed labels, update MAX_PUBLIC_MISMATCHES (ratchet DOWN only)."
        )

        # Absent in CI and on any checkout without the full dataset. Skip rather
        # than pass, so a run that never opened the file cannot read as a clean
        # bill of health for it.
        if not (_REPO_ROOT / vs.DEFAULT_SPLITS["test_hidden"]).exists():
            pytest.skip("hidden split not present — its bound was not checked")

        assert hidden <= MAX_HIDDEN_MISMATCHES, (
            f"sub-test consistency regressed on the hidden split: {hidden} mismatches "
            f"> baseline {MAX_HIDDEN_MISMATCHES}. Ratchet DOWN only."
        )
