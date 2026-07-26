"""Tests for hallmark.dataset.blinding and the blinding invariant it enforces.

Regression cover for the defect these tests exist to prevent: dispatch-time
blinding lived in a single method that a runner could skip, and one did — an
ad-hoc resume script read the corpus JSONL directly and serialized ``url``
into the prompt. The invariant under test is that no entry reaches a verifier
carrying a field on the blind-list, whichever path it took to get there.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hallmark.dataset.blinding import (
    BLIND_EXCLUDED_FIELDS,
    BlindingViolationError,
    assert_blinded,
    blind_fields,
    blind_record,
    find_blind_violations,
    scrub_bibtex,
)
from hallmark.dataset.schema import BenchmarkEntry, BlindEntry

REPO = Path(__file__).resolve().parents[1]

# --- Fixtures ---


def make_entry_with_url(url: str = "https://jmlr.org/papers/v24/22-0582.html") -> BenchmarkEntry:
    """A VALID entry shaped like the released ones that still carry a URL."""
    return BenchmarkEntry(
        bibtex_key="faa1ea092c70",
        bibtex_type="inproceedings",
        fields={
            "title": "Bayesian Nonparametric Learning of Stochastic Differential Equations",
            "author": "Doe, Jane and Roe, Richard",
            "year": "2023",
            "booktitle": "J. Mach. Learn. Res.",
            "url": url,
        },
        label="VALID",
    )


def record_with_url(url: str = "https://openreview.net/forum?id=HylxE1HKwS") -> dict:
    """A raw corpus JSONL record, as an ad-hoc runner would load it."""
    return {
        "bibtex_key": "ac3b1baad664",
        "bibtex_type": "inproceedings",
        "fields": {"title": "Once-for-All", "author": "Cai, Han", "year": "2023", "url": url},
        "raw_bibtex": None,
        "label": "HALLUCINATED",
    }


# --- The blind-list is declared once ---


class TestBlindList:
    def test_url_is_on_the_blind_list(self):
        assert "url" in BLIND_EXCLUDED_FIELDS

    def test_blind_fields_drops_only_listed_fields(self):
        out = blind_fields({"title": "T", "doi": "10.1/x", "url": "https://e.com/abc12345"})
        assert out == {"title": "T", "doi": "10.1/x"}

    def test_to_blind_follows_the_constant_not_a_hardcoded_pop(self, monkeypatch):
        """Extending the blind-list must change what to_blind() withholds."""
        monkeypatch.setattr(
            "hallmark.dataset.blinding.BLIND_EXCLUDED_FIELDS",
            frozenset({"url", "doi"}),
        )
        entry = make_entry_with_url()
        entry.fields["doi"] = "10.5555/1234567"
        blind = entry.to_blind()
        assert "doi" not in blind.fields
        assert "url" not in blind.fields


# --- An entry carrying a url is blinded before dispatch ---


class TestEntryIsBlindedBeforeDispatch:
    def test_to_blind_strips_url(self):
        entry = make_entry_with_url()
        assert entry.fields["url"]  # the corpus entry still stores it
        blind = entry.to_blind()
        assert "url" not in blind.fields

    def test_blinded_bibtex_carries_no_url_token(self):
        entry = make_entry_with_url()
        bibtex = entry.to_blind().to_bibtex()
        assert "22-0582.html" not in bibtex
        assert "url" not in bibtex

    def test_to_blind_leaves_the_source_entry_untouched(self):
        entry = make_entry_with_url()
        entry.to_blind()
        assert entry.fields["url"] == "https://jmlr.org/papers/v24/22-0582.html"

    def test_blind_entry_built_by_hand_still_strips_url(self):
        """The ablation-runner bypass: constructing BlindEntry from raw fields."""
        entry = make_entry_with_url()
        blind = BlindEntry(
            bibtex_key=entry.bibtex_key,
            bibtex_type=entry.bibtex_type,
            fields=dict(entry.fields),
            raw_bibtex=entry.raw_bibtex,
        )
        assert "url" not in blind.fields
        assert "22-0582.html" not in blind.to_bibtex()

    def test_openreview_query_id_is_stripped_too(self):
        """The id lives in the query string, which is where the leak was worst."""
        blind = make_entry_with_url("https://openreview.net/forum?id=HylxE1HKwS").to_blind()
        assert "HylxE1HKwS" not in blind.to_bibtex()


# --- raw_bibtex cannot smuggle a blinded field through ---


class TestRawBibtexCannotSmuggle:
    RAW = (
        "@inproceedings{faa1ea092c70,\n"
        "  title = {Bayesian Nonparametric Learning},\n"
        "  url = {https://jmlr.org/papers/v24/22-0582.html},\n"
        "  year = {2023},\n"
        "}"
    )

    def test_scrub_removes_the_url_assignment(self):
        out = scrub_bibtex(self.RAW)
        assert out is not None
        assert "22-0582.html" not in out
        assert "title = {Bayesian Nonparametric Learning}," in out
        assert "year = {2023}," in out

    def test_to_blind_scrubs_raw_bibtex(self):
        entry = make_entry_with_url()
        entry.raw_bibtex = self.RAW
        blind = entry.to_blind()
        assert "22-0582.html" not in blind.to_bibtex()

    def test_to_bibtex_prefers_raw_but_the_raw_is_clean(self):
        """to_bibtex() returns raw_bibtex verbatim, so raw_bibtex must be clean."""
        entry = make_entry_with_url()
        entry.raw_bibtex = self.RAW
        rendered = entry.to_blind().to_bibtex()
        assert rendered.startswith("@inproceedings{faa1ea092c70,")
        assert "url" not in rendered

    def test_quoted_and_bare_values_are_scrubbed(self):
        raw = '@article{k,\n  url = "https://e.com/abc12345",\n  year = 2023,\n}'
        out = scrub_bibtex(raw)
        assert out is not None and "abc12345" not in out and "year = 2023" in out

    def test_unparsable_value_nulls_the_string_rather_than_leaking(self):
        raw = "@article{k,\n  url = {https://e.com/abc12345,\n  year = {2023},\n}"
        assert scrub_bibtex(raw) is None

    def test_scrub_is_a_noop_when_there_is_nothing_to_strip(self):
        raw = "@article{k,\n  title = {T},\n}"
        assert scrub_bibtex(raw) == raw

    def test_none_and_empty_pass_through(self):
        assert scrub_bibtex(None) is None
        assert scrub_bibtex("") == ""


# --- The guard fires when a runner skips blinding ---


class TestGuard:
    def test_assert_blinded_raises_on_an_unblinded_record(self):
        with pytest.raises(BlindingViolationError, match="url"):
            assert_blinded(record_with_url(), context="test")

    def test_the_error_names_the_offending_entry_and_context(self):
        with pytest.raises(BlindingViolationError) as exc:
            assert_blinded(record_with_url(), context="my_runner")
        assert "my_runner" in str(exc.value)
        assert "ac3b1baad664" in str(exc.value)

    def test_assert_blinded_passes_on_a_blinded_record(self):
        assert_blinded(blind_record(record_with_url()), context="test")

    def test_assert_blinded_catches_url_hiding_in_raw_bibtex(self):
        rec = record_with_url()
        rec["fields"].pop("url")
        rec["raw_bibtex"] = "@article{k,\n  url = {https://e.com/abc12345},\n}"
        with pytest.raises(BlindingViolationError, match="raw_bibtex"):
            assert_blinded(rec, context="test")

    def test_to_bibtex_guard_catches_post_construction_mutation(self):
        blind = make_entry_with_url().to_blind()
        blind.fields["url"] = "https://e.com/abc12345"  # a runner mutating after blinding
        with pytest.raises(BlindingViolationError, match=r"BlindEntry\.to_bibtex"):
            blind.to_bibtex()

    def test_find_blind_violations_reports_nothing_when_clean(self):
        assert find_blind_violations(make_entry_with_url().to_blind()) == []

    def test_blind_record_blinds_fields_and_raw_bibtex(self):
        rec = record_with_url()
        rec["raw_bibtex"] = "@article{k,\n  url = {https://e.com/abc12345},\n  year = {2023},\n}"
        out = blind_record(rec)
        assert "url" not in out["fields"]
        assert "abc12345" not in (out["raw_bibtex"] or "")
        assert out["label"] == "HALLUCINATED"  # non-field keys are the caller's business

    def test_blind_record_does_not_mutate_the_input(self):
        rec = record_with_url()
        blind_record(rec)
        assert "url" in rec["fields"]


# --- The runner that had the demonstrated bypass ---


def _load_parallel_resume_module():
    path = REPO / "scripts" / "parallel_resume_test_public.py"
    spec = importlib.util.spec_from_file_location("parallel_resume_test_public", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestParallelResumeRunner:
    """The runner that read corpus JSONL directly and skipped to_blind()."""

    def test_entry_to_bibtex_refuses_an_unblinded_record(self):
        module = _load_parallel_resume_module()
        with pytest.raises(BlindingViolationError):
            module.entry_to_bibtex(record_with_url())

    def test_entry_to_bibtex_renders_a_blinded_record_without_the_url(self):
        module = _load_parallel_resume_module()
        bibtex = module.entry_to_bibtex(blind_record(record_with_url()))
        assert "HylxE1HKwS" not in bibtex
        assert "url" not in bibtex
        assert "title = {Once-for-All}," in bibtex

    def test_rendering_matches_the_registry_dispatch_path(self):
        """Prompt parity: the ad-hoc runner and BlindEntry must render alike."""
        module = _load_parallel_resume_module()
        entry = make_entry_with_url()
        via_runner = module.entry_to_bibtex(
            blind_record(
                {
                    "bibtex_key": entry.bibtex_key,
                    "bibtex_type": entry.bibtex_type,
                    "fields": dict(entry.fields),
                    "raw_bibtex": None,
                }
            )
        )
        assert via_runner == entry.to_blind().to_bibtex()
