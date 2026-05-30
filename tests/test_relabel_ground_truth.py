"""Unit tests for the deterministic ground-truth relabeller (no network).

Covers the pure decision logic, sub-test recomputation, and the VALID-entry
cleanup detector. Network resolution is exercised via synthetic resolution
dicts so these tests are fully offline and deterministic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# The relabeller lives in scripts/ (not an installed package); load it by path.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "relabel_ground_truth.py"
_spec = importlib.util.spec_from_file_location("relabel_ground_truth", _SCRIPT)
assert _spec is not None and _spec.loader is not None
rgt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rgt)


def _resolution(best, sim, *, errored=False, sources=None, arxiv_doi_id=None):
    cands = [best] if best else []
    # "Sources responded" is True whenever we were NOT blocked by a transient
    # error — even when nothing matched (a definitive 'looked, found nothing').
    return {
        "candidates": cands,
        "best": best,
        "best_sim": sim,
        "arxiv_doi_id": arxiv_doi_id,
        "errored": errored,
        "any_response": bool(cands),
        "authoritative_responded": not errored,
        "authoritative_errored": errored,
        "own_doi_resolves": None,
        "own_doi_record": None,
        "live_scoped": True,
    }


def _record(real_title, authors, year, source="arxiv", rid="x"):
    return {
        "source": source,
        "id": rid,
        "title": real_title,
        "authors": authors,
        "year": year,
        "venue": "arXiv",
        "doi": None,
    }


# --------------------------------------------------------------------------- #
# Text / author utilities
# --------------------------------------------------------------------------- #


def test_title_similarity_identical():
    assert rgt.title_similarity("Attention Is All You Need", "attention is all you need") == 1.0


def test_extract_last_names_both_formats():
    assert rgt.extract_last_names("Jane Doe and Smith, John") == {"doe", "smith"}


def test_author_overlap_partial():
    # entry has {zhang, roller}; real has {zhang, goyal} -> overlap 1/min(2,2)=0.5
    ov = rgt.author_overlap("Susan Zhang and Stephen Roller", ["Susan Zhang", "Naman Goyal"])
    assert ov == pytest.approx(0.5)


def test_parse_year():
    assert rgt.parse_year("Published 2022 at NeurIPS") == 2022
    assert rgt.parse_year("n/a") is None


# --------------------------------------------------------------------------- #
# Decision rules
# --------------------------------------------------------------------------- #


def _entry(title, author, year, doi=None, htype="plausible_fabrication"):
    fields = {"title": title, "author": author, "year": year}
    if doi:
        fields["doi"] = doi
    return {
        "bibtex_key": "k",
        "bibtex_type": "inproceedings",
        "label": "HALLUCINATED",
        "hallucination_type": htype,
        "fields": fields,
    }


def test_flip_valid_when_title_authors_year_match():
    entry = _entry(
        "OPT: Open Pre-trained Transformer Language Models",
        "Susan Zhang and Stephen Roller",
        "2022",
    )
    best = _record(
        "OPT: Open Pre-trained Transformer Language Models", ["Susan Zhang", "Stephen Roller"], 2022
    )
    d = rgt.decide(entry, _resolution(best, 1.0))
    assert d["decision"] == "flip_valid"
    assert d["new_label"] == "VALID"
    assert d["new_type"] is None


def test_retype_swapped_authors_when_title_matches_authors_dont():
    entry = _entry(
        "Diffusion Models Beat GANs on Image Synthesis", "Suresh Kumar and Vinoth S", "2021"
    )
    best = _record(
        "Diffusion Models Beat GANs on Image Synthesis", ["Prafulla Dhariwal", "Alex Nichol"], 2021
    )
    d = rgt.decide(entry, _resolution(best, 1.0))
    assert d["decision"] == "retype"
    assert d["new_type"] == rgt.HallucinationType.AUTHOR_MISMATCH.value


def test_retype_future_date_when_year_in_future():
    future = rgt.CURRENT_YEAR + 3
    entry = _entry("Some Real Paper Title Here", "Real Author and Other Author", str(future))
    best = _record("Some Real Paper Title Here", ["Real Author", "Other Author"], rgt.CURRENT_YEAR)
    d = rgt.decide(entry, _resolution(best, 1.0))
    assert d["decision"] == "retype"
    assert d["new_type"] == rgt.HallucinationType.FUTURE_DATE.value


def test_keep_fabrication_when_no_title_match():
    entry = _entry("A Totally Invented Paper About Nothing", "Nobody Real", "2022")
    best = _record("An Entirely Different Real Paper", ["Someone Else"], 2022)
    d = rgt.decide(entry, _resolution(best, 0.42))
    assert d["decision"] == "keep"
    assert d["new_type"] == "plausible_fabrication"


def test_keep_fabrication_when_no_candidate_but_sources_responded():
    entry = _entry("Invented", "Nobody", "2022")
    d = rgt.decide(entry, _resolution(None, 0.0, errored=False))
    assert d["decision"] == "keep"


def test_unresolved_when_all_sources_errored():
    entry = _entry("Real Paper That We Could Not Look Up", "Author", "2022")
    d = rgt.decide(entry, _resolution(None, 0.0, errored=True))
    assert d["decision"] == "unresolved"
    assert d["low_confidence"] is True


def test_year_drift_within_tolerance_still_flips():
    entry = _entry("A Real Paper", "Alice Smith and Bob Jones", "2022")
    best = _record("A Real Paper", ["Alice Smith", "Bob Jones"], 2021)  # delta 1
    d = rgt.decide(entry, _resolution(best, 1.0))
    assert d["decision"] == "flip_valid"


def test_year_drift_beyond_tolerance_retypes_version_mismatch():
    entry = _entry("A Real Paper", "Alice Smith and Bob Jones", "2025")
    best = _record("A Real Paper", ["Alice Smith", "Bob Jones"], 2020)  # delta 5, not future-of-now
    # Make sure 2025 is not in the future relative to CURRENT_YEAR for this test's intent;
    # if CURRENT_YEAR < 2025 this would be future_date, so guard:
    d = rgt.decide(entry, _resolution(best, 1.0))
    if rgt.CURRENT_YEAR < 2025:
        assert d["new_type"] == rgt.HallucinationType.FUTURE_DATE.value
    else:
        assert d["new_type"] == rgt.HallucinationType.ARXIV_VERSION_MISMATCH.value
    assert d["decision"] == "retype"


# --------------------------------------------------------------------------- #
# Author-faithfulness classification (BUG-1: truncation vs partial vs swap)
# --------------------------------------------------------------------------- #


def test_classify_authors_exact_set_is_faithful():
    cls = rgt.classify_authors("Susan Zhang and Stephen Roller", ["Susan Zhang", "Stephen Roller"])
    assert cls["verdict"] == "faithful"
    assert cls["set_equal"] is True


def test_classify_authors_truncated_prefix_is_faithful():
    # 3 correct leading authors + "and others"; real has 10. A FAITHFUL truncated
    # citation of a many-author paper — must NOT be a defect.
    real = [f"Author{i} Surname{i}" for i in range(10)]
    real[0], real[1], real[2] = "Alice Zhang", "Bob Roller", "Carol Goyal"
    entry = "Alice Zhang and Bob Roller and Carol Goyal and others"
    cls = rgt.classify_authors(entry, real)
    assert cls["verdict"] == "faithful"
    assert cls["truncated"] is True


def test_classify_authors_et_al_marker_is_faithful():
    real = [f"First{i} Last{i}" for i in range(8)]
    real[0], real[1] = "Jane Doe", "John Smith"
    cls = rgt.classify_authors("Jane Doe and John Smith and et al.", real)
    assert cls["verdict"] == "faithful"
    assert cls["truncated"] is True


def test_classify_authors_diacritic_latex_normalized_is_faithful():
    # LaTeX-escaped vs Unicode accents must collapse (Schärli / Rozière cohort).
    real = ["Denny Zhou", "Nathanael Schärli", "Le Hou"]
    entry = r"Denny Zhou and Nathanael Sch{\"a}rli and Le Hou"
    cls = rgt.classify_authors(entry, real)
    assert cls["verdict"] == "faithful"


def test_classify_authors_partial_no_marker_is_partial():
    # First + last author kept, middle dropped, NO marker -> partial_author_list.
    real = ["Alice First", "Bob Mid", "Carol Mid2", "Dave Last"]
    cls = rgt.classify_authors("Alice First and Dave Last", real)
    assert cls["verdict"] == "partial_no_marker"
    assert cls["truncated"] is False


def test_classify_authors_swapped_is_different():
    # Authors from a different paper entirely -> swapped/different.
    real = ["Prafulla Dhariwal", "Alex Nichol"]
    cls = rgt.classify_authors("Suresh Kumar and Vinoth Selvam", real)
    assert cls["verdict"] == "different"


# --------------------------------------------------------------------------- #
# Over-flip guards: list length alone is NOT faithfulness. Each shape below was
# wrongly admitted as ``faithful`` by the bare long-list / set-equal shortcut and
# must now resolve to an author defect (``different``). Mirrors the 5 confirmed
# false flips (wrong-lead, fabricated-insertion, collision-masked substitution).
# --------------------------------------------------------------------------- #


def test_classify_authors_wrong_lead_is_different():
    # Make-A-Video shape: a cherry-picked list that DROPS the true first author
    # (Singer) and leads with a non-lead author (Gafni). Mostly real names, long
    # real list — but the lead is wrong, so it is NOT faithful.
    real = [
        "Singer, Uriel",
        "Polyak, Adam",
        "Hayes, Thomas",
        "Yin, Xi",
        "An, Jie",
        "Zhang, Songyang",
        "Hu, Qiyuan",
        "Yang, Harry",
        "Ashual, Oron",
        "Gafni, Oran",
        "Parikh, Devi",
        "Taigman, Yaniv",
        "Singer, U.",
    ]
    entry = (
        "Oran Gafni and Adam Polyak and Oron Ashual and Shelly Sheynin and "
        "Devi Parikh and Yaniv Taigman"
    )
    cls = rgt.classify_authors(entry, real)
    assert cls["verdict"] == "different"
    assert cls["lead_ok"] is False


def test_classify_authors_fabricated_insertion_shallow_is_different():
    # Open Problems RLHF shape: correct lead (Casper) but a FABRICATED author
    # (Halpern, not in the 32-author real list) appears right after a SHALLOW
    # verified prefix. Long real list must not rescue it.
    real = [
        "Casper, Stephen",
        "Davies, Xander",
        "Shi, Claudia",
        "Gilbert, Thomas",
        "Scheurer, Jeremy",
        "Rando, Javier",
        "Freedman, Rachel",
        "Korbak, Tomasz",
        "Krueger, David",
        "Hadfield-Menell, Dylan",
        "Dragan, Anca",
    ] + [f"Extra{i} Author{i}" for i in range(21)]
    entry = (
        "Stephen Casper and Rachel Freedman and Yoni Halpern and David Krueger and "
        "Dylan Hadfield-Menell and Anca D. Dragan"
    )
    cls = rgt.classify_authors(entry, real)
    assert cls["verdict"] == "different"


def test_classify_authors_fabricated_insertion_after_midprefix_is_different():
    # Discovering LM Behaviors shape: first 5 are a real prefix, but a fabricated
    # author (Steinhardt) sits at position 7 while the given-aware verified prefix
    # is short — a cherry-pick, not a faithful truncation.
    real = [
        "Perez, Ethan",
        "Ringer, Sam",
        "Lukosiute, Kamile",
        "Nguyen, Karina",
        "Chen, Edwin",
        "Heiner, Scott",
        "Pettit, Craig",
        "Olsson, Catherine",
    ] + [f"More{i} Person{i}" for i in range(55)]
    entry = (
        "Perez, Ethan and Ringer, Sam and Lukosiute, Krista and Nguyen, Kamile and "
        "Chen, Daphne and Jones, Andrew and Bowman, Samuel R. and "
        "Steinhardt, Jacob and Olsson, Catherine"
    )
    cls = rgt.classify_authors(entry, real)
    assert cls["verdict"] == "different"


def test_classify_authors_collision_masked_lead_substitution_is_different():
    # Least-to-Most shape: last name 'Zhou' collides, masking a LEAD substitution
    # (entry 'Shunyu Zhou' vs real 'Denny Zhou'); 'Chung' is also inserted. The
    # given-name corroboration on the lead catches it despite the collision.
    real = [
        "Zhou, Denny",
        "Scharli, Nathanael",
        "Hou, Le",
        "Wei, Jason",
        "Scales, Nathan",
        "Wang, Xuezhi",
        "Schuurmans, Dale",
        "Cui, Claire",
        "Bousquet, Olivier",
        "Le, Quoc",
        "Chi, Ed",
    ]
    entry = (
        "Shunyu Zhou and Nathanael Scharli and Le Hou and Jason Wei and "
        "Nathan Scales and Hyung Won Chung and Xuezhi Wang and Quoc V. Le and "
        "Ed H. Chi and Dale Schuurmans and Denny Zhou"
    )
    cls = rgt.classify_authors(entry, real)
    assert cls["verdict"] == "different"
    assert cls["lead_given_conflict"] is True


def test_classify_authors_collision_masked_interior_substitution_is_different():
    # ConvMAE shape: same-length set match by last name, but a collision on 'Lin'
    # masks a substitution (entry 'Dahua Lin' vs real 'Ziyi Lin') at position 4,
    # plus an extra 'Hongyang Li'. The given-aware prefix breaks at the masked sub.
    real = [
        "Gao, Peng",
        "Ma, Teli",
        "Li, Hongsheng",
        "Lin, Ziyi",
        "Dai, Jifeng",
        "Qiao, Yu",
    ]
    entry = (
        "Peng Gao and Teli Ma and Hongsheng Li and Dahua Lin and Yu Qiao and "
        "Jifeng Dai and Hongyang Li"
    )
    cls = rgt.classify_authors(entry, real)
    assert cls["verdict"] == "different"


# --------------------------------------------------------------------------- #
# Cohort shapes that MUST stay faithful (genuine truncated / reordered citations
# of many-author landmark papers). The over-flip guards keep the diacritic /
# initials / nickname tolerance and the deep verified prefix.
# --------------------------------------------------------------------------- #


def test_classify_authors_truncated_cohort_with_extra_real_tail_is_faithful():
    # OPT shape: 10 correct leading authors + "and others"; the resolver's real
    # list is reordered/abbreviated so a couple of entry names are absent from it,
    # but the truncation marker + a deep verified prefix make it faithful.
    real = [
        "Zhang, Susan",
        "Roller, Stephen",
        "Goyal, Naman",
        "Artetxe, Mikel",
        "Chen, Moya",
        "Chen, Shuohui",
        "Dewan, Christopher",
        "Diab, Mona",
    ] + [f"Tail{i} Member{i}" for i in range(11)]
    entry = (
        "Susan Zhang and Stephen Roller and Naman Goyal and Mikel Artetxe and "
        "Moya Chen and Shuohui Chen and Mona Dewan and Mona Diab and "
        "Emily Dinan and Zhiguo Du and others"
    )
    cls = rgt.classify_authors(entry, real)
    assert cls["verdict"] == "faithful"
    assert cls["truncated"] is True


def test_classify_authors_deep_reorder_with_late_absences_is_faithful():
    # PaLM/LLaMA-2 shape: a near-complete list of a 60+ author paper. A deep
    # given-aware verified prefix corroborates it; later names that the truncated
    # real list reorders/omits sit AFTER the verified prefix, so they are not
    # fabrications. The entry's later authors (Sutton/Gehrmann/Le) ARE real, just
    # positioned past where the resolver's reordered list happens to place them.
    # Real list: a clean 10-deep lead the entry reproduces exactly, then a long
    # reordered tail that the resolver returns in a DIFFERENT order than the entry
    # (so a few entry tail-names are absent from the real set, all AFTER pos 10).
    lead = [
        "Alayrac, Jean-Baptiste",
        "Donahue, Jeff",
        "Luc, Pauline",
        "Miech, Antoine",
        "Barr, Iain",
        "Hasson, Yana",
        "Lenc, Karel",
        "Mensch, Arthur",
        "Millican, Katie",
        "Reynolds, Malcolm",
    ]
    real = lead + [f"Tail{i} Person{i}" for i in range(17)]
    entry = " and ".join(
        [
            "Jean-Baptiste Alayrac",
            "Jeff Donahue",
            "Pauline Luc",
            "Antoine Miech",
            "Iain Barr",
            "Yana Hasson",
            "Karel Lenc",
            "Arthur Mensch",
            "Katie Millican",
            "Malcolm Reynolds",
            # tail names NOT in the resolver's reordered real list -> absent, but
            # they sit AFTER the deep verified prefix, so not fabricated insertions.
            # Kept to <=25% of the list so subset-dominance still holds.
            "Serkan Cabi",
            "Tengda Han",
            "Zhitao Gong",
        ]
    )
    cls = rgt.classify_authors(entry, real)
    assert cls["verdict"] == "faithful"
    assert cls["gcontig"] >= rgt.AUTHOR_VERIFY_RUN


def test_classify_authors_nickname_given_variant_is_faithful():
    # InstructGPT shape: 'Wu, Jeffrey' (entry) vs 'Wu, Jeff' (real) is the SAME
    # person — a nickname/prefix variant must not be read as a substitution.
    real = ["Ouyang, Long", "Wu, Jeff", "Jiang, Xu", "Almeida, Diogo", "Wainwright, Carroll L."]
    entry = "Long Ouyang and Jeffrey Wu and Xu Jiang and Diogo Almeida and Carroll Wainwright"
    cls = rgt.classify_authors(entry, real)
    assert cls["verdict"] == "faithful"


def test_flip_valid_for_truncated_faithful_long_list():
    # End-to-end: title exact + truncated faithful prefix + arXiv DOI -> flip.
    real_authors = [f"Real{i} Name{i}" for i in range(20)]
    real_authors[0], real_authors[1] = "Susan Zhang", "Stephen Roller"
    entry = _entry(
        "OPT: Open Pre-trained Transformer Language Models",
        "Susan Zhang and Stephen Roller and others",
        "2022",
        doi="10.48550/arXiv.2205.01068",
    )
    best = _record(
        "OPT: Open Pre-trained Transformer Language Models", real_authors, 2022, source="datacite"
    )
    res = _resolution(best, 1.0)
    res["arxiv_doi_id"] = "2205.01068"
    d = rgt.decide(entry, res)
    assert d["decision"] == "flip_valid"
    assert d["new_label"] == "VALID"
    assert d["match_scores"]["author_faithful"] is True


def test_partial_author_list_does_not_flip():
    # A marker-less partial list of a real paper -> retype partial_author_list,
    # NEVER flip to VALID (it is a genuine author defect).
    real_authors = ["Alice First", "Bob Mid", "Carol Mid2", "Dave Last"]
    entry = _entry("A Real Multi Author Paper", "Alice First and Dave Last", "2022")
    best = _record("A Real Multi Author Paper", real_authors, 2022)
    d = rgt.decide(entry, _resolution(best, 1.0))
    assert d["decision"] == "retype"
    assert d["new_type"] == rgt.HallucinationType.PARTIAL_AUTHOR_LIST.value


def test_arxiv_doi_publisher_does_not_block_flip_on_venue():
    # BUG-2/venue false-negative: a real venue (entry) vs arXiv-DataCite publisher
    # "arXiv" must NOT count as a venue mismatch — the flip still goes through.
    entry = _entry(
        "Flamingo: a Visual Language Model for Few-Shot Learning",
        "Jean-Baptiste Alayrac and Jeff Donahue and others",
        "2022",
        doi="10.48550/arXiv.2204.14198",
    )
    entry["fields"]["booktitle"] = "NeurIPS"
    real_authors = [f"R{i} N{i}" for i in range(27)]
    real_authors[0], real_authors[1] = "Jean-Baptiste Alayrac", "Jeff Donahue"
    best = {
        "source": "datacite",
        "id": "10.48550/arXiv.2204.14198",
        "title": "Flamingo: a Visual Language Model for Few-Shot Learning",
        "authors": real_authors,
        "year": 2022,
        "venue": "arXiv",  # arXiv publisher — NOT a venue authority
        "doi": "10.48550/arXiv.2204.14198",
    }
    res = _resolution(best, 1.0)
    res["arxiv_doi_id"] = "2204.14198"
    d = rgt.decide(entry, res)
    assert d["decision"] == "flip_valid"
    assert d["match_scores"]["venue_refuted"] is False


def test_authors_match_but_venue_wrong_retypes_wrong_venue_not_swapped():
    # BUG-2 branch-ordering (ImageBind): authors set-equal, real venue from a TRUE
    # source (OpenAlex) differs -> wrong_venue, NEVER swapped_authors.
    entry = _entry(
        "ImageBind: One Embedding Space To Bind Them All",
        "Rohit Girdhar and Alaaeldin El-Nouby",
        "2023",
    )
    entry["fields"]["booktitle"] = "ECCV"  # wrong; real is CVPR
    best = {
        "source": "openalex",
        "id": "W123",
        "title": "ImageBind: One Embedding Space To Bind Them All",
        "authors": ["Rohit Girdhar", "Alaaeldin El-Nouby"],
        "year": 2023,
        "venue": "2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
        "doi": None,
    }
    d = rgt.decide(entry, _resolution(best, 1.0))
    assert d["decision"] == "retype"
    assert d["new_type"] == rgt.HallucinationType.WRONG_VENUE.value


def test_scoped_cache_only_deliberate_keeps_not_unresolved():
    # SCOPING: a perturbation entry resolved cache-only with no candidate is KEPT
    # (HALLUCINATED by construction), never UNRESOLVED.
    entry = _entry("Some Real Paper", "Donor Authors", "2022", htype="swapped_authors")
    entry["generation_method"] = "perturbation"
    res = _resolution(None, 0.0)
    res["live_scoped"] = False
    d = rgt.decide(entry, res)
    assert d["decision"] == "keep"
    assert d["new_label"] == "HALLUCINATED"


# --------------------------------------------------------------------------- #
# Transient-cache purge (cache-poisoning guard)
# --------------------------------------------------------------------------- #


def test_purge_transient_cache_drops_429_and_timeout_keeps_body_and_404():
    cache = {
        "GET::a": {"body": "<xml/>", "error": None},  # keep (success)
        "GET::b": {"body": None, "error": "HTTP 404"},  # keep (definitive)
        "GET::c": {"body": None, "error": "HTTP 429"},  # purge (transient)
        "GET::d": {"body": None, "error": "HTTP 503"},  # purge (transient)
        "GET::e": {"body": None, "error": "TimeoutError: timed out"},  # purge
    }
    n = rgt.purge_transient_cache(cache)
    assert n == 3
    assert set(cache.keys()) == {"GET::a", "GET::b"}


def test_needs_live_resolution_scopes_correctly():
    assert rgt.needs_live_resolution(
        {"generation_method": "llm_generated", "hallucination_type": "swapped_authors"}
    )
    assert rgt.needs_live_resolution(
        {"generation_method": "perturbation", "hallucination_type": "plausible_fabrication"}
    )
    assert not rgt.needs_live_resolution(
        {"generation_method": "perturbation", "hallucination_type": "swapped_authors"}
    )
    assert rgt.needs_live_resolution(
        {"generation_method": "real_world", "hallucination_type": "placeholder_authors"}
    )


# --------------------------------------------------------------------------- #
# Sub-test recomputation
# --------------------------------------------------------------------------- #


def test_subtests_for_valid_arxiv_doi_resolves_true():
    fields = {"title": "T", "author": "A", "year": "2022", "doi": "10.48550/arXiv.2205.01068"}
    st = rgt.subtests_for_valid(fields, {"doi": "10.48550/arXiv.2205.01068"})
    assert st["doi_resolves"] is True
    assert st["title_exists"] is True
    assert st["authors_match"] is True
    assert st["cross_db_agreement"] is True


def test_subtests_for_valid_no_doi_is_none():
    fields = {"title": "T", "author": "A", "year": "2022"}
    st = rgt.subtests_for_valid(fields, {})
    assert st["doi_resolves"] is None


def test_subtests_for_type_swapped_authors():
    fields = {"title": "T", "author": "A", "year": "2022"}
    st = rgt.subtests_for_type(rgt.HallucinationType.AUTHOR_MISMATCH.value, fields)
    assert st["title_exists"] is True
    assert st["authors_match"] is False


def test_subtests_for_fabrication():
    st = rgt.subtests_for_fabrication({"title": "T", "author": "A", "year": "2022"})
    assert st["title_exists"] is False
    assert st["authors_match"] is False


# --------------------------------------------------------------------------- #
# VALID-entry cleanup detection
# --------------------------------------------------------------------------- #


def test_valid_cleanup_flagged_when_retains_hallucination_type():
    entry = {"label": "VALID", "hallucination_type": "swapped_authors", "subtests": {}}
    assert rgt.valid_needs_cleanup(entry) is True


def test_valid_cleanup_flagged_when_relabeled_with_stale_subtests():
    entry = {
        "label": "VALID",
        "relabeled_by": "mislabel-audit-2026-05-29",
        "subtests": {"title_exists": False, "authors_match": False},
    }
    assert rgt.valid_needs_cleanup(entry) is True


def test_valid_cleanup_not_flagged_for_benign_valid():
    # legitimate VALID entry: cross_db_agreement False is benign, no relabel marker
    entry = {"label": "VALID", "subtests": {"cross_db_agreement": False, "title_exists": True}}
    assert rgt.valid_needs_cleanup(entry) is False


def test_valid_cleanup_not_flagged_for_hallucinated():
    entry = {"label": "HALLUCINATED", "hallucination_type": "swapped_authors", "subtests": {}}
    assert rgt.valid_needs_cleanup(entry) is False


def test_cleanup_valid_record_drops_type_and_fixes_subtests():
    rec = {
        "label": "VALID",
        "hallucination_type": "swapped_authors",
        "difficulty_tier": 2,
        "fields": {"title": "T", "author": "A", "year": "2022", "doi": "10.48550/arXiv.1234.5678"},
        "subtests": {"title_exists": False, "authors_match": False, "doi_resolves": False},
    }
    rgt.cleanup_valid_record(rec)
    assert "hallucination_type" not in rec
    assert "difficulty_tier" not in rec
    assert rec["subtests"]["title_exists"] is True
    assert rec["subtests"]["authors_match"] is True
    assert rec["subtests"]["doi_resolves"] is True  # arXiv DOI


# --------------------------------------------------------------------------- #
# apply_decision_to_record produces schema-valid records
# --------------------------------------------------------------------------- #


def test_apply_flip_valid_produces_loadable_entry():
    from hallmark.dataset.schema import BenchmarkEntry

    rec = _entry("OPT", "Susan Zhang and Stephen Roller", "2022", doi="10.48550/arXiv.2205.01068")
    rec["difficulty_tier"] = 3
    rec["subtests"] = {}
    best = _record("OPT", ["Susan Zhang", "Stephen Roller"], 2022)
    res = _resolution(best, 1.0, arxiv_doi_id="2205.01068")
    d = rgt.decide(rec, res)
    rgt.apply_decision_to_record(rec, d, res)
    assert rec["label"] == "VALID"
    assert "hallucination_type" not in rec
    # The schema loader must accept the result without raising.
    entry = BenchmarkEntry.from_dict(rec)
    assert entry.label == "VALID"


def test_apply_retype_produces_loadable_entry():
    from hallmark.dataset.schema import BenchmarkEntry

    rec = _entry("Diffusion Models Beat GANs", "Wrong Person", "2021")
    rec["difficulty_tier"] = 3
    rec["subtests"] = {}
    best = _record("Diffusion Models Beat GANs", ["Prafulla Dhariwal", "Alex Nichol"], 2021)
    res = _resolution(best, 1.0)
    d = rgt.decide(rec, res)
    rgt.apply_decision_to_record(rec, d, res)
    assert rec["label"] == "HALLUCINATED"
    assert rec["hallucination_type"] == rgt.HallucinationType.AUTHOR_MISMATCH.value
    assert rec["difficulty_tier"] == 2
    entry = BenchmarkEntry.from_dict(rec)
    assert entry.label == "HALLUCINATED"
