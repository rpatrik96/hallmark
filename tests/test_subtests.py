"""Tests for hallmark.evaluation.subtests."""


from hallmark.evaluation.subtests import (
    _extract_last_names,
    _normalize_title,
    _required_fields_for_type,
    check_authors_match,
    check_cross_db_agreement,
    check_fields_complete,
    check_title_exists,
    check_venue_real,
)


class TestNormalizeTitle:
    def test_lowercase(self):
        assert _normalize_title("Hello World") == "hello world"

    def test_strip_latex(self):
        assert _normalize_title(r"\textbf{Bold} and \emph{italic}") == "bold and italic"

    def test_strip_braces(self):
        assert _normalize_title("{A} {Title}") == "a title"

    def test_strip_punctuation(self):
        assert _normalize_title("Hello, World!") == "hello world"

    def test_collapse_whitespace(self):
        assert _normalize_title("hello   world") == "hello world"


class TestExtractLastNames:
    def test_family_given_format(self):
        names = _extract_last_names("Vaswani, Ashish and Shazeer, Noam")
        assert "vaswani" in names
        assert "shazeer" in names

    def test_given_family_format(self):
        names = _extract_last_names("Ashish Vaswani and Noam Shazeer")
        assert "vaswani" in names
        assert "shazeer" in names

    def test_single_author(self):
        names = _extract_last_names("John Smith")
        assert "smith" in names

    def test_empty_string(self):
        assert _extract_last_names("") == set()


class TestRequiredFields:
    def test_article(self):
        fields = _required_fields_for_type("article")
        assert "author" in fields
        assert "title" in fields
        assert "journal" in fields
        assert "year" in fields

    def test_inproceedings(self):
        fields = _required_fields_for_type("inproceedings")
        assert "booktitle" in fields

    def test_unknown_type(self):
        fields = _required_fields_for_type("unknown")
        assert "author" in fields
        assert "title" in fields
        assert "year" in fields


class TestCheckTitleExists:
    def test_exact_match(self):
        results = [{"title": "Attention Is All You Need"}]
        r = check_title_exists("Attention Is All You Need", results)
        assert r.passed is True
        assert r.score >= 0.9

    def test_fuzzy_match(self):
        results = [{"title": "Attention is All You Need"}]  # different case
        r = check_title_exists("attention is all you need", results)
        assert r.passed is True

    def test_no_match(self):
        results = [{"title": "Completely Different Paper"}]
        r = check_title_exists("Attention Is All You Need", results)
        assert r.passed is False

    def test_no_results(self):
        r = check_title_exists("Test Title", None)
        assert r.passed is None

    def test_empty_title(self):
        r = check_title_exists("", [{"title": "Something"}])
        assert r.passed is None


class TestCheckAuthorsMatch:
    def test_exact_match(self):
        r = check_authors_match(
            "Vaswani, Ashish and Shazeer, Noam",
            ["vaswani", "shazeer"],
        )
        assert r.passed is True

    def test_partial_match(self):
        r = check_authors_match(
            "Vaswani, Ashish and Shazeer, Noam and Parmar, Niki",
            ["vaswani", "shazeer"],
            threshold=0.5,
        )
        assert r.passed is True

    def test_no_match(self):
        r = check_authors_match(
            "John Doe and Jane Smith",
            ["vaswani", "shazeer"],
        )
        assert r.passed is False

    def test_no_api_authors(self):
        r = check_authors_match("Vaswani, Ashish", None)
        assert r.passed is None


class TestCheckVenueReal:
    def test_known_venue(self):
        r = check_venue_real(
            "NeurIPS",
            known_venues={"NeurIPS", "ICML", "ICLR"},
        )
        assert r.passed is True

    def test_api_match(self):
        r = check_venue_real(
            "Neural Information Processing Systems",
            api_venue="Advances in Neural Information Processing Systems",
        )
        assert r.passed is True

    def test_unknown_venue(self):
        r = check_venue_real(
            "International Conference on Advanced AI Systems",
            known_venues={"NeurIPS", "ICML", "ICLR"},
        )
        assert r.passed is False

    def test_no_reference_data(self):
        r = check_venue_real("Some Venue")
        assert r.passed is None


class TestCheckFieldsComplete:
    def test_all_present(self):
        fields = {
            "author": "John Smith",
            "title": "A Paper",
            "journal": "Nature",
            "year": "2024",
        }
        r = check_fields_complete("article", fields)
        assert r.passed is True

    def test_missing_required(self):
        fields = {"title": "A Paper", "year": "2024"}
        r = check_fields_complete("article", fields)
        assert r.passed is False
        assert "Missing" in r.detail

    def test_malformed_year(self):
        fields = {
            "author": "A",
            "title": "T",
            "journal": "J",
            "year": "twenty24",
        }
        r = check_fields_complete("article", fields)
        assert r.passed is False
        assert "Malformed" in r.detail

    def test_malformed_doi(self):
        fields = {
            "author": "A",
            "title": "T",
            "journal": "J",
            "year": "2024",
            "doi": "not-a-doi",
        }
        r = check_fields_complete("article", fields)
        assert r.passed is False


class TestCheckCrossDbAgreement:
    def test_agreement(self):
        results = {
            "crossref": {"title": "Attention Is All You Need"},
            "dblp": {"title": "Attention Is All You Need"},
        }
        r = check_cross_db_agreement(results)
        assert r.passed is True
        assert r.score >= 0.9

    def test_disagreement(self):
        results = {
            "crossref": {"title": "Attention Is All You Need"},
            "dblp": {"title": "A Completely Different Paper Title"},
        }
        r = check_cross_db_agreement(results)
        assert r.passed is False

    def test_single_source(self):
        results = {"crossref": {"title": "Attention Is All You Need"}}
        r = check_cross_db_agreement(results)
        assert r.passed is None
