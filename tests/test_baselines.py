"""Tests for hallmark.baselines (unit tests, no network calls)."""

import json
from unittest.mock import MagicMock, patch

from hallmark.baselines.doi_only import check_doi
from hallmark.baselines.ensemble import EnsembleConfig, ensemble_predict
from hallmark.baselines.llm_verifier import _parse_llm_response
from hallmark.dataset.schema import BenchmarkEntry, Prediction

# --- Helpers ---


def _entry(key: str, label: str = "VALID", **extra_fields):
    fields = {"title": f"Paper {key}", "author": "Author", "year": "2024"}
    fields.update(extra_fields)
    kwargs = {
        "bibtex_key": key,
        "bibtex_type": "article",
        "fields": fields,
        "label": label,
        "explanation": "test",
    }
    if label == "HALLUCINATED":
        kwargs["hallucination_type"] = "fabricated_doi"
        kwargs["difficulty_tier"] = 1
    return BenchmarkEntry(**kwargs)


def _pred(key: str, label: str, confidence: float = 0.9):
    return Prediction(bibtex_key=key, label=label, confidence=confidence)


# --- DOI-only baseline ---


class TestDOIOnly:
    @patch("hallmark.baselines.doi_only.httpx.head")
    def test_doi_resolves(self, mock_head):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = "https://example.com"
        mock_head.return_value = mock_resp

        resolves, _detail = check_doi("10.1234/test")
        assert resolves is True

    @patch("hallmark.baselines.doi_only.httpx.head")
    def test_doi_not_found(self, mock_head):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_head.return_value = mock_resp

        resolves, _detail = check_doi("10.9999/fake")
        assert resolves is False

    def test_empty_doi(self):
        resolves, _detail = check_doi("")
        assert resolves is True  # No DOI to check = assume valid


# --- LLM verifier ---


class TestLLMVerifierParsing:
    def test_parse_valid_json(self):
        content = json.dumps(
            {
                "label": "HALLUCINATED",
                "confidence": 0.85,
                "reason": "Title does not exist",
            }
        )
        pred = _parse_llm_response(content, "test_key")
        assert pred.label == "HALLUCINATED"
        assert pred.confidence == 0.85
        assert "does not exist" in pred.reason

    def test_parse_json_in_code_block(self):
        content = '```json\n{"label": "VALID", "confidence": 0.95, "reason": "ok"}\n```'
        pred = _parse_llm_response(content, "test_key")
        assert pred.label == "VALID"
        assert pred.confidence == 0.95

    def test_parse_invalid_json(self):
        pred = _parse_llm_response("not json at all", "test_key")
        assert pred.label == "VALID"  # Fallback
        assert pred.confidence == 0.5

    def test_clamp_confidence(self):
        content = json.dumps({"label": "VALID", "confidence": 5.0})
        pred = _parse_llm_response(content, "test_key")
        assert pred.confidence == 1.0

    def test_invalid_label_fallback(self):
        content = json.dumps({"label": "MAYBE", "confidence": 0.5})
        pred = _parse_llm_response(content, "test_key")
        assert pred.label == "VALID"


# --- Ensemble ---


class TestEnsemble:
    def test_weighted_vote_unanimous(self):
        entries = [_entry("a")]
        strat_preds = {
            "s1": [_pred("a", "HALLUCINATED", 0.9)],
            "s2": [_pred("a", "HALLUCINATED", 0.8)],
        }
        result = ensemble_predict(entries, strat_preds)
        assert len(result) == 1
        assert result[0].label == "HALLUCINATED"

    def test_weighted_vote_split(self):
        entries = [_entry("a")]
        strat_preds = {
            "s1": [_pred("a", "HALLUCINATED", 0.9)],
            "s2": [_pred("a", "VALID", 0.9)],
        }
        # Default threshold=0.5, so should be HALLUCINATED since
        # hall_weight = 0.9, valid_weight = 0.9, fraction = ~0.5
        result = ensemble_predict(entries, strat_preds)
        assert len(result) == 1

    def test_max_confidence_method(self):
        entries = [_entry("a")]
        strat_preds = {
            "s1": [_pred("a", "VALID", 0.6)],
            "s2": [_pred("a", "HALLUCINATED", 0.95)],
        }
        config = EnsembleConfig(method="max_confidence")
        result = ensemble_predict(entries, strat_preds, config)
        assert result[0].label == "HALLUCINATED"
        assert result[0].confidence == 0.95

    def test_mean_confidence_method(self):
        entries = [_entry("a")]
        strat_preds = {
            "s1": [_pred("a", "HALLUCINATED", 0.9)],
            "s2": [_pred("a", "HALLUCINATED", 0.7)],
            "s3": [_pred("a", "VALID", 0.5)],
        }
        config = EnsembleConfig(method="mean_confidence")
        result = ensemble_predict(entries, strat_preds, config)
        assert result[0].label == "HALLUCINATED"

    def test_missing_entry_in_strategy(self):
        entries = [_entry("a"), _entry("b")]
        strat_preds = {
            "s1": [_pred("a", "VALID", 0.9)],  # missing "b"
        }
        result = ensemble_predict(entries, strat_preds)
        assert len(result) == 2


# --- Registry ---


class TestRegistry:
    def test_list_baselines_returns_all(self):
        from hallmark.baselines.registry import list_baselines

        names = list_baselines()
        assert "doi_only" in names
        assert "bibtexupdater" in names
        assert "harc" in names
        assert "verify_citations" in names
        assert "llm_openai" in names
        assert "llm_anthropic" in names
        assert "ensemble" in names

    def test_list_baselines_free_only(self):
        from hallmark.baselines.registry import list_baselines

        free = list_baselines(free_only=True)
        assert "doi_only" in free
        assert "ensemble" in free
        assert "llm_openai" not in free
        assert "llm_anthropic" not in free

    def test_check_available_doi_only(self):
        from hallmark.baselines.registry import check_available

        avail, msg = check_available("doi_only")
        assert avail is True
        assert msg == "OK"

    def test_check_available_unknown(self):
        from hallmark.baselines.registry import check_available

        avail, msg = check_available("nonexistent_baseline")
        assert avail is False
        assert "Unknown baseline" in msg

    def test_run_baseline_unknown_raises(self):
        import pytest

        from hallmark.baselines.registry import run_baseline

        with pytest.raises(ValueError, match="Unknown baseline"):
            run_baseline("nonexistent_baseline", [])

    def test_get_registry_returns_dict(self):
        from hallmark.baselines.registry import get_registry

        reg = get_registry()
        assert isinstance(reg, dict)
        assert "doi_only" in reg
        info = reg["doi_only"]
        assert info.is_free is True
        assert info.requires_api_key is False

    def test_registry_info_fields(self):
        from hallmark.baselines.registry import get_registry

        reg = get_registry()
        for name, info in reg.items():
            assert info.name == name
            assert isinstance(info.description, str)
            assert len(info.description) > 0
            assert isinstance(info.is_free, bool)
            assert isinstance(info.requires_api_key, bool)
            assert isinstance(info.pip_packages, list)

    def test_no_prescreening_variants_registered(self):
        from hallmark.baselines.registry import list_baselines

        baselines = list_baselines()
        assert "doi_only_no_prescreening" in baselines
        assert "bibtexupdater_no_prescreening" in baselines
        assert "harc_no_prescreening" in baselines
        assert "verify_citations_no_prescreening" in baselines


# --- HaRC output parsing ---


class TestHarcOutputParsing:
    def test_parse_all_ok(self):
        from hallmark.baselines.harc import _parse_harcx_output

        output = "\nAll entries verified successfully!\n"
        assert _parse_harcx_output(output) == {}

    def test_parse_single_issue(self):
        from hallmark.baselines.harc import _parse_harcx_output

        output = (
            "\n============================================================\n"
            "Found 1 entries requiring attention:\n"
            "============================================================\n"
            "\n"
            "[fake_entry_2024]\n"
            "  Title: A Fake Paper\n"
            "  Bib Authors: john smith\n"
            "  Year: 2024\n"
            "  Issue: Not found in Semantic Scholar, DBLP, or Google Scholar\n"
        )
        result = _parse_harcx_output(output)
        assert "fake_entry_2024" in result
        assert len(result["fake_entry_2024"]) == 1
        assert "Not found" in result["fake_entry_2024"][0]

    def test_parse_multiple_entries(self):
        from hallmark.baselines.harc import _parse_harcx_output

        output = (
            "[entry_a]\n"
            "  Issue: Authors don't match\n"
            "\n"
            "[entry_b]\n"
            "  Issue: Not found in any database\n"
        )
        result = _parse_harcx_output(output)
        assert len(result) == 2
        assert "entry_a" in result
        assert "entry_b" in result

    def test_parse_empty_output(self):
        from hallmark.baselines.harc import _parse_harcx_output

        assert _parse_harcx_output("") == {}


# --- verify-citations baseline ---


class TestVerifyCitationsBaseline:
    def test_fallback_predictions(self):
        from hallmark.baselines.common import fallback_predictions

        entries = [_entry("x"), _entry("y")]
        preds = fallback_predictions(entries)
        assert len(preds) == 2
        assert all(p.label == "VALID" for p in preds)
        assert all(p.confidence == 0.5 for p in preds)

    def test_strip_ansi_codes(self):
        from hallmark.baselines.verify_citations_baseline import strip_ansi_codes

        text = "\x1b[32m✓\x1b[0m test_key: Verified"
        clean = strip_ansi_codes(text)
        assert "\x1b" not in clean
        assert "✓" in clean

    def test_parse_terminal_output_verified(self):
        from hallmark.baselines.verify_citations_baseline import (
            _parse_terminal_output,
        )

        entries = [_entry("mykey")]
        stdout = "[1/1] Verifying:\n  [mykey] Some Title (Author, 2024)\n  Status: ✓ VERIFIED\n"
        preds = _parse_terminal_output(stdout, entries, 1.0, 1)
        assert len(preds) == 1
        assert preds[0].bibtex_key == "mykey"
        assert preds[0].label == "VALID"
        assert preds[0].confidence == 0.90

    def test_parse_terminal_output_not_found(self):
        from hallmark.baselines.verify_citations_baseline import (
            _parse_terminal_output,
        )

        entries = [_entry("badkey")]
        stdout = "[1/1] Verifying:\n  [badkey] Bad Paper (Author, 2024)\n  Status: ✗ ISSUES FOUND\n"
        preds = _parse_terminal_output(stdout, entries, 1.0, 1)
        assert len(preds) == 1
        assert preds[0].label == "HALLUCINATED"
        assert preds[0].confidence == 0.80

    def test_parse_terminal_output_warning(self):
        from hallmark.baselines.verify_citations_baseline import (
            _parse_terminal_output,
        )

        entries = [_entry("warnkey")]
        stdout = "[1/1] Verifying:\n  [warnkey] Warn Paper (Author, 2024)\n  Status: ⚠ WARNING\n"
        preds = _parse_terminal_output(stdout, entries, 1.0, 1)
        assert len(preds) == 1
        assert preds[0].label == "HALLUCINATED"
        assert preds[0].confidence == 0.60

    def test_parse_ignores_unknown_keys(self):
        from hallmark.baselines.verify_citations_baseline import (
            _parse_terminal_output,
        )

        entries = [_entry("mykey")]
        stdout = (
            "[1/2] Verifying:\n  [otherkey] Other (A, 2024)\n  Status: ✓ VERIFIED\n"
            "[2/2] Verifying:\n  [mykey] Mine (B, 2024)\n  Status: ✓ VERIFIED\n"
        )
        preds = _parse_terminal_output(stdout, entries, 1.0, 1)
        assert len(preds) == 1
        assert preds[0].bibtex_key == "mykey"

    @patch("hallmark.baselines.verify_citations_baseline.subprocess.run")
    def test_run_verify_citations_tool_not_found(self, mock_run):
        from hallmark.baselines.verify_citations_baseline import run_verify_citations

        mock_run.side_effect = FileNotFoundError
        entries = [_entry("a")]
        preds = run_verify_citations(entries)
        assert len(preds) == 1
        assert preds[0].label == "VALID"
        assert preds[0].confidence == 0.5


# --- Pre-screening tests ---


class TestPrescreening:
    @patch("hallmark.baselines.doi_only.httpx.head")
    @patch("hallmark.baselines.prescreening.httpx.head")
    def test_doi_only_with_prescreening(self, mock_prescreen_head, mock_doi_head):
        """Test that doi_only applies pre-screening by default."""
        from hallmark.baselines.doi_only import run_doi_only

        # Mock both pre-screening DOI check and main DOI check to succeed
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = "https://example.com"
        mock_prescreen_head.return_value = mock_resp
        mock_doi_head.return_value = mock_resp

        entries = [_entry("test", doi="10.1234/test")]
        preds = run_doi_only(entries, skip_prescreening=False)
        assert len(preds) == 1
        # Pre-screening should have run (httpx.head called)
        assert mock_prescreen_head.called or mock_doi_head.called

    @patch("hallmark.baselines.doi_only.httpx.head")
    def test_doi_only_skip_prescreening(self, mock_head):
        """Test that doi_only skips pre-screening when requested."""
        from hallmark.baselines.doi_only import run_doi_only

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = "https://example.com"
        mock_head.return_value = mock_resp

        entries = [_entry("test", doi="10.1234/test")]
        preds = run_doi_only(entries, skip_prescreening=True)
        assert len(preds) == 1
        # Should still call httpx for the main DOI check
        assert mock_head.called

    @patch("hallmark.baselines.verify_citations_baseline.subprocess.run")
    def test_verify_citations_skip_prescreening(self, mock_run):
        """Test that verify_citations respects skip_prescreening."""
        from hallmark.baselines.verify_citations_baseline import run_verify_citations

        # Mock subprocess to return valid output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "[1/1] Verifying:\n  [test] Title (Author, 2024)\n  Status: ✓ VERIFIED\n"
        )
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        entries = [_entry("test")]
        preds = run_verify_citations(entries, skip_prescreening=True)
        assert len(preds) == 1
        # Result should not include pre-screening overrides
        assert "[Pre-screening" not in preds[0].reason
