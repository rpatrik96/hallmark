"""Tests for citebench.baselines (unit tests, no network calls)."""

import json
from unittest.mock import MagicMock, patch

from citebench.baselines.doi_only import check_doi
from citebench.baselines.ensemble import EnsembleConfig, ensemble_predict
from citebench.baselines.llm_verifier import _parse_llm_response
from citebench.dataset.schema import BenchmarkEntry, Prediction

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
    @patch("citebench.baselines.doi_only.httpx.head")
    def test_doi_resolves(self, mock_head):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = "https://example.com"
        mock_head.return_value = mock_resp

        resolves, detail = check_doi("10.1234/test")
        assert resolves is True

    @patch("citebench.baselines.doi_only.httpx.head")
    def test_doi_not_found(self, mock_head):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_head.return_value = mock_resp

        resolves, detail = check_doi("10.9999/fake")
        assert resolves is False

    def test_empty_doi(self):
        resolves, detail = check_doi("")
        assert resolves is True  # No DOI to check = assume valid


# --- LLM verifier ---


class TestLLMVerifierParsing:
    def test_parse_valid_json(self):
        content = json.dumps({
            "label": "HALLUCINATED",
            "confidence": 0.85,
            "reason": "Title does not exist",
        })
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
