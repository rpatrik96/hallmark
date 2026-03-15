"""Tests for tool-augmented LLM baseline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hallmark.baselines.llm_tool_augmented import (
    TOOL_AUGMENTED_PROMPT,
    format_tool_evidence,
    load_tool_evidence,
    verify_tool_augmented,
)
from hallmark.dataset.schema import BlindEntry


def _make_entry(key: str = "test2024") -> BlindEntry:
    return BlindEntry(
        bibtex_key=key,
        bibtex_type="article",
        fields={"title": "Test Paper", "author": "Doe, J.", "year": "2024"},
        raw_bibtex=f"@article{{{key}, title={{Test Paper}}, author={{Doe, J.}}, year={{2024}}}}",
    )


class TestFormatToolEvidence:
    def test_verified(self) -> None:
        record = {
            "status": "verified",
            "confidence": 0.95,
            "api_sources": ["crossref", "dblp"],
            "mismatched_fields": [],
        }
        result = format_tool_evidence(record)
        assert "Status: verified" in result
        assert "Confidence: 0.95" in result
        assert "crossref" in result
        assert "Mismatched fields" not in result

    def test_mismatch(self) -> None:
        record = {
            "status": "title_mismatch",
            "confidence": 0.85,
            "mismatched_fields": ["title", "year"],
            "api_sources": ["crossref"],
        }
        result = format_tool_evidence(record)
        assert "title_mismatch" in result
        assert "title" in result
        assert "year" in result

    def test_api_error(self) -> None:
        record = {
            "status": "api_error",
            "errors": ["ConnectionTimeout"],
        }
        result = format_tool_evidence(record)
        assert "could not verify" in result.lower()
        assert "ConnectionTimeout" in result

    def test_empty_record(self) -> None:
        result = format_tool_evidence({})
        assert "No tool results available" in result

    def test_missing_optional_fields(self) -> None:
        record = {"status": "not_found"}
        result = format_tool_evidence(record)
        assert "Status: not_found" in result
        assert "Confidence" not in result


class TestLoadToolEvidence:
    def test_round_trip(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "evidence.jsonl"
        records = [
            {"key": "paper1", "status": "verified", "confidence": 0.95},
            {"key": "paper2", "status": "not_found", "confidence": 0.80},
        ]
        jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        loaded = load_tool_evidence(jsonl_path)
        assert len(loaded) == 2
        assert loaded["paper1"]["status"] == "verified"
        assert loaded["paper2"]["confidence"] == 0.80

    def test_empty_file(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")
        loaded = load_tool_evidence(jsonl_path)
        assert loaded == {}


class TestAugmentedPromptConstruction:
    def test_prompt_contains_both_sections(self) -> None:
        bibtex = "@article{test, title={Test}}"
        evidence = "Status: verified\nConfidence: 0.95"
        prompt = TOOL_AUGMENTED_PROMPT.format(bibtex=bibtex, tool_evidence=evidence)
        assert bibtex in prompt
        assert "Status: verified" in prompt
        assert "Confidence: 0.95" in prompt
        assert '"label"' in prompt
        assert '"confidence"' in prompt


class TestToolAugmentedRegistered:
    def test_in_registry(self) -> None:
        from hallmark.baselines.registry import get_registry

        registry = get_registry()
        assert "llm_tool_augmented" in registry
        info = registry["llm_tool_augmented"]
        assert info.confidence_type == "probabilistic"
        assert "openai" in info.pip_packages
        assert "bibtex-check" in info.cli_commands
        assert info.env_var == "OPENAI_API_KEY"
        assert info.requires_api_key is True
        assert info.is_free is False


class TestToolAugmentedMockFlow:
    def test_end_to_end_with_mock(self, tmp_path: Path) -> None:
        """Mock OpenAI client + mock evidence, verify end-to-end prediction."""
        # Write mock evidence
        evidence_path = tmp_path / "evidence.jsonl"
        evidence_records = [
            {
                "key": "test2024",
                "status": "title_mismatch",
                "confidence": 0.85,
                "mismatched_fields": ["title"],
                "api_sources": ["crossref"],
            },
        ]
        evidence_path.write_text("\n".join(json.dumps(r) for r in evidence_records) + "\n")

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "label": "HALLUCINATED",
                "confidence": 0.92,
                "reason": "Title mismatch confirmed by tool and semantic analysis",
            }
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        entries = [_make_entry()]
        with patch.dict("sys.modules", {"openai": mock_openai}):
            predictions = verify_tool_augmented(
                entries,
                model="gpt-5.1",
                api_key="test-key",
                tool_evidence_path=evidence_path,
            )

        assert len(predictions) == 1
        pred = predictions[0]
        assert pred.bibtex_key == "test2024"
        assert pred.label == "HALLUCINATED"
        assert pred.confidence == pytest.approx(0.92)

        # Verify the prompt sent to the LLM contained tool evidence
        call_args = mock_client.chat.completions.create.call_args
        prompt_sent = call_args.kwargs["messages"][0]["content"]
        assert "title_mismatch" in prompt_sent
        assert "crossref" in prompt_sent


class TestParseJsonlToRaw:
    def test_basic(self, tmp_path: Path) -> None:
        from hallmark.baselines.bibtexupdater import parse_jsonl_to_raw

        jsonl_path = tmp_path / "results.jsonl"
        records = [
            {"key": "a", "status": "verified", "confidence": 0.95, "api_sources": ["crossref"]},
            {"key": "b", "status": "not_found", "mismatched_fields": ["title"]},
        ]
        jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        result = parse_jsonl_to_raw(jsonl_path)
        assert len(result) == 2
        assert result["a"]["status"] == "verified"
        assert result["b"]["mismatched_fields"] == ["title"]

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        from hallmark.baselines.bibtexupdater import parse_jsonl_to_raw

        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text('{"key": "valid", "status": "verified"}\nnot json\n')

        result = parse_jsonl_to_raw(jsonl_path)
        assert len(result) == 1
        assert "valid" in result

    def test_skips_empty_key(self, tmp_path: Path) -> None:
        from hallmark.baselines.bibtexupdater import parse_jsonl_to_raw

        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text('{"key": "", "status": "verified"}\n{"key": "ok", "status": "ok"}\n')

        result = parse_jsonl_to_raw(jsonl_path)
        assert len(result) == 1
        assert "ok" in result
