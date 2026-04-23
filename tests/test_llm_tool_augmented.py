"""Tests for tool-augmented LLM baseline."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hallmark.baselines.llm_tool_augmented import (
    TOOL_AUGMENTED_PROMPT,
    format_tool_evidence,
    load_tool_evidence,
    save_tool_evidence,
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


# ---------------------------------------------------------------------------
# Priority 3 — save_tool_evidence branches
# ---------------------------------------------------------------------------


class TestSaveToolEvidence:
    def test_missing_binary_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When bibtex-check binary is absent, save_tool_evidence returns {}."""
        monkeypatch.setattr("shutil.which", lambda _: None)
        entries = [_make_entry()]
        output_path = tmp_path / "out.jsonl"
        result = save_tool_evidence(entries, output_path)
        assert result == {}
        assert not output_path.exists()

    def test_timeout_returns_partial_or_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TimeoutExpired during subprocess.run → warning logged, returns {}."""
        monkeypatch.setattr("shutil.which", lambda _: "/fake/bibtex-check")

        def _raise_timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd="bibtex-check", timeout=7200.0)

        monkeypatch.setattr("subprocess.run", _raise_timeout)
        entries = [_make_entry()]
        output_path = tmp_path / "out.jsonl"
        result = save_tool_evidence(entries, output_path)
        # No JSONL was produced, so result should be empty
        assert result == {}

    def test_successful_run_writes_and_returns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Successful subprocess.run that writes a JSONL file → parsed dict returned."""
        import shutil as _shutil

        output_path = tmp_path / "out.jsonl"
        record = {"key": "test2024", "status": "verified", "confidence": 0.95, "api_sources": []}

        def _fake_run(cmd, **_kw):
            # Write the JSONL to the path that would have been passed as --jsonl arg
            idx = cmd.index("--jsonl")
            jsonl = Path(cmd[idx + 1])
            jsonl.parent.mkdir(parents=True, exist_ok=True)
            jsonl.write_text(json.dumps(record) + "\n")
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("shutil.which", lambda _: "/fake/bibtex-check")
        monkeypatch.setattr("subprocess.run", _fake_run)

        # Also patch shutil.copy2 to copy from the tmp JSONL to output_path
        original_copy2 = _shutil.copy2

        def _copy2(src, dst):
            original_copy2(src, dst)

        monkeypatch.setattr("shutil.copy2", _copy2)

        entries = [_make_entry()]
        result = save_tool_evidence(entries, output_path)

        assert "test2024" in result
        assert result["test2024"]["status"] == "verified"
        assert output_path.exists()


# ---------------------------------------------------------------------------
# Priority 3 — verify_tool_augmented branching
# ---------------------------------------------------------------------------


class TestToolAugmentedMockFlowExtended:
    def _mock_openai(self, label: str = "HALLUCINATED", confidence: float = 0.9) -> MagicMock:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"label": label, "confidence": confidence, "reason": "mocked"}
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        return mock_openai

    def test_no_evidence_path_calls_save_tool_evidence(self, tmp_path: Path) -> None:
        """tool_evidence_path=None triggers live generation via save_tool_evidence."""
        pre_built_evidence = {
            "test2024": {
                "status": "not_found",
                "confidence": 0.1,
                "mismatched_fields": [],
                "api_sources": [],
            }
        }
        mock_openai = self._mock_openai()

        entries = [_make_entry()]
        with (
            patch.dict("sys.modules", {"openai": mock_openai}),
            patch(
                "hallmark.baselines.llm_tool_augmented.save_tool_evidence",
                return_value=pre_built_evidence,
            ) as mock_save,
        ):
            preds = verify_tool_augmented(
                entries,
                model="gpt-5.1",
                api_key="test-key",
                tool_evidence_path=None,
            )

        mock_save.assert_called_once()
        assert len(preds) == 1

    def test_missing_evidence_path_triggers_live_generation(self, tmp_path: Path) -> None:
        """Passing a path that does not exist triggers save_tool_evidence (live branch)."""
        nonexistent_path = tmp_path / "does_not_exist.jsonl"
        assert not nonexistent_path.exists()

        pre_built_evidence = {
            "test2024": {
                "status": "verified",
                "confidence": 0.95,
                "mismatched_fields": [],
                "api_sources": ["crossref"],
            }
        }
        mock_openai = self._mock_openai(label="VALID", confidence=0.95)

        entries = [_make_entry()]
        with (
            patch.dict("sys.modules", {"openai": mock_openai}),
            patch(
                "hallmark.baselines.llm_tool_augmented.save_tool_evidence",
                return_value=pre_built_evidence,
            ) as mock_save,
        ):
            preds = verify_tool_augmented(
                entries,
                model="gpt-5.1",
                api_key="test-key",
                tool_evidence_path=nonexistent_path,
            )

        # save_tool_evidence must have been called with the nonexistent path
        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert (
            call_args.args[1] == nonexistent_path
            or call_args.kwargs.get("output_path") == nonexistent_path
        )
        assert len(preds) == 1


# ---------------------------------------------------------------------------
# Bug-fix: verify_tool_augmented must forward **kwargs to _verify_with_openai_compatible
# ---------------------------------------------------------------------------


class TestToolAugmentedKwargsForwarding:
    """verify_tool_augmented(**kwargs) must pass extra kwargs to the inner call."""

    def test_extra_kwargs_forwarded(self, tmp_path: Path) -> None:
        """retry_failed (and any future kwarg) must reach _verify_with_openai_compatible."""
        from hallmark.dataset.schema import Prediction

        dummy_pred = Prediction(
            bibtex_key="test2024",
            label="VALID",
            confidence=0.9,
            reason="mocked",
        )

        with patch(
            "hallmark.baselines.llm_verifier._verify_with_openai_compatible",
            return_value=[dummy_pred],
        ) as mock_verify:
            verify_tool_augmented(
                [_make_entry()],
                model="gpt-5.1",
                api_key="test-key",
                retry_failed=True,
            )

        mock_verify.assert_called_once()
        _, call_kwargs = mock_verify.call_args
        assert call_kwargs.get("retry_failed") is True
