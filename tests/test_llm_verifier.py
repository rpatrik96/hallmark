"""Unit tests for the llm_verifier baseline.

Focuses on the cutoff-aware prompt ablation introduced for the temporal
robustness analysis (H2).  All API calls are mocked — no live keys consumed
in CI.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from hallmark.baselines.llm_verifier import (
    CUTOFF_AWARE_ADDENDUM,
    VERIFICATION_PROMPT,
    _build_verification_prompt,
    _parse_llm_response,
    verify_with_anthropic,
    verify_with_openai,
)
from hallmark.dataset.schema import BlindEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(key: str = "test2024") -> BlindEntry:
    return BlindEntry(
        bibtex_key=key,
        bibtex_type="article",
        fields={
            "title": "Deep Learning for Citation Verification",
            "author": "Smith, John and Doe, Jane",
            "year": "2024",
            "doi": "10.1234/fake",
        },
        raw_bibtex=(
            f"@article{{{key}, title={{Deep Learning for Citation Verification}}, "
            f"author={{Smith, John and Doe, Jane}}, year={{2024}}}}"
        ),
    )


def _make_openai_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_anthropic_response(content: str) -> MagicMock:
    block = MagicMock()
    block.text = content
    resp = MagicMock()
    resp.content = [block]
    return resp


# ---------------------------------------------------------------------------
# Cutoff-aware prompt ablation
# ---------------------------------------------------------------------------


class TestCutoffAwarePrompt:
    """Tests for the cutoff-aware prompt addendum and its wiring."""

    def test_addendum_contains_key_phrases(self) -> None:
        """The addendum text must mention both the cutoff and UNCERTAIN label."""
        assert "knowledge cutoff" in CUTOFF_AWARE_ADDENDUM
        assert "UNCERTAIN" in CUTOFF_AWARE_ADDENDUM

    def test_build_prompt_default_is_base(self) -> None:
        """Default (cutoff_aware=False) returns the base VERIFICATION_PROMPT."""
        entry = _make_entry()
        prompt = _build_verification_prompt(entry, cutoff_aware=False)
        assert CUTOFF_AWARE_ADDENDUM not in prompt
        # base template placeholders are filled
        assert "{bibtex}" not in prompt
        # The structural instructions from VERIFICATION_PROMPT survive
        assert "citation verification expert" in prompt

    def test_build_prompt_cutoff_aware_appends_addendum(self) -> None:
        entry = _make_entry()
        prompt = _build_verification_prompt(entry, cutoff_aware=True)
        assert CUTOFF_AWARE_ADDENDUM in prompt
        # base prompt content is preserved; addendum is appended, not a replacement
        assert "citation verification expert" in prompt

    def test_cutoff_aware_baseline_uses_addendum(self) -> None:
        """verify_with_openai(cutoff_aware=True) sends a prompt containing the addendum."""
        resp = _make_openai_response(
            json.dumps({"label": "UNCERTAIN", "confidence": 0.5, "reason": "post-cutoff"})
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            verify_with_openai(
                [_make_entry()],
                model="gpt-5.1",
                api_key="test-key",
                cutoff_aware=True,
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        sent = call_kwargs["messages"][0]["content"]
        assert CUTOFF_AWARE_ADDENDUM in sent

    def test_default_baseline_unchanged(self) -> None:
        """verify_with_openai() default path must NOT contain the addendum."""
        resp = _make_openai_response(
            json.dumps({"label": "VALID", "confidence": 0.8, "reason": "ok"})
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            verify_with_openai(
                [_make_entry()],
                model="gpt-5.1",
                api_key="test-key",
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        sent = call_kwargs["messages"][0]["content"]
        assert CUTOFF_AWARE_ADDENDUM not in sent

    def test_cutoff_aware_anthropic_uses_addendum(self) -> None:
        """verify_with_anthropic(cutoff_aware=True) sends the addendum too."""
        resp = _make_anthropic_response(
            json.dumps({"label": "UNCERTAIN", "confidence": 0.5, "reason": "post-cutoff"})
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = resp

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            verify_with_anthropic(
                [_make_entry()],
                model="claude-sonnet-4-5-20250929",
                api_key="test-key",
                cutoff_aware=True,
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        sent = call_kwargs["messages"][0]["content"]
        assert CUTOFF_AWARE_ADDENDUM in sent

    def test_verification_prompt_template_valid(self) -> None:
        """Sanity check that VERIFICATION_PROMPT has a {bibtex} placeholder."""
        assert "{bibtex}" in VERIFICATION_PROMPT


class TestParseLlmResponseUncertain:
    """UNCERTAIN is a valid model output (already supported)."""

    def test_explicit_uncertain_label_preserved(self) -> None:
        pred = _parse_llm_response(
            '{"label": "UNCERTAIN", "confidence": 0.3, "reason": "post-cutoff"}',
            "k",
        )
        assert pred.label == "UNCERTAIN"
        assert pred.confidence == pytest.approx(0.3)

    def test_unknown_label_falls_back_to_uncertain(self) -> None:
        pred = _parse_llm_response(
            '{"label": "MAYBE", "confidence": 0.5, "reason": ""}',
            "k",
        )
        assert pred.label == "UNCERTAIN"


class TestCutoffAwareRegistry:
    """Registry entries for the cutoff-aware ablation baselines."""

    def test_openai_cutoff_aware_registered(self) -> None:
        from hallmark.baselines.registry import get_registry

        assert "llm_openai_cutoff_aware" in get_registry()

    def test_anthropic_cutoff_aware_registered(self) -> None:
        from hallmark.baselines.registry import get_registry

        assert "llm_anthropic_cutoff_aware" in get_registry()

    def test_openrouter_subset_registered(self) -> None:
        from hallmark.baselines.registry import get_registry

        reg = get_registry()
        assert "llm_openrouter_gemini_flash_cutoff_aware" in reg
        assert "llm_openrouter_qwen_cutoff_aware" in reg

    def test_deepseek_variants_not_registered(self) -> None:
        """DeepSeek is intentionally skipped — it already saturates at UNCERTAIN."""
        from hallmark.baselines.registry import get_registry

        reg = get_registry()
        assert "llm_openrouter_deepseek_r1_cutoff_aware" not in reg
        assert "llm_openrouter_deepseek_v3_cutoff_aware" not in reg

    def test_cutoff_aware_metadata_matches_default(self) -> None:
        from hallmark.baselines.registry import get_registry

        reg = get_registry()
        default = reg["llm_openai"]
        ablation = reg["llm_openai_cutoff_aware"]
        assert ablation.env_var == default.env_var
        assert ablation.requires_api_key is True
        assert ablation.is_free is False
        assert ablation.confidence_type == "probabilistic"
