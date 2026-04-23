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
    ANTHROPIC_MODELS,
    CUTOFF_AWARE_ADDENDUM,
    OPENAI_MODELS,
    OPENROUTER_MODELS,
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


# ---------------------------------------------------------------------------
# Expanded model registry (new baselines: Llama 4, Gemini Pro, Qwen-Max)
# ---------------------------------------------------------------------------


class TestExpandedModelRegistry:
    """Tests for the three new OpenRouter baselines and two new reference dicts."""

    def test_llama_4_maverick_registered(self) -> None:
        from hallmark.baselines.registry import get_registry

        assert "llm_openrouter_llama_4_maverick" in get_registry()

    def test_gemini_pro_registered(self) -> None:
        from hallmark.baselines.registry import get_registry

        assert "llm_openrouter_gemini_pro" in get_registry()

    def test_qwen_max_registered(self) -> None:
        from hallmark.baselines.registry import get_registry

        assert "llm_openrouter_qwen_max" in get_registry()

    def test_new_models_share_openrouter_env_var(self) -> None:
        """All three new baselines must require OPENROUTER_API_KEY."""
        from hallmark.baselines.registry import get_registry

        reg = get_registry()
        for name in (
            "llm_openrouter_llama_4_maverick",
            "llm_openrouter_gemini_pro",
            "llm_openrouter_qwen_max",
        ):
            assert reg[name].env_var == "OPENROUTER_API_KEY", f"{name} has wrong env_var"

    def test_openrouter_models_includes_new_entries(self) -> None:
        assert "llama-4-maverick" in OPENROUTER_MODELS
        assert "gemini-pro" in OPENROUTER_MODELS
        assert "qwen-max" in OPENROUTER_MODELS

    def test_cutoff_aware_qwen_key_still_present(self) -> None:
        """The 'qwen' key must remain — _CUTOFF_AWARE_OPENROUTER references it."""
        assert "qwen" in OPENROUTER_MODELS

    def test_openai_models_dict_exists(self) -> None:
        assert isinstance(OPENAI_MODELS, dict)
        assert "gpt-5.1" in OPENAI_MODELS

    def test_anthropic_models_dict_exists(self) -> None:
        assert isinstance(ANTHROPIC_MODELS, dict)
        assert "claude-sonnet-4-6" in ANTHROPIC_MODELS

    def test_anthropic_default_model_refreshed(self) -> None:
        """Default should now be claude-sonnet-4-6."""
        import inspect

        assert (
            inspect.signature(verify_with_anthropic).parameters["model"].default
            == "claude-sonnet-4-6"
        )

    def test_llama_4_maverick_end_to_end_model_id(self) -> None:
        """Registered runner passes the correct Llama 4 model ID to the API call."""
        from hallmark.baselines.registry import run_baseline
        from hallmark.dataset.schema import BenchmarkEntry

        entry = BenchmarkEntry(
            bibtex_key="test2024",
            bibtex_type="article",
            fields={
                "title": "Test Paper",
                "author": "Smith, John",
                "year": "2024",
            },
            raw_bibtex="@article{test2024, title={Test Paper}}",
            label="VALID",
        )

        captured_model: list[str] = []

        mock_resp = MagicMock()
        mock_resp.choices[
            0
        ].message.content = '{"label": "VALID", "confidence": 0.9, "reason": "ok"}'

        mock_client = MagicMock()

        def _capture_create(**kwargs: object) -> MagicMock:
            captured_model.append(str(kwargs.get("model", "")))
            return mock_resp

        mock_client.chat.completions.create.side_effect = _capture_create

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        import os

        with (
            patch.dict("sys.modules", {"openai": mock_openai}),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
        ):
            run_baseline("llm_openrouter_llama_4_maverick", [entry])

        assert len(captured_model) == 1
        assert captured_model[0] == OPENROUTER_MODELS["llama-4-maverick"]


# ---------------------------------------------------------------------------
# Auto-injection of per-baseline API key from env_var (registry fix)
# ---------------------------------------------------------------------------


class TestApiKeyAutoInjection:
    """run_baseline must inject api_key from the baseline's declared env_var.

    Previously, openai-SDK-based OpenRouter baselines silently 401ed from the
    CLI because the SDK defaulted to OPENAI_API_KEY (which is for api.openai.com,
    not openrouter.ai). run_baseline now reads info.env_var and forwards it.
    """

    def test_openrouter_baseline_uses_openrouter_api_key(self) -> None:
        from hallmark.baselines.registry import run_baseline
        from hallmark.dataset.schema import BenchmarkEntry

        entry = BenchmarkEntry(
            bibtex_key="k",
            bibtex_type="article",
            fields={"title": "t", "author": "a", "year": "2024"},
            raw_bibtex="@article{k,title={t}}",
            label="VALID",
        )

        mock_resp = MagicMock()
        mock_resp.choices[
            0
        ].message.content = '{"label": "VALID", "confidence": 0.9, "reason": "ok"}'
        captured_api_key: list[str] = []

        def _capture_openai(**kwargs: object) -> MagicMock:
            captured_api_key.append(str(kwargs.get("api_key", "")))
            client = MagicMock()
            client.chat.completions.create.return_value = mock_resp
            return client

        mock_openai = MagicMock()
        mock_openai.OpenAI.side_effect = _capture_openai

        import os

        # Both keys set to distinguishable values; OPENROUTER must win for
        # an llm_openrouter_* baseline.
        with (
            patch.dict("sys.modules", {"openai": mock_openai}),
            patch.dict(
                os.environ,
                {
                    "OPENROUTER_API_KEY": "or-sentinel",
                    "OPENAI_API_KEY": "openai-sentinel",
                },
            ),
        ):
            run_baseline("llm_openrouter_llama_4_maverick", [entry])

        assert captured_api_key == ["or-sentinel"], (
            f"Expected OPENROUTER_API_KEY to be forwarded; got {captured_api_key}"
        )

    def test_openai_baseline_uses_openai_api_key(self) -> None:
        from hallmark.baselines.registry import run_baseline
        from hallmark.dataset.schema import BenchmarkEntry

        entry = BenchmarkEntry(
            bibtex_key="k",
            bibtex_type="article",
            fields={"title": "t", "author": "a", "year": "2024"},
            raw_bibtex="@article{k,title={t}}",
            label="VALID",
        )

        mock_resp = MagicMock()
        mock_resp.choices[
            0
        ].message.content = '{"label": "VALID", "confidence": 0.9, "reason": "ok"}'
        captured_api_key: list[str] = []

        def _capture_openai(**kwargs: object) -> MagicMock:
            captured_api_key.append(str(kwargs.get("api_key", "")))
            client = MagicMock()
            client.chat.completions.create.return_value = mock_resp
            return client

        mock_openai = MagicMock()
        mock_openai.OpenAI.side_effect = _capture_openai

        import os

        with (
            patch.dict("sys.modules", {"openai": mock_openai}),
            patch.dict(
                os.environ,
                {
                    "OPENROUTER_API_KEY": "or-sentinel",
                    "OPENAI_API_KEY": "openai-sentinel",
                },
            ),
        ):
            run_baseline("llm_openai", [entry])

        assert captured_api_key == ["openai-sentinel"]

    def test_explicit_api_key_kwarg_overrides_env(self) -> None:
        """Explicit api_key= passed to run_baseline must not be clobbered by env."""
        from hallmark.baselines.registry import run_baseline
        from hallmark.dataset.schema import BenchmarkEntry

        entry = BenchmarkEntry(
            bibtex_key="k",
            bibtex_type="article",
            fields={"title": "t", "author": "a", "year": "2024"},
            raw_bibtex="@article{k,title={t}}",
            label="VALID",
        )

        mock_resp = MagicMock()
        mock_resp.choices[
            0
        ].message.content = '{"label": "VALID", "confidence": 0.9, "reason": "ok"}'
        captured_api_key: list[str] = []

        def _capture_openai(**kwargs: object) -> MagicMock:
            captured_api_key.append(str(kwargs.get("api_key", "")))
            client = MagicMock()
            client.chat.completions.create.return_value = mock_resp
            return client

        mock_openai = MagicMock()
        mock_openai.OpenAI.side_effect = _capture_openai

        import os

        with (
            patch.dict("sys.modules", {"openai": mock_openai}),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-from-env"}),
        ):
            run_baseline(
                "llm_openrouter_llama_4_maverick",
                [entry],
                api_key="explicit-kwarg",
            )

        assert captured_api_key == ["explicit-kwarg"]


# ---------------------------------------------------------------------------
# Salvage parser for truncated JSON (verbose reasoning models)
# ---------------------------------------------------------------------------


class TestSalvageParser:
    """_parse_llm_response must salvage label+confidence from truncated JSON."""

    def test_salvages_label_from_truncated_response(self) -> None:
        # Simulates Gemini 2.5 Pro hitting max_completion_tokens mid-"reason"
        truncated = (
            '```json\n{\n  "label": "HALLUCINATED",\n  "confidence": 0.92,\n  '
            '"reason": "The DOI 10.1234/fake does not resolve and the authors are'
        )
        pred = _parse_llm_response(truncated, "k")
        assert pred.label == "HALLUCINATED"
        assert pred.confidence == pytest.approx(0.92)
        assert "Salvaged" in pred.reason

    def test_salvages_valid_label(self) -> None:
        truncated = '{"label":"VALID","confidence":1.0,"reason":"The DOI'
        pred = _parse_llm_response(truncated, "k")
        assert pred.label == "VALID"
        assert pred.confidence == pytest.approx(1.0)

    def test_complete_json_still_parses_normally(self) -> None:
        """Complete JSON must go through the fast path, not the salvage path."""
        complete = '{"label": "HALLUCINATED", "confidence": 0.8, "reason": "all checks fail"}'
        pred = _parse_llm_response(complete, "k")
        assert pred.label == "HALLUCINATED"
        assert pred.confidence == pytest.approx(0.8)
        assert pred.reason == "all checks fail"
        assert "Salvaged" not in pred.reason

    def test_unrecoverable_response_falls_back(self) -> None:
        """Content with no parseable label at all still falls back to UNCERTAIN."""
        garbage = "I cannot answer this question."
        pred = _parse_llm_response(garbage, "k")
        assert pred.label == "UNCERTAIN"
        assert "Error fallback" in pred.reason


# ---------------------------------------------------------------------------
# Checkpoint resume + retry-failed behavior
# ---------------------------------------------------------------------------


class TestCheckpointRetryFailed:
    """_load_checkpoint must skip [Error fallback] entries when skip_failed=True."""

    def test_skip_failed_drops_error_fallback_entries(self, tmp_path: object) -> None:
        from hallmark.baselines.llm_verifier import _load_checkpoint

        path = tmp_path / "ckpt.jsonl"  # type: ignore[attr-defined]
        records = [
            {
                "bibtex_key": "ok-1",
                "label": "VALID",
                "confidence": 0.9,
                "reason": "looks real",
            },
            {
                "bibtex_key": "failed-1",
                "label": "UNCERTAIN",
                "confidence": 0.5,
                "reason": "[Error fallback] API error: ConnectionError",
            },
            {
                "bibtex_key": "ok-2",
                "label": "HALLUCINATED",
                "confidence": 0.8,
                "reason": "DOI does not resolve",
            },
        ]
        path.write_text("\n".join(json.dumps(r) for r in records))

        # Default: all three kept
        loaded_all = _load_checkpoint(path)
        assert set(loaded_all.keys()) == {"ok-1", "failed-1", "ok-2"}

        # skip_failed=True: only the two good ones
        loaded_retry = _load_checkpoint(path, skip_failed=True)
        assert set(loaded_retry.keys()) == {"ok-1", "ok-2"}
        assert "failed-1" not in loaded_retry

    def test_skip_failed_keeps_later_success_for_same_key(self, tmp_path: object) -> None:
        """If a key was retried successfully after a fallback, keep the success."""
        from hallmark.baselines.llm_verifier import _load_checkpoint

        path = tmp_path / "ckpt.jsonl"  # type: ignore[attr-defined]
        # Same key appears twice: once as a fallback, once as a successful retry
        records = [
            {
                "bibtex_key": "key-a",
                "label": "UNCERTAIN",
                "confidence": 0.5,
                "reason": "[Error fallback] transient",
            },
            {
                "bibtex_key": "key-a",
                "label": "VALID",
                "confidence": 0.95,
                "reason": "resolved after retry",
            },
        ]
        path.write_text("\n".join(json.dumps(r) for r in records))

        loaded = _load_checkpoint(path, skip_failed=True)
        assert loaded["key-a"].label == "VALID"
        assert loaded["key-a"].confidence == pytest.approx(0.95)

    def test_retry_failed_kwarg_flows_through_openrouter(self, tmp_path: object) -> None:
        """verify_with_openrouter(retry_failed=True) must honor the flag."""
        from hallmark.baselines.llm_verifier import (
            _append_checkpoint,
            verify_with_openrouter,
        )

        ckpt_dir = tmp_path / "ckpt"  # type: ignore[attr-defined]
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "openrouter_meta-llama_llama-4-maverick.jsonl"

        # Pre-seed: one failed record for an entry we'll re-attempt
        _append_checkpoint(
            ckpt_path,
            _parse_llm_response(
                '{"label":"UNCERTAIN","confidence":0.5,"reason":"[Error fallback] x"}',
                "retry-me",
            ),
        )

        entry = _make_entry(key="retry-me")
        resp = _make_openai_response(
            json.dumps({"label": "VALID", "confidence": 0.9, "reason": "ok"})
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            preds = verify_with_openrouter(
                [entry],
                model="meta-llama/llama-4-maverick",
                api_key="test-key",
                checkpoint_dir=ckpt_dir,
                retry_failed=True,
            )

        # The API should have been re-invoked (since we cleared the fallback)
        assert mock_client.chat.completions.create.call_count == 1
        assert len(preds) == 1
        assert preds[0].label == "VALID"
