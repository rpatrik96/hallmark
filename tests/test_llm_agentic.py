"""Unit tests for the llm_agentic baseline.

All API calls are mocked — no live keys consumed in CI.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hallmark.baselines._agentic_cache import AgenticToolCache, cache_key
from hallmark.baselines._agentic_tools import TOOL_DEFINITIONS, TOOL_REGISTRY, _normalise
from hallmark.baselines.llm_agentic import (
    ANTHROPIC_MODEL,
    MAX_TOOL_CALLS,
    _dispatch_tool,
    _load_checkpoint,
    _write_checkpoint,
    verify_agentic_anthropic,
    verify_agentic_openai,
)
from hallmark.dataset.schema import BlindEntry, Prediction

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


# ---------------------------------------------------------------------------
# _agentic_cache tests
# ---------------------------------------------------------------------------


class TestAgenticToolCache:
    def test_miss_returns_none(self, tmp_path: Path) -> None:
        db = tmp_path / "test.sqlite"
        with AgenticToolCache(db) as cache:
            assert cache.get("nonexistent") is None

    def test_set_then_get(self, tmp_path: Path) -> None:
        db = tmp_path / "test.sqlite"
        with AgenticToolCache(db) as cache:
            key = "abc123"
            value = {"title": "Test Paper", "authors": "Doe"}
            cache.set(key, value)
            assert cache.get(key) == value

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        db = tmp_path / "test.sqlite"
        with AgenticToolCache(db) as cache:
            cache.set("k", {"v": 1})
            cache.set("k", {"v": 2})
            assert cache.get("k") == {"v": 2}

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        db = tmp_path / "test.sqlite"
        with AgenticToolCache(db) as cache:
            cache.set("persistent_key", {"data": "hello"})
        # Open a second instance pointing at the same DB
        with AgenticToolCache(db) as cache2:
            assert cache2.get("persistent_key") == {"data": "hello"}

    def test_ttl_expiry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import time

        db = tmp_path / "ttl_test.sqlite"
        monkeypatch.setenv("CACHE_TTL_DAYS", "0.000001")  # ~86ms TTL

        with AgenticToolCache(db) as cache:
            cache.set("expiry_key", {"x": 1})
            # Patch time to simulate expiry — capture original before replacing
            original_time = time.time
            monkeypatch.setattr(time, "time", lambda: original_time() + 200)
            # Re-open to pick up monkeypatched env
            cache2 = AgenticToolCache(db)
            assert cache2.get("expiry_key") is None
            cache2.close()

    def test_schema_creates_table(self, tmp_path: Path) -> None:
        db = tmp_path / "schema_test.sqlite"
        with AgenticToolCache(db):
            pass
        conn = sqlite3.connect(str(db))
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
        assert ("tool_cache",) in tables


class TestCacheKey:
    def test_deterministic(self) -> None:
        k1 = cache_key("resolve_doi", {"doi": "10.1234/abc"})
        k2 = cache_key("resolve_doi", {"doi": "10.1234/abc"})
        assert k1 == k2

    def test_args_order_independent(self) -> None:
        k1 = cache_key("search_crossref", {"query": "deep learning", "limit": 5})
        k2 = cache_key("search_crossref", {"limit": 5, "query": "deep learning"})
        assert k1 == k2

    def test_different_tools_differ(self) -> None:
        k1 = cache_key("resolve_doi", {"doi": "10.1234/abc"})
        k2 = cache_key("search_crossref", {"doi": "10.1234/abc"})
        assert k1 != k2

    def test_different_args_differ(self) -> None:
        k1 = cache_key("resolve_doi", {"doi": "10.1234/abc"})
        k2 = cache_key("resolve_doi", {"doi": "10.1234/xyz"})
        assert k1 != k2


# ---------------------------------------------------------------------------
# _agentic_tools tests
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_truncates_long_fields(self) -> None:
        long_str = "x" * 600
        result = _normalise({"title": long_str, "authors": "", "venue": "", "year": "", "doi": ""})
        assert len(result["title"]) == 500

    def test_none_becomes_empty_string(self) -> None:
        result = _normalise(
            {"title": None, "authors": None, "venue": None, "year": None, "doi": None}
        )
        for v in result.values():
            assert v == ""

    def test_missing_keys_become_empty(self) -> None:
        result = _normalise({})
        assert result == {"authors": "", "title": "", "venue": "", "year": "", "doi": ""}


class TestToolDefinitions:
    def test_all_tools_registered(self) -> None:
        names = {t["name"] for t in TOOL_DEFINITIONS}
        assert names == {"resolve_doi", "search_crossref", "search_openalex", "search_arxiv"}

    def test_each_tool_has_required_keys(self) -> None:
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert tool["parameters"]["type"] == "object"

    def test_registry_matches_definitions(self) -> None:
        for tool in TOOL_DEFINITIONS:
            assert tool["name"] in TOOL_REGISTRY


class TestDispatchTool:
    def test_cache_hit_skips_call(self, tmp_path: Path) -> None:
        db = tmp_path / "test.sqlite"
        cached_value = {"title": "cached", "authors": "", "venue": "", "year": "", "doi": ""}
        key = cache_key("resolve_doi", {"doi": "10.9999/cached"})

        with AgenticToolCache(db) as cache:
            cache.set(key, cached_value)
            result_str = _dispatch_tool("resolve_doi", {"doi": "10.9999/cached"}, cache)
            result = json.loads(result_str)
            assert result["title"] == "cached"

    def test_unknown_tool_returns_error(self, tmp_path: Path) -> None:
        db = tmp_path / "test.sqlite"
        with AgenticToolCache(db) as cache:
            result_str = _dispatch_tool("nonexistent_tool", {}, cache)
            result = json.loads(result_str)
            assert "error" in result

    def test_tool_failure_returns_error_json(self, tmp_path: Path) -> None:
        db = tmp_path / "test.sqlite"
        with (
            AgenticToolCache(db) as cache,
            patch(
                "hallmark.baselines.llm_agentic.retry_with_backoff",
                side_effect=RuntimeError("network failure"),
            ),
        ):
            result_str = _dispatch_tool("resolve_doi", {"doi": "10.9999/bad"}, cache)
            result = json.loads(result_str)
            assert "error" in result

    def test_successful_call_stored_in_cache(self, tmp_path: Path) -> None:
        db = tmp_path / "test.sqlite"
        fake_result = {"title": "stored", "authors": "", "venue": "", "year": "", "doi": "10.1/x"}
        with AgenticToolCache(db) as cache:
            with patch(
                "hallmark.baselines.llm_agentic.retry_with_backoff",
                return_value=fake_result,
            ):
                _dispatch_tool("resolve_doi", {"doi": "10.1/x"}, cache)
            # Re-read from cache without mocking — should return stored value
            key = cache_key("resolve_doi", {"doi": "10.1/x"})
            assert cache.get(key) == fake_result


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_write_and_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.jsonl"
        pred = Prediction(
            bibtex_key="paper42",
            label="HALLUCINATED",
            confidence=0.9,
            reason="test reason",
            wall_clock_seconds=1.5,
            api_calls=3,
            api_sources_queried=["openai/gpt-5.1", "tool:resolve_doi"],
        )
        _write_checkpoint(path, pred)
        loaded = _load_checkpoint(path)
        assert "paper42" in loaded
        assert loaded["paper42"].label == "HALLUCINATED"
        assert loaded["paper42"].confidence == pytest.approx(0.9)
        assert loaded["paper42"].api_calls == 3

    def test_write_none_path_is_noop(self) -> None:
        pred = Prediction(bibtex_key="x", label="VALID", confidence=0.8, reason="ok")
        _write_checkpoint(None, pred)  # must not raise

    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        result = _load_checkpoint(tmp_path / "missing.jsonl")
        assert result == {}

    def test_load_skips_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.jsonl"
        path.write_text(
            'not-json\n{"bibtex_key":"ok","label":"VALID","confidence":0.7,"reason":""}\n'
        )
        loaded = _load_checkpoint(path)
        assert "ok" in loaded
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestAgenticRegistry:
    def test_both_baselines_registered(self) -> None:
        from hallmark.baselines.registry import get_registry

        registry = get_registry()
        assert "llm_agentic_openai" in registry
        assert "llm_agentic_anthropic" in registry

    def test_openai_baseline_metadata(self) -> None:
        from hallmark.baselines.registry import get_registry

        info = get_registry()["llm_agentic_openai"]
        assert info.confidence_type == "probabilistic"
        assert "openai" in info.pip_packages
        assert info.env_var == "OPENAI_API_KEY"
        assert info.requires_api_key is True
        assert info.is_free is False

    def test_anthropic_baseline_metadata(self) -> None:
        from hallmark.baselines.registry import get_registry

        info = get_registry()["llm_agentic_anthropic"]
        assert info.confidence_type == "probabilistic"
        assert "anthropic" in info.pip_packages
        assert info.env_var == "ANTHROPIC_API_KEY"
        assert info.requires_api_key is True
        assert info.is_free is False


# ---------------------------------------------------------------------------
# OpenAI agentic loop (mocked)
# ---------------------------------------------------------------------------


def _make_openai_text_response(text: str) -> MagicMock:
    """Build a mock OpenAI chat completion that returns text (no tool calls)."""
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = text
    msg.model_dump.return_value = {"role": "assistant", "content": text}
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.total_tokens = 100
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_openai_tool_response(
    tool_name: str, tool_args: dict, call_id: str = "call_1"
) -> MagicMock:
    """Build a mock OpenAI chat completion that requests a tool call."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(tool_args)

    msg = MagicMock()
    msg.tool_calls = [tc]
    msg.content = None
    msg.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [
            {"id": call_id, "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}
        ],
    }
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.total_tokens = 150
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


class TestVerifyAgenticOpenAI:
    def test_direct_verdict_no_tools(self, tmp_path: Path) -> None:
        """Model returns verdict on first turn (no tools)."""
        verdict = json.dumps({"label": "VALID", "confidence": 0.85, "reason": "Looks real"})
        mock_resp = _make_openai_text_response(verdict)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        entries = [_make_entry()]
        with patch.dict("sys.modules", {"openai": mock_openai}):
            preds = verify_agentic_openai(
                entries,
                model="gpt-5.1",
                api_key="test-key",
                cache_db_path=tmp_path / "cache.sqlite",
            )

        assert len(preds) == 1
        assert preds[0].label == "VALID"
        assert preds[0].confidence == pytest.approx(0.85)
        assert "parametric" in preds[0].reason

    def test_one_tool_call_then_verdict(self, tmp_path: Path) -> None:
        """Model calls one tool, then emits verdict."""
        tool_resp = _make_openai_tool_response(
            "search_crossref", {"query": "deep learning citations"}, call_id="c1"
        )
        verdict_resp = _make_openai_text_response(
            json.dumps({"label": "HALLUCINATED", "confidence": 0.92, "reason": "Not found"})
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [tool_resp, verdict_resp]

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        fake_tool_result = [{"title": "", "authors": "", "venue": "", "year": "", "doi": ""}]

        entries = [_make_entry()]
        with (
            patch.dict("sys.modules", {"openai": mock_openai}),
            patch(
                "hallmark.baselines.llm_agentic._dispatch_tool",
                return_value=json.dumps(fake_tool_result),
            ),
        ):
            preds = verify_agentic_openai(
                entries,
                model="gpt-5.1",
                api_key="test-key",
                cache_db_path=tmp_path / "cache.sqlite",
            )

        assert len(preds) == 1
        assert preds[0].label == "HALLUCINATED"
        assert "tool" in preds[0].reason
        assert "search_crossref" in preds[0].reason

    def test_max_tool_calls_cap(self, tmp_path: Path) -> None:
        """Loop aborts with UNCERTAIN after MAX_TOOL_CALLS tool invocations."""
        # Alternate tool calls indefinitely
        tool_resps = [
            _make_openai_tool_response("resolve_doi", {"doi": "10.x/y"}, call_id=f"c{i}")
            for i in range(MAX_TOOL_CALLS + 2)
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = tool_resps

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        entries = [_make_entry()]
        with (
            patch.dict("sys.modules", {"openai": mock_openai}),
            patch(
                "hallmark.baselines.llm_agentic._dispatch_tool",
                return_value=json.dumps({"error": "not found"}),
            ),
        ):
            preds = verify_agentic_openai(
                entries,
                model="gpt-5.1",
                api_key="test-key",
                cache_db_path=tmp_path / "cache.sqlite",
            )

        assert len(preds) == 1
        assert preds[0].label == "UNCERTAIN"
        assert "Max tool calls" in preds[0].reason

    def test_consecutive_failures_abort(self, tmp_path: Path) -> None:
        """Consecutive API failures trigger early abort with UNCERTAIN for all entries."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        entries = [_make_entry(f"k{i}") for i in range(5)]
        with patch.dict("sys.modules", {"openai": mock_openai}):
            preds = verify_agentic_openai(
                entries,
                model="gpt-5.1",
                api_key="test-key",
                max_consecutive_failures=2,
                cache_db_path=tmp_path / "cache.sqlite",
            )

        assert len(preds) == 5
        assert all(p.label == "UNCERTAIN" for p in preds)

    def test_checkpoint_resume(self, tmp_path: Path) -> None:
        """Entries already in the checkpoint are not re-queried."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "agentic_openai_gpt-5.1.jsonl"
        # Pre-populate checkpoint with one entry
        pred = Prediction(
            bibtex_key="test2024",
            label="VALID",
            confidence=0.9,
            reason="pre-cached",
        )
        _write_checkpoint(ckpt_path, pred)

        mock_client = MagicMock()
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        entries = [_make_entry("test2024")]
        with patch.dict("sys.modules", {"openai": mock_openai}):
            preds = verify_agentic_openai(
                entries,
                model="gpt-5.1",
                api_key="test-key",
                checkpoint_dir=ckpt_dir,
                cache_db_path=tmp_path / "cache.sqlite",
            )

        # API should NOT have been called (entry already in checkpoint)
        mock_client.chat.completions.create.assert_not_called()
        assert preds[0].label == "VALID"

    def test_missing_openai_returns_fallback(self, tmp_path: Path) -> None:
        """When openai import fails inside the runner, returns fallback predictions."""
        import sys

        entries = [_make_entry()]
        original = sys.modules.pop("openai", None)
        try:
            preds = verify_agentic_openai(
                entries,
                model="gpt-5.1",
                api_key="test-key",
                cache_db_path=tmp_path / "cache.sqlite",
            )
        finally:
            if original is not None:
                sys.modules["openai"] = original

        assert len(preds) == 1
        assert preds[0].label in ("VALID", "UNCERTAIN")


# ---------------------------------------------------------------------------
# Anthropic agentic loop (mocked)
# ---------------------------------------------------------------------------


def _make_anthropic_text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    usage = MagicMock()
    usage.input_tokens = 80
    usage.output_tokens = 40
    resp = MagicMock()
    resp.content = [block]
    resp.usage = usage
    return resp


def _make_anthropic_tool_response(
    tool_name: str, tool_args: dict, call_id: str = "tu_1"
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = call_id
    block.name = tool_name
    block.input = tool_args
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 20
    resp = MagicMock()
    resp.content = [block]
    resp.usage = usage
    return resp


class TestVerifyAgenticAnthropic:
    def test_direct_verdict_no_tools(self, tmp_path: Path) -> None:
        verdict = json.dumps({"label": "HALLUCINATED", "confidence": 0.88, "reason": "Fake DOI"})
        mock_resp = _make_anthropic_text_response(verdict)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        entries = [_make_entry()]
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            preds = verify_agentic_anthropic(
                entries,
                model=ANTHROPIC_MODEL,
                api_key="test-key",
                cache_db_path=tmp_path / "cache.sqlite",
            )

        assert len(preds) == 1
        assert preds[0].label == "HALLUCINATED"
        assert "parametric" in preds[0].reason

    def test_one_tool_call_then_verdict(self, tmp_path: Path) -> None:
        tool_resp = _make_anthropic_tool_response(
            "search_openalex", {"query": "attention is all you need"}, call_id="tu_1"
        )
        verdict_resp = _make_anthropic_text_response(
            json.dumps({"label": "VALID", "confidence": 0.95, "reason": "Found in OpenAlex"})
        )

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [tool_resp, verdict_resp]

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        entries = [_make_entry()]
        dispatch_result = json.dumps(
            [{"title": "Attention", "authors": "", "venue": "", "year": "", "doi": ""}]
        )
        with (
            patch.dict("sys.modules", {"anthropic": mock_anthropic}),
            patch(
                "hallmark.baselines.llm_agentic._dispatch_tool",
                return_value=dispatch_result,
            ),
        ):
            preds = verify_agentic_anthropic(
                entries,
                model=ANTHROPIC_MODEL,
                api_key="test-key",
                cache_db_path=tmp_path / "cache.sqlite",
            )

        assert len(preds) == 1
        assert preds[0].label == "VALID"
        assert "search_openalex" in preds[0].reason

    def test_max_tool_calls_cap(self, tmp_path: Path) -> None:
        tool_resps = [
            _make_anthropic_tool_response("resolve_doi", {"doi": "10.x/y"}, call_id=f"tu_{i}")
            for i in range(MAX_TOOL_CALLS + 2)
        ]

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = tool_resps

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        entries = [_make_entry()]
        with (
            patch.dict("sys.modules", {"anthropic": mock_anthropic}),
            patch(
                "hallmark.baselines.llm_agentic._dispatch_tool",
                return_value=json.dumps({"error": "not found"}),
            ),
        ):
            preds = verify_agentic_anthropic(
                entries,
                model=ANTHROPIC_MODEL,
                api_key="test-key",
                cache_db_path=tmp_path / "cache.sqlite",
            )

        assert preds[0].label == "UNCERTAIN"
        assert "Max tool calls" in preds[0].reason

    def test_consecutive_failures_abort(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("Anthropic down")

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        entries = [_make_entry(f"k{i}") for i in range(4)]
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            preds = verify_agentic_anthropic(
                entries,
                model=ANTHROPIC_MODEL,
                api_key="test-key",
                max_consecutive_failures=2,
                cache_db_path=tmp_path / "cache.sqlite",
            )

        assert len(preds) == 4
        assert all(p.label == "UNCERTAIN" for p in preds)


# ---------------------------------------------------------------------------
# JSON parsing edge cases (reused from llm_verifier, stress-tested here)
# ---------------------------------------------------------------------------


class TestJsonParsingEdgeCases:
    """Verify _parse_llm_response handles outputs the agentic model may emit."""

    def _parse(self, text: str, key: str = "k") -> Prediction:
        from hallmark.baselines.llm_verifier import _parse_llm_response

        return _parse_llm_response(text, key)

    def test_valid_json(self) -> None:
        pred = self._parse('{"label": "VALID", "confidence": 0.9, "reason": "ok"}')
        assert pred.label == "VALID"
        assert pred.confidence == pytest.approx(0.9)

    def test_hallucinated_json(self) -> None:
        pred = self._parse('{"label": "HALLUCINATED", "confidence": 0.7, "reason": "fake doi"}')
        assert pred.label == "HALLUCINATED"

    def test_invalid_label_becomes_uncertain(self) -> None:
        pred = self._parse('{"label": "MAYBE", "confidence": 0.6, "reason": "unsure"}')
        assert pred.label == "UNCERTAIN"

    def test_confidence_clamped(self) -> None:
        pred = self._parse('{"label": "VALID", "confidence": 1.5, "reason": "x"}')
        assert pred.confidence <= 1.0
        pred2 = self._parse('{"label": "VALID", "confidence": -0.1, "reason": "x"}')
        assert pred2.confidence >= 0.0

    def test_completely_invalid_json(self) -> None:
        pred = self._parse("This is not JSON at all.")
        assert pred.label == "UNCERTAIN"
        assert pred.confidence == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# BTU-as-tool agentic variants
# ---------------------------------------------------------------------------


class TestBtuTool:
    """Tests for verify_with_bibtex_updater (BTU-as-tool wrapper)."""

    def test_no_binary_raises_runtimeerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """verify_with_bibtex_updater raises RuntimeError when bibtex-check missing."""
        from hallmark.baselines._agentic_tools import verify_with_bibtex_updater

        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(RuntimeError, match="bibtex-check not found"):
            verify_with_bibtex_updater("@article{k, title={t}}")

    def test_successful_call_returns_flat_dict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parses one-line JSONL output from bibtex-check into the flat canonical dict."""
        from hallmark.baselines import _agentic_tools as at

        record = {
            "status": "title_mismatch",
            "confidence": 0.85,
            "mismatched_fields": ["title", "venue"],
            "api_sources": ["crossref", "dblp"],
            "errors": [],
        }

        def _fake_run(cmd, **_kw):  # type: ignore[no-untyped-def]
            # bibtex-check writes JSONL to the path after --jsonl
            idx = cmd.index("--jsonl")
            jsonl = Path(cmd[idx + 1])
            jsonl.write_text(json.dumps(record) + "\n")
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(at, "_trunc", lambda v: str(v) if v is not None else "")
        monkeypatch.setattr("shutil.which", lambda _: "/fake/bibtex-check")
        monkeypatch.setattr("subprocess.run", _fake_run)

        result = at.verify_with_bibtex_updater("@article{k, title={fake}}")
        assert result["status"] == "title_mismatch"
        assert result["mismatched_fields"] == "title, venue"
        assert result["api_sources"] == "crossref, dblp"

    def test_tool_listed_in_registry(self) -> None:
        from hallmark.baselines._agentic_tools import (
            BTU_TOOL_DEFINITION,
            TOOL_REGISTRY,
        )

        assert "verify_with_bibtex_updater" in TOOL_REGISTRY
        assert BTU_TOOL_DEFINITION["name"] == "verify_with_bibtex_updater"
        assert "bibtex" in BTU_TOOL_DEFINITION["parameters"]["properties"]


class TestVerifyAgenticBtuOpenai:
    """Mocked tests for verify_agentic_btu_openai (tool_defs=[BTU_TOOL_DEFINITION])."""

    def test_btu_variant_uses_only_btu_tool(self, tmp_path: Path) -> None:
        """OpenAI call receives exactly one tool: verify_with_bibtex_updater."""
        from hallmark.baselines.llm_agentic import verify_agentic_btu_openai

        verdict = json.dumps(
            {"label": "HALLUCINATED", "confidence": 0.9, "reason": "BTU status=not_found"}
        )
        resp = _make_openai_text_response(verdict)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        entries = [_make_entry()]
        with patch.dict("sys.modules", {"openai": mock_openai}):
            preds = verify_agentic_btu_openai(
                entries,
                model="gpt-5.1",
                api_key="test-key",
                cache_db_path=tmp_path / "cache.sqlite",
            )

        assert len(preds) == 1
        assert preds[0].label == "HALLUCINATED"

        # Verify the tool set passed to the API contained only BTU
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        tools = call_kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "verify_with_bibtex_updater"

    def test_btu_variant_uses_btu_system_prompt(self, tmp_path: Path) -> None:
        from hallmark.baselines.llm_agentic import BTU_SYSTEM_PROMPT, verify_agentic_btu_openai

        verdict = json.dumps({"label": "VALID", "confidence": 0.9, "reason": "ok"})
        resp = _make_openai_text_response(verdict)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            verify_agentic_btu_openai(
                [_make_entry()],
                model="gpt-5.1",
                api_key="test-key",
                cache_db_path=tmp_path / "cache.sqlite",
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        system_msg = call_kwargs["messages"][0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == BTU_SYSTEM_PROMPT

    def test_btu_variant_checkpoint_name_distinct(self, tmp_path: Path) -> None:
        """BTU variant writes to a different checkpoint file than the multi-tool variant."""
        from hallmark.baselines.llm_agentic import verify_agentic_btu_openai

        verdict = json.dumps({"label": "VALID", "confidence": 0.8, "reason": "ok"})
        resp = _make_openai_text_response(verdict)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        ckpt_dir = tmp_path / "ckpt"
        with patch.dict("sys.modules", {"openai": mock_openai}):
            verify_agentic_btu_openai(
                [_make_entry()],
                model="gpt-5.1",
                api_key="test-key",
                checkpoint_dir=ckpt_dir,
                cache_db_path=tmp_path / "cache.sqlite",
            )

        # Checkpoint file should be prefixed "agentic_btu_openai_"
        files = list(ckpt_dir.iterdir())
        assert len(files) == 1
        assert files[0].name.startswith("agentic_btu_openai_")


class TestVerifyAgenticBtuAnthropic:
    def test_btu_anthropic_uses_only_btu_tool(self, tmp_path: Path) -> None:
        from hallmark.baselines.llm_agentic import verify_agentic_btu_anthropic

        verdict = json.dumps({"label": "VALID", "confidence": 0.9, "reason": "BTU verified"})
        resp = _make_anthropic_text_response(verdict)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = resp
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            preds = verify_agentic_btu_anthropic(
                [_make_entry()],
                model=ANTHROPIC_MODEL,
                api_key="test-key",
                cache_db_path=tmp_path / "cache.sqlite",
            )

        assert preds[0].label == "VALID"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        tools = call_kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "verify_with_bibtex_updater"

    def test_consecutive_failures_abort(self, tmp_path: Path) -> None:
        """BTU Anthropic variant aborts after N consecutive API failures."""
        from hallmark.baselines.llm_agentic import verify_agentic_btu_anthropic

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("BTU Anthropic down")

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        entries = [_make_entry(f"k{i}") for i in range(5)]
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            preds = verify_agentic_btu_anthropic(
                entries,
                model=ANTHROPIC_MODEL,
                api_key="test-key",
                max_consecutive_failures=2,
                cache_db_path=tmp_path / "cache.sqlite",
            )

        assert len(preds) == 5
        assert all(p.label == "UNCERTAIN" for p in preds)


# ---------------------------------------------------------------------------
# Priority 1 — missing anthropic package fallback
# ---------------------------------------------------------------------------


class TestVerifyAgenticAnthropicMissingPackage:
    def test_missing_anthropic_returns_fallback(self, tmp_path: Path) -> None:
        """When anthropic import fails, returns fallback UNCERTAIN predictions."""
        import sys

        entries = [_make_entry()]
        original = sys.modules.pop("anthropic", None)
        try:
            preds = verify_agentic_anthropic(
                entries,
                model=ANTHROPIC_MODEL,
                api_key="test-key",
                cache_db_path=tmp_path / "cache.sqlite",
            )
        finally:
            if original is not None:
                sys.modules["anthropic"] = original

        assert len(preds) == 1
        assert preds[0].label in ("VALID", "UNCERTAIN")


# ---------------------------------------------------------------------------
# Priority 2 — network tool functions (mocked httpx)
# ---------------------------------------------------------------------------


class TestResolveDoi:
    def test_success_returns_normalised(self) -> None:
        """200 response with CrossRef work body → normalised dict."""
        from hallmark.baselines._agentic_tools import resolve_doi

        payload = {
            "message": {
                "author": [{"family": "Smith", "given": "John"}],
                "title": ["Deep Learning"],
                "container-title": ["Nature"],
                "published": {"date-parts": [[2022]]},
                "DOI": "10.1234/abc",
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload

        with patch("httpx.get", return_value=mock_resp):
            result = resolve_doi("10.1234/abc")

        assert result["title"] == "Deep Learning"
        assert "Smith" in result["authors"]
        assert result["venue"] == "Nature"
        assert result["year"] == "2022"
        assert result["doi"] == "10.1234/abc"

    def test_404_raises_value_error(self) -> None:
        """404 response → ValueError (DOI not found)."""
        from hallmark.baselines._agentic_tools import resolve_doi

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with (
            patch("httpx.get", return_value=mock_resp),
            pytest.raises(ValueError, match="DOI not found"),
        ):
            resolve_doi("10.9999/notexist")

    def test_500_raises_runtime_error(self) -> None:
        """Non-200/non-404 response → RuntimeError."""
        from hallmark.baselines._agentic_tools import resolve_doi

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("httpx.get", return_value=mock_resp), pytest.raises(RuntimeError, match="500"):
            resolve_doi("10.1234/abc")

    def test_network_error_raises_runtime_error(self) -> None:
        """httpx.RequestError → RuntimeError."""
        import httpx

        from hallmark.baselines._agentic_tools import resolve_doi

        with (
            patch("httpx.get", side_effect=httpx.RequestError("connection refused")),
            pytest.raises(RuntimeError, match="Network error"),
        ):
            resolve_doi("10.1234/abc")

    def test_malformed_doi_raises_value_error(self) -> None:
        """String with no valid DOI pattern → ValueError."""
        from hallmark.baselines._agentic_tools import resolve_doi

        with pytest.raises(ValueError, match="Malformed DOI"):
            resolve_doi("not-a-doi-at-all")


class TestSearchCrossref:
    def test_success_returns_list(self) -> None:
        """200 response with items → list of normalised dicts."""
        from hallmark.baselines._agentic_tools import search_crossref

        payload = {
            "message": {
                "items": [
                    {
                        "author": [{"family": "Doe", "given": "Jane"}],
                        "title": ["Attention Mechanisms"],
                        "container-title": ["ICML"],
                        "published": {"date-parts": [[2023]]},
                        "DOI": "10.5555/1234",
                    }
                ]
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload

        with patch("httpx.get", return_value=mock_resp):
            results = search_crossref("attention mechanisms")

        assert len(results) == 1
        assert results[0]["title"] == "Attention Mechanisms"
        assert "Doe" in results[0]["authors"]
        assert results[0]["venue"] == "ICML"
        assert results[0]["year"] == "2023"
        assert results[0]["doi"] == "10.5555/1234"

    def test_non_200_raises_runtime_error(self) -> None:
        """Non-200 response → RuntimeError."""
        from hallmark.baselines._agentic_tools import search_crossref

        mock_resp = MagicMock()
        mock_resp.status_code = 503

        with patch("httpx.get", return_value=mock_resp), pytest.raises(RuntimeError, match="503"):
            search_crossref("some query")

    def test_network_error_raises_runtime_error(self) -> None:
        """httpx.RequestError → RuntimeError."""
        import httpx

        from hallmark.baselines._agentic_tools import search_crossref

        with (
            patch("httpx.get", side_effect=httpx.RequestError("timeout")),
            pytest.raises(RuntimeError, match="network error"),
        ):
            search_crossref("some query")

    def test_empty_items_returns_empty_list(self) -> None:
        """Response with no items → empty list (no error)."""
        from hallmark.baselines._agentic_tools import search_crossref

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"items": []}}

        with patch("httpx.get", return_value=mock_resp):
            results = search_crossref("nonexistent paper xyz")

        assert results == []


class TestSearchOpenalex:
    def test_success_returns_list(self) -> None:
        """200 response with results → list of normalised dicts."""
        from hallmark.baselines._agentic_tools import search_openalex

        payload = {
            "results": [
                {
                    "title": "Graph Neural Networks",
                    "authorships": [
                        {"author": {"display_name": "Alice Chen"}},
                        {"author": {"display_name": "Bob Lee"}},
                    ],
                    "primary_location": {"source": {"display_name": "NeurIPS"}},
                    "publication_year": 2021,
                    "doi": "https://doi.org/10.9999/gnn",
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload

        with patch("httpx.get", return_value=mock_resp):
            results = search_openalex("graph neural networks")

        assert len(results) == 1
        assert results[0]["title"] == "Graph Neural Networks"
        assert "Alice Chen" in results[0]["authors"]
        assert results[0]["venue"] == "NeurIPS"
        assert results[0]["year"] == "2021"
        # doi prefix stripped
        assert results[0]["doi"] == "10.9999/gnn"

    def test_non_200_raises_runtime_error(self) -> None:
        """Non-200 response → RuntimeError."""
        from hallmark.baselines._agentic_tools import search_openalex

        mock_resp = MagicMock()
        mock_resp.status_code = 429

        with patch("httpx.get", return_value=mock_resp), pytest.raises(RuntimeError, match="429"):
            search_openalex("test")

    def test_network_error_raises_runtime_error(self) -> None:
        """httpx.RequestError → RuntimeError."""
        import httpx

        from hallmark.baselines._agentic_tools import search_openalex

        with (
            patch("httpx.get", side_effect=httpx.RequestError("connection reset")),
            pytest.raises(RuntimeError, match="network error"),
        ):
            search_openalex("test")


class TestSearchArxiv:
    _ATOM_RESPONSE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <title>Transformers for Citation Verification</title>
    <author><name>Alice Wang</name></author>
    <author><name>Bob Kim</name></author>
    <published>2023-06-15T00:00:00Z</published>
    <link title="doi" href="https://doi.org/10.48550/arXiv.2306.12345"/>
  </entry>
</feed>"""

    def test_success_returns_list(self) -> None:
        """200 Atom response → list of normalised dicts."""
        from hallmark.baselines._agentic_tools import search_arxiv

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = self._ATOM_RESPONSE

        with patch("httpx.get", return_value=mock_resp):
            results = search_arxiv("citation verification")

        assert len(results) == 1
        assert results[0]["title"] == "Transformers for Citation Verification"
        assert "Alice Wang" in results[0]["authors"]
        assert results[0]["venue"] == "arXiv"
        assert results[0]["year"] == "2023"
        assert "arXiv.2306.12345" in results[0]["doi"]

    def test_non_200_raises_runtime_error(self) -> None:
        """Non-200 response → RuntimeError."""
        from hallmark.baselines._agentic_tools import search_arxiv

        mock_resp = MagicMock()
        mock_resp.status_code = 503

        with patch("httpx.get", return_value=mock_resp), pytest.raises(RuntimeError, match="503"):
            search_arxiv("test query")

    def test_network_error_raises_runtime_error(self) -> None:
        """httpx.RequestError → RuntimeError."""
        import httpx

        from hallmark.baselines._agentic_tools import search_arxiv

        with (
            patch("httpx.get", side_effect=httpx.RequestError("arXiv unreachable")),
            pytest.raises(RuntimeError, match="network error"),
        ):
            search_arxiv("test query")

    def test_empty_feed_returns_empty_list(self) -> None:
        """Atom feed with no entries → empty list."""
        from hallmark.baselines._agentic_tools import search_arxiv

        empty_atom = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = empty_atom

        with patch("httpx.get", return_value=mock_resp):
            results = search_arxiv("completely obscure title xyz")

        assert results == []


# ---------------------------------------------------------------------------
# Regression: agentic Anthropic loop must thread the system_prompt argument
# through to the Anthropic messages.create(system=...) kwarg.  Previously
# the loop hard-coded `system=SYSTEM_PROMPT`, silently ignoring the
# BTU_SYSTEM_PROMPT threaded in by verify_agentic_btu_anthropic.
# ---------------------------------------------------------------------------


class TestAgenticBtuAnthropicSystemPrompt:
    def test_uses_btu_system_prompt(self, tmp_path: Path) -> None:
        """verify_agentic_btu_anthropic sends BTU_SYSTEM_PROMPT, not SYSTEM_PROMPT."""
        from hallmark.baselines.llm_agentic import (
            BTU_SYSTEM_PROMPT,
            SYSTEM_PROMPT,
            verify_agentic_btu_anthropic,
        )

        verdict = json.dumps({"label": "VALID", "confidence": 0.9, "reason": "BTU verified"})
        resp = _make_anthropic_text_response(verdict)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = resp
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            verify_agentic_btu_anthropic(
                [_make_entry()],
                model=ANTHROPIC_MODEL,
                api_key="test-key",
                cache_db_path=tmp_path / "cache.sqlite",
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == BTU_SYSTEM_PROMPT
        # And it must not be the generic multi-tool prompt
        assert call_kwargs["system"] != SYSTEM_PROMPT

    def test_non_btu_uses_generic_system_prompt(self, tmp_path: Path) -> None:
        """verify_agentic_anthropic (multi-tool) still sends SYSTEM_PROMPT."""
        from hallmark.baselines.llm_agentic import (
            BTU_SYSTEM_PROMPT,
            SYSTEM_PROMPT,
            verify_agentic_anthropic,
        )

        verdict = json.dumps({"label": "VALID", "confidence": 0.9, "reason": "ok"})
        resp = _make_anthropic_text_response(verdict)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = resp
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            verify_agentic_anthropic(
                [_make_entry()],
                model=ANTHROPIC_MODEL,
                api_key="test-key",
                cache_db_path=tmp_path / "cache.sqlite",
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == SYSTEM_PROMPT
        assert call_kwargs["system"] != BTU_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Bug-fix: agentic _load_checkpoint skip_failed + retry_failed threading
# ---------------------------------------------------------------------------


class TestAgenticCheckpointSkipFailed:
    """_load_checkpoint(skip_failed=True) must drop [Agentic error] entries
    while preserving successful ones, including the priority-preserve case."""

    def _write_checkpoint(self, path: Path, records: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in records))

    def _make_pred_record(self, key: str, label: str, confidence: float, reason: str) -> dict:
        return {
            "bibtex_key": key,
            "label": label,
            "confidence": confidence,
            "reason": reason,
            "wall_clock_seconds": 1.0,
            "api_calls": 1,
            "api_sources_queried": [],
        }

    def test_skip_failed_drops_agentic_error_entries(self, tmp_path: Path) -> None:
        """Entries with [Agentic error] reason are excluded when skip_failed=True."""
        path = tmp_path / "ckpt.jsonl"
        records = [
            self._make_pred_record("ok-1", "VALID", 0.9, "resolved"),
            self._make_pred_record("err-1", "UNCERTAIN", 0.5, "[Agentic error] timeout"),
        ]
        self._write_checkpoint(path, records)

        loaded = _load_checkpoint(path, skip_failed=True)
        assert "ok-1" in loaded
        assert "err-1" not in loaded

    def test_skip_failed_keeps_later_success_for_same_key(self, tmp_path: Path) -> None:
        """[Agentic error] followed by success for same key: success is kept."""
        path = tmp_path / "ckpt.jsonl"
        records = [
            self._make_pred_record("key-a", "UNCERTAIN", 0.5, "[Agentic error] transient"),
            self._make_pred_record("key-a", "HALLUCINATED", 0.88, "confirmed bad DOI"),
        ]
        self._write_checkpoint(path, records)

        loaded = _load_checkpoint(path, skip_failed=True)
        assert loaded["key-a"].label == "HALLUCINATED"
        assert loaded["key-a"].confidence == pytest.approx(0.88)

    def test_skip_failed_preserves_prior_success_when_error_follows(self, tmp_path: Path) -> None:
        """Success seen first, then [Agentic error] for same key: success is kept.

        This mirrors the Bug 3 scenario but for agentic checkpoints.
        """
        path = tmp_path / "ckpt.jsonl"
        records = [
            self._make_pred_record("key-b", "VALID", 0.95, "all fields match"),
            self._make_pred_record("key-b", "UNCERTAIN", 0.5, "[Agentic error] network drop"),
        ]
        self._write_checkpoint(path, records)

        loaded = _load_checkpoint(path, skip_failed=True)
        assert "key-b" in loaded
        assert loaded["key-b"].label == "VALID"
        assert loaded["key-b"].confidence == pytest.approx(0.95)

    def test_retry_failed_flows_through_verify_agentic_openai(self, tmp_path: Path) -> None:
        """verify_agentic_openai(retry_failed=True) must re-run [Agentic error] entries."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "agentic_openai_gpt-5.1.jsonl"

        # Write a checkpoint with one error entry
        record = {
            "bibtex_key": "test2024",
            "label": "UNCERTAIN",
            "confidence": 0.5,
            "reason": "[Agentic error] simulated",
            "wall_clock_seconds": 0.1,
            "api_calls": 1,
            "api_sources_queried": [],
        }
        ckpt_path.write_text(json.dumps(record) + "\n")

        verdict = json.dumps({"label": "VALID", "confidence": 0.9, "reason": "ok"})

        def _make_openai_final_msg(content: str) -> MagicMock:
            msg = MagicMock()
            msg.content = content
            msg.tool_calls = None
            msg.refusal = None
            return msg

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(
                message=_make_openai_final_msg(verdict),
                finish_reason="stop",
            )
        ]
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            preds = verify_agentic_openai(
                [_make_entry()],
                model="gpt-5.1",
                api_key="test-key",
                checkpoint_dir=ckpt_dir,
                retry_failed=True,
                cache_db_path=tmp_path / "cache.sqlite",
            )

        # The errored entry was retried — API must have been called
        assert mock_client.chat.completions.create.call_count >= 1
        assert len(preds) == 1

    def test_retry_failed_false_preserves_agentic_error_entries(self, tmp_path: Path) -> None:
        """retry_failed=False (default) must NOT skip [Agentic error] entries."""
        path = tmp_path / "ckpt.jsonl"
        record = {
            "bibtex_key": "test2024",
            "label": "UNCERTAIN",
            "confidence": 0.5,
            "reason": "[Agentic error] simulated",
            "wall_clock_seconds": 0.1,
            "api_calls": 1,
            "api_sources_queried": [],
        }
        path.write_text(json.dumps(record) + "\n")

        loaded = _load_checkpoint(path, skip_failed=False)
        assert "test2024" in loaded
        assert loaded["test2024"].reason.startswith("[Agentic error]")
