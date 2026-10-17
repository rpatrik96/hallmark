"""Provider dispatch in scripts/parallel_resume_test_public.py.

The native-vs-OpenRouter comparison relies on both arms going through the same
call_one(); these tests pin what each provider is sent and what is recorded.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

anthropic = pytest.importorskip("anthropic")

_ENTRY = {
    "bibtex_key": "k0",
    "bibtex_type": "article",
    "fields": {"title": "T", "author": "A", "year": "2024"},
}
_REPLY = '{"label": "VALID", "confidence": 0.9, "reason": "ok"}'


def _load_script() -> Any:
    script = Path(__file__).resolve().parent.parent / "scripts" / "parallel_resume_test_public.py"
    spec = importlib.util.spec_from_file_location("parallel_resume_providers", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeMessages:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="..."),
                SimpleNamespace(type="text", text=_REPLY),
            ]
        )


def test_anthropic_native_call() -> None:
    module = _load_script()
    client = anthropic.Anthropic(api_key="test-key")
    fake = _FakeMessages()
    client.messages = fake  # type: ignore[assignment]

    rec = module.call_one(client, "claude-sonnet-4-6", _ENTRY, 30.0, 3, 1024, 0.0, 42, "anthropic")

    assert rec["label"] == "VALID"
    assert rec["upstream_provider"] == "anthropic"
    assert rec["api_sources_queried"] == ["anthropic/claude-sonnet-4-6"]
    assert fake.kwargs["max_tokens"] == 1024
    assert fake.kwargs["temperature"] == 0.0
    assert "seed" not in fake.kwargs
    # Same prompt as the OpenRouter arm.
    assert fake.kwargs["messages"][0]["content"] == module.build_prompt(_ENTRY, "claude-sonnet-4-6")


def test_anthropic_opus_4_7_omits_temperature() -> None:
    module = _load_script()
    client = anthropic.Anthropic(api_key="test-key")
    fake = _FakeMessages()
    client.messages = fake  # type: ignore[assignment]

    rec = module.call_one(client, "claude-opus-4-7", _ENTRY, 30.0, 3, 1024, 0.0, 42, "anthropic")

    assert rec["label"] == "VALID"
    assert "temperature" not in fake.kwargs
    assert fake.kwargs["max_tokens"] == 1024


def test_openrouter_records_upstream_provider() -> None:
    module = _load_script()
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=_REPLY))],
        model_extra={"provider": "Amazon Bedrock"},
    )
    sent: dict[str, Any] = {}

    def _create(**kwargs: Any) -> Any:
        sent.update(kwargs)
        return resp

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    rec = module.call_one(
        client, "anthropic/claude-sonnet-4.6", _ENTRY, 30.0, 3, 1024, 0.0, 42, "openrouter"
    )

    assert rec["upstream_provider"] == "Amazon Bedrock"
    assert sent["max_completion_tokens"] == 1024
    assert sent["temperature"] == 0.0


def test_anthropic_rate_limit_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    client = anthropic.Anthropic(api_key="test-key")
    calls = {"n": 0}
    ok = _FakeMessages()

    def _create(**kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise anthropic.RateLimitError(
                "slow down",
                response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
                body=None,
            )
        return ok.create(**kwargs)

    client.messages = SimpleNamespace(create=_create)  # type: ignore[assignment]
    rec = module.call_one(client, "claude-opus-4-7", _ENTRY, 30.0, 3, 1024, 0.0, 42, "anthropic")
    assert calls["n"] == 2
    assert rec["label"] == "VALID"
