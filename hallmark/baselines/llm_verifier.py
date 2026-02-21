"""LLM-based verification baseline.

Uses GPT-5.1 (OpenAI), Claude Sonnet 4.5 (Anthropic), or OpenRouter models
to verify BibTeX entries by prompting the model to assess whether a citation
appears genuine or hallucinated.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from hallmark.baselines.common import fallback_predictions
from hallmark.dataset.schema import BenchmarkEntry, Prediction

logger = logging.getLogger(__name__)

VERIFICATION_PROMPT = """\
You are a citation verification expert. Analyze the following BibTeX entry \
and determine if it is a VALID real publication or a HALLUCINATED (fabricated) citation.

BibTeX entry:
```bibtex
{bibtex}
```

Consider:
1. Is the title plausible and does it match known work by these authors?
2. Are the authors real researchers in this field?
3. Is the venue (journal/conference) real?
4. Does the year make sense?
5. If a DOI is present, does it look properly formatted?

Respond with JSON only:
{{
    "label": "VALID" or "HALLUCINATED",
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation"
}}"""

OPENROUTER_MODELS: dict[str, str] = {
    "deepseek-r1": "deepseek/deepseek-r1",
    "deepseek-v3": "deepseek/deepseek-v3.2",
    "qwen": "qwen/qwen3-235b-a22b-2507",
    "mistral": "mistralai/mistral-large-2512",
    "gemini-flash": "google/gemini-2.5-flash",
}


def _verify_entries(
    entries: list[BenchmarkEntry],
    call_fn: Callable[[str], str],
    source_prefix: str,
    model: str,
) -> list[Prediction]:
    """Shared verification loop for all LLM providers.

    Args:
        entries: Benchmark entries to verify.
        call_fn: Function that takes a prompt string and returns raw response text.
        source_prefix: Provider name for metadata (e.g. "openai", "anthropic").
        model: Model identifier for metadata.
    """
    predictions = []

    for entry in entries:
        start = time.time()
        bibtex = entry.to_bibtex()
        prompt = VERIFICATION_PROMPT.format(bibtex=bibtex)

        try:
            content = call_fn(prompt)
            pred = _parse_llm_response(content, entry.bibtex_key)
        except Exception as e:
            logger.warning(f"{source_prefix} API error for {entry.bibtex_key}: {e}")
            pred = Prediction(
                bibtex_key=entry.bibtex_key,
                label="UNCERTAIN",
                confidence=0.5,
                reason=f"[Error fallback] API error: {e}",
            )

        pred.wall_clock_seconds = time.time() - start
        pred.api_calls = 1
        pred.api_sources_queried = [f"{source_prefix}/{model}"]
        predictions.append(pred)

    return predictions


def _verify_with_openai_compatible(
    entries: list[BenchmarkEntry],
    model: str,
    api_key: str | None,
    base_url: str | None,
    source_prefix: str,
    **kwargs: Any,
) -> list[Prediction]:
    """Verification via OpenAI-SDK-compatible providers."""
    try:
        import openai
    except ImportError:
        logger.error("openai package not installed. Install with: pip install openai")
        return fallback_predictions(entries, reason="LLM baseline unavailable")

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    def call_fn(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=256,
            seed=42,
        )
        return str(resp.choices[0].message.content).strip()

    return _verify_entries(entries, call_fn, source_prefix, model)


def verify_with_openai(
    entries: list[BenchmarkEntry],
    model: str = "gpt-5.1",
    api_key: str | None = None,
    **kwargs: Any,
) -> list[Prediction]:
    """Verify entries using OpenAI API."""
    return _verify_with_openai_compatible(
        entries,
        model=model,
        api_key=api_key,
        base_url=None,
        source_prefix="openai",
    )


def verify_with_openrouter(
    entries: list[BenchmarkEntry],
    model: str = "deepseek/deepseek-r1",
    api_key: str | None = None,
    **kwargs: Any,
) -> list[Prediction]:
    """Verify entries using OpenRouter API (100+ models via OpenAI-compatible endpoint)."""
    return _verify_with_openai_compatible(
        entries,
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        source_prefix="openrouter",
    )


def verify_with_anthropic(
    entries: list[BenchmarkEntry],
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str | None = None,
) -> list[Prediction]:
    """Verify entries using Anthropic API."""
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed. Install with: pip install anthropic")
        return fallback_predictions(entries, reason="LLM baseline unavailable")

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def call_fn(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(resp.content[0].text).strip()

    return _verify_entries(entries, call_fn, "anthropic", model)


def _parse_llm_response(content: str, bibtex_key: str) -> Prediction:
    """Parse LLM JSON response into a Prediction."""
    original_content = content

    def _try_parse(text: str) -> dict | None:
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "label" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    # Try bare JSON first (no fences)
    if "```" not in content:
        data = _try_parse(content)
    else:
        # Iterate over all fenced blocks (odd-indexed after split on ```)
        data = None
        blocks = content.split("```")
        for i in range(1, len(blocks), 2):
            block = blocks[i]
            # Strip language identifier (e.g., "json\n" or "JSON\n")
            if block.lower().startswith("json"):
                block = block[4:]
            data = _try_parse(block)
            if data is not None:
                break

    if data is None:
        logger.warning(f"Failed to parse LLM response for {bibtex_key}")
        return Prediction(
            bibtex_key=bibtex_key,
            label="UNCERTAIN",
            confidence=0.5,
            reason=f"[Error fallback] Parse error: {original_content[:100]}",
        )

    try:
        label = data.get("label", "UNCERTAIN").upper()
        if label not in ("VALID", "HALLUCINATED"):
            label = "UNCERTAIN"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return Prediction(
            bibtex_key=bibtex_key,
            label=label,
            confidence=confidence,
            reason=data.get("reason", ""),
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to extract fields from LLM response for {bibtex_key}: {e}")
        return Prediction(
            bibtex_key=bibtex_key,
            label="UNCERTAIN",
            confidence=0.5,
            reason=f"[Error fallback] Parse error: {original_content[:100]}",
        )
