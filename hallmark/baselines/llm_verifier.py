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
from pathlib import Path
from typing import Any

from hallmark.baselines.common import fallback_predictions
from hallmark.dataset.schema import BlindEntry, Prediction

logger = logging.getLogger(__name__)


def _log_api_call(
    log_dir: Path,
    bibtex_key: str,
    prompt: str,
    raw_response: str,
    parsed: Prediction,
    model: str,
    elapsed: float,
) -> None:
    """Write full API call details to a JSON file for debugging."""
    from datetime import datetime, timezone

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{bibtex_key}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bibtex_key": bibtex_key,
        "model": model,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed": {
            "label": parsed.label,
            "confidence": parsed.confidence,
            "reason": parsed.reason,
        },
        "elapsed_seconds": elapsed,
    }
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


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


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Prediction]:
    """Load previously saved predictions from a JSONL checkpoint file."""
    if not checkpoint_path.exists():
        return {}
    existing: dict[str, Prediction] = {}
    for line in checkpoint_path.read_text().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        existing[data["bibtex_key"]] = Prediction(
            bibtex_key=data["bibtex_key"],
            label=data["label"],
            confidence=data["confidence"],
            reason=data.get("reason", ""),
            wall_clock_seconds=data.get("wall_clock_seconds", 0.0),
            api_calls=data.get("api_calls", 0),
            api_sources_queried=data.get("api_sources_queried", []),
        )
    return existing


def _append_checkpoint(checkpoint_path: Path, pred: Prediction) -> None:
    """Append a single prediction to the JSONL checkpoint file."""
    data = {
        "bibtex_key": pred.bibtex_key,
        "label": pred.label,
        "confidence": pred.confidence,
        "reason": pred.reason,
        "wall_clock_seconds": pred.wall_clock_seconds,
        "api_calls": pred.api_calls,
        "api_sources_queried": pred.api_sources_queried,
    }
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(data) + "\n")


def _verify_entries(
    entries: list[BlindEntry],
    call_fn: Callable[[str], str],
    source_prefix: str,
    model: str,
    log_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    max_consecutive_failures: int = 3,
) -> list[Prediction]:
    """Shared verification loop for all LLM providers.

    Args:
        entries: Benchmark entries to verify.
        call_fn: Function that takes a prompt string and returns raw response text.
        source_prefix: Provider name for metadata (e.g. "openai", "anthropic").
        model: Model identifier for metadata.
        log_dir: Optional directory to write per-entry JSON logs for debugging.
        checkpoint_dir: Directory for JSONL checkpoint files. When provided,
            completed predictions are saved after each entry and resumed on
            subsequent calls.
        max_consecutive_failures: Abort after this many consecutive API errors.
    """
    checkpoint_path: Path | None = None
    completed: dict[str, Prediction] = {}

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model.replace("/", "_")
        checkpoint_path = checkpoint_dir / f"{source_prefix}_{safe_model}.jsonl"
        completed = _load_checkpoint(checkpoint_path)
        if completed:
            logger.info(
                "Resuming %s/%s: %d entries already completed",
                source_prefix,
                model,
                len(completed),
            )

    predictions = list(completed.values())
    consecutive_failures = 0

    for entry in entries:
        if entry.bibtex_key in completed:
            continue

        start = time.time()
        bibtex = entry.to_bibtex()
        prompt = VERIFICATION_PROMPT.format(bibtex=bibtex)

        try:
            content = call_fn(prompt)
            pred = _parse_llm_response(content, entry.bibtex_key)
            consecutive_failures = 0
            if log_dir is not None:
                _log_api_call(
                    log_dir, entry.bibtex_key, prompt, content, pred, model, time.time() - start
                )
        except Exception as e:
            logger.warning(f"{source_prefix} API error for {entry.bibtex_key}: {e}")
            consecutive_failures += 1
            pred = Prediction(
                bibtex_key=entry.bibtex_key,
                label="UNCERTAIN",
                confidence=0.5,
                reason=f"[Error fallback] API error: {e}",
            )
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    "Aborting %s: %d consecutive API failures — likely a persistent issue",
                    source_prefix,
                    max_consecutive_failures,
                )
                pred.wall_clock_seconds = time.time() - start
                pred.api_calls = 1
                pred.api_sources_queried = [f"{source_prefix}/{model}"]
                predictions.append(pred)
                if checkpoint_path is not None:
                    _append_checkpoint(checkpoint_path, pred)
                # Fill remaining entries with fallback predictions
                processed_keys = {p.bibtex_key for p in predictions}
                remaining = [e for e in entries if e.bibtex_key not in processed_keys]
                for rem_entry in remaining:
                    if rem_entry.bibtex_key in completed:
                        continue
                    fallback = Prediction(
                        bibtex_key=rem_entry.bibtex_key,
                        label="UNCERTAIN",
                        confidence=0.5,
                        reason=f"[Error fallback] Skipped after {max_consecutive_failures} consecutive API failures",
                        api_sources_queried=[f"{source_prefix}/{model}"],
                    )
                    predictions.append(fallback)
                    if checkpoint_path is not None:
                        _append_checkpoint(checkpoint_path, fallback)
                return predictions

        pred.wall_clock_seconds = time.time() - start
        pred.api_calls = 1
        pred.api_sources_queried = [f"{source_prefix}/{model}"]
        predictions.append(pred)

        if checkpoint_path is not None:
            _append_checkpoint(checkpoint_path, pred)

    return predictions


def _verify_with_openai_compatible(
    entries: list[BlindEntry],
    model: str,
    api_key: str | None,
    base_url: str | None,
    source_prefix: str,
    log_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
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
    client_kwargs.setdefault("max_retries", 5)
    client_kwargs.setdefault("timeout", 120.0)
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

    return _verify_entries(
        entries, call_fn, source_prefix, model, log_dir=log_dir, checkpoint_dir=checkpoint_dir
    )


def verify_with_openai(
    entries: list[BlindEntry],
    model: str = "gpt-5.1",
    api_key: str | None = None,
    log_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    **kwargs: Any,
) -> list[Prediction]:
    """Verify entries using OpenAI API."""
    return _verify_with_openai_compatible(
        entries,
        model=model,
        api_key=api_key,
        base_url=None,
        source_prefix="openai",
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
    )


def verify_with_openrouter(
    entries: list[BlindEntry],
    model: str = "deepseek/deepseek-r1",
    api_key: str | None = None,
    log_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    **kwargs: Any,
) -> list[Prediction]:
    """Verify entries using OpenRouter API (100+ models via OpenAI-compatible endpoint)."""
    return _verify_with_openai_compatible(
        entries,
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        source_prefix="openrouter",
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
    )


def verify_with_anthropic(
    entries: list[BlindEntry],
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str | None = None,
    log_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> list[Prediction]:
    """Verify entries using Anthropic API."""
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed. Install with: pip install anthropic")
        return fallback_predictions(entries, reason="LLM baseline unavailable")

    client_kwargs_anth: dict[str, Any] = {"max_retries": 5, "timeout": 120.0}
    if api_key:
        client_kwargs_anth["api_key"] = api_key
    client = anthropic.Anthropic(**client_kwargs_anth)

    def call_fn(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(resp.content[0].text).strip()

    return _verify_entries(
        entries, call_fn, "anthropic", model, log_dir=log_dir, checkpoint_dir=checkpoint_dir
    )


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
