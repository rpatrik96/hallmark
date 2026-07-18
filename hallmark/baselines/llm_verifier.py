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

When the entry is HALLUCINATED, classify the hallucination mode using exactly one of:
`fabricated_doi`, `nonexistent_venue`, `placeholder_authors`, `future_date`,
`chimeric_title`, `wrong_venue`, `swapped_authors`, `preprint_as_published`,
`hybrid_fabrication`, `near_miss_title`, `plausible_fabrication`,
`merged_citation`, `partial_author_list`, `arxiv_version_mismatch`.
Brief definitions:
- `fabricated_doi`: DOI does not resolve / is invented.
- `nonexistent_venue`: venue/journal does not exist.
- `placeholder_authors`: authors are placeholders ("Author1", "et al." alone, etc.).
- `future_date`: year is in the future relative to publication.
- `chimeric_title`: title combines fragments from multiple real works.
- `wrong_venue`: real paper but cited at wrong venue.
- `swapped_authors`: authors swapped or mismatched against the real paper.
- `preprint_as_published`: arXiv preprint cited as published in a venue.
- `hybrid_fabrication`: real DOI but other metadata (authors/title) doesn't match the DOI target.
- `near_miss_title`: title differs from a real paper by small but meaningful edits.
- `plausible_fabrication`: entirely fabricated yet plausible-sounding paper.
- `merged_citation`: metadata combined from two real papers.
- `partial_author_list`: real paper but author list is incomplete.
- `arxiv_version_mismatch`: arXiv version cited as a different version (or as published).

Respond with JSON only:
{{
    "label": "VALID" or "HALLUCINATED" or "UNCERTAIN",
    "confidence": 0.0 to 1.0,
    "predicted_hallucination_type": "<one of the 14 types above, or null>",
    "reason": "brief explanation"
}}
`predicted_hallucination_type` MUST be null when label is VALID or UNCERTAIN."""

# Addendum appended to VERIFICATION_PROMPT when cutoff_aware=True. Tests the
# H2 hypothesis: when explicitly reminded of the training cutoff, do LLMs route
# post-cutoff citations to UNCERTAIN rather than over-flagging them as
# HALLUCINATED?  The wording deliberately permits UNCERTAIN as a third label
# and instructs the model not to guess.
CUTOFF_AWARE_ADDENDUM = (
    "Note: your training data has a knowledge cutoff. If the citation could "
    "post-date your training data, or if you cannot recall the paper with "
    "confidence, respond with UNCERTAIN rather than HALLUCINATED or VALID. "
    "Do not guess."
)


def _build_verification_prompt(entry: BlindEntry, *, cutoff_aware: bool = False) -> str:
    """Return the verification prompt string for a single entry.

    When ``cutoff_aware`` is True, the :data:`CUTOFF_AWARE_ADDENDUM` is
    appended to the default prompt so the model is explicitly told that
    UNCERTAIN is a valid output for post-cutoff citations.
    """
    prompt = VERIFICATION_PROMPT.format(bibtex=entry.to_bibtex())
    if cutoff_aware:
        prompt = prompt + "\n\n" + CUTOFF_AWARE_ADDENDUM
    return prompt


OPENROUTER_MODELS: dict[str, str] = {
    "deepseek-r1": "deepseek/deepseek-r1",
    "deepseek-v3": "deepseek/deepseek-v3.2",
    "qwen": "qwen/qwen3-235b-a22b-2507",
    "mistral": "mistralai/mistral-large-2512",
    "gemini-flash": "google/gemini-2.5-flash",
    # --- New entries (confirmed live via curl /api/v1/models + 5-token probe) ---
    # meta-llama/llama-4-maverick: confirmed live 2026-04-22
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    # google/gemini-2.5-pro: GA non-thinking tier (confirmed live 2026-04-22).
    # gemini-3.1-pro-preview emits unbounded thinking tokens that overrun the
    # 256-token max_completion_tokens budget, causing JSON parse failures;
    # 2.5-pro is the closest reliable "Pro-tier" alternative without that issue.
    "gemini-pro": "google/gemini-2.5-pro",
    # qwen/qwen3-vl-235b-a22b-instruct: same 235B class as the existing "qwen"
    # baseline (qwen3-235b-a22b-2507 / July 2025) but a newer release via
    # DeepInfra, explicit -instruct suffix (non-thinking). Smoke-tested
    # 2026-04-23 (provider=DeepInfra, reasoning_tokens=0). Previous attempts:
    #   qwen3.5-397b-a17b: ~40% empty responses (thinking mode ate budget)
    #   qwen3-max / qwen3.6-plus / qwen-plus: 404 Alibaba-only
    #   qwen3.5-122b-a10b: works but emits 500+ reasoning tokens per reply
    "qwen-max": "qwen/qwen3-vl-235b-a22b-instruct",
    # --- Q2 2026 frontier additions (confirmed via OpenRouter /api/v1/models 2026-05-04) ---
    # openai/gpt-5.5: non-reasoning chat model, released 2026-04-24. $5/$30 per M tok.
    "gpt-5.5": "openai/gpt-5.5",
    # openai/gpt-5.5-pro: high-capability non-reasoning variant, released 2026-04-24.
    # $30/$180 per M tok — expensive; include for completeness but budget accordingly.
    "gpt-5.5-pro": "openai/gpt-5.5-pro",
    # anthropic/claude-sonnet-4.6 via OpenRouter mirror (native path: llm_anthropic).
    # Released 2026-02-17. $3/$15 per M tok.
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    # anthropic/claude-opus-4.7 via OpenRouter mirror (native path: llm_anthropic_opus_4_7).
    # Released 2026-04-16. $5/$25 per M tok.
    "claude-opus-4-7": "anthropic/claude-opus-4.7",
    # Skipped models (Q2 2026):
    #   deepseek/deepseek-v4-pro  — thinking model (reasoning_effort high/xhigh supported);
    #                               risk of reasoning tokens consuming the 1024-tok budget.
    #   deepseek/deepseek-v4-flash — same thinking-mode issue as V4-pro.
    #   google/gemini-3.1-flash-lite-preview — supports full thinking levels (minimal→high);
    #                               same unbounded-thinking risk as gemini-3.1-pro-preview.
    #   google/gemini-3.1-pro-preview — already rejected (see comment above gemini-pro).
}

# Reference dict of OpenAI model IDs for documentation and kwarg overrides.
# The llm_openai baseline defaults to "gpt-5.1"; pass model= to use any entry here.
# Confirmed live via GET /v1/models on 2026-04-22: gpt-5.1, gpt-5.2, gpt-5.4,
# gpt-5.4-mini. Listed for future use (not yet verified): gpt-5.1-mini, gpt-5.4-nano.
OPENAI_MODELS: dict[str, str] = {
    "gpt-5.1": "gpt-5.1",
    "gpt-5.1-mini": "gpt-5.1-mini",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.4-mini": "gpt-5.4-mini",
}

# Reference dict of Anthropic model IDs for documentation and kwarg overrides.
# The llm_anthropic baseline defaults to "claude-sonnet-4-6"; pass model= to override.
# IDs sourced from Anthropic model reference (no live API verification performed).
ANTHROPIC_MODELS: dict[str, str] = {
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-opus-4-6": "claude-opus-4-6",
    "claude-opus-4-7": "claude-opus-4-7",
    "claude-haiku-4-5": "claude-haiku-4-5",
}


def _load_checkpoint(checkpoint_path: Path, *, skip_failed: bool = False) -> dict[str, Prediction]:
    """Load previously saved predictions from a JSONL checkpoint file.

    When ``skip_failed=True``, entries whose reason starts with
    ``[Error fallback]`` are omitted from the returned dict, so the main loop
    treats them as incomplete and re-attempts the API call. Use this to patch
    transient failures (e.g. network drops) without rerunning clean entries.
    """
    if not checkpoint_path.exists():
        return {}
    existing: dict[str, Prediction] = {}
    for line in checkpoint_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            # A process killed mid-write (e.g. laptop sleep/disconnect) can leave
            # a truncated final line. Skip it; the entry is simply re-attempted.
            logger.warning("Skipping malformed checkpoint line in %s", checkpoint_path)
            continue
        reason = data.get("reason", "")
        if skip_failed and reason.startswith("[Error fallback]"):
            # Intentionally drop from the "completed" set so it gets retried.
            # Keep earlier non-failed record if we already saw one for this key.
            prior = existing.get(data["bibtex_key"])
            prior_reason = prior.reason if prior is not None else ""
            if prior is None or prior_reason.startswith("[Error fallback]"):
                existing.pop(data["bibtex_key"], None)
            continue
        existing[data["bibtex_key"]] = Prediction(
            bibtex_key=data["bibtex_key"],
            label=data["label"],
            confidence=data["confidence"],
            reason=reason,
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
    prompt_fn: Callable[[BlindEntry], str] | None = None,
    retry_failed: bool = False,
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
        prompt_fn: Optional function that takes a BlindEntry and returns a prompt
            string. When provided, overrides the default VERIFICATION_PROMPT.
        retry_failed: When loading a checkpoint, treat previous
            ``[Error fallback]`` predictions as incomplete so they get
            re-attempted. Useful after a transient network outage.
    """
    checkpoint_path: Path | None = None
    completed: dict[str, Prediction] = {}

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model.replace("/", "_")
        checkpoint_path = checkpoint_dir / f"{source_prefix}_{safe_model}.jsonl"
        completed = _load_checkpoint(checkpoint_path, skip_failed=retry_failed)
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
        if prompt_fn is not None:
            prompt = prompt_fn(entry)
        else:
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
    cutoff_aware: bool = False,
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

    # Raised from 256 to 1024 so that reasoning-capable Gemini/OpenAI models
    # can emit hidden thinking tokens before the JSON verdict without the
    # response being truncated mid-structure. Non-thinking models still emit
    # ~50-80 tokens of actual content, so the increase has negligible cost.
    max_completion_tokens = int(kwargs.pop("max_completion_tokens", 1024))
    # GPT-5.5 family rejects temperature != 1 (API returns 400). Default to 1.0
    # for those models, 0.0 elsewhere; callers may still override via kwargs.
    _default_temp = 1.0 if "gpt-5.5" in model else 0.0
    temperature = float(kwargs.pop("temperature", _default_temp))
    # GPT-5.5 is a reasoning model; the default effort burns through the
    # 1024-token budget on hidden thinking and leaves the JSON output empty
    # (finish_reason='length'). Verification doesn't need chain-of-thought, so
    # default to 'none' for the gpt-5.5 family. Other models don't accept
    # reasoning_effort, so don't pass it.
    extra_call_kwargs: dict[str, Any] = {}
    if "gpt-5.5" in model:
        extra_call_kwargs["reasoning_effort"] = kwargs.pop("reasoning_effort", "none")

    def call_fn(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            seed=42,
            **extra_call_kwargs,
        )
        return str(resp.choices[0].message.content).strip()

    entry_prompt_fn: Callable[[BlindEntry], str] | None = kwargs.pop("prompt_fn", None)
    if entry_prompt_fn is None and cutoff_aware:

        def entry_prompt_fn(entry: BlindEntry) -> str:
            return _build_verification_prompt(entry, cutoff_aware=True)

    retry_failed = bool(kwargs.pop("retry_failed", False))

    return _verify_entries(
        entries,
        call_fn,
        source_prefix,
        model,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        prompt_fn=entry_prompt_fn,
        retry_failed=retry_failed,
    )


def verify_with_openai(
    entries: list[BlindEntry],
    model: str = "gpt-5.1",
    api_key: str | None = None,
    log_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    cutoff_aware: bool = False,
    **kwargs: Any,
) -> list[Prediction]:
    """Verify entries using OpenAI API.

    When ``cutoff_aware`` is True, the default prompt is extended with
    :data:`CUTOFF_AWARE_ADDENDUM`, explicitly telling the model to route
    post-cutoff citations to UNCERTAIN.
    """
    return _verify_with_openai_compatible(
        entries,
        model=model,
        api_key=api_key,
        base_url=None,
        source_prefix="openai",
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        cutoff_aware=cutoff_aware,
        **kwargs,
    )


def verify_with_openrouter(
    entries: list[BlindEntry],
    model: str = "deepseek/deepseek-r1",
    api_key: str | None = None,
    log_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    cutoff_aware: bool = False,
    **kwargs: Any,
) -> list[Prediction]:
    """Verify entries using OpenRouter API (100+ models via OpenAI-compatible endpoint).

    When ``cutoff_aware`` is True, the default prompt is extended with
    :data:`CUTOFF_AWARE_ADDENDUM`.
    """
    return _verify_with_openai_compatible(
        entries,
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        source_prefix="openrouter",
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        cutoff_aware=cutoff_aware,
        **kwargs,
    )


def verify_with_anthropic(
    entries: list[BlindEntry],
    model: str = "claude-sonnet-4-6",
    api_key: str | None = None,
    log_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    cutoff_aware: bool = False,
    retry_failed: bool = False,
    **kwargs: Any,
) -> list[Prediction]:
    """Verify entries using Anthropic API.

    The default model is ``claude-sonnet-4-6``.  Pass ``model=`` with a value
    from :data:`ANTHROPIC_MODELS` to override.

    When ``cutoff_aware`` is True, the default prompt is extended with
    :data:`CUTOFF_AWARE_ADDENDUM`.
    """
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

    prompt_fn: Callable[[BlindEntry], str] | None = None
    if cutoff_aware:

        def prompt_fn(entry: BlindEntry) -> str:
            return _build_verification_prompt(entry, cutoff_aware=True)

    return _verify_entries(
        entries,
        call_fn,
        "anthropic",
        model,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        prompt_fn=prompt_fn,
        retry_failed=retry_failed,
    )


_VALID_HALLUCINATION_TYPES: frozenset[str] = frozenset(
    {
        "fabricated_doi",
        "nonexistent_venue",
        "placeholder_authors",
        "future_date",
        "chimeric_title",
        "wrong_venue",
        "swapped_authors",
        "preprint_as_published",
        "hybrid_fabrication",
        "near_miss_title",
        "plausible_fabrication",
        "merged_citation",
        "partial_author_list",
        "arxiv_version_mismatch",
    }
)


def _parse_hallucination_type(raw: object, bibtex_key: str, label: str) -> str | None:
    """Validate and return ``predicted_hallucination_type`` from LLM JSON output.

    Rules:
    - If label is not HALLUCINATED, always return None.
    - If raw is None / "null" / "None", return None.
    - If raw is a string not in the 14 valid types, log a warning and return None.
    - Otherwise return the validated string.
    """
    if label != "HALLUCINATED":
        return None
    if raw is None or str(raw).lower() in ("null", "none", ""):
        return None
    if isinstance(raw, str) and raw in _VALID_HALLUCINATION_TYPES:
        return raw
    logger.warning("Unknown predicted_hallucination_type %r for %s — ignoring", raw, bibtex_key)
    return None


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
        # Salvage path: truncated JSON from verbose reasoning models (e.g.,
        # Gemini 2.5 Pro) often contains a parseable label+confidence before
        # the response is cut off. Regex-extract them rather than giving up.
        import re

        label_match = re.search(
            r'"label"\s*:\s*"(VALID|HALLUCINATED|UNCERTAIN)"',
            original_content,
            re.IGNORECASE,
        )
        conf_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', original_content)
        if label_match is not None:
            from typing import Literal, cast

            salvaged_label = cast(
                Literal["VALID", "HALLUCINATED", "UNCERTAIN"],
                label_match.group(1).upper(),
            )
            salvaged_conf = float(conf_match.group(1)) if conf_match else 0.5
            salvaged_conf = max(0.0, min(1.0, salvaged_conf))
            return Prediction(
                bibtex_key=bibtex_key,
                label=salvaged_label,
                confidence=salvaged_conf,
                reason="[Salvaged] label/confidence extracted from truncated JSON",
            )

        logger.warning(f"Failed to parse LLM response for {bibtex_key}")
        return Prediction(
            bibtex_key=bibtex_key,
            label="UNCERTAIN",
            confidence=0.5,
            reason=f"[Error fallback] Parse error: {original_content[:100]}",
        )

    try:
        label = data.get("label", "UNCERTAIN").upper()
        if label not in ("VALID", "HALLUCINATED", "UNCERTAIN"):
            label = "UNCERTAIN"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        predicted_hallucination_type = _parse_hallucination_type(
            data.get("predicted_hallucination_type"), bibtex_key, label
        )

        return Prediction(
            bibtex_key=bibtex_key,
            label=label,
            confidence=confidence,
            reason=data.get("reason", ""),
            predicted_hallucination_type=predicted_hallucination_type,
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to extract fields from LLM response for {bibtex_key}: {e}")
        return Prediction(
            bibtex_key=bibtex_key,
            label="UNCERTAIN",
            confidence=0.5,
            reason=f"[Error fallback] Parse error: {original_content[:100]}",
        )
