"""Agentic LLM citation-verification baseline.

The model receives a BibTeX entry and may call up to MAX_TOOL_CALLS tools
(resolve_doi, search_crossref, search_openalex, search_arxiv) before emitting
a structured JSON verdict.  This closes the compute-asymmetry gap with
bibtex-updater, which also makes multiple API calls per entry.

Two backends are supported:
- OpenAI (GPT-5.1) via ``verify_agentic_openai``
- Anthropic (Claude Sonnet 4.5) via ``verify_agentic_anthropic``

Tool results are cached in SQLite (.cache/agentic_tools.sqlite) keyed on
sha256(tool_name::canonical_args_json) for reproducibility.

Attribution tracking fields logged per-prediction:
    tool_sequence      list of tool names called, in order
    total_tokens       sum of input+output tokens (if reported by API)
    tool_call_count    number of tool invocations
    final_verdict_source  "parametric" | "tool"
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from hallmark.baselines._agentic_cache import AgenticToolCache, cache_key
from hallmark.baselines._agentic_tools import (
    BTU_TOOL_DEFINITION,
    TOOL_DEFINITIONS,
    TOOL_REGISTRY,
)
from hallmark.baselines._cache import retry_with_backoff
from hallmark.baselines.common import fallback_predictions
from hallmark.baselines.llm_verifier import _parse_llm_response
from hallmark.dataset.schema import BlindEntry, Prediction

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 5

# Model identifiers — pinned to date-stamped versions for reproducibility.
OPENAI_MODEL = "gpt-5.1"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"

SYSTEM_PROMPT = """\
You are a citation verification expert with access to bibliographic lookup tools.
Your task: determine whether a BibTeX entry is a VALID real publication or a \
HALLUCINATED (fabricated) citation.

Strategy:
1. Inspect the entry for obvious red flags (fake DOI prefix, future year, \
placeholder authors).
2. Use tools to cross-reference: resolve the DOI, or search by title/author.
3. After gathering evidence (or after finding sufficient signal), emit your verdict.

When you are ready to give your final answer, output ONLY valid JSON — no prose, \
no markdown fences:
{
    "label": "VALID" or "HALLUCINATED",
    "confidence": 0.0 to 1.0,
    "reason": "concise explanation citing the evidence you found"
}

Do NOT output the JSON until you have used enough tools or determined that \
parametric knowledge is sufficient.
"""

BTU_SYSTEM_PROMPT = """\
You are a citation verification expert with access to a specialised tool, \
`verify_with_bibtex_updater`, which cross-references a BibTeX entry against \
CrossRef, DBLP, and Semantic Scholar and returns a structured verdict.

Strategy:
1. For almost every entry, call `verify_with_bibtex_updater` once with the \
exact BibTeX string you were given.
2. Interpret the returned `status` field: statuses like `verified`, \
`url_verified`, or `published_version_exists` suggest VALID; statuses like \
`not_found`, `title_mismatch`, `author_mismatch`, `hallucinated`, `future_date`, \
`doi_not_found`, or `venue_mismatch` suggest HALLUCINATED.
3. If the tool returns `api_error` or you suspect the tool is wrong (e.g. it \
reports `verified` but the entry still looks suspicious on inspection), apply \
your own judgment — the tool is evidence, not an oracle.
4. If the first call is unambiguous, do NOT call again. Extra calls waste \
budget.

When you are ready to give your final answer, output ONLY valid JSON — no \
prose, no markdown fences:
{
    "label": "VALID" or "HALLUCINATED",
    "confidence": 0.0 to 1.0,
    "reason": "concise explanation referencing the tool status and any disagreement"
}
"""

INITIAL_USER_PROMPT = """\
Please verify this BibTeX entry:

```bibtex
{bibtex}
```

Use the available tools to look up information, then output your JSON verdict.
"""


def _dispatch_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    cache: AgenticToolCache,
) -> str:
    """Call a tool (with cache + retry) and return its JSON-serialised result."""
    key = cache_key(tool_name, tool_args)
    cached = cache.get(key)
    if cached is not None:
        return json.dumps(cached, ensure_ascii=False)

    tool_fn = TOOL_REGISTRY.get(tool_name)
    if tool_fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        result = retry_with_backoff(
            lambda: tool_fn(**tool_args),  # type: ignore[operator]
            max_retries=3,
            base_delay=1.0,
            exceptions=(RuntimeError, OSError),
        )
    except Exception as exc:
        logger.warning("Tool %s failed: %s", tool_name, exc)
        return json.dumps({"error": str(exc)})

    cache.set(key, result)
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# OpenAI agentic loop
# ---------------------------------------------------------------------------


def _run_agentic_openai(
    entry: BlindEntry,
    client: Any,
    cache: AgenticToolCache,
    model: str,
    checkpoint_path: Path | None,
    tool_defs: list[dict] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> Prediction:
    """Run one agentic verification step via the OpenAI tool-use API."""
    bibtex = entry.to_bibtex()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": INITIAL_USER_PROMPT.format(bibtex=bibtex)},
    ]

    tool_sequence: list[str] = []
    total_tokens = 0
    start = time.time()

    active_tool_defs = tool_defs if tool_defs is not None else TOOL_DEFINITIONS
    # Convert tool defs to OpenAI function-tool format
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in active_tool_defs
    ]

    for _call_idx in range(MAX_TOOL_CALLS + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            temperature=0.0,
            seed=42,
            max_completion_tokens=1024,
        )
        total_tokens += resp.usage.total_tokens if resp.usage else 0
        choice = resp.choices[0]
        msg = choice.message

        # Append assistant turn
        messages.append(msg.model_dump(exclude_none=True))

        # Check if model wants to call tools
        if msg.tool_calls:
            if len(tool_sequence) >= MAX_TOOL_CALLS:
                # Hard cap exceeded — abort and return UNCERTAIN
                logger.warning(
                    "Max tool calls (%d) exceeded for %s — returning UNCERTAIN",
                    MAX_TOOL_CALLS,
                    entry.bibtex_key,
                )
                pred = Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="UNCERTAIN",
                    confidence=0.5,
                    reason=f"[Agentic] Max tool calls ({MAX_TOOL_CALLS}) exceeded",
                    api_sources_queried=[f"openai/{model}", *tool_sequence],
                    api_calls=len(tool_sequence) + 1,
                    wall_clock_seconds=time.time() - start,
                )
                _write_checkpoint(checkpoint_path, pred)
                return pred

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    fn_args = {}
                tool_sequence.append(fn_name)
                tool_result = _dispatch_tool(fn_name, fn_args, cache)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )
        else:
            # No tool calls — model emitted text; try to parse as verdict
            content = (msg.content or "").strip()
            pred = _parse_llm_response(content, entry.bibtex_key)
            final_source = "parametric" if not tool_sequence else "tool"
            pred.wall_clock_seconds = time.time() - start
            pred.api_calls = len(tool_sequence) + 1
            pred.api_sources_queried = [f"openai/{model}", *[f"tool:{t}" for t in tool_sequence]]
            pred.reason = (
                f"[Agentic|{final_source}|tools={','.join(tool_sequence) or 'none'}|"
                f"tokens={total_tokens}] {pred.reason}"
            )
            _write_checkpoint(checkpoint_path, pred)
            return pred

    # Exhausted iterations without a verdict
    pred = Prediction(
        bibtex_key=entry.bibtex_key,
        label="UNCERTAIN",
        confidence=0.5,
        reason=f"[Agentic] Loop exhausted after {MAX_TOOL_CALLS} tool calls",
        api_sources_queried=[f"openai/{model}", *tool_sequence],
        api_calls=len(tool_sequence) + 1,
        wall_clock_seconds=time.time() - start,
    )
    _write_checkpoint(checkpoint_path, pred)
    return pred


# ---------------------------------------------------------------------------
# Anthropic agentic loop
# ---------------------------------------------------------------------------


def _run_agentic_anthropic(
    entry: BlindEntry,
    client: Any,
    cache: AgenticToolCache,
    model: str,
    checkpoint_path: Path | None,
    tool_defs: list[dict] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> Prediction:
    """Run one agentic verification step via the Anthropic tool-use API."""
    bibtex = entry.to_bibtex()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": INITIAL_USER_PROMPT.format(bibtex=bibtex)},
    ]

    active_tool_defs = tool_defs if tool_defs is not None else TOOL_DEFINITIONS
    # Convert tool defs to Anthropic tool format
    anthropic_tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in active_tool_defs
    ]

    tool_sequence: list[str] = []
    total_tokens = 0
    start = time.time()

    for _call_idx in range(MAX_TOOL_CALLS + 1):
        resp = client.messages.create(
            model=model,
            system=system_prompt,
            messages=messages,
            tools=anthropic_tools,
            tool_choice={"type": "auto"},
            temperature=0.0,
            max_tokens=1024,
        )
        total_tokens += (resp.usage.input_tokens + resp.usage.output_tokens) if resp.usage else 0

        # Append assistant message
        messages.append({"role": "assistant", "content": resp.content})

        # Find tool uses and text blocks
        tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
        text_blocks = [b for b in resp.content if b.type == "text"]

        if tool_use_blocks:
            if len(tool_sequence) >= MAX_TOOL_CALLS:
                logger.warning(
                    "Max tool calls (%d) exceeded for %s — returning UNCERTAIN",
                    MAX_TOOL_CALLS,
                    entry.bibtex_key,
                )
                pred = Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="UNCERTAIN",
                    confidence=0.5,
                    reason=f"[Agentic] Max tool calls ({MAX_TOOL_CALLS}) exceeded",
                    api_sources_queried=[f"anthropic/{model}", *tool_sequence],
                    api_calls=len(tool_sequence) + 1,
                    wall_clock_seconds=time.time() - start,
                )
                _write_checkpoint(checkpoint_path, pred)
                return pred

            tool_results = []
            for tb in tool_use_blocks:
                fn_name = tb.name
                fn_args = tb.input if isinstance(tb.input, dict) else {}
                tool_sequence.append(fn_name)
                tool_result = _dispatch_tool(fn_name, fn_args, cache)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tb.id,
                        "content": tool_result,
                    }
                )
            messages.append({"role": "user", "content": tool_results})
        else:
            # No tool use — parse text as verdict
            content = " ".join(b.text for b in text_blocks if hasattr(b, "text")).strip()
            pred = _parse_llm_response(content, entry.bibtex_key)
            final_source = "parametric" if not tool_sequence else "tool"
            pred.wall_clock_seconds = time.time() - start
            pred.api_calls = len(tool_sequence) + 1
            pred.api_sources_queried = [f"anthropic/{model}", *[f"tool:{t}" for t in tool_sequence]]
            pred.reason = (
                f"[Agentic|{final_source}|tools={','.join(tool_sequence) or 'none'}|"
                f"tokens={total_tokens}] {pred.reason}"
            )
            _write_checkpoint(checkpoint_path, pred)
            return pred

    pred = Prediction(
        bibtex_key=entry.bibtex_key,
        label="UNCERTAIN",
        confidence=0.5,
        reason=f"[Agentic] Loop exhausted after {MAX_TOOL_CALLS} tool calls",
        api_sources_queried=[f"anthropic/{model}", *tool_sequence],
        api_calls=len(tool_sequence) + 1,
        wall_clock_seconds=time.time() - start,
    )
    _write_checkpoint(checkpoint_path, pred)
    return pred


# ---------------------------------------------------------------------------
# Checkpoint helpers (JSONL, same format as llm_verifier)
# ---------------------------------------------------------------------------


def _load_checkpoint(path: Path) -> dict[str, Prediction]:
    if not path.exists():
        return {}
    completed: dict[str, Prediction] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            completed[d["bibtex_key"]] = Prediction(
                bibtex_key=d["bibtex_key"],
                label=d["label"],
                confidence=d["confidence"],
                reason=d.get("reason", ""),
                wall_clock_seconds=d.get("wall_clock_seconds", 0.0),
                api_calls=d.get("api_calls", 0),
                api_sources_queried=d.get("api_sources_queried", []),
            )
        except Exception:
            pass
    return completed


def _write_checkpoint(path: Path | None, pred: Prediction) -> None:
    if path is None:
        return
    data = {
        "bibtex_key": pred.bibtex_key,
        "label": pred.label,
        "confidence": pred.confidence,
        "reason": pred.reason,
        "wall_clock_seconds": pred.wall_clock_seconds,
        "api_calls": pred.api_calls,
        "api_sources_queried": pred.api_sources_queried,
    }
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def verify_agentic_openai(
    entries: list[BlindEntry],
    model: str = OPENAI_MODEL,
    api_key: str | None = None,
    checkpoint_dir: Path | None = None,
    cache_db_path: Path | None = None,
    max_consecutive_failures: int = 3,
    tool_defs: list[dict] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    checkpoint_name: str = "agentic_openai",
    **_kwargs: Any,
) -> list[Prediction]:
    """Run agentic citation verification via OpenAI tool-use API.

    Args:
        entries: Blind entries to verify.
        model: OpenAI model identifier.
        api_key: API key (falls back to OPENAI_API_KEY env var).
        checkpoint_dir: Directory for JSONL resume checkpoints.
        cache_db_path: Override SQLite cache path.
        max_consecutive_failures: Abort after N consecutive errors.
    """
    try:
        import openai
    except ImportError:
        logger.error("openai package not installed. Install with: pip install openai")
        return fallback_predictions(entries, reason="llm_agentic_openai unavailable")

    client_kwargs: dict[str, Any] = {"max_retries": 3, "timeout": 120.0}
    if api_key:
        client_kwargs["api_key"] = api_key
    client = openai.OpenAI(**client_kwargs)

    checkpoint_path: Path | None = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model.replace("/", "_")
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}_{safe_model}.jsonl"

    completed = _load_checkpoint(checkpoint_path) if checkpoint_path else {}
    if completed:
        logger.info("Resuming %s: %d entries already done", checkpoint_name, len(completed))

    predictions = list(completed.values())
    consecutive_failures = 0

    with AgenticToolCache(cache_db_path) as tool_cache:
        for entry in entries:
            if entry.bibtex_key in completed:
                continue
            try:
                pred = _run_agentic_openai(
                    entry,
                    client,
                    tool_cache,
                    model,
                    checkpoint_path,
                    tool_defs=tool_defs,
                    system_prompt=system_prompt,
                )
                consecutive_failures = 0
            except Exception as exc:
                logger.warning("%s error for %s: %s", checkpoint_name, entry.bibtex_key, exc)
                consecutive_failures += 1
                pred = Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="UNCERTAIN",
                    confidence=0.5,
                    reason=f"[Agentic error] {exc}",
                    api_sources_queried=[f"openai/{model}"],
                    api_calls=1,
                )
                _write_checkpoint(checkpoint_path, pred)
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        "Aborting agentic_openai after %d consecutive failures",
                        max_consecutive_failures,
                    )
                    predictions.append(pred)
                    processed = {p.bibtex_key for p in predictions}
                    for rem in entries:
                        if rem.bibtex_key not in processed:
                            fb = Prediction(
                                bibtex_key=rem.bibtex_key,
                                label="UNCERTAIN",
                                confidence=0.5,
                                reason=f"[Agentic] Skipped after {max_consecutive_failures} consecutive failures",
                                api_sources_queried=[f"openai/{model}"],
                            )
                            predictions.append(fb)
                            _write_checkpoint(checkpoint_path, fb)
                    return predictions
            predictions.append(pred)

    return predictions


def verify_agentic_anthropic(
    entries: list[BlindEntry],
    model: str = ANTHROPIC_MODEL,
    api_key: str | None = None,
    checkpoint_dir: Path | None = None,
    cache_db_path: Path | None = None,
    max_consecutive_failures: int = 3,
    tool_defs: list[dict] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    checkpoint_name: str = "agentic_anthropic",
    **_kwargs: Any,
) -> list[Prediction]:
    """Run agentic citation verification via Anthropic tool-use API.

    Args:
        entries: Blind entries to verify.
        model: Anthropic model identifier.
        api_key: API key (falls back to ANTHROPIC_API_KEY env var).
        checkpoint_dir: Directory for JSONL resume checkpoints.
        cache_db_path: Override SQLite cache path.
        max_consecutive_failures: Abort after N consecutive errors.
    """
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed. Install with: pip install anthropic")
        return fallback_predictions(entries, reason="llm_agentic_anthropic unavailable")

    client_kwargs: dict[str, Any] = {"max_retries": 3, "timeout": 120.0}
    if api_key:
        client_kwargs["api_key"] = api_key
    client = anthropic.Anthropic(**client_kwargs)

    checkpoint_path: Path | None = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model.replace("/", "_")
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}_{safe_model}.jsonl"

    completed = _load_checkpoint(checkpoint_path) if checkpoint_path else {}
    if completed:
        logger.info("Resuming %s: %d entries already done", checkpoint_name, len(completed))

    predictions = list(completed.values())
    consecutive_failures = 0

    with AgenticToolCache(cache_db_path) as tool_cache:
        for entry in entries:
            if entry.bibtex_key in completed:
                continue
            try:
                pred = _run_agentic_anthropic(
                    entry,
                    client,
                    tool_cache,
                    model,
                    checkpoint_path,
                    tool_defs=tool_defs,
                    system_prompt=system_prompt,
                )
                consecutive_failures = 0
            except Exception as exc:
                logger.warning("%s error for %s: %s", checkpoint_name, entry.bibtex_key, exc)
                consecutive_failures += 1
                pred = Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="UNCERTAIN",
                    confidence=0.5,
                    reason=f"[Agentic error] {exc}",
                    api_sources_queried=[f"anthropic/{model}"],
                    api_calls=1,
                )
                _write_checkpoint(checkpoint_path, pred)
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        "Aborting agentic_anthropic after %d consecutive failures",
                        max_consecutive_failures,
                    )
                    predictions.append(pred)
                    processed = {p.bibtex_key for p in predictions}
                    for rem in entries:
                        if rem.bibtex_key not in processed:
                            fb = Prediction(
                                bibtex_key=rem.bibtex_key,
                                label="UNCERTAIN",
                                confidence=0.5,
                                reason=f"[Agentic] Skipped after {max_consecutive_failures} consecutive failures",
                                api_sources_queried=[f"anthropic/{model}"],
                            )
                            predictions.append(fb)
                            _write_checkpoint(checkpoint_path, fb)
                    return predictions
            predictions.append(pred)

    return predictions


# ---------------------------------------------------------------------------
# BTU-as-tool agentic variants
# ---------------------------------------------------------------------------


def verify_agentic_btu_openai(
    entries: list[BlindEntry],
    model: str = OPENAI_MODEL,
    api_key: str | None = None,
    checkpoint_dir: Path | None = None,
    cache_db_path: Path | None = None,
    max_consecutive_failures: int = 3,
    **kwargs: Any,
) -> list[Prediction]:
    """Agentic baseline where the LLM can call ``bibtex-updater`` as its only tool.

    Isolates the "LLM as BTU dispatcher" signal from the full multi-source
    agentic loop: the model receives one tool (``verify_with_bibtex_updater``)
    and a BTU-specific system prompt telling it how to interpret statuses.
    Compared to ``llm_tool_augmented`` (which always calls BTU upfront), this
    variant lets the model decide whether to call the tool at all and how to
    interpret disagreement between parametric knowledge and tool output.
    """
    return verify_agentic_openai(
        entries,
        model=model,
        api_key=api_key,
        checkpoint_dir=checkpoint_dir,
        cache_db_path=cache_db_path,
        max_consecutive_failures=max_consecutive_failures,
        tool_defs=[BTU_TOOL_DEFINITION],
        system_prompt=BTU_SYSTEM_PROMPT,
        checkpoint_name="agentic_btu_openai",
        **kwargs,
    )


def verify_agentic_btu_anthropic(
    entries: list[BlindEntry],
    model: str = ANTHROPIC_MODEL,
    api_key: str | None = None,
    checkpoint_dir: Path | None = None,
    cache_db_path: Path | None = None,
    max_consecutive_failures: int = 3,
    **kwargs: Any,
) -> list[Prediction]:
    """Anthropic variant of the BTU-as-tool agentic baseline."""
    return verify_agentic_anthropic(
        entries,
        model=model,
        api_key=api_key,
        checkpoint_dir=checkpoint_dir,
        cache_db_path=cache_db_path,
        max_consecutive_failures=max_consecutive_failures,
        tool_defs=[BTU_TOOL_DEFINITION],
        system_prompt=BTU_SYSTEM_PROMPT,
        checkpoint_name="agentic_btu_anthropic",
        **kwargs,
    )
