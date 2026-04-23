"""Tool-augmented LLM baseline: GPT-5.1 + bibtex-updater evidence injection.

Runs bibtex-updater first, formats its structured output (status, mismatched
fields, APIs queried), and injects it into an augmented LLM prompt. The LLM
makes the final classification with both the raw BibTeX and tool evidence.

This is the realistic deployment scenario (LLM + verification APIs) and tests
whether combining both signals improves over either alone.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hallmark.baselines.common import entries_to_bib
from hallmark.dataset.schema import BlindEntry, Prediction

logger = logging.getLogger(__name__)

TOOL_AUGMENTED_PROMPT = """\
You are a citation verification expert. Analyze the following BibTeX entry \
and determine if it is a VALID real publication or a HALLUCINATED (fabricated) citation.

BibTeX entry:
```bibtex
{bibtex}
```

An automated verification tool checked this entry against CrossRef, DBLP, \
and Semantic Scholar. Results:

{tool_evidence}

Use the tool's findings as evidence, but apply your own judgment:
- If the tool found mismatches, are they real errors or API artifacts?
- If the tool verified the entry, could it still be a sophisticated fabrication?
- Consider author plausibility, venue existence, and title coherence.

Respond with JSON only:
{{"label": "VALID" or "HALLUCINATED", "confidence": 0.0 to 1.0, "reason": "..."}}"""


def format_tool_evidence(record: dict[str, Any]) -> str:
    """Convert a raw bibtex-check JSONL record into readable text for prompt injection.

    Handles missing records, api_error status, and partial results gracefully.
    """
    if not record:
        return "No tool results available for this entry."

    status = record.get("status", "unknown")

    if status == "api_error":
        errors = record.get("errors", [])
        error_str = "; ".join(str(e) for e in errors) if errors else "unknown error"
        return f"Tool could not verify this entry — API unreachable ({error_str})."

    parts: list[str] = [f"Status: {status}"]

    confidence = record.get("confidence")
    if confidence is not None:
        parts.append(f"Confidence: {confidence:.2f}")

    mismatched = record.get("mismatched_fields", [])
    if mismatched:
        parts.append(f"Mismatched fields: {', '.join(str(f) for f in mismatched)}")

    api_sources = record.get("api_sources", [])
    if api_sources:
        parts.append(f"APIs consulted: {', '.join(api_sources)}")

    errors = record.get("errors", [])
    if errors:
        parts.append(f"Warnings: {'; '.join(str(e) for e in errors)}")

    return "\n".join(parts)


def load_tool_evidence(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    """Load pre-saved raw bibtex-check JSONL into a keyed dict.

    Returns:
        Mapping from bibtex_key to raw record dict.
    """
    from hallmark.baselines.bibtexupdater import parse_jsonl_to_raw

    return parse_jsonl_to_raw(jsonl_path)


def save_tool_evidence(
    entries: list[BlindEntry],
    output_path: Path,
    extra_args: list[str] | None = None,
    timeout: float = 7200.0,
    rate_limit: int = 120,
    academic_only: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run bibtex-updater and save raw JSONL output for later evidence injection.

    Returns:
        Mapping from bibtex_key to raw record dict.
    """
    import os

    tmpdir = tempfile.mkdtemp()
    bib_path = Path(tmpdir) / "input.bib"
    jsonl_path = Path(tmpdir) / "results.jsonl"

    try:
        bib_content = entries_to_bib(entries)
        bib_path.write_text(bib_content)

        # Find bibtex-check binary, preferring pipx-installed version
        bibtex_check_bin = shutil.which("bibtex-check")
        if bibtex_check_bin is None:
            logger.error("bibtex-check not found. Install with: pipx install bibtex-updater")
            return {}

        cmd = [
            bibtex_check_bin,
            str(bib_path),
            "--jsonl",
            str(jsonl_path),
            "--rate-limit",
            str(rate_limit),
        ]
        if academic_only:
            cmd.append("--academic-only")
        s2_key = os.environ.get("S2_API_KEY")
        if s2_key:
            cmd.extend(["--s2-api-key", s2_key])
        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"Running bibtex-check for tool evidence: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            logger.error("bibtex-check not found. Install with: pipx install bibtex-updater")
            return {}
        except subprocess.TimeoutExpired:
            logger.warning(f"bibtex-check timed out after {timeout}s (partial results may exist)")

        if jsonl_path.exists():
            # Copy to output path before cleanup
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(jsonl_path, output_path)
            from hallmark.baselines.bibtexupdater import parse_jsonl_to_raw

            return parse_jsonl_to_raw(output_path)

        logger.warning("No JSONL output produced by bibtex-check")
        return {}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def verify_tool_augmented(
    entries: list[BlindEntry],
    model: str = "gpt-5.1",
    api_key: str | None = None,
    tool_evidence_path: Path | str | None = None,
    log_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    **kwargs: Any,
) -> list[Prediction]:
    """LLM verification augmented with bibtex-updater tool evidence.

    Args:
        entries: Benchmark entries to verify.
        model: LLM model identifier (default: gpt-5.1).
        api_key: OpenAI API key (falls back to OPENAI_API_KEY env var).
        tool_evidence_path: Path to pre-computed bibtex-check raw JSONL.
            If None, runs bibtex-check live to generate evidence.
        log_dir: Optional directory for per-entry API call logs.
        checkpoint_dir: Directory for JSONL checkpoint files.
        **kwargs: Additional arguments forwarded to the LLM call.
    """
    from hallmark.baselines.llm_verifier import _verify_with_openai_compatible

    # Load or compute tool evidence
    evidence: dict[str, dict[str, Any]]
    if tool_evidence_path is not None:
        evidence_path = Path(tool_evidence_path)
        if evidence_path.exists():
            evidence = load_tool_evidence(evidence_path)
            logger.info(f"Loaded tool evidence for {len(evidence)} entries from {evidence_path}")
        else:
            logger.warning(f"Tool evidence file not found: {evidence_path}, running live")
            evidence = save_tool_evidence(entries, evidence_path)
    else:
        # Generate evidence to a temp file
        tmpdir = tempfile.mkdtemp()
        try:
            tmp_evidence_path = Path(tmpdir) / "tool_evidence.jsonl"
            evidence = save_tool_evidence(entries, tmp_evidence_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # Build per-entry prompt function that injects tool evidence
    def _make_prompt(entry: BlindEntry) -> str:
        bibtex = entry.to_bibtex()
        record = evidence.get(entry.bibtex_key, {})
        tool_text = format_tool_evidence(record)
        return TOOL_AUGMENTED_PROMPT.format(bibtex=bibtex, tool_evidence=tool_text)

    return _verify_with_openai_compatible(
        entries,
        model=model,
        api_key=api_key,
        base_url=None,
        source_prefix="openai_tool_augmented",
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        prompt_fn=_make_prompt,
        **kwargs,
    )
