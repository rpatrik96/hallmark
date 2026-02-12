"""LLM-based verification baseline.

Uses GPT-4 or Claude to verify BibTeX entries by prompting the model
to assess whether a citation appears genuine or hallucinated.
"""

from __future__ import annotations

import json
import logging
import time

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


def verify_with_openai(
    entries: list[BenchmarkEntry],
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> list[Prediction]:
    """Verify entries using OpenAI API."""
    try:
        import openai
    except ImportError:
        logger.error("openai package not installed. Install with: pip install openai")
        return fallback_predictions(entries, reason="LLM baseline unavailable")

    client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
    predictions = []

    for entry in entries:
        start = time.time()
        bibtex = entry.to_bibtex()
        prompt = VERIFICATION_PROMPT.format(bibtex=bibtex)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            content = response.choices[0].message.content.strip()
            pred = _parse_llm_response(content, entry.bibtex_key)
        except Exception as e:
            logger.warning(f"OpenAI API error for {entry.bibtex_key}: {e}")
            pred = Prediction(
                bibtex_key=entry.bibtex_key,
                label="VALID",
                confidence=0.5,
                reason=f"API error: {e}",
            )

        pred.wall_clock_seconds = time.time() - start
        pred.api_calls = 1
        pred.api_sources_queried = [f"openai/{model}"]
        predictions.append(pred)

    return predictions


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
    predictions = []

    for entry in entries:
        start = time.time()
        bibtex = entry.to_bibtex()
        prompt = VERIFICATION_PROMPT.format(bibtex=bibtex)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()
            pred = _parse_llm_response(content, entry.bibtex_key)
        except Exception as e:
            logger.warning(f"Anthropic API error for {entry.bibtex_key}: {e}")
            pred = Prediction(
                bibtex_key=entry.bibtex_key,
                label="VALID",
                confidence=0.5,
                reason=f"API error: {e}",
            )

        pred.wall_clock_seconds = time.time() - start
        pred.api_calls = 1
        pred.api_sources_queried = [f"anthropic/{model}"]
        predictions.append(pred)

    return predictions


def _parse_llm_response(content: str, bibtex_key: str) -> Prediction:
    """Parse LLM JSON response into a Prediction."""
    # Try to extract JSON from response
    try:
        # Handle markdown code blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data = json.loads(content)
        label = data.get("label", "VALID").upper()
        if label not in ("VALID", "HALLUCINATED"):
            label = "VALID"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return Prediction(
            bibtex_key=bibtex_key,
            label=label,
            confidence=confidence,
            reason=data.get("reason", ""),
        )
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        logger.warning(f"Failed to parse LLM response for {bibtex_key}: {e}")
        return Prediction(
            bibtex_key=bibtex_key,
            label="VALID",
            confidence=0.5,
            reason=f"Parse error: {content[:100]}",
        )
