#!/usr/bin/env python3
"""A5 inter-annotator-agreement: run 3 independent LLM raters over the substrate.

Each rater independently labels every blinded entry VALID / HALLUCINATED /
UNCERTAIN (and, where HALLUCINATED, a hallucination_type) from blinded metadata
only. This is an AUTOMATED multi-rater reliability proxy -- NOT human IAA. The
human-IAA follow-up uses the kit emitted by ``a5_export_human_kit.py``.

Raters (OpenRouter, deliberately NOT GPT -- OpenAI key is out of quota):
  * anthropic/claude-sonnet-4.6
  * google/gemini-2.5-pro
  * deepseek/deepseek-v3.2

ENDPOINT-DRIFT POLICY: any fresh LLM run is a NEW dated snapshot, not a
reproduction of the published eval. Every output file records the run date and
the endpoint; nothing here overwrites published aggregates.

Per-entry predictions (label + confidence + type + raw response) are persisted
for every rater for full reproducibility. Checkpointing makes the run
resumable: completed keys are skipped on re-invocation.

Usage:
    OPENROUTER_API_KEY=... uv run python scripts/a5_run_raters.py
    uv run python scripts/a5_run_raters.py --raters claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "ablations" / "a5_kappa"
PRED_DIR = OUT / "rater_predictions"
CKPT_DIR = OUT / "checkpoints"
ENDPOINT = "https://openrouter.ai/api/v1"

# rater id -> OpenRouter model slug. IDs are filesystem-safe.
RATERS: dict[str, str] = {
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "gemini-2-5-pro": "google/gemini-2.5-pro",
    "deepseek-v3-2": "deepseek/deepseek-v3.2",
}

# Per-rater completion-token budget. Thinking models (Gemini-2.5-pro) spend the
# budget on hidden reasoning before the JSON verdict; 1024 truncates the JSON
# mid-object (closing braces cut off -> parse failure). Give them room.
RATER_MAX_TOKENS: dict[str, int] = {
    "gemini-2-5-pro": 8192,
}
DEFAULT_MAX_TOKENS = 1024

VALID_TYPES = {
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

PROMPT = """\
You are a citation verification expert acting as an independent annotator. \
Analyze the following BibTeX entry and decide whether it is a VALID real \
publication or a HALLUCINATED (fabricated or corrupted) citation. Judge only \
from the metadata shown; do not assume access to external databases.

BibTeX entry:
```bibtex
{bibtex}
```

Consider: (1) is the title plausible and attributable to these authors? \
(2) are the authors real researchers? (3) is the venue real? (4) is the year \
sensible? (5) if a DOI is present, is it well formed?

When the entry is HALLUCINATED, classify the mode using exactly one of: \
fabricated_doi, nonexistent_venue, placeholder_authors, future_date, \
chimeric_title, wrong_venue, swapped_authors, preprint_as_published, \
hybrid_fabrication, near_miss_title, plausible_fabrication, merged_citation, \
partial_author_list, arxiv_version_mismatch.

Respond with JSON only:
{{
    "label": "VALID" or "HALLUCINATED" or "UNCERTAIN",
    "confidence": 0.0 to 1.0,
    "predicted_hallucination_type": "<one type above, or null>",
    "reason": "brief explanation"
}}
predicted_hallucination_type MUST be null when label is VALID or UNCERTAIN."""


def _to_bibtex(entry: dict) -> str:
    if entry.get("raw_bibtex"):
        return str(entry["raw_bibtex"])
    lines = [f"@{entry['bibtex_type']}{{{entry['bibtex_key']},"]
    for key, value in sorted(entry["fields"].items()):
        lines.append(f"  {key} = {{{value}}},")
    lines.append("}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply (handles code fences).

    Returns ``None`` when no parseable complete object is present (e.g. a
    thinking model that truncated the closing braces); the regex fallback in
    :func:`_parse` then recovers the fields field-by-field.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _regex_recover(text: str) -> dict:
    """Recover label/type/confidence from a truncated/unclosed JSON reply.

    Thinking models (e.g. Gemini-2.5-pro) sometimes emit a valid-looking JSON
    prefix that the token budget cuts off before the closing brace, so
    ``json.loads`` fails. The verdict itself is present, so pull the fields out
    with regex rather than discarding a real decision as a parse error.
    """
    label_m = re.search(r'"label"\s*:\s*"?(VALID|HALLUCINATED|UNCERTAIN)"?', text, re.IGNORECASE)
    type_m = re.search(r'"predicted_hallucination_type"\s*:\s*"?([a-z_]+)"?', text, re.IGNORECASE)
    conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    return {
        "label": label_m.group(1).upper() if label_m else "UNCERTAIN",
        "predicted_hallucination_type": type_m.group(1).lower() if type_m else None,
        "confidence": conf_m.group(1) if conf_m else 0.5,
        "reason": "[recovered from truncated JSON]",
    }


def _parse(text: str) -> dict:
    obj = _extract_json(text)
    recovered = False
    if obj is None:
        obj = _regex_recover(text)
        recovered = True
    label = str(obj.get("label", "UNCERTAIN")).strip().upper()
    if label not in {"VALID", "HALLUCINATED", "UNCERTAIN"}:
        label = "UNCERTAIN"
    htype = obj.get("predicted_hallucination_type")
    if isinstance(htype, str):
        htype = htype.strip().lower()
        if htype in {"", "null", "none"}:
            htype = None
    if label != "HALLUCINATED" or htype not in VALID_TYPES:
        htype = None
    try:
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    reason = str(obj.get("reason", ""))[:500]
    return {
        "label": label,
        "confidence": conf,
        "predicted_hallucination_type": htype,
        "reason": reason,
        "recovered": recovered,
    }


def _load_done(ckpt: Path) -> set[str]:
    if not ckpt.exists():
        return set()
    return {
        json.loads(line)["bibtex_key"] for line in ckpt.read_text().splitlines() if line.strip()
    }


def run_rater(rater_id: str, model: str, entries: list[dict], client) -> None:
    ckpt = CKPT_DIR / f"{rater_id}.jsonl"
    done = _load_done(ckpt)
    pending = [e for e in entries if e["bibtex_key"] not in done]
    max_tokens = RATER_MAX_TOKENS.get(rater_id, DEFAULT_MAX_TOKENS)
    print(
        f"[{rater_id}] {model}: {len(done)} done, {len(pending)} pending (max_tokens={max_tokens})"
    )

    for i, entry in enumerate(pending, 1):
        key = entry["bibtex_key"]
        prompt = PROMPT.format(bibtex=_to_bibtex(entry))
        start = time.time()
        raw = ""
        err = None
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=max_tokens,
                seed=42,
            )
            raw = str(resp.choices[0].message.content or "").strip()
            parsed = _parse(raw)
        except Exception as e:  # record and continue
            err = str(e)
            parsed = {
                "label": "UNCERTAIN",
                "confidence": 0.5,
                "predicted_hallucination_type": None,
                "reason": f"[error] {err}",
                "recovered": False,
            }
        rec = {
            "bibtex_key": key,
            "rater_id": rater_id,
            "model": model,
            "endpoint": ENDPOINT,
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "label": parsed["label"],
            "confidence": parsed["confidence"],
            "predicted_hallucination_type": parsed["predicted_hallucination_type"],
            "reason": parsed["reason"],
            "recovered_from_truncation": parsed.get("recovered", False),
            "raw_response": raw,
            "error": err,
            "elapsed_seconds": round(time.time() - start, 3),
        }
        with ckpt.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 10 == 0 or i == len(pending):
            print(f"  [{rater_id}] {i}/{len(pending)}  last={key} -> {parsed['label']}")

    # Consolidate checkpoint -> final per-rater prediction file (dedup, last wins).
    final: dict[str, dict] = {}
    for line in ckpt.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            final[r["bibtex_key"]] = r
    records = sorted(final.values(), key=lambda r: r["bibtex_key"])
    pred_path = PRED_DIR / f"rater_{rater_id}.jsonl"
    pred_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    print(f"[{rater_id}] wrote {len(records)} predictions -> {pred_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raters",
        nargs="*",
        default=list(RATERS),
        choices=list(RATERS),
        help="subset of rater ids to run (default: all 3)",
    )
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set (source /tmp/.or_env)")

    try:
        import openai
    except ImportError:
        sys.exit("openai package not installed")

    substrate = OUT / "substrate_blinded.jsonl"
    if not substrate.exists():
        sys.exit(f"missing {substrate}; run a5_build_substrate.py first")
    entries = [json.loads(line) for line in substrate.read_text().splitlines() if line.strip()]

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    client = openai.OpenAI(api_key=api_key, base_url=ENDPOINT, max_retries=5, timeout=120.0)
    for rater_id in args.raters:
        run_rater(rater_id, RATERS[rater_id], entries, client)


if __name__ == "__main__":
    main()
