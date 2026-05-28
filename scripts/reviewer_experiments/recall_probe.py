"""E1 Pass B -- recall probe.

For each VALID paper in the E1 sample, give the model ONLY the title + year and
ask it (with NO external lookup) to recall the full author list and venue. Score
"recalled"=1 when the Jaccard overlap of predicted-vs-truth author last-names is
>= RECALL_THRESHOLD (0.5). Also record a (secondary) venue match.

Direct clients are used (not the hallmark verifier), because the verifier always
feeds the full BibTeX -- which includes the authors -- and that would defeat the
probe. Here the prompt deliberately omits authors and venue.

Per-entry results are checkpointed to a JSONL so the run resumes for free.

Usage:
    python recall_probe.py --model gpt-5.1
    python recall_probe.py --model anthropic/claude-sonnet-4.6
    python recall_probe.py --model anthropic/claude-opus-4.7
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_common as c

PROBE_PROMPT = """\
You are an expert on the machine-learning / computer-science research literature.
I will give you ONLY the title and year of a paper. From your own internal
knowledge, WITHOUT using any external tools, search, or lookup, recall who wrote
it and where it was published.

Paper title: {title}
Year: {year}

If you do not actually know this specific paper, do not invent authors -- return
an empty author list and an empty venue instead.

Respond with JSON only, no prose:
{{
  "known": true or false,
  "authors": ["Full Name", "Full Name", ...],
  "venue": "conference or journal name (short form is fine), or empty string"
}}"""


def get_client_and_model(model: str):
    """Return (client, api_model_id, temperature, token_param).

    ``token_param`` is the name of the max-output-tokens kwarg the provider
    accepts: OpenAI gpt-5.x rejects ``max_tokens`` and requires
    ``max_completion_tokens``; the OpenRouter/anthropic path accepts
    ``max_tokens``.
    """
    import openai

    if model.startswith("anthropic/"):
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY not set")
        client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        # Anthropic via OpenRouter accepts temperature 0.0; deterministic-ish.
        return client, model, 0.0, "max_tokens"
    # OpenAI direct (gpt-5.x). GPT-5 family is forced to temperature=1.0 by API.
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=key)
    temp = 1.0  # gpt-5 family
    return client, model, temp, "max_completion_tokens"


def parse_probe_response(content: str) -> dict:
    """Extract {known, authors, venue} from a (possibly fenced) JSON reply."""
    text = content.strip()
    if "```" in text:
        blocks = text.split("```")
        for i in range(1, len(blocks), 2):
            b = blocks[i]
            if b.lower().startswith("json"):
                b = b[4:]
            try:
                d = json.loads(b.strip())
                if isinstance(d, dict):
                    text = b.strip()
                    break
            except (json.JSONDecodeError, ValueError):
                continue
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Salvage: regex out an authors array if present.
        m = re.search(r'"authors"\s*:\s*\[(.*?)\]', content, re.DOTALL)
        authors: list[str] = []
        if m:
            authors = [a.strip().strip('"') for a in m.group(1).split(",") if a.strip()]
        vm = re.search(r'"venue"\s*:\s*"([^"]*)"', content)
        return {
            "known": bool(authors),
            "authors": authors,
            "venue": vm.group(1) if vm else "",
            "_parse": "salvaged",
        }
    if not isinstance(d, dict):
        return {"known": False, "authors": [], "venue": "", "_parse": "non_dict"}
    authors = d.get("authors") or []
    if not isinstance(authors, list):
        authors = []
    return {
        "known": bool(d.get("known", bool(authors))),
        "authors": [str(a) for a in authors],
        "venue": str(d.get("venue", "") or ""),
        "_parse": "ok",
    }


def run(model: str) -> dict:
    sample = c.load_sample()
    client, api_model, temperature, token_param = get_client_and_model(model)

    safe = model.replace("/", "_")
    ckpt = c.OUT_DIR / f"recall_{safe}.jsonl"
    completed: dict[str, dict] = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                completed[r["bibtex_key"]] = r
    print(f"[{model}] resuming with {len(completed)} completed of {len(sample)}")

    api_calls = 0
    for entry in sample:
        if entry.bibtex_key in completed:
            continue
        prompt = PROBE_PROMPT.format(title=entry.title, year=entry.year)
        start = time.time()
        try:
            call_kwargs = {
                "model": api_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "seed": 42,
                token_param: 400,
            }
            resp = client.chat.completions.create(**call_kwargs)
            api_calls += 1
            raw = (resp.choices[0].message.content or "").strip()
            parsed = parse_probe_response(raw)
            err = ""
        except Exception as e:
            api_calls += 1
            raw = ""
            parsed = {"known": False, "authors": [], "venue": "", "_parse": "api_error"}
            err = str(e)

        gt_authors = c.last_name_set(c.split_authors(entry.author))
        pred_authors = c.last_name_set(parsed["authors"])
        jac = c.jaccard(pred_authors, gt_authors)
        recalled = int(jac >= c.RECALL_THRESHOLD)
        vmatch = c.venue_match(parsed["venue"], entry.venue)

        rec = {
            "bibtex_key": entry.bibtex_key,
            "year": entry.year,
            "title": entry.title,
            "true_authors": sorted(gt_authors),
            "true_venue": entry.venue,
            "pred_authors_raw": parsed["authors"],
            "pred_authors_norm": sorted(pred_authors),
            "pred_venue": parsed["venue"],
            "model_says_known": parsed["known"],
            "author_jaccard": round(jac, 4),
            "recalled": recalled,
            "venue_match": vmatch,
            "parse_status": parsed.get("_parse", "ok"),
            "error": err,
            "elapsed_s": round(time.time() - start, 2),
            "raw_response": raw[:1000],
        }
        with open(ckpt, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        completed[entry.bibtex_key] = rec

    # Reload full set in sample order for the summary.
    records = [completed[e.bibtex_key] for e in sample if e.bibtex_key in completed]
    n = len(records)
    n_recalled = sum(r["recalled"] for r in records)
    n_venue = sum(1 for r in records if r["venue_match"])
    summary = {
        "model": model,
        "n": n,
        "recall_rate": round(n_recalled / n, 4) if n else None,
        "n_recalled": n_recalled,
        "venue_match_rate": round(n_venue / n, 4) if n else None,
        "new_api_calls_this_run": api_calls,
        "checkpoint": str(ckpt),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    run(args.model)
