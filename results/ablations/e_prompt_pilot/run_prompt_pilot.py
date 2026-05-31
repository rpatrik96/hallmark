"""Ablation 1 pilot: prompt structure / phrasing sensitivity on n=60 dev_public.

Compares the DEFAULT VERIFICATION_PROMPT against three variants:
  (a) uncertain   - explicit "UNCERTAIN allowed" wording (no taxonomy)
  (b) taxonomy    - default minus taxonomy block ... wait, default ALREADY has
                    the 14-type taxonomy. The printed app:prompt-template OMITS
                    it, so "taxonomy-in-prompt" == the default code prompt. To
                    make the ablation meaningful we instead test the inverse:
                    `notaxo` (no taxonomy, the printed/paper variant) as the
                    contrast, plus an explicit `taxonomy` reinforcement.
  (c) terse       - minimal one-line prompt, JSON-only.

We reuse the repo verify path (`_verify_with_openai_compatible`) with a custom
`prompt_fn` per variant rather than reimplementing the API loop. Scoring uses
`hallmark.evaluation.evaluate`. temperature=0, seed=42 are fixed inside the
helper for comparability.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from hallmark.baselines.llm_verifier import (
    VERIFICATION_PROMPT,
    _verify_with_openai_compatible,
)
from hallmark.dataset.schema import BenchmarkEntry, BlindEntry
from hallmark.evaluation import evaluate

SAMPLE = Path(
    "/Users/patrik.reizinger/Documents/GitHub/hallmark/results/ablations/pilot_sample_dev60.jsonl"
)
OUTDIR = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark/results/ablations/e_prompt_pilot")
MODEL = "deepseek/deepseek-v3.2"  # cheapest live OpenRouter model; prompt effect is model-agnostic

# --- Prompt variants -------------------------------------------------------

# (default) is VERIFICATION_PROMPT verbatim (taxonomy present, UNCERTAIN allowed
# in the JSON schema but no explicit instruction to use it).

# notaxo: the prompt as PRINTED in app:prompt-template — taxonomy block removed,
# UNCERTAIN still a permitted JSON value. This is the variant the paper's figure
# actually shows, so it is the most reviewer-relevant contrast.
NOTAXO_PROMPT = """\
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
    "label": "VALID" or "HALLUCINATED" or "UNCERTAIN",
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation"
}}"""

# uncertain: NOTAXO + an explicit instruction that UNCERTAIN is allowed and
# preferred over guessing. Isolates the "permit abstention" wording effect.
UNCERTAIN_PROMPT = (
    NOTAXO_PROMPT + "\n\nIf you cannot determine with confidence whether the entry is real or "
    "fabricated, respond with UNCERTAIN rather than guessing VALID or HALLUCINATED."
)

# terse: minimal one-line prompt.
TERSE_PROMPT = """\
Is this BibTeX citation a real publication or fabricated?
```bibtex
{bibtex}
```
Reply with JSON only: {{"label": "VALID" or "HALLUCINATED" or "UNCERTAIN", \
"confidence": 0.0-1.0, "reason": "..."}}"""

VARIANTS: dict[str, str] = {
    "default": VERIFICATION_PROMPT,  # taxonomy-in-prompt (code default)
    "notaxo": NOTAXO_PROMPT,  # printed paper variant: no taxonomy
    "uncertain": UNCERTAIN_PROMPT,  # explicit UNCERTAIN-allowed wording
    "terse": TERSE_PROMPT,  # minimal prompt
}


def load_entries() -> list[BenchmarkEntry]:
    entries = []
    for line in SAMPLE.read_text().splitlines():
        if line.strip():
            entries.append(BenchmarkEntry.from_dict(json.loads(line)))
    return entries


def make_prompt_fn(template: str):
    def _fn(entry: BlindEntry) -> str:
        return template.format(bibtex=entry.to_bibtex())

    return _fn


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    entries = load_entries()
    blind = [e.to_blind() for e in entries]
    n_hall = sum(1 for e in entries if e.label == "HALLUCINATED")
    n_valid = sum(1 for e in entries if e.label == "VALID")
    print(f"Loaded {len(entries)} entries: {n_hall} HALL / {n_valid} VALID", flush=True)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set; source /tmp/.or_env first")

    summary: dict[str, dict] = {}
    for name, template in VARIANTS.items():
        print(f"\n=== variant: {name} ===", flush=True)
        ckpt = OUTDIR / "checkpoints" / name
        preds = _verify_with_openai_compatible(
            blind,
            model=MODEL,
            api_key=api_key,  # OpenRouter key passed explicitly (base_url is OR)
            base_url="https://openrouter.ai/api/v1",
            source_prefix="openrouter",
            checkpoint_dir=ckpt,
            prompt_fn=make_prompt_fn(template),
            temperature=0.0,
        )
        res = evaluate(entries, preds, tool_name=f"prompt-{name}", split_name="pilot_dev60")
        n_unc = sum(1 for p in preds if p.label == "UNCERTAIN")
        n_err = sum(1 for p in preds if "[Error fallback]" in (p.reason or ""))
        rec = {
            "variant": name,
            "n": len(preds),
            "detection_rate": res.detection_rate,
            "false_positive_rate": res.false_positive_rate,
            "f1_hallucination": res.f1_hallucination,
            "ece": res.ece,
            "num_uncertain": n_unc,
            "uncertain_rate": n_unc / len(preds) if preds else 0.0,
            "num_error_fallback": n_err,
            "coverage": res.coverage,
        }
        summary[name] = rec
        print(json.dumps(rec, indent=2), flush=True)

    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Delta table vs default
    d = summary["default"]
    print("\n=== DELTA vs default ===", flush=True)
    print(
        f"{'variant':<12}{'DR':>8}{'dDR':>8}{'FPR':>8}{'dFPR':>8}{'UNC%':>8}{'dUNC%':>8}",
        flush=True,
    )
    for name, r in summary.items():
        fpr = r["false_positive_rate"]
        dfpr = (
            (fpr - d["false_positive_rate"])
            if (fpr is not None and d["false_positive_rate"] is not None)
            else None
        )
        print(
            f"{name:<12}"
            f"{r['detection_rate']:>8.3f}"
            f"{r['detection_rate'] - d['detection_rate']:>8.3f}"
            f"{(fpr if fpr is not None else float('nan')):>8.3f}"
            f"{(dfpr if dfpr is not None else float('nan')):>8.3f}"
            f"{r['uncertain_rate'] * 100:>8.1f}"
            f"{(r['uncertain_rate'] - d['uncertain_rate']) * 100:>8.1f}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
