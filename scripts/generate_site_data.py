#!/usr/bin/env python3
"""Generate site/data/site_data.js for the companion website (site/).

Reads only released, tracked artifacts:
  - data/v1.0/metadata.json                    (corpus composition)
  - data/v1.0/dev_public.jsonl                 (example entries; public labels)
  - data/v1.0/baseline_results/*.json          (dev/test leaderboards)
  - results/crossdomain_llms/result_*.json     (cross-domain split, n=500)
  - results/relabel_delta/btu_v1_2_0/test_crossdomain_metrics.json
  - results/temporal_supplement/*              (canonical 448-entry supplement)

Deterministic: example sampling is seeded, iteration orders are sorted.
Run from the repo root (or anywhere): python scripts/generate_site_data.py
"""

# ruff: noqa: RUF001  # en-dashes in strings are intentional site copy

from __future__ import annotations

import datetime
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data" / "site_data.js"
SEED = 8042  # matches the corpus build seed

BR = ROOT / "data" / "v1.0" / "baseline_results"
TS = ROOT / "results" / "temporal_supplement"
XD = ROOT / "results" / "crossdomain_llms"

# ---------------------------------------------------------------- model roster

CAT_DB = "db"
CAT_LLM = "llm"
CAT_AGENTIC = "agentic"

CATEGORIES = [
    {"key": CAT_LLM, "name": "LLM zero-shot", "slot": 1},
    {"key": CAT_AGENTIC, "name": "Agentic & cascade", "slot": 2},
    {"key": CAT_DB, "name": "Rule-based & DB tools", "slot": 3},
]

# tool key -> (display name, category, ranked, tag)
# ranked=False rows are excluded from best-in-column bolding, mirroring the
# paper's ranking exclusions (co-designed block; HaRC's coverage caveat).
MODELS = [
    ("doi_only", "DOI-only", CAT_DB, True, None),
    ("harc_with_s2key", "HaRC (S2 key)", CAT_DB, False, "excluded in paper"),
    ("bibtexupdater", "bibtex-updater", CAT_DB, False, "co-designed"),
    ("llm_openai", "GPT-5.1", CAT_LLM, True, None),
    ("llm_openai_gpt54", "GPT-5.4", CAT_LLM, True, None),
    ("llm_openrouter_claude_opus_4_7", "Claude Opus 4.7", CAT_LLM, True, None),
    ("llm_openrouter_claude_sonnet_4_6", "Claude Sonnet 4.6", CAT_LLM, True, None),
    ("llm_openrouter_deepseek_r1", "DeepSeek-R1", CAT_LLM, True, None),
    ("llm_openrouter_deepseek_v3", "DeepSeek-V3.2", CAT_LLM, True, None),
    ("llm_openrouter_gemini_flash", "Gemini 2.5 Flash", CAT_LLM, True, None),
    ("llm_openrouter_gemini_pro", "Gemini 2.5 Pro", CAT_LLM, True, None),
    ("llm_openrouter_llama_4_maverick", "Llama 4 Maverick", CAT_LLM, True, None),
    ("llm_openrouter_mistral", "Mistral Large", CAT_LLM, True, None),
    ("llm_openrouter_qwen", "Qwen3-235B", CAT_LLM, True, None),
    ("llm_openrouter_qwen_max", "Qwen3-VL-235B", CAT_LLM, True, None),
    (
        "llm_agentic_openai",
        "GPT-5.1 + CrossRef/OpenAlex/arXiv",
        CAT_AGENTIC,
        True,
        None,
    ),
    ("llm_agentic_btu_openai", "GPT-5.1 + bibtex-updater", CAT_AGENTIC, True, None),
    (
        "llm_agentic_btu_sonnet_4_6",
        "Sonnet 4.6 + bibtex-updater",
        CAT_AGENTIC,
        True,
        None,
    ),
    (
        "llm_tool_augmented",
        "GPT-5.1 + bibtex-updater (always-call)",
        CAT_AGENTIC,
        False,
        "co-designed",
    ),
    (
        "cascade_db_diagnosis",
        "Cascade: btu → Sonnet (conservative)",
        CAT_AGENTIC,
        False,
        "co-designed",
    ),
    (
        "cascade_db_diagnosis_aggressive",
        "Cascade: btu → Sonnet (aggressive)",
        CAT_AGENTIC,
        False,
        "co-designed",
    ),
]

# split key -> list of (tool_key, path, schema)
RESULT_SOURCES: dict[str, list[tuple[str, Path, str]]] = {
    "dev_public": [
        ("bibtexupdater", BR / "bibtexupdater_dev_public.json", "flat"),
        ("cascade_db_diagnosis", BR / "cascade_db_diagnosis_dev_public.json", "flat"),
        (
            "cascade_db_diagnosis_aggressive",
            BR / "cascade_db_diagnosis_aggressive_dev_public.json",
            "flat",
        ),
        ("doi_only", BR / "doi_only_dev_public.json", "flat"),
        ("harc_with_s2key", BR / "harc_with_s2key_dev_public.json", "flat"),
        ("llm_openai", BR / "llm_openai_dev_public.json", "flat"),
        ("llm_openai_gpt54", BR / "llm_openai_gpt54_dev_public.json", "flat"),
        ("llm_agentic_openai", BR / "llm_agentic_openai_dev_public.json", "flat"),
        (
            "llm_agentic_btu_openai",
            BR / "llm_agentic_btu_openai_dev_public.json",
            "flat",
        ),
        (
            "llm_agentic_btu_sonnet_4_6",
            BR / "llm_agentic_btu_sonnet_4_6_dev_public.json",
            "flat",
        ),
    ]
    + [
        (f"llm_openrouter_{m}", BR / f"llm_openrouter_{m}_dev_public.json", "flat")
        for m in [
            "claude_opus_4_7",
            "claude_sonnet_4_6",
            "deepseek_r1",
            "deepseek_v3",
            "gemini_flash",
            "gemini_pro",
            "llama_4_maverick",
            "mistral",
            "qwen",
            "qwen_max",
        ]
    ],
    "test_public": [
        ("bibtexupdater", BR / "bibtexupdater_test_public.json", "flat"),
        ("cascade_db_diagnosis", BR / "cascade_db_diagnosis_test_public.json", "flat"),
        (
            "cascade_db_diagnosis_aggressive",
            BR / "cascade_db_diagnosis_aggressive_test_public.json",
            "flat",
        ),
        ("doi_only", BR / "doi_only_test_public.json", "flat"),
        ("llm_openai", BR / "llm_openai_test_public.json", "flat"),
        ("llm_openai_gpt54", BR / "llm_openai_gpt54_test_public.json", "flat"),
        ("llm_agentic_openai", BR / "llm_agentic_openai_test_public.json", "flat"),
        (
            "llm_agentic_btu_openai",
            BR / "llm_agentic_btu_openai_test_public.json",
            "flat",
        ),
        (
            "llm_agentic_btu_sonnet_4_6",
            BR / "llm_agentic_btu_sonnet_4_6_test_public.json",
            "flat",
        ),
        ("llm_tool_augmented", BR / "llm_tool_augmented_test_public.json", "flat"),
    ]
    + [
        (f"llm_openrouter_{m}", BR / f"llm_openrouter_{m}_test_public.json", "flat")
        for m in [
            "claude_opus_4_7",
            "claude_sonnet_4_6",
            "deepseek_r1",
            "deepseek_v3",
            "gemini_flash",
            "gemini_pro",
            "llama_4_maverick",
            "mistral",
            "qwen",
            "qwen_max",
        ]
    ],
    "test_crossdomain": [
        (
            "bibtexupdater",
            ROOT / "results/relabel_delta/btu_v1_2_0/test_crossdomain_metrics.json",
            "btu_flat",
        ),
        ("llm_openai", XD / "result_llm_openai.json", "wrapper"),
        ("llm_openai_gpt54", XD / "result_llm_openai_gpt54.json", "wrapper"),
    ]
    + [
        (f"llm_openrouter_{m}", XD / f"result_llm_openrouter_{m}.json", "wrapper")
        for m in [
            "claude_opus_4_7",
            "claude_sonnet_4_6",
            "deepseek_r1",
            "deepseek_v3",
            "gemini_flash",
            "gemini_pro",
            "llama_4_maverick",
            "mistral",
            "qwen",
            "qwen_max",
        ]
    ],
    "temporal_448": [
        ("llm_openai", TS / "llm_openai_temporal_v1subset.json", "flat"),
        (
            "llm_openai_gpt54",
            TS / "gpt54_448/gpt54_temporal_448_summary.json",
            "confusion",
        ),
        (
            "llm_openrouter_claude_sonnet_4_6",
            TS / "llm_openrouter_claude_sonnet_4_6_temporal_supplement_eval.json",
            "flat",
        ),
        (
            "llm_openrouter_claude_opus_4_7",
            TS / "llm_openrouter_claude_opus_4_7_temporal_supplement_eval.json",
            "flat",
        ),
        (
            "llm_openrouter_gemini_pro",
            TS / "llm_openrouter_gemini_pro_temporal_supplement_eval.json",
            "flat",
        ),
        (
            "llm_openrouter_llama_4_maverick",
            TS / "llm_openrouter_llama_4_maverick_temporal_supplement_eval.json",
            "flat",
        ),
        (
            "llm_openrouter_qwen_max",
            TS / "llm_openrouter_qwen_max_temporal_supplement_eval.json",
            "flat",
        ),
        (
            "llm_openrouter_deepseek_r1",
            TS / "llm_openrouter_deepseek_r1_temporal_v1subset.json",
            "flat",
        ),
        (
            "llm_openrouter_deepseek_v3",
            TS / "llm_openrouter_deepseek_v3_temporal_v1subset.json",
            "flat",
        ),
        (
            "llm_openrouter_gemini_flash",
            TS / "llm_openrouter_gemini_flash_temporal_v1subset.json",
            "flat",
        ),
        (
            "llm_openrouter_mistral",
            TS / "llm_openrouter_mistral_temporal_v1subset.json",
            "flat",
        ),
        (
            "llm_openrouter_qwen",
            TS / "llm_openrouter_qwen_temporal_v1subset.json",
            "flat",
        ),
    ],
}

SPLIT_LABELS = {
    "dev_public": "dev_public",
    "test_public": "test_public",
    "test_crossdomain": "cross-domain",
    "temporal_448": "temporal 2024–25",
}

SPLIT_NOTES = {
    "dev_public": (
        "dev_public: 1,119 entries (513 valid / 606 hallucinated), ML venues "
        "2021–2023; the main leaderboard split."
    ),
    "test_public": (
        "test_public: 831 entries (312 valid / 519 hallucinated), the held-out public test split."
    ),
    "test_crossdomain": (
        "test_crossdomain: 500 evaluation-only entries (200 valid / 300 "
        "hallucinated) from PubMed/bioRxiv and non-ML CS venues. The "
        "bibtex-updater row is the released v1.2.0 metrics file (coverage 74%); "
        "a recency-matched rebuild returns its FPR to 0.112, so out of domain "
        "its cost is coverage, not precision. LLM FPRs here are dominated by "
        "post-cutoff recency: 89% of biomedical false positives cite the "
        "future date."
    ),
    "temporal_448": (
        "Canonical 448-entry 2024–25 temporal supplement (300 valid / 148 "
        "hallucinated), disjoint from the 2021–23 corpus. Some models abstain "
        "heavily here, so watch the coverage column; the paper footnotes the "
        "abstention conventions for this split. The GPT-5.4 row comes from a "
        "confusion-matrix summary without tier/type breakdowns."
    ),
}

# --------------------------------------------------------------- taxonomy copy

# (key, name, tier, stress, short, description) — wording follows the paper.
TAXONOMY = [
    (
        "fabricated_doi",
        "Fabricated DOI",
        1,
        False,
        "DOI does not resolve",
        "A fabricated DOI attached to otherwise plausible metadata.",
    ),
    (
        "nonexistent_venue",
        "Nonexistent venue",
        1,
        False,
        "Invented conference or journal",
        "A venue name that does not exist, however plausible it sounds.",
    ),
    (
        "placeholder_authors",
        "Placeholder authors",
        1,
        False,
        "Generic or fake author names",
        "Stock placeholder names (e.g. John Doe and Jane Smith) on a real or fabricated paper.",
    ),
    (
        "future_date",
        "Future date",
        1,
        False,
        "Publication year in the future",
        "A publication date past the present: instantly checkable, and the "
        "reason LLMs give for 89% of their false positives on the 2026 "
        "biomedical cross-domain entries.",
    ),
    (
        "chimeric_title",
        "Chimeric title",
        2,
        False,
        "Real authors + fabricated title",
        "Pairs a real author list with a fabricated title.",
    ),
    (
        "wrong_venue",
        "Wrong venue",
        2,
        False,
        "Real paper, wrong venue",
        "Assigns a real paper to the wrong conference or journal.",
    ),
    (
        "swapped_authors",
        "Author mismatch",
        2,
        False,
        "Correct title, wrong authors",
        "A real title carrying an author list that does not match the actual publication.",
    ),
    (
        "preprint_as_published",
        "Preprint as published",
        2,
        False,
        "arXiv preprint cited as a venue paper",
        "An arXiv-only paper cited as if published at a conference or journal.",
    ),
    (
        "hybrid_fabrication",
        "Hybrid fabrication",
        2,
        False,
        "Real DOI + fake metadata",
        "A real DOI whose resolved record does not match the BibTeX metadata.",
    ),
    (
        "merged_citation",
        "Merged citation",
        2,
        True,
        "Metadata from 2+ papers",
        "Authors from paper A, title from paper B, venue from paper C; "
        "theoretically motivated with zero documented real-world instances.",
    ),
    (
        "partial_author_list",
        "Partial author list",
        2,
        True,
        "Subset of the real authors",
        "Fewer than half of the real authors listed without an 'et al.' "
        "indicator; theoretically motivated stress-test type.",
    ),
    (
        "near_miss_title",
        "Near-miss title",
        3,
        False,
        "Title off by 1–2 words",
        "Differs from a real paper's title by one or two words (e.g. "
        "'Attention is All you Require').",
    ),
    (
        "plausible_fabrication",
        "Plausible fabrication",
        3,
        False,
        "Entirely invented but realistic",
        "All fields invented yet realistic: the predominant real-world LLM "
        "failure mode (55% of mapped incidents).",
    ),
    (
        "arxiv_version_mismatch",
        "arXiv version mismatch",
        3,
        True,
        "Wrong version, venue, or year",
        "Cites a preprint with the wrong venue and a shifted year; "
        "theoretically motivated stress-test type.",
    ),
]

TIERS = [
    (
        1,
        "Tier 1 · Easy",
        "Detectable by a single API lookup: a DOI that does not resolve, a "
        "nonexistent venue, placeholder authors, or a future date.",
    ),
    (
        2,
        "Tier 2 · Medium",
        "Requires cross-referencing metadata fields: chimeric titles, wrong "
        "venues, author mismatches, or a real DOI resolving to different "
        "metadata.",
    ),
    (
        3,
        "Tier 3 · Hard",
        "Requires deep verification or semantic reasoning: near-miss titles, "
        "plausible fabrications, and arXiv version mismatches.",
    ),
]

GENERATION_METHODS = [
    ("scraped", "scraped (real venue metadata)"),
    ("perturbation", "perturbation of a real entry"),
    ("llm_generated", "LLM-generated"),
    ("adversarial", "adversarial (hand-crafted)"),
    ("real_world", "real-world incident"),
]

# curated must-include example keys (dev_public), picked for illustrative value
MUST_INCLUDE = [
    "d46c5fa004ad",
    "fd9da8902f80",
    "f9247e4251cf",
    "b67497cbd9ea",
    "ed4c058bf525",
    "fdb41768eae3",
    "f8c3fa3bca65",
    "ab0f3b832ff6",
    "d1fe214f878b",
    "d4471f53482e",
    "c25b080c90c2",
    "da82cc87f879",
    "ffe715a15b16",
    "cd588085bf52",
    "08ebaa54f666",
    "f5ed78046e20",
    "d0a040eb49c2",
    "cb603ef5f1fe",
    "db228049d7a9",
    "d1e09089ae24",
    "e9fdba573807",
    "dae1eb71d49a",
    "e9e08922a057",
    "a2974be79850",
]

# ------------------------------------------------------------------- helpers


def rnd(x: Any, digits: int = 4) -> Any:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return round(float(x), digits)
    return None


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_flat(obj: dict) -> dict:
    n = obj.get("num_entries")
    unc = obj.get("num_uncertain")
    # The released files carry coverage=1.0 even when num_uncertain > 0,
    # so coverage is recomputed from the abstention count.
    coverage = None
    if n and unc is not None:
        coverage = 1.0 - unc / n
    elif obj.get("coverage") is not None:
        coverage = obj["coverage"]
    per_tier = None
    if obj.get("per_tier_metrics"):
        per_tier = {
            t: {"dr": rnd(v.get("detection_rate"))}
            for t, v in sorted(obj["per_tier_metrics"].items())
            if t in ("1", "2", "3")
        }
    per_type = None
    if obj.get("per_type_metrics"):
        per_type = {
            t: {"dr": rnd(v.get("detection_rate")), "count": v.get("count")}
            for t, v in sorted(obj["per_type_metrics"].items())
            if t != "valid"
        }
    return {
        "dr": rnd(obj.get("detection_rate")),
        "fpr": rnd(obj.get("false_positive_rate")),
        "f1": rnd(obj.get("f1_hallucination")),
        "twf1": rnd(obj.get("tier_weighted_f1")),
        "tier3_f1": rnd(obj.get("tier3_f1")),
        "ece": rnd(obj.get("ece")),
        "auroc": rnd(obj.get("auroc")),
        "coverage": rnd(coverage),
        "n": n,
        "per_tier": per_tier,
        "per_type": per_type,
    }


def extract_btu_flat(obj: dict) -> dict:
    f1 = obj.get("f1")
    return {
        "dr": rnd(obj.get("detection_rate")),
        "fpr": rnd(obj.get("fpr")),
        "f1": rnd(f1),
        "twf1": None,
        "tier3_f1": None,
        "ece": None,
        "auroc": None,
        "coverage": rnd(obj.get("coverage")),
        "n": obj.get("n"),
        "per_tier": None,
        "per_type": None,
    }


def extract_confusion(obj: dict) -> dict:
    tp, fp = obj["TP"], obj["FP"]
    fn = obj["FN"]
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    n = obj.get("n")
    decided = tp + fp + fn + obj.get("TN", 0)
    return {
        "dr": rnd(obj.get("detection_rate")),
        "fpr": rnd(obj.get("fpr")),
        "f1": rnd(f1),
        "twf1": None,
        "tier3_f1": None,
        "ece": None,
        "auroc": None,
        "coverage": rnd(decided / n) if n else None,
        "n": n,
        "per_tier": None,
        "per_type": None,
    }


def extract(path: Path, schema: str) -> dict:
    obj = load_json(path)
    if schema == "wrapper":
        return extract_flat(obj["overall"])
    if schema == "btu_flat":
        return extract_btu_flat(obj)
    if schema == "confusion":
        return extract_confusion(obj)
    return extract_flat(obj)


# --------------------------------------------------------------- example prep


def truncate(text: str, limit: int = 480) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …"


def entry_to_example(e: dict) -> dict:
    label = e["label"]
    explanation = e.get("explanation") or ""
    # Relabel trap: entries flipped HALLUCINATED→VALID in the ground-truth
    # audit keep their old diagnosis-style explanation; show the audit reason.
    if label == "VALID" and e.get("relabeled_from"):
        reason = e.get("relabel_reason") or ""
        explanation = "Ground-truth audit (relabeled): " + reason
    return {
        "key": e["bibtex_key"],
        "label": label,
        "type": e.get("hallucination_type"),
        "tier": e.get("difficulty_tier"),
        "method": e.get("generation_method"),
        "bibtex_type": e.get("bibtex_type", "misc"),
        "fields": e.get("fields", {}),
        "explanation": truncate(explanation),
        "subtests": e.get("subtests") or {},
        "venue": e.get("source_conference"),
    }


def sample_examples(entries: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    by_key = {e["bibtex_key"]: e for e in entries}
    chosen: list[dict] = []
    seen: set[str] = set()

    def add(e: dict) -> None:
        if e["bibtex_key"] in seen:
            return
        seen.add(e["bibtex_key"])
        chosen.append(e)

    for key in MUST_INCLUDE:
        if key in by_key:
            add(by_key[key])

    type_keys = [t[0] for t in TAXONOMY]
    for t in type_keys:
        pool = sorted(
            (
                e
                for e in entries
                if e.get("hallucination_type") == t and e["bibtex_key"] not in seen
            ),
            key=lambda e: e["bibtex_key"],
        )
        for e in rng.sample(pool, min(8, len(pool))):
            add(e)

    valid = [e for e in entries if e["label"] == "VALID"]
    relabeled = sorted(
        (e for e in valid if e.get("relabeled_from")),
        key=lambda e: e["bibtex_key"],
    )
    suspicious = sorted(
        (
            e
            for e in valid
            if not e.get("relabeled_from")
            and (
                (e.get("subtests") or {}).get("cross_db_agreement") is False
                or ("booktitle" not in e.get("fields", {}) and "journal" not in e.get("fields", {}))
            )
        ),
        key=lambda e: e["bibtex_key"],
    )
    plain = sorted(
        (
            e
            for e in valid
            if not e.get("relabeled_from")
            and e["bibtex_key"] not in {x["bibtex_key"] for x in suspicious}
        ),
        key=lambda e: e["bibtex_key"],
    )
    for pool, k in ((relabeled, 10), (suspicious, 10), (plain, 12)):
        avail = [e for e in pool if e["bibtex_key"] not in seen]
        for e in rng.sample(avail, min(k, len(avail))):
            add(e)

    examples = [entry_to_example(e) for e in chosen]
    # hallucinated first (by tier, then type), then valid
    examples.sort(
        key=lambda x: (
            x["label"] != "HALLUCINATED",
            x["tier"] or 9,
            x["type"] or "zzz",
            x["key"],
        )
    )
    return examples


# ------------------------------------------------------------------ assembly


def ppv(dr: float | None, fpr: float | None, base_rate: float = 0.02) -> float | None:
    if dr is None or fpr is None:
        return None
    denom = dr * base_rate + fpr * (1 - base_rate)
    if denom == 0:
        return None
    return dr * base_rate / denom


def main() -> None:
    metadata = load_json(ROOT / "data" / "v1.0" / "metadata.json")

    results: dict[str, list[dict]] = {}
    for split, sources in RESULT_SOURCES.items():
        rows = []
        for tool, path, schema in sources:
            if not path.exists():
                raise FileNotFoundError(f"{split}: missing {path}")
            row = extract(path, schema)
            row["tool"] = tool
            rows.append(row)
        results[split] = rows

    def row(split: str, tool: str) -> dict:
        return next(r for r in results[split] if r["tool"] == tool)

    # taxonomy counts over the public splits (dev + test)
    type_counts: dict[str, int] = {}
    for split in ("dev_public", "test_public"):
        for t, c in metadata["splits"][split]["type_distribution"].items():
            type_counts[t] = type_counts.get(t, 0) + c

    taxonomy = [
        {
            "key": k,
            "name": name,
            "tier": tier,
            "stress": stress,
            "short": short,
            "description": desc,
            "count_public": type_counts.get(k, 0),
        }
        for k, name, tier, stress, short, desc in TAXONOMY
    ]

    tiers = [{"tier": t, "name": n, "description": d} for t, n, d in TIERS]

    md_splits = metadata["splits"]
    xd_meta = metadata["v1_1_extensions"]["splits"]["test_crossdomain"]
    splits = [
        {
            "key": "dev_public",
            "name": "dev_public",
            "valid": md_splits["dev_public"]["valid"],
            "hallucinated": md_splits["dev_public"]["hallucinated"],
            "extension": False,
            "description": (
                "The public development split and main leaderboard: ML-venue "
                "BibTeX (NeurIPS, ICML, ICLR, AAAI, CVPR; 2021–2023) covering "
                "all 14 hallucination types."
            ),
        },
        {
            "key": "test_public",
            "name": "test_public",
            "valid": md_splits["test_public"]["valid"],
            "hallucinated": md_splits["test_public"]["hallucinated"],
            "extension": False,
            "description": (
                "The held-out public test split; its ΔFPR against dev_public "
                "carries the cross-split robustness story."
            ),
        },
        {
            "key": "test_hidden",
            "name": "test_hidden",
            "valid": md_splits["test_hidden"]["valid"],
            "hallucinated": md_splits["test_hidden"]["hallucinated"],
            "extension": False,
            "description": (
                "The contamination-resistant held-out split: kept private, "
                "spread over all 14 types."
            ),
        },
        {
            "key": "stress_test",
            "name": "stress_test",
            "valid": md_splits["stress_test"]["valid"],
            "hallucinated": md_splits["stress_test"]["hallucinated"],
            "extension": False,
            "description": (
                "Extra depth for the three theoretically-motivated stress-test "
                "types; the single valid entry is the contamination canary."
            ),
        },
        {
            "key": "test_crossdomain",
            "name": "test_crossdomain",
            "valid": xd_meta["valid"],
            "hallucinated": xd_meta["hallucinated"],
            "extension": True,
            "description": (
                "Cross-domain generalization: PubMed/bioRxiv and non-ML CS "
                "venues, all 14 types, evaluation-only."
            ),
        },
        {
            "key": "temporal_448",
            "name": "temporal supplement (2024–25)",
            "valid": 300,
            "hallucinated": 148,
            "extension": True,
            "description": (
                "Post-cutoff behavior: 300 valid 2024–25 DBLP entries from six "
                "ML venues plus 148 perturbation-generated hallucinations "
                "(failure mode iii)."
            ),
        },
        {
            "key": "walters_wilder",
            "name": "ChatGPT citations (Walters–Wilder)",
            "valid": 172,
            "hallucinated": 169,
            "extension": True,
            "description": (
                "Authentic ChatGPT-generated citations hand-verified by Walters "
                "and Wilder across 42 multidisciplinary topics: hallucinations "
                "the benchmark authors did not construct."
            ),
        },
        {
            "key": "temporal_probe",
            "name": "temporal probe (2026)",
            "valid": 30,
            "hallucinated": 30,
            "extension": True,
            "description": (
                "A 60-entry probe extending post-cutoff coverage to 2026 arXiv "
                "submissions; reproduces the FPR-multiplier pattern."
            ),
        },
    ]

    # ---- failure-mode visuals (computed from the loaded result rows) ----
    mode1_rows = [
        {
            "name": "GPT-5.1 (multi-source)",
            "zero_shot_fpr": row("dev_public", "llm_openai")["fpr"],
            "agentic_fpr": row("dev_public", "llm_agentic_openai")["fpr"],
        },
        {
            "name": "GPT-5.1 (+btu)",
            "zero_shot_fpr": row("dev_public", "llm_openai")["fpr"],
            "agentic_fpr": row("dev_public", "llm_agentic_btu_openai")["fpr"],
        },
        {
            "name": "Sonnet 4.6 (+btu)",
            "zero_shot_fpr": row("dev_public", "llm_openrouter_claude_sonnet_4_6")["fpr"],
            "agentic_fpr": row("dev_public", "llm_agentic_btu_sonnet_4_6")["fpr"],
        },
    ]

    mode2_tools = [
        ("llm_openrouter_claude_opus_4_7", "Claude Opus 4.7"),
        ("llm_openrouter_gemini_pro", "Gemini 2.5 Pro"),
        ("bibtexupdater", "bibtex-updater"),
        ("llm_openrouter_claude_sonnet_4_6", "Claude Sonnet 4.6"),
        ("llm_openai", "GPT-5.1"),
        ("llm_openrouter_deepseek_v3", "DeepSeek-V3.2"),
    ]
    mode2_rows = []
    for tool, name in mode2_tools:
        r = row("dev_public", tool)
        p = ppv(r["dr"], r["fpr"])
        if p is not None:
            mode2_rows.append({"name": name, "ppv": rnd(p)})

    mode3_tools = [
        ("llm_openai", "GPT-5.1"),
        ("llm_openrouter_gemini_flash", "Gemini 2.5 Flash"),
        ("llm_openrouter_qwen", "Qwen3-235B"),
        ("llm_openrouter_deepseek_v3", "DeepSeek-V3.2"),
        ("llm_openrouter_claude_sonnet_4_6", "Claude Sonnet 4.6"),
        ("llm_openrouter_claude_opus_4_7", "Claude Opus 4.7"),
    ]
    mode3_rows = [
        {
            "name": name,
            "fpr_indist": row("dev_public", tool)["fpr"],
            "fpr_post": row("temporal_448", tool)["fpr"],
        }
        for tool, name in mode3_tools
    ]

    failure_modes = {
        "mode1": {
            "rows": mode1_rows,
            "note": (
                "FPR on dev_public valid entries, zero-shot vs the agentic "
                "harness on the same base model. A deterministic re-aggregation "
                "over three of the harness's four sources cuts FPR ~15× "
                "(0.729 → 0.049); the any-vs-consensus ordering is what "
                "transfers, not the absolute level."
            ),
        },
        "mode2": {
            "rows": mode2_rows,
            "note": (
                "Precision when 2% of a bibliography is hallucinated "
                "(venue-realistic base rate), from dev_public detection and "
                "false-positive rates."
            ),
        },
        "mode3": {
            "rows": mode3_rows,
            "note": (
                "FPR on valid entries: 2021–23 corpus (dev_public) vs the "
                "448-entry 2024–25 supplement. Descriptive: confounded with "
                "possible recall of those entries."
            ),
        },
    }

    n_verifiers = len({r["tool"] for rows in results.values() for r in rows})
    kpis = [
        {
            "label": "BibTeX entries",
            "value": f"{metadata['total_entries']:,}",
            "sub": "core corpus, + 1,349 evaluation-only extension entries",
        },
        {"label": "Hallucination types", "value": "14", "sub": "3 stress-test"},
        {
            "label": "Difficulty tiers",
            "value": "3",
            "sub": "single lookup → semantic reasoning",
        },
        {
            "label": "Verifier configurations",
            "value": str(n_verifiers),
            "sub": "zero-shot LLMs, agents, cascades, rule-based tools",
        },
        {
            "label": "Diagnostic sub-tests",
            "value": "6",
            "sub": "per entry, from DOI resolution to cross-DB agreement",
        },
    ]

    entries = []
    dev_path = ROOT / "data" / "v1.0" / "dev_public.jsonl"
    with open(dev_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    examples = sample_examples(entries)

    data = {
        "generated": datetime.date.today().isoformat(),
        "corpus_version": metadata["version"],
        "default_split": "dev_public",
        "corpus_note": "seeded sample of the 1,119-entry public dev split",
        "kpis": kpis,
        "categories": CATEGORIES,
        "models": [
            {
                "tool": tool,
                "name": name,
                "category": cat,
                "ranked": ranked,
                "tag": tag,
                "default_on": True,
            }
            for tool, name, cat, ranked, tag in MODELS
        ],
        "taxonomy": taxonomy,
        "tiers": tiers,
        "splits": splits,
        "split_labels": SPLIT_LABELS,
        "split_notes": SPLIT_NOTES,
        "results": results,
        "failure_modes": failure_modes,
        "generation_methods": [{"key": k, "name": n} for k, n in GENERATION_METHODS],
        "examples": examples,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # Harden against </script> breakout: corpus strings are adversarial by
    # design, so no data string may ever close the enclosing script element.
    payload = payload.replace("</", "<\\/")
    OUT.write_text(
        "// Generated by scripts/generate_site_data.py — do not edit by hand.\n"
        f"window.HALLMARK_DATA = {payload};\n",
        encoding="utf-8",
    )
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)} ({size_kb:.0f} KB)")
    print(f"splits: {[(s, len(r)) for s, r in results.items()]}")
    print(f"examples: {len(examples)}")


if __name__ == "__main__":
    main()
