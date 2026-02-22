#!/usr/bin/env python3
"""Generate hallucinated entries with year=2026 to fix year leakage.

The benchmark has VALID entries with year=2026 but zero HALLUCINATED entries,
creating a deterministic label oracle (year=2026 → always VALID). This script
generates hallucinated entries for both dev_public and test_public splits.

Two strategies:
  Part 1 — Perturbation: run standard generators on existing 2026 VALID sources.
            Year is preserved because generators only touch title/authors/venue/DOI.
  Part 2 — LLM: call GPT-4o-mini to fabricate plausible_fabrication entries
            with year=2026 and real venue names.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hallmark.dataset.generators._pools import (  # noqa: E402
    CHIMERIC_TITLE_TEMPLATES,
    VALID_CONFERENCES,
)
from hallmark.dataset.generators.tier1 import (  # noqa: E402
    generate_fabricated_doi,
    generate_nonexistent_venue,
    generate_placeholder_authors,
)
from hallmark.dataset.generators.tier2 import (  # noqa: E402
    generate_chimeric_title,
    generate_wrong_venue,
)
from hallmark.dataset.generators.tier3 import generate_near_miss_title  # noqa: E402
from hallmark.dataset.schema import (  # noqa: E402
    EXPECTED_SUBTESTS,
    BenchmarkEntry,
    DifficultyTier,
    GenerationMethod,
    HallucinationType,
    load_entries,
    save_entries,
)

DATA_DIR = ROOT / "data" / "v1.0"


def _stable_key(tag: str, idx: int) -> str:
    """Deterministic, collision-resistant bibtex_key."""
    h = hashlib.md5(f"{tag}_{idx}".encode()).hexdigest()[:12]
    return f"gen2026_{tag}_{h}"


def _pick_different_venue(source_venue: str, candidates: list[str], rng: random.Random) -> str:
    """Pick a venue that is different from the source entry's venue."""
    others = [v for v in candidates if v != source_venue]
    return rng.choice(others) if others else rng.choice(candidates)


def generate_perturbation_entries(
    sources: list[BenchmarkEntry],
    split_name: str,
    rng: random.Random,
) -> list[BenchmarkEntry]:
    """Produce hallucinated entries from 2026 VALID sources via perturbation."""
    results: list[BenchmarkEntry] = []

    # We define generators as (tag, callable) pairs.
    # Each callable takes (source, rng) and returns BenchmarkEntry.
    # Generators that need extra args are wrapped here.

    def gen_near_miss(src: BenchmarkEntry, r: random.Random) -> BenchmarkEntry:
        e = generate_near_miss_title(src, rng=r)
        return e

    def gen_fabricated_doi(src: BenchmarkEntry, r: random.Random) -> BenchmarkEntry:
        return generate_fabricated_doi(src, rng=r)

    def gen_nonexistent_venue(src: BenchmarkEntry, r: random.Random) -> BenchmarkEntry:
        return generate_nonexistent_venue(src, rng=r)

    def gen_placeholder_authors(src: BenchmarkEntry, r: random.Random) -> BenchmarkEntry:
        return generate_placeholder_authors(src, rng=r)

    def gen_wrong_venue(src: BenchmarkEntry, r: random.Random) -> BenchmarkEntry:
        source_venue = src.fields.get("booktitle", "") or src.fields.get("journal", "")
        wrong_venue = _pick_different_venue(source_venue, VALID_CONFERENCES, r)
        return generate_wrong_venue(src, wrong_venue, rng=r)

    def gen_chimeric(src: BenchmarkEntry, r: random.Random) -> BenchmarkEntry:
        fake_title = r.choice(CHIMERIC_TITLE_TEMPLATES)
        return generate_chimeric_title(src, fake_title, rng=r)

    generators = [
        ("near_miss", gen_near_miss),
        ("fabricated_doi", gen_fabricated_doi),
        ("nonexistent_venue", gen_nonexistent_venue),
        ("placeholder_authors", gen_placeholder_authors),
        ("wrong_venue", gen_wrong_venue),
        ("chimeric", gen_chimeric),
    ]

    global_idx = 0
    for _src_idx, source in enumerate(sources):
        # Apply 2-3 generators per source (keeps entries-per-source manageable)
        n_gens = rng.randint(2, 3)
        selected = rng.sample(generators, n_gens)

        for tag, gen_fn in selected:
            try:
                result = gen_fn(source, rng)

                # Enforce year=2026 (most generators don't touch year, but be explicit)
                result.fields["year"] = "2026"

                # Replace auto-generated bibtex_key with a stable, unique one
                result.bibtex_key = _stable_key(f"{split_name}_{tag}", global_idx)
                global_idx += 1

                result.added_to_benchmark = "2026-02-22"
                results.append(result)
            except Exception as exc:
                print(f"  Warning: {tag} failed on {source.bibtex_key}: {exc}")

    return results


def generate_llm_entries(
    n_entries: int,
    split_name: str,
    rng: random.Random,
) -> list[BenchmarkEntry]:
    """Fabricate plausible_fabrication entries with year=2026 via GPT-4o-mini."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  Skipping LLM generation: OPENAI_API_KEY not set")
        return []

    try:
        import openai
    except ImportError:
        print("  Skipping LLM generation: openai package not installed")
        return []

    client = openai.OpenAI(api_key=api_key)

    venues_sample = ", ".join(VALID_CONFERENCES[:6])
    prompt = (
        f"Generate {n_entries} realistic but FAKE BibTeX entries for ML papers "
        "supposedly published in 2026.\n\n"
        "Requirements:\n"
        "- Each entry must be a plausible ML paper that does NOT actually exist\n"
        "- Use realistic author names (mix of Western and East Asian names)\n"
        f"- Use only these real venue names: {venues_sample}\n"
        "- year must be 2026 for all entries\n"
        "- Titles should sound like real ML papers (use current trends: LLMs, "
        "diffusion models, RL from human feedback, mechanistic interpretability, etc.)\n"
        "- Include fields: title, author, booktitle, year\n\n"
        "Return a JSON array of objects, each with keys: title, author, booktitle, year.\n"
        "Only return the JSON array, no other text."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=2000,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:])
            if content.endswith("```"):
                content = content[: content.rfind("```")]

        papers = json.loads(content)
        results: list[BenchmarkEntry] = []

        subtests = dict(EXPECTED_SUBTESTS[HallucinationType.PLAUSIBLE_FABRICATION])
        subtests["fields_complete"] = False  # no identifier

        for i, paper in enumerate(papers):
            key = _stable_key(f"{split_name}_llm", i)
            entry = BenchmarkEntry(
                bibtex_key=key,
                bibtex_type="inproceedings",
                label="HALLUCINATED",
                fields={
                    "title": paper["title"],
                    "author": paper["author"],
                    "booktitle": paper["booktitle"],
                    "year": "2026",
                },
                hallucination_type=HallucinationType.PLAUSIBLE_FABRICATION.value,
                difficulty_tier=DifficultyTier.HARD.value,
                generation_method=GenerationMethod.LLM_GENERATED.value,
                explanation="LLM-fabricated paper with plausible 2026 metadata",
                subtests=subtests,
                added_to_benchmark="2026-02-22",
            )
            results.append(entry)

        return results

    except Exception as exc:
        print(f"  LLM generation error: {exc}")
        return []


def update_metadata(meta: dict, split_name: str, entries: list[BenchmarkEntry]) -> None:
    """Recompute and update metadata for a split."""
    labels = Counter(e.label for e in entries)
    types = Counter(e.hallucination_type for e in entries if e.label == "HALLUCINATED")
    tiers = Counter(e.difficulty_tier for e in entries if e.label == "HALLUCINATED")
    methods = Counter(e.generation_method for e in entries)

    s = meta["splits"][split_name]
    s["count"] = len(entries)
    s["total"] = len(entries)
    s["valid"] = labels["VALID"]
    s["hallucinated"] = labels["HALLUCINATED"]
    s["type_distribution"] = {str(k): v for k, v in sorted(types.items()) if k is not None}
    s["tier_distribution"] = {str(k): v for k, v in sorted(tiers.items()) if k is not None}
    s["generation_methods"] = {str(k): v for k, v in sorted(methods.items()) if k is not None}


def main() -> None:
    # Load OPENAI_API_KEY from /tmp/.or_env if present (CI / local convenience)
    env_file = Path("/tmp/.or_env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("export "):
                kv = line[7:].split("=", 1)
                if len(kv) == 2:
                    os.environ[kv[0]] = kv[1].strip('"').strip("'")

    rng = random.Random(2026)

    for split_name in ["dev_public", "test_public"]:
        print(f"\n{'=' * 50}")
        print(f"Processing {split_name}")
        print(f"{'=' * 50}")

        entries = load_entries(DATA_DIR / f"{split_name}.jsonl")
        sources_2026 = [e for e in entries if e.fields.get("year") == "2026" and e.label == "VALID"]
        print(f"  VALID entries with year=2026: {len(sources_2026)}")

        # Part 1: perturbation-based entries
        perturb = generate_perturbation_entries(sources_2026, split_name, rng)
        print(f"  Perturbation entries generated: {len(perturb)}")

        # Part 2: LLM-based entries (5 per split)
        llm = generate_llm_entries(5, split_name, rng)
        print(f"  LLM entries generated:          {len(llm)}")

        new_entries = perturb + llm
        all_entries = entries + new_entries

        # Sanity: show year=2026 breakdown
        y2026_hall = [
            e for e in all_entries if e.fields.get("year") == "2026" and e.label == "HALLUCINATED"
        ]
        y2026_valid = [
            e for e in all_entries if e.fields.get("year") == "2026" and e.label == "VALID"
        ]
        print(f"  year=2026 VALID:      {len(y2026_valid)}")
        print(f"  year=2026 HALLUCINATED: {len(y2026_hall)}")
        print(f"  Total entries in split: {len(all_entries)}")

        save_entries(all_entries, DATA_DIR / f"{split_name}.jsonl")
        print(f"  Saved {split_name}.jsonl")

    # Update metadata.json
    meta_path = DATA_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text())

    for split_name in ["dev_public", "test_public"]:
        entries = load_entries(DATA_DIR / f"{split_name}.jsonl")
        update_metadata(meta, split_name, entries)

    meta["total_entries"] = sum(s.get("count", s.get("total", 0)) for s in meta["splits"].values())

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"\nMetadata updated. Total entries: {meta['total_entries']}")
    print("Done!")


if __name__ == "__main__":
    main()
