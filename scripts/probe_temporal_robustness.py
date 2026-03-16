"""Multi-model temporal robustness probe.  [analysis]

Scrapes fresh valid entries from DBLP and arXiv, generates hallucinated
variants, then evaluates multiple LLM baselines on the same probe set.
Compares each model's temporal probe performance against its full-dataset
baseline to quantify how much each model degrades on recent papers.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark.baselines.llm_tool_augmented import verify_tool_augmented
from hallmark.baselines.llm_verifier import (
    verify_with_openai,
    verify_with_openrouter,
)
from hallmark.dataset.generator import (
    generate_chimeric_title,
    generate_fabricated_doi,
    generate_hybrid_fabrication,
    generate_near_miss_title,
    generate_nonexistent_venue,
    generate_placeholder_authors,
    generate_plausible_fabrication,
    generate_swapped_authors,
    generate_wrong_venue,
)
from hallmark.dataset.schema import (
    VALID_SUBTESTS,
    BenchmarkEntry,
    GenerationMethod,
)
from hallmark.dataset.scraper import (
    dblp_hit_to_entry,
)
from hallmark.evaluation.metrics import evaluate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


ARXIV_API_BASE = "https://export.arxiv.org/api/query"

# arXiv ML categories to search
ARXIV_ML_CATEGORIES = [
    "cs.LG",
    "cs.CL",
    "cs.CV",
    "cs.AI",
    "stat.ML",
]

# ── Model dispatch table ─────────────────────────────────────────────
# Maps model key -> (verify_fn, model_id, api_key_env, baseline_results_file)
MODEL_DISPATCH: dict[str, tuple[str, str, str, str | None]] = {
    "openai": (
        "openai",
        "gpt-5.1",
        "OPENAI_API_KEY",
        "llm_openai_dev_public_ci.json",
    ),
    "deepseek_r1": (
        "openrouter",
        "deepseek/deepseek-r1",
        "OPENROUTER_API_KEY",
        "llm_openrouter_deepseek_r1_dev_public.json",
    ),
    "deepseek_v3": (
        "openrouter",
        "deepseek/deepseek-v3.2",
        "OPENROUTER_API_KEY",
        "llm_openrouter_deepseek_v3_dev_public.json",
    ),
    "qwen": (
        "openrouter",
        "qwen/qwen3-235b-a22b-2507",
        "OPENROUTER_API_KEY",
        "llm_openrouter_qwen_dev_public.json",
    ),
    "mistral": (
        "openrouter",
        "mistralai/mistral-large-2512",
        "OPENROUTER_API_KEY",
        "llm_openrouter_mistral_dev_public.json",
    ),
    "gemini_flash": (
        "openrouter",
        "google/gemini-2.5-flash",
        "OPENROUTER_API_KEY",
        "llm_openrouter_gemini_flash_dev_public.json",
    ),
    "tool_augmented": (
        "tool_augmented",
        "gpt-5.1",
        "OPENAI_API_KEY",
        None,
    ),
}

# Display names for reporting
DISPLAY_NAMES: dict[str, str] = {
    "openai": "GPT-5.1",
    "deepseek_r1": "DeepSeek-R1",
    "deepseek_v3": "DeepSeek-V3",
    "qwen": "Qwen3-235B",
    "mistral": "Mistral Large",
    "gemini_flash": "Gemini Flash",
    "tool_augmented": "GPT-5.1 + BTU",
}


def scrape_arxiv_2026(
    n_target: int = 15,
) -> list[BenchmarkEntry]:
    """Scrape recent 2026 papers from arXiv ML categories.

    Uses the arXiv API to fetch papers submitted in 2026.
    """
    today = date.today().isoformat()
    all_entries: list[BenchmarkEntry] = []
    seen_titles: set[str] = set()

    for cat in ARXIV_ML_CATEGORIES:
        if len(all_entries) >= n_target:
            break

        # Query arXiv for recent papers in this category
        query = f"cat:{cat}"
        params = {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": 20,
        }

        try:
            resp = httpx.get(ARXIV_API_BASE, params=params, timeout=30.0)
            resp.raise_for_status()
        except (httpx.HTTPError, httpx.TransportError) as e:
            logger.warning(f"arXiv query failed for {cat}: {e}")
            continue

        # Parse Atom XML
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry_el in root.findall("atom:entry", ns):
            if len(all_entries) >= n_target:
                break

            title_el = entry_el.find("atom:title", ns)
            if title_el is None or title_el.text is None:
                continue
            title = " ".join(title_el.text.strip().split())

            # Skip duplicates
            if title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())

            # Parse published date — only keep 2026
            pub_el = entry_el.find("atom:published", ns)
            if pub_el is None or pub_el.text is None:
                continue
            pub_date = pub_el.text[:10]  # YYYY-MM-DD
            pub_year = pub_date[:4]
            if pub_year != "2026":
                continue

            # Parse authors
            authors = []
            for author_el in entry_el.findall("atom:author", ns):
                name_el = author_el.find("atom:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())
            author_str = " and ".join(authors)

            # Parse arXiv ID for DOI-like identifier
            id_el = entry_el.find("atom:id", ns)
            arxiv_url = id_el.text.strip() if id_el is not None else ""
            # Extract arXiv ID: http://arxiv.org/abs/XXXX.XXXXX -> XXXX.XXXXX
            arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else ""

            # Build BibTeX fields
            first_author_last = authors[0].split()[-1] if authors else "unknown"
            first_word = title.split()[0].lower() if title else "untitled"
            bibtex_key = f"{first_author_last}{pub_year}{first_word}"

            fields: dict[str, str] = {
                "title": title,
                "author": author_str,
                "year": pub_year,
            }
            if arxiv_id:
                fields["doi"] = f"10.48550/arXiv.{arxiv_id}"
                fields["url"] = arxiv_url

            # Determine category tag for venue field
            primary_cat = None
            for cat_el in entry_el.findall("atom:category", ns):
                term = cat_el.get("term", "")
                if term.startswith("cs.") or term.startswith("stat."):
                    primary_cat = term
                    break

            if primary_cat:
                fields["note"] = f"arXiv preprint, {primary_cat}"

            be = BenchmarkEntry(
                bibtex_key=bibtex_key,
                bibtex_type="misc",
                fields=fields,
                label="VALID",
                explanation="Valid arXiv preprint (2026)",
                generation_method=GenerationMethod.SCRAPED.value,
                source_conference="arXiv",
                publication_date=pub_date,
                added_to_benchmark=today,
                subtests={
                    **VALID_SUBTESTS,
                    "doi_resolves": bool(arxiv_id),
                },
            )
            all_entries.append(be)

        # Rate-limit between categories
        time.sleep(3.0)

    logger.info(f"Scraped {len(all_entries)} arXiv 2026 entries")
    return all_entries[:n_target]


def scrape_recent_entries(n_target: int = 30) -> list[BenchmarkEntry]:
    """Scrape recent (2024-2026) valid entries from DBLP and arXiv."""
    # First: arXiv 2026 papers
    n_arxiv = n_target // 2
    arxiv_entries = scrape_arxiv_2026(n_arxiv)

    # Remaining: DBLP 2024-2025 papers
    # Use stream: query format (venue:... returns 0 for recent years)
    n_dblp = n_target - len(arxiv_entries)
    dblp_entries: list[BenchmarkEntry] = []

    dblp_stream_keys = {
        "NeurIPS": "conf/nips",
        "ICML": "conf/icml",
        "ICLR": "conf/iclr",
        "AAAI": "conf/aaai",
        "CVPR": "conf/cvpr",
        "ACL": "conf/acl",
    }

    for venue, stream_key in dblp_stream_keys.items():
        if len(dblp_entries) >= n_dblp:
            break

        for year in [2025, 2024]:
            query = f"stream:{stream_key}: year:{year}"
            try:
                resp = httpx.get(
                    "https://dblp.org/search/publ/api",
                    params={"q": query, "format": "json", "h": 20},
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                logger.warning(f"DBLP query failed: {e}")
                continue

            hits_data = data.get("result", {}).get("hits", {})
            hits = hits_data.get("hit", [])
            infos = [h.get("info", {}) for h in hits if "info" in h]
            logger.info(f"  {venue} {year}: {len(infos)} hits from DBLP")

            for hit in infos:
                entry = dblp_hit_to_entry(hit, venue)
                if entry is not None:
                    dblp_entries.append(entry)
                    if len(dblp_entries) >= n_dblp:
                        break

            time.sleep(0.5)
            if len(dblp_entries) >= n_dblp:
                break

    all_entries = arxiv_entries + dblp_entries[:n_dblp]
    logger.info(
        f"Scraped {len(all_entries)} recent valid entries"
        f" ({len(arxiv_entries)} arXiv 2026,"
        f" {len(dblp_entries)} DBLP 2024-2025)"
    )
    return all_entries[:n_target]


def generate_hallucinated_probe(
    valid_entries: list[BenchmarkEntry], n_target: int = 30
) -> list[BenchmarkEntry]:
    """Generate hallucinated entries across all tiers from recent valid entries."""
    rng = random.Random(42)
    hallucinated: list[BenchmarkEntry] = []

    # Tier 1: Easy (fabricated_doi, nonexistent_venue, placeholder_authors)
    tier1_generators = [
        generate_fabricated_doi,
        generate_nonexistent_venue,
        generate_placeholder_authors,
    ]

    # Tier 2: Medium (wrong_venue, swapped_authors, chimeric_title, hybrid_fabrication)
    venues = ["NeurIPS", "ICML", "ICLR", "AAAI", "ACL", "CVPR"]
    ml_buzzwords = [
        "Attention",
        "Transformer",
        "Contrastive",
        "Diffusion",
        "Meta-Learning",
    ]

    # Tier 3: Hard (near_miss_title, plausible_fabrication)

    # Distribute roughly equally across tiers
    n_per_tier = n_target // 3

    # Tier 1
    for i in range(n_per_tier):
        source = rng.choice(valid_entries)
        gen = rng.choice(tier1_generators)
        entry = gen(source, rng)
        entry.bibtex_key = f"probe_t1_{i}_{entry.bibtex_key}"
        hallucinated.append(entry)

    # Tier 2
    for i in range(n_per_tier):
        source = rng.choice(valid_entries)
        method = rng.choice(
            ["wrong_venue", "swapped_authors", "chimeric_title", "hybrid_fabrication"]
        )
        if method == "wrong_venue":
            entry = generate_wrong_venue(source, rng.choice(venues), rng=rng)
        elif method == "swapped_authors":
            donor = rng.choice(valid_entries)
            while donor.bibtex_key == source.bibtex_key and len(valid_entries) > 1:
                donor = rng.choice(valid_entries)
            entry = generate_swapped_authors(source, donor, rng)
        elif method == "chimeric_title":
            fake_title = f"{rng.choice(ml_buzzwords)} for {rng.choice(['Classification', 'Generation', 'Reasoning'])}"
            entry = generate_chimeric_title(source, fake_title, rng)
        else:
            entry = generate_hybrid_fabrication(source, rng)
        entry.bibtex_key = f"probe_t2_{i}_{entry.bibtex_key}"
        hallucinated.append(entry)

    # Tier 3
    remaining = n_target - len(hallucinated)
    for i in range(remaining):
        source = rng.choice(valid_entries)
        method = rng.choice(["near_miss_title", "plausible_fabrication"])
        if method == "near_miss_title":
            entry = generate_near_miss_title(source, rng)
        else:
            entry = generate_plausible_fabrication(source, rng)
        entry.bibtex_key = f"probe_t3_{i}_{entry.bibtex_key}"
        hallucinated.append(entry)

    logger.info(f"Generated {len(hallucinated)} hallucinated entries for probe")
    return hallucinated


# ── Probe set persistence ─────────────────────────────────────────────


def save_probe_set(entries: list[BenchmarkEntry], path: Path) -> None:
    """Save probe entries to JSONL for reuse across models."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(entry.to_json() + "\n")
    logger.info(f"Saved {len(entries)} probe entries to {path}")


def load_probe_set(path: Path) -> list[BenchmarkEntry]:
    """Load cached probe entries from JSONL."""
    entries: list[BenchmarkEntry] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(BenchmarkEntry.from_json(line))
    logger.info(f"Loaded {len(entries)} probe entries from {path}")
    return entries


# ── Model evaluation ──────────────────────────────────────────────────


def load_baseline_metrics(results_dir: Path, baseline_file: str | None) -> dict:
    """Load full-dataset baseline metrics for comparison."""
    if baseline_file is None:
        return {}
    path = results_dir / baseline_file
    if not path.exists():
        logger.warning(f"Baseline file not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def run_model(
    model_key: str,
    probe_entries: list[BenchmarkEntry],
    results_dir: Path,
    checkpoint_dir: Path | None = None,
) -> dict:
    """Run a single model on the probe set and return the output data dict."""
    import os

    provider, model_id, env_var, baseline_file = MODEL_DISPATCH[model_key]
    api_key = os.environ.get(env_var)
    if not api_key:
        logger.error(f"Missing {env_var} for {model_key}")
        sys.exit(1)

    display = DISPLAY_NAMES.get(model_key, model_key)
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Running {display} ({model_id})")
    logger.info(f"{'=' * 70}")

    # Convert to blind entries for verification
    blind_entries = [e.to_blind() for e in probe_entries]

    # Dispatch to the right verify function
    if provider == "openai":
        predictions = verify_with_openai(
            blind_entries,
            model=model_id,
            api_key=api_key,
            checkpoint_dir=checkpoint_dir,
        )
    elif provider == "openrouter":
        predictions = verify_with_openrouter(
            blind_entries,
            model=model_id,
            api_key=api_key,
            checkpoint_dir=checkpoint_dir,
        )
    elif provider == "tool_augmented":
        predictions = verify_tool_augmented(
            blind_entries,
            model=model_id,
            api_key=api_key,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Evaluate
    result = evaluate(
        probe_entries,
        predictions,
        tool_name=f"temporal_probe_{model_key}",
        split_name="temporal_probe",
    )

    # Load baseline
    baseline = load_baseline_metrics(results_dir, baseline_file)

    # Report
    n_v = sum(1 for e in probe_entries if e.label == "VALID")
    n_h = sum(1 for e in probe_entries if e.label == "HALLUCINATED")
    logger.info(f"\n--- {display} Probe Results ---")
    logger.info(f"  Detection Rate:    {result.detection_rate:.3f}")
    logger.info(f"  FPR:               {result.false_positive_rate:.3f}")
    logger.info(f"  F1-Hallucination:  {result.f1_hallucination:.3f}")
    logger.info(
        f"  ECE:               {result.ece:.3f}"
        if result.ece is not None
        else "  ECE:               N/A"
    )

    if baseline:
        base_dr = baseline.get("detection_rate", "N/A")
        base_fpr = baseline.get("false_positive_rate", "N/A")
        logger.info(f"  Baseline DR:       {base_dr}")
        logger.info(f"  Baseline FPR:      {base_fpr}")
        if isinstance(base_fpr, (int, float)) and base_fpr > 0:
            fpr_mult = result.false_positive_rate / base_fpr
            logger.info(f"  FPR multiplier:    {fpr_mult:.1f}x")

    # Build output
    output = {
        "model_key": model_key,
        "display_name": display,
        "model_id": model_id,
        "probe_metrics": result.to_dict(),
        "full_baseline": baseline if baseline else {},
        "probe_config": {
            "n_valid": n_v,
            "n_hallucinated": n_h,
            "n_total": len(probe_entries),
        },
    }

    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Multi-model temporal robustness probe")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys or 'all' or 'none' (generate probe set only)",
    )
    parser.add_argument(
        "--probe-set",
        default="results/temporal_probe_set.jsonl",
        help="Path to cached probe set JSONL",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing baseline result JSONs",
    )
    parser.add_argument(
        "--n-valid",
        type=int,
        default=30,
        help="Number of valid entries to scrape",
    )
    parser.add_argument(
        "--n-hallucinated",
        type=int,
        default=30,
        help="Number of hallucinated entries",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="results/temporal_checkpoints",
        help="Directory for checkpoint files",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    probe_set_path = Path(args.probe_set)
    checkpoint_dir = Path(args.checkpoint_dir)

    # ── Step 1: Load or generate probe set ────────────────────────────
    if probe_set_path.exists():
        logger.info("=== Loading cached probe set ===")
        probe_entries = load_probe_set(probe_set_path)
    else:
        logger.info("=== Step 1: Scraping recent valid entries ===")
        valid_entries = scrape_recent_entries(args.n_valid)
        if len(valid_entries) < 10:
            logger.error(f"Only scraped {len(valid_entries)} valid entries — insufficient")
            sys.exit(1)

        logger.info("\n=== Step 2: Generating hallucinated entries ===")
        hallucinated_entries = generate_hallucinated_probe(valid_entries, args.n_hallucinated)

        probe_entries = valid_entries + hallucinated_entries
        save_probe_set(probe_entries, probe_set_path)

    n_v = sum(1 for e in probe_entries if e.label == "VALID")
    n_h = sum(1 for e in probe_entries if e.label == "HALLUCINATED")
    logger.info(f"Probe set: {n_v} valid + {n_h} hallucinated = {len(probe_entries)} total")

    # ── Step 2: Determine which models to run ─────────────────────────
    if args.models == "none":
        logger.info("Probe set generated. Exiting (--models none).")
        return

    if args.models == "all":
        model_keys = list(MODEL_DISPATCH.keys())
    else:
        model_keys = [k.strip() for k in args.models.split(",")]
        for k in model_keys:
            if k not in MODEL_DISPATCH:
                logger.error(f"Unknown model key: {k}. Valid: {list(MODEL_DISPATCH.keys())}")
                sys.exit(1)

    # ── Step 3: Run each model ────────────────────────────────────────
    for model_key in model_keys:
        output_path = results_dir / f"temporal_probe_{model_key}.json"

        # Skip if already completed
        if output_path.exists():
            logger.info(f"Skipping {model_key} — {output_path} already exists")
            continue

        output = run_model(model_key, probe_entries, results_dir, checkpoint_dir)

        # Save per-model results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {output_path}")

    # ── Summary ───────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 70}")
    logger.info("TEMPORAL ROBUSTNESS SUMMARY")
    logger.info(f"{'=' * 70}")
    for model_key in model_keys:
        output_path = results_dir / f"temporal_probe_{model_key}.json"
        if not output_path.exists():
            continue
        with open(output_path) as f:
            data = json.load(f)
        pm = data["probe_metrics"]
        bl = data.get("full_baseline", {})
        display = data.get("display_name", model_key)
        base_fpr = bl.get("false_positive_rate", None)
        probe_fpr = pm.get("false_positive_rate", 0)
        fpr_mult = f"{probe_fpr / base_fpr:.1f}x" if base_fpr and base_fpr > 0 else "N/A"
        logger.info(
            f"  {display:20s}  DR={pm['detection_rate']:.3f}  "
            f"FPR={probe_fpr:.3f}  "
            f"FPR mult={fpr_mult}  "
            f"ECE={pm.get('ece') or 0:.3f}"
        )


if __name__ == "__main__":
    main()
