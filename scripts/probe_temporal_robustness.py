"""Probe GPT-5.1 temporal robustness on recent (2025-2026) papers.  [analysis]

Scrapes fresh valid entries from DBLP and arXiv, generates hallucinated
variants, runs GPT-5.1, and compares with the full-dataset baseline.
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

from hallmark.baselines.llm_verifier import verify_with_openai
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


def load_full_dataset_baseline() -> dict:
    """Load the known full-dataset GPT-5.1 performance for comparison."""
    # These are the known values from the recent evaluation
    return {
        "detection_rate": 0.80,
        "f1_hallucination": None,  # Will load from results if available
        "false_positive_rate": None,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Probe GPT-5.1 temporal robustness")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--n-valid", type=int, default=30, help="Number of valid entries to scrape")
    parser.add_argument(
        "--n-hallucinated", type=int, default=30, help="Number of hallucinated entries"
    )
    parser.add_argument("--model", default="gpt-5.1", help="OpenAI model to use")
    parser.add_argument("--output", default="results/temporal_probe.json", help="Output file")
    args = parser.parse_args()

    # Step 1: Scrape recent valid entries
    logger.info("=== Step 1: Scraping recent valid entries from DBLP ===")
    valid_entries = scrape_recent_entries(args.n_valid)
    if len(valid_entries) < 10:
        logger.error(f"Only scraped {len(valid_entries)} valid entries — insufficient for probe")
        sys.exit(1)

    # Step 2: Generate hallucinated variants
    logger.info("\n=== Step 2: Generating hallucinated entries ===")
    hallucinated_entries = generate_hallucinated_probe(valid_entries, args.n_hallucinated)

    # Combine
    probe_entries = valid_entries + hallucinated_entries
    n_v, n_h = len(valid_entries), len(hallucinated_entries)
    logger.info(f"\nProbe set: {n_v} valid + {n_h} hallucinated = {len(probe_entries)} total")

    # Step 3: Run GPT-5.1
    logger.info("\n=== Step 3: Running GPT-5.1 on probe set ===")
    predictions = verify_with_openai(probe_entries, model=args.model, api_key=args.api_key)

    # Step 4: Evaluate
    logger.info("\n=== Step 4: Evaluating ===")
    result = evaluate(
        probe_entries,
        predictions,
        tool_name=f"llm_openai_{args.model}",
        split_name="temporal_probe_2025",
    )

    # Load full-dataset baseline for comparison
    full_baseline_dr = 0.80  # Known from earlier evaluation
    # Try to load actual results
    results_path = Path("results/llm_openai_dev_public_ci.json")
    full_baseline = {}
    if results_path.exists():
        with open(results_path) as f:
            full_baseline = json.load(f)

    # Step 5: Report
    logger.info("\n" + "=" * 70)
    logger.info("TEMPORAL ROBUSTNESS PROBE RESULTS")
    logger.info("=" * 70)
    logger.info(f"Probe set: {len(probe_entries)} entries ({n_v} valid, {n_h} hallucinated)")
    logger.info("  Years in probe: 2024-2026 (DBLP + arXiv)")
    logger.info("")

    # Probe metrics
    logger.info("--- Probe Metrics (2025 papers) ---")
    logger.info(f"  Detection Rate:    {result.detection_rate:.3f}")
    logger.info(f"  FPR:               {result.false_positive_rate:.3f}")
    logger.info(f"  F1-Hallucination:  {result.f1_hallucination:.3f}")
    logger.info(f"  Tier-weighted F1:  {result.tier_weighted_f1:.3f}")
    logger.info(f"  ECE:               {result.ece:.3f}")

    # Full-dataset comparison
    logger.info("")
    logger.info("--- Full Dataset Baseline (2021-2023 papers) ---")
    full_dr = full_baseline.get("detection_rate", full_baseline_dr)
    full_fpr = full_baseline.get("false_positive_rate", "N/A")
    full_f1 = full_baseline.get("f1_hallucination", "N/A")
    logger.info(f"  Detection Rate:    {full_dr}")
    logger.info(f"  FPR:               {full_fpr}")
    logger.info(f"  F1-Hallucination:  {full_f1}")

    # Discrepancy analysis
    logger.info("")
    logger.info("--- Discrepancy Analysis ---")
    dr_diff = result.detection_rate - float(full_dr) if isinstance(full_dr, (int, float)) else None
    if dr_diff is not None:
        if dr_diff < -0.05:
            verdict = "PROBE WORSE"
        elif dr_diff > 0.05:
            verdict = "PROBE BETTER"
        else:
            verdict = "COMPARABLE"
        logger.info(f"  DR difference:     {dr_diff:+.3f} ({verdict})")
        if abs(dr_diff) > 0.05:
            logger.info(
                "  >>> SIGNIFICANT DISCREPANCY DETECTED — benchmark extension may be warranted"
            )
        else:
            logger.info(
                "  >>> No significant discrepancy — benchmark extension NOT needed for temporal coverage"
            )

    # Per-tier breakdown
    logger.info("")
    logger.info("--- Per-Tier Breakdown (Probe) ---")
    for tier, metrics in sorted(result.per_tier_metrics.items()):
        if tier == 0:
            continue  # Skip valid entries tier
        dr = metrics["detection_rate"]
        f1 = metrics["f1"]
        cnt = metrics["count"]
        logger.info(f"  Tier {tier}: DR={dr:.3f}, F1={f1:.3f}, count={cnt:.0f}")

    # Per-type breakdown
    logger.info("")
    logger.info("--- Per-Type Breakdown (Probe) ---")
    for h_type, metrics in sorted(result.per_type_metrics.items()):
        if h_type == "valid":
            continue
        logger.info(
            f"  {h_type:30s}: DR={metrics['detection_rate']:.3f}, count={metrics['count']:.0f}"
        )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "probe_metrics": result.to_dict(),
        "full_baseline": full_baseline if full_baseline else {"detection_rate": full_baseline_dr},
        "discrepancy": {
            "dr_diff": dr_diff,
            "significant": abs(dr_diff) > 0.05 if dr_diff is not None else None,
            "recommendation": (
                "EXTEND benchmark with 2025+ entries"
                if dr_diff is not None and abs(dr_diff) > 0.05
                else "No extension needed — performance is comparable"
            ),
        },
        "probe_config": {
            "n_valid": len(valid_entries),
            "n_hallucinated": len(hallucinated_entries),
            "model": args.model,
            "years_scraped": [2024, 2025],
        },
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
