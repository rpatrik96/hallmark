"""Scale up HALLMARK dataset to ensure ≥10 hallucinated entries per type per split.

Loads existing splits, computes gaps, generates entries using existing generators,
appends to split files, and updates metadata.json.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from hallmark.dataset.generator import (
    generate_arxiv_version_mismatch,
    generate_chimeric_title,
    generate_fabricated_doi,
    generate_future_date,
    generate_hybrid_fabrication,
    generate_near_miss_title,
    generate_nonexistent_venue,
    generate_placeholder_authors,
    generate_plausible_fabrication,
    generate_preprint_as_published,
    generate_swapped_authors,
    generate_wrong_venue,
    is_preprint_source,
)
from hallmark.dataset.schema import (
    BenchmarkEntry,
    HallucinationType,
    load_entries,
    save_entries,
)

SEED = 2026_02_12
ADDED_DATE = "2026-02-12"
MIN_PER_TYPE = 30

# ── Additional arXiv-to-venue mappings for arxiv_version_mismatch ────────────────
# Split into separate pools to prevent cross-split contamination


# ── ML buzzwords for chimeric titles ────────────────────────────────────────

ML_BUZZWORDS = [
    "Self-Attention Mechanisms for Low-Resource Temporal Reasoning",
    "Cross-Modal Representation Learning in Heterogeneous Domains",
    "Contrastive Self-Supervised Methods for Dense Prediction",
    "Few-Shot Meta-Learning with Task-Adaptive Initialization",
    "Diffusion-Based Generative Models for Molecular Design",
    "Neural Architecture Search with Hardware Constraints",
    "Multi-Task Transfer Learning Across Modalities",
    "Graph Neural Network Architectures for Combinatorial Optimization",
    "Prompt-Tuning Strategies for Instruction-Following Models",
    "Retrieval-Augmented Generation for Long-Form Question Answering",
    "Vision-Language Alignment via Contrastive Pre-Training",
    "Causal Inference Methods for Treatment Effect Estimation",
    "Federated Learning with Heterogeneous Client Distributions",
    "Sparse Mixture-of-Experts for Efficient Inference",
    "Test-Time Adaptation Under Distribution Shift",
    "Token-Free Language Modeling with Character-Level Transformers",
    "Reward Modeling for Alignment of Large Language Models",
    "Low-Rank Adaptation for Parameter-Efficient Fine-Tuning",
    "Continual Learning Without Catastrophic Forgetting",
    "Efficient Attention via Linear Complexity Approximations",
    "Denoising Diffusion Probabilistic Models for Image Restoration",
    "Self-Supervised Speech Representation Learning",
    "Equivariant Neural Networks for Physical Simulations",
    "Bayesian Optimization for Hyperparameter Tuning",
    "Multi-Agent Reinforcement Learning in Cooperative Settings",
    "Knowledge Distillation for Model Compression",
    "Adversarial Robustness Through Certified Defenses",
    "Neural Radiance Fields for Novel View Synthesis",
    "Temporal Graph Networks for Dynamic Interaction Modeling",
    "Data Augmentation Strategies for Low-Resource NLP",
    "Hierarchical Reinforcement Learning with Temporal Abstraction",
    "Attention Mechanisms for Sequential Decision Making",
    "Uncertainty Quantification in Deep Neural Networks",
    "Efficient Transformer Architectures for Long Sequences",
    "Multi-Modal Fusion for Visual Question Answering",
    "Domain Adaptation via Adversarial Training",
    "Neural Program Synthesis from Input-Output Examples",
    "Explainable AI Through Attention Visualization",
    "Graph Transformers for Molecular Property Prediction",
    "Few-Shot Learning via Prototypical Networks",
    "Meta-Reinforcement Learning for Task Distribution",
    "Causal Discovery from Observational Data",
    "Self-Supervised Learning for Medical Imaging",
    "Neural Architecture Search with Evolutionary Algorithms",
    "Multimodal Pre-Training for Vision and Language",
    "Efficient Neural Network Pruning Techniques",
    "Gradient-Based Meta-Learning for Quick Adaptation",
    "Contrastive Learning for Self-Supervised Representation",
    "Neural Ordinary Differential Equations for Time Series",
    "Adversarial Training for Distribution Robustness",
    "Knowledge Graph Completion via Relation Prediction",
    "Transformer-Based Models for Code Generation",
    "Curriculum Learning for Complex Task Training",
    "Neural Scene Representation and Rendering",
    "Multi-Agent Communication with Emergent Protocols",
    "Active Learning Strategies for Label Efficiency",
    "Deep Learning for Combinatorial Optimization Problems",
    "Variational Autoencoders for Anomaly Detection",
    "Neural Machine Translation with Attention",
    "Graph Neural Networks for Traffic Forecasting",
    "Reinforcement Learning from Human Feedback",
    "Diffusion Models for High-Resolution Image Synthesis",
    "Vision Transformers for Dense Prediction Tasks",
    "Neural Architecture Search Under Resource Constraints",
    "Self-Supervised Speech Representation via Contrastive Learning",
    "Probabilistic Forecasting with Deep Learning",
    "Neural Tangent Kernel Theory and Applications",
    "Multi-Objective Optimization in Neural Architecture Search",
    "Geometric Deep Learning on Manifolds and Graphs",
    "Neural Sequence-to-Sequence Models with Copy Mechanism",
]

# ── Venues for wrong_venue and preprint_as_published ────────────────────────

VENUES = [
    "NeurIPS",
    "ICML",
    "ICLR",
    "AAAI",
    "ACL",
    "CVPR",
    "ECCV",
    "EMNLP",
    "AISTATS",
    "UAI",
    "IJCAI",
    "COLT",
    "KDD",
    "WWW",
    "SIGIR",
]


# ── Per-type generation logic ───────────────────────────────────────────────


def _gen_fabricated_doi(
    source: BenchmarkEntry, rng: random.Random, split: str, idx: int
) -> BenchmarkEntry:
    entry = generate_fabricated_doi(source, rng)
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_fabdoi_{split}_{idx}"
    return entry


def _gen_nonexistent_venue(
    source: BenchmarkEntry, rng: random.Random, split: str, idx: int
) -> BenchmarkEntry:
    entry = generate_nonexistent_venue(source, rng)
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_nonvenue_{split}_{idx}"
    return entry


def _gen_placeholder_authors(
    source: BenchmarkEntry, rng: random.Random, split: str, idx: int
) -> BenchmarkEntry:
    entry = generate_placeholder_authors(source, rng)
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_placeholder_{split}_{idx}"
    return entry


def _gen_future_date(
    source: BenchmarkEntry, rng: random.Random, split: str, idx: int
) -> BenchmarkEntry:
    entry = generate_future_date(source, rng)
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_future_{split}_{idx}"
    return entry


def _gen_chimeric_title(
    source: BenchmarkEntry, rng: random.Random, split: str, idx: int
) -> BenchmarkEntry:
    fake_title = rng.choice(ML_BUZZWORDS)
    entry = generate_chimeric_title(source, fake_title, rng)
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_chimeric_{split}_{idx}"
    return entry


def _gen_wrong_venue(
    source: BenchmarkEntry, rng: random.Random, split: str, idx: int
) -> BenchmarkEntry:
    # Pick a venue different from the source's
    current_venue = source.venue
    candidates = [v for v in VENUES if v != current_venue]
    wrong_v = rng.choice(candidates)
    entry = generate_wrong_venue(source, wrong_v, rng=rng)
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_wrongvenue_{split}_{idx}"
    return entry


def _gen_swapped_authors(
    source: BenchmarkEntry,
    valid_entries: list[BenchmarkEntry],
    rng: random.Random,
    split: str,
    idx: int,
) -> BenchmarkEntry:
    donor = rng.choice(valid_entries)
    while donor.bibtex_key == source.bibtex_key and len(valid_entries) > 1:
        donor = rng.choice(valid_entries)
    entry = generate_swapped_authors(source, donor, rng)
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_swapped_{split}_{idx}"
    return entry


def _gen_preprint_as_published(
    source: BenchmarkEntry, rng: random.Random, split: str, idx: int
) -> BenchmarkEntry:
    fake_venue = rng.choice(VENUES)
    entry = generate_preprint_as_published(source, fake_venue, rng)
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_preprint_{split}_{idx}"
    return entry


def _gen_hybrid_fabrication(
    source: BenchmarkEntry, rng: random.Random, split: str, idx: int
) -> BenchmarkEntry:
    entry = generate_hybrid_fabrication(source, rng)
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_hybrid_{split}_{idx}"
    return entry


def _gen_near_miss_title(
    source: BenchmarkEntry, rng: random.Random, split: str, idx: int
) -> BenchmarkEntry:
    entry = generate_near_miss_title(source, rng)
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_nearmiss_{split}_{idx}"
    return entry


def _gen_plausible_fabrication(
    source: BenchmarkEntry, rng: random.Random, split: str, idx: int
) -> BenchmarkEntry:
    entry = generate_plausible_fabrication(source, rng)
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_plausible_{split}_{idx}"
    return entry


def _gen_arxiv_version_mismatch(
    source: BenchmarkEntry,
    rng: random.Random,
    split: str,
    idx: int,
) -> BenchmarkEntry:
    # Pick a venue different from the source's actual venue
    current_venue = source.venue
    candidates = [v for v in VENUES if v != current_venue]
    wrong_venue = rng.choice(candidates)
    entry = generate_arxiv_version_mismatch(source, wrong_venue, rng)
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_version_{split}_{idx}"
    return entry


# ── Main logic ──────────────────────────────────────────────────────────────


def decontaminate_splits(
    dev_entries: list[BenchmarkEntry],
    test_entries: list[BenchmarkEntry],
    rng: random.Random,
) -> int:
    """Replace overlapping arxiv_version_mismatch eprints in dev with fresh ones.

    Since eprints are now fabricated randomly, collisions are extremely rare.
    Modifies dev_entries in-place. Returns number of entries replaced.
    """
    test_version_eprints = {
        e.fields.get("eprint", "")
        for e in test_entries
        if e.hallucination_type == "arxiv_version_mismatch"
    }

    replaced = 0
    for e in dev_entries:
        if (
            e.hallucination_type == "arxiv_version_mismatch"
            and e.fields.get("eprint", "") in test_version_eprints
        ):
            # Regenerate a unique fabricated eprint
            year = int(e.fields.get("year", "2020"))
            arxiv_yymm = f"{year % 100:02d}{rng.randint(1, 12):02d}"
            arxiv_seq = f"{rng.randint(1, 9999):05d}"
            e.fields["eprint"] = f"{arxiv_yymm}.{arxiv_seq}"
            replaced += 1

    return replaced


def compute_gaps(entries: list[BenchmarkEntry], target: int = MIN_PER_TYPE) -> dict[str, int]:
    """Compute per-type gap to reach target count."""
    type_counts: Counter[str] = Counter()
    for e in entries:
        if e.label == "HALLUCINATED" and e.hallucination_type:
            type_counts[e.hallucination_type] += 1

    gaps: dict[str, int] = {}
    for ht in HallucinationType:
        current = type_counts.get(ht.value, 0)
        gap = max(0, target - current)
        if gap > 0:
            gaps[ht.value] = gap
    return gaps


def generate_entries_for_gaps(
    gaps: dict[str, int],
    valid_entries: list[BenchmarkEntry],
    split_name: str,
    rng: random.Random,
    other_split_entries: list[BenchmarkEntry] | None = None,
    existing_entries: list[BenchmarkEntry] | None = None,
) -> list[BenchmarkEntry]:
    """Generate exactly the needed entries to fill gaps."""
    new_entries: list[BenchmarkEntry] = []

    # Compute offset for unique key generation based on existing scaleup entries
    existing_entries = existing_entries or []
    existing_keys = {e.bibtex_key for e in existing_entries}
    key_offset = 0
    chimeric_title_idx = 0

    def make_unique_key(prefix: str, idx: int) -> str:
        """Generate unique key by incrementing until no collision."""
        nonlocal key_offset
        candidate = f"{prefix}_{split_name}_{idx + key_offset}"
        while candidate in existing_keys:
            key_offset += 1
            candidate = f"{prefix}_{split_name}_{idx + key_offset}"
        existing_keys.add(candidate)
        return candidate

    def get_next_chimeric_title() -> str:
        """Get next unique chimeric title from shuffled pool."""
        nonlocal chimeric_title_idx
        title = available_buzzwords[chimeric_title_idx % len(available_buzzwords)]
        chimeric_title_idx += 1
        return title

    # Track used chimeric titles to ensure diversity
    existing_chimeric_titles = {
        e.fields.get("title", "")
        for e in existing_entries
        if e.hallucination_type == "chimeric_title"
    }
    available_buzzwords = [b for b in ML_BUZZWORDS if b not in existing_chimeric_titles]
    rng.shuffle(available_buzzwords)

    for type_val, count in sorted(gaps.items()):
        for i in range(count):
            source = rng.choice(valid_entries)

            if type_val == HallucinationType.FABRICATED_DOI.value:
                entry = _gen_fabricated_doi(source, rng, split_name, i)
                entry.bibtex_key = make_unique_key("scaleup_fabdoi", i)
            elif type_val == HallucinationType.NONEXISTENT_VENUE.value:
                entry = _gen_nonexistent_venue(source, rng, split_name, i)
                entry.bibtex_key = make_unique_key("scaleup_nonvenue", i)
            elif type_val == HallucinationType.PLACEHOLDER_AUTHORS.value:
                entry = _gen_placeholder_authors(source, rng, split_name, i)
                entry.bibtex_key = make_unique_key("scaleup_placeholder", i)
            elif type_val == HallucinationType.FUTURE_DATE.value:
                entry = _gen_future_date(source, rng, split_name, i)
                entry.bibtex_key = make_unique_key("scaleup_future", i)
            elif type_val == HallucinationType.CHIMERIC_TITLE.value:
                # Use sequential selection from shuffled pool for diversity
                fake_title = get_next_chimeric_title()
                entry = generate_chimeric_title(source, fake_title, rng)
                entry.added_to_benchmark = ADDED_DATE
                entry.bibtex_key = make_unique_key("scaleup_chimeric", i)
            elif type_val == HallucinationType.WRONG_VENUE.value:
                entry = _gen_wrong_venue(source, rng, split_name, i)
                entry.bibtex_key = make_unique_key("scaleup_wrongvenue", i)
            elif type_val == HallucinationType.AUTHOR_MISMATCH.value:
                entry = _gen_swapped_authors(source, valid_entries, rng, split_name, i)
                entry.bibtex_key = make_unique_key("scaleup_swapped", i)
            elif type_val == HallucinationType.PREPRINT_AS_PUBLISHED.value:
                # Filter for genuine preprint sources (no conference DOI)
                preprint_sources = [e for e in valid_entries if is_preprint_source(e)]
                if preprint_sources:
                    source = rng.choice(preprint_sources)
                entry = _gen_preprint_as_published(source, rng, split_name, i)
                entry.bibtex_key = make_unique_key("scaleup_preprint", i)
            elif type_val == HallucinationType.HYBRID_FABRICATION.value:
                entry = _gen_hybrid_fabrication(source, rng, split_name, i)
                entry.bibtex_key = make_unique_key("scaleup_hybrid", i)
            elif type_val == HallucinationType.NEAR_MISS_TITLE.value:
                entry = _gen_near_miss_title(source, rng, split_name, i)
                entry.bibtex_key = make_unique_key("scaleup_nearmiss", i)
            elif type_val == HallucinationType.PLAUSIBLE_FABRICATION.value:
                entry = _gen_plausible_fabrication(source, rng, split_name, i)
                entry.bibtex_key = make_unique_key("scaleup_plausible", i)
            elif type_val == HallucinationType.ARXIV_VERSION_MISMATCH.value:
                entry = _gen_arxiv_version_mismatch(source, rng, split_name, i)
                entry.bibtex_key = make_unique_key("scaleup_version", i)
            else:
                raise ValueError(f"Unknown hallucination type: {type_val}")

            new_entries.append(entry)

    return new_entries


def compute_type_distribution(entries: list[BenchmarkEntry]) -> dict[str, int]:
    """Count hallucinated entries by type."""
    counts: Counter[str] = Counter()
    for e in entries:
        if e.label == "HALLUCINATED" and e.hallucination_type:
            counts[e.hallucination_type] += 1
    return dict(sorted(counts.items()))


def compute_tier_distribution(entries: list[BenchmarkEntry]) -> dict[str, int]:
    """Count hallucinated entries by difficulty tier."""
    counts: Counter[int] = Counter()
    for e in entries:
        if e.label == "HALLUCINATED" and e.difficulty_tier is not None:
            counts[e.difficulty_tier] += 1
    return {str(k): v for k, v in sorted(counts.items())}


def update_metadata(
    metadata_path: Path,
    dev_entries: list[BenchmarkEntry],
    test_entries: list[BenchmarkEntry],
) -> None:
    """Update metadata.json with new split statistics."""
    with open(metadata_path) as f:
        metadata = json.load(f)

    for split_name, entries in [
        ("dev_public", dev_entries),
        ("test_public", test_entries),
    ]:
        valid = sum(1 for e in entries if e.label == "VALID")
        hallucinated = sum(1 for e in entries if e.label == "HALLUCINATED")
        metadata["splits"][split_name]["total"] = len(entries)
        metadata["splits"][split_name]["valid"] = valid
        metadata["splits"][split_name]["hallucinated"] = hallucinated
        metadata["splits"][split_name]["tier_distribution"] = compute_tier_distribution(entries)
        metadata["splits"][split_name]["type_distribution"] = compute_type_distribution(entries)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        f.write("\n")


def print_summary(
    split_name: str,
    entries: list[BenchmarkEntry],
    gaps: dict[str, int],
    num_new: int,
) -> None:
    """Print summary table for a split."""
    type_dist = compute_type_distribution(entries)
    tier_dist = compute_tier_distribution(entries)
    valid = sum(1 for e in entries if e.label == "VALID")
    hallucinated = sum(1 for e in entries if e.label == "HALLUCINATED")

    print(f"\n{'=' * 60}")
    print(f"  {split_name}: {len(entries)} total ({valid} valid, {hallucinated} hallucinated)")
    print(f"  Added: {num_new} new hallucinated entries")
    print(f"  Tier distribution: {tier_dist}")
    print(f"{'=' * 60}")
    print(f"  {'Type':<28} {'Count':>5}  {'Added':>5}")
    print(f"  {'-' * 40}")
    for ht in HallucinationType:
        count = type_dist.get(ht.value, 0)
        added = gaps.get(ht.value, 0)
        marker = " *" if added > 0 else ""
        print(f"  {ht.value:<28} {count:>5}  +{added:<4}{marker}")
    print()


def main() -> None:
    """Scale up dataset to ≥10 hallucinated entries per type per split."""
    data_dir = Path("data/v1.0")
    dev_path = data_dir / "dev_public.jsonl"
    test_path = data_dir / "test_public.jsonl"
    metadata_path = data_dir / "metadata.json"

    rng = random.Random(SEED)

    # Load existing entries
    print("Loading existing entries...")
    dev_entries = load_entries(dev_path)
    test_entries = load_entries(test_path)

    dev_valid = [e for e in dev_entries if e.label == "VALID"]
    test_valid = [e for e in test_entries if e.label == "VALID"]

    print(f"  dev_public: {len(dev_entries)} entries ({len(dev_valid)} valid)")
    print(f"  test_public: {len(test_entries)} entries ({len(test_valid)} valid)")

    # Decontaminate: replace overlapping retracted/version entries in dev
    n_replaced = decontaminate_splits(dev_entries, test_entries, rng)
    if n_replaced:
        print(f"\nDecontamination: replaced {n_replaced} overlapping dev entries")

    # Compute gaps
    dev_gaps = compute_gaps(dev_entries)
    test_gaps = compute_gaps(test_entries)

    print(f"\nGaps to fill (target: {MIN_PER_TYPE} per type):")
    print(f"  dev_public: {sum(dev_gaps.values())} entries across {len(dev_gaps)} types")
    print(f"  test_public: {sum(test_gaps.values())} entries across {len(test_gaps)} types")

    # Generate new entries (pass other split to exclude cross-split contamination)
    print("\nGenerating entries for dev_public...")
    dev_new = generate_entries_for_gaps(
        dev_gaps,
        dev_valid,
        "dev",
        rng,
        other_split_entries=test_entries,
        existing_entries=dev_entries,
    )

    print("Generating entries for test_public...")
    test_new = generate_entries_for_gaps(
        test_gaps,
        test_valid,
        "test",
        rng,
        other_split_entries=dev_entries,
        existing_entries=test_entries,
    )

    # Append and save
    dev_entries.extend(dev_new)
    test_entries.extend(test_new)

    print("\nSaving updated dataset files...")
    save_entries(dev_entries, dev_path)
    save_entries(test_entries, test_path)

    # Update metadata
    print("Updating metadata.json...")
    update_metadata(metadata_path, dev_entries, test_entries)

    # Print summaries
    print_summary("dev_public", dev_entries, dev_gaps, len(dev_new))
    print_summary("test_public", test_entries, test_gaps, len(test_new))

    # Verify minimum counts
    print("Verification:")
    all_ok = True
    for split_name, entries in [("dev_public", dev_entries), ("test_public", test_entries)]:
        type_dist = compute_type_distribution(entries)
        for ht in HallucinationType:
            count = type_dist.get(ht.value, 0)
            if count < MIN_PER_TYPE:
                print(f"  FAIL: {split_name} {ht.value} has {count} < {MIN_PER_TYPE}")
                all_ok = False

    if all_ok:
        print(f"  OK: All types have >= {MIN_PER_TYPE} entries in both splits.")
    else:
        print("  FAILED: Some types are below minimum. Check generation logic.")
        raise SystemExit(1)

    # Grand totals
    total = len(dev_entries) + len(test_entries)
    total_halluc = sum(1 for e in dev_entries if e.label == "HALLUCINATED") + sum(
        1 for e in test_entries if e.label == "HALLUCINATED"
    )
    print(f"\nGrand total (dev + test): {total} entries ({total_halluc} hallucinated)")
    print("Done.")


if __name__ == "__main__":
    main()
