"""Stage 4: Scale up hallucinated entries to meet minimum per-type thresholds.

Computes gaps per hallucination type and generates entries using the canonical
generators from hallmark.dataset.generator. Reuses logic from scale_up_dataset.py.
"""

from __future__ import annotations

import logging
import random
from collections import Counter

from hallmark.dataset.generator import (
    generate_chimeric_title,
    generate_fabricated_doi,
    generate_future_date,
    generate_hybrid_fabrication,
    generate_merged_citation,
    generate_near_miss_title,
    generate_nonexistent_venue,
    generate_partial_author_list,
    generate_placeholder_authors,
    generate_plausible_fabrication,
    generate_preprint_as_published,
    generate_swapped_authors,
    generate_version_confusion,
    generate_wrong_venue,
    is_preprint_source,
)
from hallmark.dataset.schema import BenchmarkEntry, HallucinationType

logger = logging.getLogger(__name__)

# ML buzzwords for chimeric titles
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


def compute_gaps(
    entries: list[BenchmarkEntry],
    target: int,
) -> dict[str, int]:
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


def _generate_for_type(
    type_val: str,
    source: BenchmarkEntry,
    valid_entries: list[BenchmarkEntry],
    rng: random.Random,
    split_name: str,
    idx: int,
    chimeric_titles: list[str],
    chimeric_idx: list[int],
    build_date: str,
) -> BenchmarkEntry:
    """Generate a single entry of the given hallucination type."""
    if type_val == HallucinationType.FABRICATED_DOI.value:
        entry = generate_fabricated_doi(source, rng)
    elif type_val == HallucinationType.NONEXISTENT_VENUE.value:
        entry = generate_nonexistent_venue(source, rng)
    elif type_val == HallucinationType.PLACEHOLDER_AUTHORS.value:
        entry = generate_placeholder_authors(source, rng)
    elif type_val == HallucinationType.FUTURE_DATE.value:
        entry = generate_future_date(source, rng)
    elif type_val == HallucinationType.CHIMERIC_TITLE.value:
        title = chimeric_titles[chimeric_idx[0] % len(chimeric_titles)]
        chimeric_idx[0] += 1
        entry = generate_chimeric_title(source, title, rng)
    elif type_val == HallucinationType.WRONG_VENUE.value:
        current = source.venue
        candidates = [v for v in VENUES if v != current]
        entry = generate_wrong_venue(source, rng.choice(candidates), rng=rng)
    elif type_val == HallucinationType.AUTHOR_MISMATCH.value:
        donor = rng.choice(valid_entries)
        while donor.bibtex_key == source.bibtex_key and len(valid_entries) > 1:
            donor = rng.choice(valid_entries)
        entry = generate_swapped_authors(source, donor, rng)
    elif type_val == HallucinationType.PREPRINT_AS_PUBLISHED.value:
        preprint_sources = [e for e in valid_entries if is_preprint_source(e)]
        if preprint_sources:
            source = rng.choice(preprint_sources)
        entry = generate_preprint_as_published(source, rng.choice(VENUES), rng)
    elif type_val == HallucinationType.HYBRID_FABRICATION.value:
        entry = generate_hybrid_fabrication(source, rng)
    elif type_val == HallucinationType.MERGED_CITATION.value:
        donor_b = rng.choice(valid_entries)
        while donor_b.bibtex_key == source.bibtex_key and len(valid_entries) > 1:
            donor_b = rng.choice(valid_entries)
        donor_c = rng.choice(valid_entries) if rng.random() < 0.5 else None
        entry = generate_merged_citation(source, donor_b, donor_c, rng)
    elif type_val == HallucinationType.PARTIAL_AUTHOR_LIST.value:
        entry = generate_partial_author_list(source, rng)
    elif type_val == HallucinationType.NEAR_MISS_TITLE.value:
        entry = generate_near_miss_title(source, rng)
    elif type_val == HallucinationType.PLAUSIBLE_FABRICATION.value:
        entry = generate_plausible_fabrication(source, rng)
    elif type_val == HallucinationType.VERSION_CONFUSION.value:
        current = source.venue
        candidates = [v for v in VENUES if v != current]
        entry = generate_version_confusion(source, rng.choice(candidates), rng)
    else:
        raise ValueError(f"Unknown hallucination type: {type_val}")

    entry.source = "perturbation_scaleup"
    entry.added_to_benchmark = build_date
    entry.bibtex_key = f"scaleup_{type_val}_{split_name}_{idx}"
    return entry


def generate_entries_for_gaps(
    gaps: dict[str, int],
    valid_entries: list[BenchmarkEntry],
    split_name: str,
    rng: random.Random,
    existing_keys: set[str],
    build_date: str,
) -> list[BenchmarkEntry]:
    """Generate exactly the needed entries to fill gaps.

    Args:
        gaps: Dict mapping hallucination type to count needed.
        valid_entries: Pool of valid entries to perturb.
        split_name: Name of the split (for key generation).
        rng: Random number generator.
        existing_keys: Set of keys already in use (mutated in-place).
        build_date: ISO date string for added_to_benchmark.

    Returns:
        List of new hallucinated entries.
    """
    new_entries: list[BenchmarkEntry] = []

    # Prepare chimeric title pool
    available_buzzwords = list(ML_BUZZWORDS)
    rng.shuffle(available_buzzwords)
    chimeric_idx = [0]  # mutable counter for closure

    for type_val, count in sorted(gaps.items()):
        for i in range(count):
            source = rng.choice(valid_entries)
            entry = _generate_for_type(
                type_val,
                source,
                valid_entries,
                rng,
                split_name,
                i,
                available_buzzwords,
                chimeric_idx,
                build_date,
            )

            # Ensure unique key
            while entry.bibtex_key in existing_keys:
                i += 1
                entry.bibtex_key = f"scaleup_{type_val}_{split_name}_{i}"
            existing_keys.add(entry.bibtex_key)

            new_entries.append(entry)

    return new_entries


def stage_scale_up(
    splits: dict[str, list[BenchmarkEntry]],
    min_per_type: int,
    seed: int,
    build_date: str,
) -> dict[str, list[BenchmarkEntry]]:
    """Scale up hallucinated entries to >= min_per_type per type per public split.

    Args:
        splits: Current split data (modified in-place).
        min_per_type: Minimum entries per hallucination type.
        seed: Random seed.
        build_date: ISO date for new entries.

    Returns:
        Updated splits dict.
    """
    rng = random.Random(seed)
    all_keys: set[str] = set()
    for entries in splits.values():
        for e in entries:
            all_keys.add(e.bibtex_key)

    for split_name in ["dev_public", "test_public"]:
        entries = splits[split_name]
        valid = [e for e in entries if e.label == "VALID"]
        gaps = compute_gaps(entries, min_per_type)

        if not gaps:
            logger.info("%s: all types already >= %d", split_name, min_per_type)
            continue

        total_gap = sum(gaps.values())
        logger.info("%s: need %d entries across %d types", split_name, total_gap, len(gaps))

        new_entries = generate_entries_for_gaps(
            gaps,
            valid,
            split_name,
            rng,
            all_keys,
            build_date,
        )
        splits[split_name] = entries + new_entries
        logger.info(
            "%s: added %d entries (now %d total)",
            split_name,
            len(new_entries),
            len(splits[split_name]),
        )

    return splits
