"""Shared utilities for pipeline stages.

Constants, type-gap computation, and the unified type→generator dispatch
used by both scale_up (Stage 4) and expand_hidden (Stage 7).
"""

from __future__ import annotations

import random
from collections import Counter

from hallmark.dataset.generator import (
    generate_arxiv_version_mismatch,
    generate_chimeric_title,
    generate_merged_citation,
    generate_preprint_as_published,
    generate_swapped_authors,
    generate_wrong_venue,
    is_preprint_source,
)
from hallmark.dataset.generators._registry import get_generator, get_generator_func
from hallmark.dataset.generators.tier1 import _current_reference_year
from hallmark.dataset.schema import BenchmarkEntry, HallucinationType

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

# Keys of fabricated entries in real_world_incidents.jsonl
FAKE_REALWORLD_KEYS = {
    "realworld_future_date_pattern",
    "realworld_nonexistent_venue",
    "realworld_fabricated_doi",
    "realworld_hybrid_fabrication",
}


def compute_type_gaps(
    entries: list[BenchmarkEntry],
    target_per_type: int,
) -> dict[str, int]:
    """Compute per-type gap to reach target count."""
    type_counts: Counter[str] = Counter()
    for e in entries:
        if e.label == "HALLUCINATED" and e.hallucination_type:
            type_counts[e.hallucination_type] += 1

    gaps: dict[str, int] = {}
    for ht in HallucinationType:
        current = type_counts.get(ht.value, 0)
        gap = max(0, target_per_type - current)
        if gap > 0:
            gaps[ht.value] = gap
    return gaps


def generate_for_type(
    type_val: str,
    source: BenchmarkEntry,
    valid_entries: list[BenchmarkEntry],
    rng: random.Random,
    key_prefix: str,
    idx: int,
    chimeric_titles: list[str],
    chimeric_idx: list[int],
    build_date: str,
) -> BenchmarkEntry:
    """Unified type→generator dispatch. Used by scale_up and expand_hidden."""
    hall_type = HallucinationType(type_val)
    spec = get_generator(hall_type)
    gen_func = get_generator_func(hall_type)

    # Determine reference_year from build_date (Change 6: thread to future_date)
    reference_year: int | None = None
    if build_date:
        try:
            reference_year = int(build_date[:4])
        except ValueError:
            reference_year = _current_reference_year()

    # Types that need a venue candidate (different from current source venue)
    def _pick_venue() -> str:
        current = source.venue
        candidates = [v for v in VENUES if v != current]
        return rng.choice(candidates)

    # Assemble kwargs based on what the generator's extra_args declare,
    # plus any dispatch-level argument preparation that can't be inferred
    # from the spec alone (donor selection, chimeric title cycling, etc.).
    if hall_type == HallucinationType.CHIMERIC_TITLE:
        title = chimeric_titles[chimeric_idx[0] % len(chimeric_titles)]
        chimeric_idx[0] += 1
        entry = generate_chimeric_title(source, title, rng)

    elif hall_type == HallucinationType.WRONG_VENUE:
        entry = generate_wrong_venue(source, _pick_venue(), rng=rng)

    elif hall_type == HallucinationType.AUTHOR_MISMATCH:
        donor = rng.choice(valid_entries)
        while donor.bibtex_key == source.bibtex_key and len(valid_entries) > 1:
            donor = rng.choice(valid_entries)
        entry = generate_swapped_authors(source, donor, rng)

    elif hall_type == HallucinationType.PREPRINT_AS_PUBLISHED:
        preprint_sources = [e for e in valid_entries if is_preprint_source(e)]
        effective_source = rng.choice(preprint_sources) if preprint_sources else source
        entry = generate_preprint_as_published(effective_source, rng.choice(VENUES), rng)

    elif hall_type == HallucinationType.MERGED_CITATION:
        donor_b = rng.choice(valid_entries)
        while donor_b.bibtex_key == source.bibtex_key and len(valid_entries) > 1:
            donor_b = rng.choice(valid_entries)
        donor_c = rng.choice(valid_entries) if rng.random() < 0.5 else None
        entry = generate_merged_citation(source, donor_b, donor_c, rng)

    elif hall_type == HallucinationType.ARXIV_VERSION_MISMATCH:
        entry = generate_arxiv_version_mismatch(source, _pick_venue(), rng)

    elif "reference_year" in spec.extra_args:
        # FUTURE_DATE — pass reference_year derived from build_date
        entry = gen_func(source, rng=rng, reference_year=reference_year)

    else:
        # All remaining types take only (entry, rng): fabricated_doi,
        # nonexistent_venue, placeholder_authors, hybrid_fabrication,
        # partial_author_list, near_miss_title, plausible_fabrication
        entry = gen_func(source, rng)

    entry.added_to_benchmark = build_date
    entry.bibtex_key = f"{key_prefix}_{type_val}_{idx}"
    return entry
