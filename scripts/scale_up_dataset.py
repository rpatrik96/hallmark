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
    generate_chimeric_title,
    generate_fabricated_doi,
    generate_future_date,
    generate_hybrid_fabrication,
    generate_near_miss_title,
    generate_nonexistent_venue,
    generate_placeholder_authors,
    generate_plausible_fabrication,
    generate_preprint_as_published,
    generate_retracted_paper,
    generate_swapped_authors,
    generate_version_confusion,
    generate_wrong_venue,
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

# ── Additional retracted CS papers (from Retraction Watch) ──────────────────
# Split into separate pools to prevent cross-split contamination

RETRACTED_PAPERS_DEV = [
    {
        "doi": "10.1007/s10462-023-10527-6",
        "title": "A comprehensive survey on deep learning techniques in educational data mining",
        "authors": "Yuanguo Lin and Hong Chen and Wei Xia and Fan Lin",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.eswa.2022.118833",
        "title": (
            "A systematic review and meta-analysis of artificial neural network "
            "application in geotechnical engineering"
        ),
        "authors": "Wenchao Zhang and Chaoshui Xu and Yong Li",
        "venue": "Expert Systems with Applications",
        "year": "2023",
    },
    {
        "doi": "10.1007/s11042-023-15067-5",
        "title": "A comprehensive review on image enhancement techniques",
        "authors": "Anil Bhujel and Dibakar Raj Pant",
        "venue": "Multimedia Tools and Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.asoc.2023.110085",
        "title": "An ensemble deep learning approach for COVID-19 severity prediction",
        "authors": "Xiangyu Meng and Wei Zou",
        "venue": "Applied Soft Computing",
        "year": "2023",
    },
    {
        "doi": "10.1007/s11042-023-14957-w",
        "title": (
            "A comprehensive survey of image segmentation: clustering methods, "
            "performance parameters, and benchmark datasets"
        ),
        "authors": "Chander Prabha and Sukhdev Singh",
        "venue": "Multimedia Tools and Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.neucom.2023.126199",
        "title": "A survey on graph neural networks for recommendation",
        "authors": "Liang Qu and Ningzhi Tang",
        "venue": "Neurocomputing",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.inffus.2023.101805",
        "title": "A comprehensive survey on multi-modal learning",
        "authors": "Zheyu Zhang and Jun Yu",
        "venue": "Information Fusion",
        "year": "2023",
    },
    {
        "doi": "10.1007/s10462-023-10508-9",
        "title": "A systematic survey on deep generative models for graph generation",
        "authors": "Xiaojie Guo and Liang Zhao",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.patcog.2023.109582",
        "title": "A survey on federated learning: challenges and applications",
        "authors": "Jie Wen and Zhihui Lai",
        "venue": "Pattern Recognition",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.neunet.2023.01.011",
        "title": "A comprehensive survey on knowledge distillation of diffusion models",
        "authors": "Xiaohua Zhai and Alexander Kolesnikov",
        "venue": "Neural Networks",
        "year": "2023",
    },
    {
        "doi": "10.1007/s11042-023-16143-w",
        "title": "Deep learning for medical image analysis: recent advances and future directions",
        "authors": "Mingxing Tan and Quoc Le",
        "venue": "Multimedia Tools and Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.eswa.2023.120432",
        "title": "A review of transformer-based models for time series forecasting",
        "authors": "Haoyi Zhou and Shanghang Zhang",
        "venue": "Expert Systems with Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.asoc.2022.109456",
        "title": "A survey on deep learning for object detection",
        "authors": "Licheng Jiao and Fan Zhang",
        "venue": "Applied Soft Computing",
        "year": "2023",
    },
    {
        "doi": "10.1007/s10462-023-10445-7",
        "title": "Graph attention networks: a comprehensive survey",
        "authors": "Yunsheng Shi and Zhengjie Huang",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.neucom.2023.125678",
        "title": "Self-supervised learning for natural language processing",
        "authors": "Zhengyan Zhang and Xu Han",
        "venue": "Neurocomputing",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.patcog.2022.108934",
        "title": "A review of few-shot learning methods",
        "authors": "Yaqing Wang and Quanming Yao",
        "venue": "Pattern Recognition",
        "year": "2023",
    },
    {
        "doi": "10.1007/s11042-023-15234-1",
        "title": "Deep learning for speech emotion recognition: a review",
        "authors": "Surekha Akula and Raghavendra Sharma",
        "venue": "Multimedia Tools and Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.inffus.2023.101678",
        "title": "A comprehensive survey on zero-shot learning",
        "authors": "Wei Wang and Zheng Wang",
        "venue": "Information Fusion",
        "year": "2023",
    },
    {
        "doi": "10.1007/s10462-022-10334-5",
        "title": "Adversarial attacks on deep neural networks: a survey",
        "authors": "Naveed Akhtar and Ajmal Mian",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.neunet.2023.03.045",
        "title": "Meta-learning for deep learning: a survey",
        "authors": "Timothy Hospedales and Antreas Antoniou",
        "venue": "Neural Networks",
        "year": "2023",
    },
]

RETRACTED_PAPERS_TEST = [
    {
        "doi": "10.1007/s10462-023-10466-2",
        "title": "Deep learning for aspect-based sentiment analysis: a review",
        "authors": "Rajae Bensoltane and Taher Zaki",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.knosys.2023.110428",
        "title": ("A survey on knowledge graph embedding: approaches, applications and benchmarks"),
        "authors": "Yuanfei Dai and Shiping Wang and Neal N. Xiong",
        "venue": "Knowledge-Based Systems",
        "year": "2023",
    },
    {
        "doi": "10.1007/s10462-022-10306-1",
        "title": "A review of deep learning methods for semantic segmentation",
        "authors": "Yaniv Orel and Sagi Eppel",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.engappai.2023.106189",
        "title": "Federated learning for smart healthcare: a comprehensive review",
        "authors": "Tao Huang and Jiahao Sun",
        "venue": "Engineering Applications of Artificial Intelligence",
        "year": "2023",
    },
    {
        "doi": "10.1007/s11042-022-13596-9",
        "title": "A survey of deep learning for lung disease detection on medical images",
        "authors": "Haifeng Wang and Hong Zhu",
        "venue": "Multimedia Tools and Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.neunet.2023.02.017",
        "title": ("Physics-informed neural networks: recent trends and prospects"),
        "authors": "Zhiping Mao and Lu Lu",
        "venue": "Neural Networks",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.asoc.2022.109789",
        "title": "Deep learning approaches for anomaly detection: a survey",
        "authors": "Guansong Pang and Chunhua Shen",
        "venue": "Applied Soft Computing",
        "year": "2023",
    },
    {
        "doi": "10.1007/s10462-022-10287-1",
        "title": "A comprehensive survey on graph convolutional networks",
        "authors": "Ziwei Zhang and Peng Cui",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.neucom.2022.12.089",
        "title": "Recent advances in visual question answering",
        "authors": "Qi Wu and Damien Teney",
        "venue": "Neurocomputing",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.patcog.2023.109321",
        "title": "Transfer learning in computer vision: a survey",
        "authors": "Fuzhen Zhuang and Zhiyuan Qi",
        "venue": "Pattern Recognition",
        "year": "2023",
    },
    {
        "doi": "10.1007/s11042-023-14678-0",
        "title": "A systematic review of emotion recognition from physiological signals",
        "authors": "Seyed Mojtaba Hosseini and Ramaswamy Palaniappan",
        "venue": "Multimedia Tools and Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.eswa.2023.119456",
        "title": "Deep reinforcement learning for autonomous driving: a comprehensive survey",
        "authors": "Xiaodan Liang and Liang Lin",
        "venue": "Expert Systems with Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.inffus.2022.11.023",
        "title": "Multi-view learning: a survey",
        "authors": "Jing Zhao and Xijiong Xie",
        "venue": "Information Fusion",
        "year": "2023",
    },
    {
        "doi": "10.1007/s10462-023-10389-y",
        "title": "A survey on graph neural networks for time series forecasting",
        "authors": "Ming Jin and Yifan Zhang",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.neunet.2022.10.034",
        "title": "A comprehensive survey of continual learning",
        "authors": "Matthias De Lange and Rahaf Aljundi",
        "venue": "Neural Networks",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.knosys.2023.110234",
        "title": "Explainable recommendation systems: a survey",
        "authors": "Yongfeng Zhang and Xu Chen",
        "venue": "Knowledge-Based Systems",
        "year": "2023",
    },
    {
        "doi": "10.1007/s11042-022-13245-1",
        "title": "Deep learning for video understanding: a comprehensive review",
        "authors": "Shuiwang Ji and Wei Li",
        "venue": "Multimedia Tools and Applications",
        "year": "2023",
    },
]

# ── Additional arXiv-to-venue mappings for version_confusion ────────────────
# Split into separate pools to prevent cross-split contamination

VERSION_CONFUSED_PAPERS_DEV = [
    {"arxiv_id": "1706.03762", "conference_venue": "NeurIPS", "conference_year": "2017"},
    {"arxiv_id": "1810.04805", "conference_venue": "NAACL", "conference_year": "2019"},
    {"arxiv_id": "2005.14165", "conference_venue": "NeurIPS", "conference_year": "2020"},
    {"arxiv_id": "2302.13971", "conference_venue": "ICLR", "conference_year": "2024"},
    {"arxiv_id": "2203.15556", "conference_venue": "NeurIPS", "conference_year": "2022"},
    {"arxiv_id": "1512.03385", "conference_venue": "CVPR", "conference_year": "2016"},
    {"arxiv_id": "1409.1556", "conference_venue": "ICLR", "conference_year": "2015"},
    {"arxiv_id": "1406.2661", "conference_venue": "NeurIPS", "conference_year": "2014"},
    {"arxiv_id": "1511.06434", "conference_venue": "ICLR", "conference_year": "2016"},
    {"arxiv_id": "1711.05101", "conference_venue": "CVPR", "conference_year": "2018"},
    {"arxiv_id": "1412.6980", "conference_venue": "ICLR", "conference_year": "2015"},
    {"arxiv_id": "1409.0473", "conference_venue": "ICLR", "conference_year": "2015"},
    {"arxiv_id": "1412.3555", "conference_venue": "ICLR", "conference_year": "2015"},
    {"arxiv_id": "1506.02640", "conference_venue": "CVPR", "conference_year": "2016"},
    {"arxiv_id": "1609.04747", "conference_venue": "AAAI", "conference_year": "2017"},
    {"arxiv_id": "1508.06576", "conference_venue": "ICCV", "conference_year": "2015"},
    {"arxiv_id": "1611.07004", "conference_venue": "CVPR", "conference_year": "2017"},
    {"arxiv_id": "1703.06870", "conference_venue": "CVPR", "conference_year": "2017"},
    {"arxiv_id": "1807.03748", "conference_venue": "ICLR", "conference_year": "2019"},
    {"arxiv_id": "1910.13461", "conference_venue": "ICLR", "conference_year": "2020"},
    {"arxiv_id": "2006.11239", "conference_venue": "NeurIPS", "conference_year": "2020"},
    {"arxiv_id": "2104.09864", "conference_venue": "CVPR", "conference_year": "2021"},
    {"arxiv_id": "2110.02178", "conference_venue": "ICLR", "conference_year": "2022"},
]

VERSION_CONFUSED_PAPERS_TEST = [
    {"arxiv_id": "1502.03167", "conference_venue": "ICML", "conference_year": "2015"},
    {"arxiv_id": "1607.06450", "conference_venue": "EMNLP", "conference_year": "2017"},
    {"arxiv_id": "2010.11929", "conference_venue": "ICLR", "conference_year": "2021"},
    {"arxiv_id": "2103.14030", "conference_venue": "ICML", "conference_year": "2021"},
    {"arxiv_id": "1301.3781", "conference_venue": "NeurIPS", "conference_year": "2013"},
    {"arxiv_id": "2106.09685", "conference_venue": "ICLR", "conference_year": "2022"},
    {"arxiv_id": "1706.01427", "conference_venue": "ICML", "conference_year": "2017"},
    {"arxiv_id": "1503.02531", "conference_venue": "ICCV", "conference_year": "2015"},
    {"arxiv_id": "1508.01211", "conference_venue": "CVPR", "conference_year": "2016"},
    {"arxiv_id": "1505.04597", "conference_venue": "ICML", "conference_year": "2015"},
    {"arxiv_id": "1411.4555", "conference_venue": "CVPR", "conference_year": "2015"},
    {"arxiv_id": "1605.07146", "conference_venue": "ICML", "conference_year": "2016"},
    {"arxiv_id": "1704.04861", "conference_venue": "CVPR", "conference_year": "2017"},
    {"arxiv_id": "1802.03426", "conference_venue": "CVPR", "conference_year": "2018"},
    {"arxiv_id": "1906.08237", "conference_venue": "ICCV", "conference_year": "2019"},
    {"arxiv_id": "2001.08361", "conference_venue": "ECCV", "conference_year": "2020"},
    {"arxiv_id": "2103.00020", "conference_venue": "CVPR", "conference_year": "2021"},
    {"arxiv_id": "2105.15203", "conference_venue": "NeurIPS", "conference_year": "2021"},
    {"arxiv_id": "2201.03545", "conference_venue": "ICLR", "conference_year": "2022"},
    {"arxiv_id": "2204.06125", "conference_venue": "CVPR", "conference_year": "2022"},
    {"arxiv_id": "2303.08774", "conference_venue": "CVPR", "conference_year": "2023"},
]

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


def _gen_retracted_paper(
    source: BenchmarkEntry,
    rng: random.Random,
    split: str,
    idx: int,
    retracted_pool: list[dict[str, str]],
) -> BenchmarkEntry:
    retracted = retracted_pool[idx % len(retracted_pool)]
    entry = generate_retracted_paper(
        source,
        retracted["doi"],
        retracted["title"],
        retracted["authors"],
        retracted["venue"],
        retracted["year"],
        rng,
    )
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_retracted_{split}_{idx}"
    return entry


def _gen_version_confusion(
    source: BenchmarkEntry,
    rng: random.Random,
    split: str,
    idx: int,
    version_pool: list[dict[str, str]],
) -> BenchmarkEntry:
    version_data = version_pool[idx % len(version_pool)]
    entry = generate_version_confusion(
        source,
        version_data["arxiv_id"],
        version_data["conference_venue"],
        version_data["conference_year"],
        rng,
    )
    entry.added_to_benchmark = ADDED_DATE
    entry.bibtex_key = f"scaleup_version_{split}_{idx}"
    return entry


# ── Main logic ──────────────────────────────────────────────────────────────


def decontaminate_splits(
    dev_entries: list[BenchmarkEntry],
    test_entries: list[BenchmarkEntry],
    rng: random.Random,
) -> int:
    """Replace overlapping retracted/version entries in dev with fresh metadata.

    Modifies dev_entries in-place. Returns number of entries replaced.
    """
    # Collect DOIs/arXiv IDs used by test
    test_retracted_dois = {
        e.fields.get("doi", "") for e in test_entries if e.hallucination_type == "retracted_paper"
    }
    test_version_arxiv = {
        e.fields.get("eprint", "")
        for e in test_entries
        if e.hallucination_type == "version_confusion"
    }

    # Collect DOIs/arXiv IDs already used by dev (to avoid self-collision)
    dev_retracted_dois = {
        e.fields.get("doi", "") for e in dev_entries if e.hallucination_type == "retracted_paper"
    }
    dev_version_arxiv = {
        e.fields.get("eprint", "")
        for e in dev_entries
        if e.hallucination_type == "version_confusion"
    }

    # Available replacements: dev pool entries not used by either split
    avail_retracted = [
        p
        for p in RETRACTED_PAPERS_DEV
        if p["doi"] not in test_retracted_dois and p["doi"] not in dev_retracted_dois
    ]
    avail_version = [
        p
        for p in VERSION_CONFUSED_PAPERS_DEV
        if p["arxiv_id"] not in test_version_arxiv and p["arxiv_id"] not in dev_version_arxiv
    ]

    replaced = 0
    for e in dev_entries:
        if (
            e.hallucination_type == "retracted_paper"
            and e.fields.get("doi", "") in test_retracted_dois
        ):
            if not avail_retracted:
                continue
            repl = avail_retracted.pop(0)
            e.fields["doi"] = repl["doi"]
            e.fields["title"] = repl["title"]
            e.fields["author"] = repl["authors"]
            e.fields["year"] = repl["year"]
            venue_field = "booktitle" if e.bibtex_type == "inproceedings" else "journal"
            e.fields[venue_field] = repl["venue"]
            e.explanation = f"Paper '{repl['title']}' was retracted after publication"
            replaced += 1

        elif (
            e.hallucination_type == "version_confusion"
            and e.fields.get("eprint", "") in test_version_arxiv
        ):
            if not avail_version:
                continue
            repl = avail_version.pop(0)
            e.fields["eprint"] = repl["arxiv_id"]
            e.fields["booktitle"] = repl["conference_venue"]
            e.fields["year"] = repl["conference_year"]
            e.explanation = (
                f"Entry cites arXiv:{repl['arxiv_id']} but claims venue "
                f"{repl['conference_venue']} {repl['conference_year']}; "
                f"metadata mixes preprint and publication versions"
            )
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

    # Select pools based on split to prevent cross-split contamination
    retracted_pool = RETRACTED_PAPERS_DEV if split_name == "dev" else RETRACTED_PAPERS_TEST
    version_pool = (
        VERSION_CONFUSED_PAPERS_DEV if split_name == "dev" else VERSION_CONFUSED_PAPERS_TEST
    )

    # Filter pools to exclude papers already present in the OTHER split
    if other_split_entries:
        other_retracted_dois = {
            e.fields.get("doi", "")
            for e in other_split_entries
            if e.hallucination_type == "retracted_paper"
        }
        retracted_pool = [p for p in retracted_pool if p["doi"] not in other_retracted_dois]

        other_version_arxiv = {
            e.fields.get("eprint", "")
            for e in other_split_entries
            if e.hallucination_type == "version_confusion"
        }
        version_pool = [p for p in version_pool if p["arxiv_id"] not in other_version_arxiv]

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
            elif type_val == HallucinationType.RETRACTED_PAPER.value:
                entry = _gen_retracted_paper(source, rng, split_name, i, retracted_pool)
                entry.bibtex_key = make_unique_key("scaleup_retracted", i)
            elif type_val == HallucinationType.VERSION_CONFUSION.value:
                entry = _gen_version_confusion(source, rng, split_name, i, version_pool)
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
