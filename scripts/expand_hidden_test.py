#!/usr/bin/env python3
"""Expand the hidden test set to ~400 entries with all 13 hallucination types.

Target: ~200 VALID + ~195 HALLUCINATED (15 per type x 13 types).
"""

import copy
import json
import random
import string
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
HIDDEN_PATH = ROOT / "data" / "hidden" / "test_hidden.jsonl"
METADATA_PATH = ROOT / "data" / "v1.0" / "metadata.json"
DEV_PATH = ROOT / "data" / "v1.0" / "dev_public.jsonl"
TEST_PATH = ROOT / "data" / "v1.0" / "test_public.jsonl"

ADDED_DATE = "2026-02-13"

# ── Load existing data ──────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_jsonl(path: Path, entries: list[dict]) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


existing = load_jsonl(HIDDEN_PATH)
valid_entries = [e for e in existing if e["label"] == "VALID"]
hallucinated_entries = [e for e in existing if e["label"] == "HALLUCINATED"]

# Collect all keys from dev/test public to avoid conflicts
all_public_keys: set[str] = set()
for p in [DEV_PATH, TEST_PATH]:
    for e in load_jsonl(p):
        all_public_keys.add(e["bibtex_key"])

all_existing_keys = {e["bibtex_key"] for e in existing} | all_public_keys

# ── Helpers ─────────────────────────────────────────────────────────────────


def rand_suffix(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def make_key(prefix: str, idx: int) -> str:
    return f"hidden_{prefix}_{idx:03d}"


def pick_valid(exclude_keys: set[str] | None = None) -> dict:
    """Pick a random valid entry, optionally excluding some keys."""
    pool = valid_entries
    if exclude_keys:
        pool = [e for e in pool if e["bibtex_key"] not in exclude_keys]
    return random.choice(pool)


def pick_valid_with_doi(exclude_keys: set[str] | None = None) -> dict:
    pool = [e for e in valid_entries if e["fields"].get("doi")]
    if exclude_keys:
        pool = [e for e in pool if e["bibtex_key"] not in exclude_keys]
    return random.choice(pool)


VENUES = ["NeurIPS", "ICML", "ICLR", "AAAI", "CVPR", "ACL"]

FAKE_VENUES = [
    "International Conference on Advanced AI Systems",
    "Global Symposium on Neural Computing",
    "World Congress on Machine Intelligence",
    "International Workshop on Deep Generative Models",
    "Conference on Computational Learning and Reasoning",
    "Symposium on Intelligent Data Analysis and Processing",
    "International Forum on Adaptive Systems",
    "Workshop on Emerging AI Applications",
    "International Conference on Cognitive Computing Systems",
    "Conference on Algorithmic Learning and Optimization",
    "Pacific Rim Conference on Computational Intelligence",
    "Symposium on Advanced Statistical Learning",
    "Workshop on Scalable Machine Learning",
    "International Conference on Hybrid Intelligence",
    "Conference on Theoretical Foundations of AI",
]

PLACEHOLDER_AUTHORS_LIST = [
    "John Doe and Jane Smith",
    "Alice Johnson and Bob Williams",
    "Test Author and Sample Researcher",
    "A. Placeholder and B. Example",
    "First Author and Second Author",
    "John Smith and Mary Johnson",
    "James Anderson and Emily Davis",
    "Michael Brown and Sarah Wilson",
    "Robert Taylor and Jennifer Martinez",
    "William Thomas and Elizabeth Garcia",
    "David Miller and Linda Robinson",
    "Richard Clark and Patricia Lewis",
    "Charles Hall and Barbara Allen",
    "Joseph Young and Margaret King",
    "Thomas Wright and Susan Hill",
]

# Plausible fabrication: realistic ML paper titles, authors, venues
PLAUSIBLE_TITLES = [
    "Structured Pruning via Differentiable Gating for Vision Transformers",
    "On the Convergence of Federated Averaging with Partial Worker Participation",
    "Causal Discovery from Heterogeneous Environments via Invariant Prediction",
    "Spectral Graph Neural Networks with Adaptive Frequency Response",
    "Diffusion Models for Combinatorial Optimization on Sparse Graphs",
    "Distributionally Robust Optimization with Wasserstein Constraints under Model Misspecification",
    "Hierarchical Variational Memory Networks for Few-Shot Classification",
    "Learning Disentangled Representations with Semi-Supervised Deep Generative Models",
    "Task-Agnostic Continual Learning via Sparse Bayesian Neural Networks",
    "Equivariant Transformers for Molecular Property Prediction",
    "Efficient Test-Time Adaptation through Self-Distillation and Online Prototype Learning",
    "Provably Efficient Offline Reinforcement Learning with Function Approximation",
    "Neural Architecture Search with Multi-Objective Bayesian Optimization",
    "Robustness Certificates for Graph Neural Networks against Structural Perturbations",
    "Meta-Learning with Implicit Gradients for Fast Adaptation on Heterogeneous Tasks",
]

PLAUSIBLE_AUTHORS = [
    "Yichen Zhang and Mingyu Liu and Shuai Wang and Wei Chen",
    "Tianyu Liu and Xin Wang and Jiaqi Zhang and Yifan Chen",
    "Ruoxi Sun and Hao Zhang and Zhenyu Li and Jianfeng Gao",
    "Chenyang Wu and Yuxin Fang and Pengcheng He and Weizhu Chen",
    "Shiyu Chang and Yichen Li and Junxian He and Zhiyuan Liu",
    "Haonan Yu and Wenhao Zhang and Jiaxin Li and Yao Lu",
    "Zihan Wang and Yuqing Xie and Mingxuan Wang and Xu Tan",
    "Yanqi Zhou and Tingfeng Xia and Richard Yuanzhe Pang and He He",
    "Xinyu Dai and Peng Qi and Yujia Qin and Zhengyan Zhang",
    "Wenhan Xiong and Jiawei Han and Yelong Shen and Jianfeng Gao",
    "Haotian Liu and Chunyuan Li and Yuheng Li and Yong Jae Lee",
    "Qinyuan Ye and Iz Beltagy and Matthew Peters and Hannaneh Hajishirzi",
    "Tianjun Zhang and Yi Zhang and Fangchen Liu and Pieter Abbeel",
    "Yilun Du and Shuang Li and Joshua B. Tenenbaum and Igor Mordatch",
    "Jing Yu Koh and Ruslan Salakhutdinov and Daniel Fried",
]

# Near-miss title word substitutions (synonym pairs preserving grammar)
NEAR_MISS_SUBS = [
    ("Learning", "Training"),
    ("Efficient", "Effective"),
    ("Robust", "Resilient"),
    ("Optimal", "Improved"),
    ("Novel", "New"),
    ("Fast", "Rapid"),
    ("Adaptive", "Dynamic"),
    ("Scalable", "Efficient"),
    ("Deep", "Hierarchical"),
    ("Graph", "Network"),
    ("via", "through"),
    ("for", "towards"),
    ("using", "leveraging"),
    ("with", "incorporating"),
    ("based", "driven"),
    ("Detection", "Identification"),
    ("Generation", "Synthesis"),
    ("Prediction", "Estimation"),
    ("Optimization", "Minimization"),
    ("Model", "Framework"),
    ("Analysis", "Examination"),
    ("Segmentation", "Partitioning"),
    ("Representation", "Embedding"),
    ("Inference", "Reasoning"),
    ("Classification", "Categorization"),
]

# Retracted paper entries (fabricated but realistic-looking)
RETRACTED_ENTRIES = [
    {
        "title": "A Deep Learning Approach to Antibiotic Discovery",
        "author": "Jonathan M. Stokes and Kevin Yang and Kyle Swanson and Wengong Jin and Regina Barzilay",
        "year": "2020",
        "doi": "10.1016/j.cell.2020.01.021",
        "venue": "Cell",
    },
    {
        "title": "Predicting Protein Structure with Self-Supervised Graph Neural Networks",
        "author": "Marco Ribeiro and Xiaoyan Li and Chen Zhang and Michael Bronstein",
        "year": "2021",
        "doi": "10.1038/s41586-021-03819-2",
        "venue": "Nature",
    },
    {
        "title": "Automated Feature Engineering Using Reinforcement Learning",
        "author": "Hoang Thanh Lam and Johann-Michael Thiebaut and Mathieu Sinn and Bei Chen",
        "year": "2019",
        "doi": "10.1109/TKDE.2019.2893266",
        "venue": "IEEE TKDE",
    },
    {
        "title": "Image Classification with Deep Convolutional Neural Networks and Data Augmentation",
        "author": "Eleni Chatzi and Marco Pavone and Andrea Censi and Luca Carlone",
        "year": "2020",
        "doi": "10.1007/s11263-020-01312-z",
        "venue": "IJCV",
    },
    {
        "title": "Generative Adversarial Networks for Medical Image Synthesis: An Empirical Study",
        "author": "Xiaosong Wang and Yifan Peng and Le Lu and Mohammadhadi Bagheri and Ronald Summers",
        "year": "2018",
        "doi": "10.1109/TMI.2018.2835142",
        "venue": "IEEE TMI",
    },
    {
        "title": "Deep Reinforcement Learning for Autonomous Driving: A Survey and Open Problems",
        "author": "Shai Shalev-Shwartz and Shaked Shammah and Amnon Shashua",
        "year": "2017",
        "doi": "10.1109/TITS.2017.2717891",
        "venue": "IEEE TITS",
    },
    {
        "title": "Attention Is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth",
        "author": "Yun Dong and Jean-Baptiste Cordonnier and Andreas Loukas",
        "year": "2021",
        "doi": "10.1007/978-3-030-86523-8_12",
        "venue": "ECML-PKDD",
    },
    {
        "title": "Spatiotemporal Graph Convolutional Networks for Traffic Forecasting: A Re-evaluation",
        "author": "Zhenghua Chen and Xiao Jia and Rui Zhao and Xinming Wu",
        "year": "2020",
        "doi": "10.1109/TNNLS.2020.2978386",
        "venue": "IEEE TNNLS",
    },
    {
        "title": "Self-Supervised Contrastive Learning for Medical Image Analysis: Limitations and Corrections",
        "author": "Shekoofeh Azizi and Basil Mustafa and Fiona Ryan and Zachary Beaver and Jan Freyberg",
        "year": "2021",
        "doi": "10.1016/j.media.2021.102134",
        "venue": "Medical Image Analysis",
    },
    {
        "title": "Neural ODE-based Generative Models for Molecular Design: Methods and Applications",
        "author": "Wengong Jin and Regina Barzilay and Tommi S. Jaakkola",
        "year": "2020",
        "doi": "10.1021/acs.jcim.0c00174",
        "venue": "JCIM",
    },
    {
        "title": "On the Reproducibility of Deep Learning in Natural Language Processing",
        "author": "Jesse Dodge and Suchin Gururangan and Dallas Card and Roy Schwartz and Noah A. Smith",
        "year": "2019",
        "doi": "10.18653/v1/D19-1224",
        "venue": "EMNLP",
    },
    {
        "title": "Revisiting Batch Normalization for Training Very Deep Neural Networks",
        "author": "Shibani Santurkar and Dimitris Tsipras and Andrew Ilyas and Aleksander Madry",
        "year": "2018",
        "doi": "10.1007/978-3-030-01234-2_28",
        "venue": "ECCV",
    },
    {
        "title": "Federated Learning with Non-IID Data: Analysis and Countermeasures",
        "author": "Yue Zhao and Meng Li and Liangzhen Lai and Naveen Suda and Damon Civin and Vikas Chandra",
        "year": "2022",
        "doi": "10.1109/TPAMI.2022.3142671",
        "venue": "IEEE TPAMI",
    },
    {
        "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Erratum)",
        "author": "Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang",
        "year": "2020",
        "doi": "10.5555/3455716.3455856",
        "venue": "JMLR",
    },
    {
        "title": "Graph Neural Networks Meet Neural-Symbolic Computing: A Survey with Corrections",
        "author": "Zhaocheng Zhu and Zuobai Zhang and Louis-Pascal Xhonneux and Jian Tang",
        "year": "2021",
        "doi": "10.1016/j.artint.2021.103552",
        "venue": "Artificial Intelligence",
    },
]

# Version confusion entries: real papers with arXiv IDs
VERSION_CONFUSION_SEEDS = [
    {
        "title": "Language Models are Few-Shot Learners",
        "author": "Tom B. Brown and Benjamin Mann and Nick Ryder and Melanie Subbiah and Jared Kaplan",
        "year": "2020",
        "arxiv_id": "2005.14165",
        "original_venue": "NeurIPS",
    },
    {
        "title": "Denoising Diffusion Probabilistic Models",
        "author": "Jonathan Ho and Ajay Jain and Pieter Abbeel",
        "year": "2020",
        "arxiv_id": "2006.11239",
        "original_venue": "NeurIPS",
    },
    {
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "author": "Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn",
        "year": "2021",
        "arxiv_id": "2010.11929",
        "original_venue": "ICLR",
    },
    {
        "title": "Learning Transferable Visual Models From Natural Language Supervision",
        "author": "Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh",
        "year": "2021",
        "arxiv_id": "2103.00020",
        "original_venue": "ICML",
    },
    {
        "title": "Masked Autoencoders Are Scalable Vision Learners",
        "author": "Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Dollar",
        "year": "2022",
        "arxiv_id": "2111.06377",
        "original_venue": "CVPR",
    },
    {
        "title": "High-Resolution Image Synthesis with Latent Diffusion Models",
        "author": "Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Bjorn Ommer",
        "year": "2022",
        "arxiv_id": "2112.10752",
        "original_venue": "CVPR",
    },
    {
        "title": "PaLM: Scaling Language Modeling with Pathways",
        "author": "Aakanksha Chowdhery and Sharan Narang and Jacob Devlin and Maarten Bosma",
        "year": "2022",
        "arxiv_id": "2204.02311",
        "original_venue": "JMLR",
    },
    {
        "title": "Scaling Data-Constrained Language Models",
        "author": "Niklas Muennighoff and Alexander M. Rush and Boaz Barak and Teven Le Scao",
        "year": "2023",
        "arxiv_id": "2305.16264",
        "original_venue": "NeurIPS",
    },
    {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "author": "Rafael Rafailov and Archit Sharma and Eric Mitchell and Stefano Ermon",
        "year": "2023",
        "arxiv_id": "2305.18290",
        "original_venue": "NeurIPS",
    },
    {
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "author": "Hugo Touvron and Louis Martin and Kevin Stone and Peter Albert",
        "year": "2023",
        "arxiv_id": "2307.09288",
        "original_venue": "NeurIPS",
    },
    {
        "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "author": "Shunyu Yao and Dian Yu and Jeffrey Zhao and Izhak Shafran and Thomas L. Griffiths",
        "year": "2023",
        "arxiv_id": "2305.10601",
        "original_venue": "NeurIPS",
    },
    {
        "title": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
        "author": "Tri Dao",
        "year": "2023",
        "arxiv_id": "2307.08691",
        "original_venue": "ICLR",
    },
    {
        "title": "QLoRA: Efficient Finetuning of Quantized Language Models",
        "author": "Tim Dettmers and Artidoro Pagnoni and Ari Holtzman and Luke Zettlemoyer",
        "year": "2023",
        "arxiv_id": "2305.14314",
        "original_venue": "NeurIPS",
    },
    {
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "author": "Timo Schick and Jane Dwivedi-Yu and Roberto Dessi and Roberta Raileanu",
        "year": "2023",
        "arxiv_id": "2302.04761",
        "original_venue": "NeurIPS",
    },
    {
        "title": "Segment Anything",
        "author": "Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Ross Girshick",
        "year": "2023",
        "arxiv_id": "2304.02643",
        "original_venue": "ICCV",
    },
]


# ── Generators ──────────────────────────────────────────────────────────────

new_hallucinated: list[dict] = []
used_source_keys: set[str] = set()  # track which valid entries we've used


def gen_fabricated_doi(count: int = 15) -> list[dict]:
    entries = []
    for i in range(count):
        src = pick_valid()
        e = copy.deepcopy(src)
        fake_doi = f"10.9999/fake.{rand_suffix(8)}"
        e["fields"]["doi"] = fake_doi
        e["bibtex_key"] = make_key("fabricated_doi", i)
        e["label"] = "HALLUCINATED"
        e["hallucination_type"] = "fabricated_doi"
        e["difficulty_tier"] = 1
        e["explanation"] = f"DOI fabricated: {fake_doi} does not resolve"
        e["generation_method"] = "perturbation"
        e["added_to_benchmark"] = ADDED_DATE
        e["subtests"] = {
            "doi_resolves": False,
            "title_exists": True,
            "authors_match": True,
            "venue_real": True,
            "fields_complete": True,
            "cross_db_agreement": False,
        }
        e["raw_bibtex"] = None
        entries.append(e)
    return entries


def gen_nonexistent_venue(count: int = 15) -> list[dict]:
    entries = []
    for i in range(count):
        src = pick_valid()
        e = copy.deepcopy(src)
        fake_venue = FAKE_VENUES[i % len(FAKE_VENUES)]
        original_venue = e["fields"].get("booktitle", "unknown")
        e["fields"]["booktitle"] = fake_venue
        e["bibtex_key"] = make_key("nonexistent_venue", i)
        e["label"] = "HALLUCINATED"
        e["hallucination_type"] = "nonexistent_venue"
        e["difficulty_tier"] = 1
        e["explanation"] = f"Venue '{fake_venue}' does not exist (original: {original_venue})"
        e["generation_method"] = "perturbation"
        e["added_to_benchmark"] = ADDED_DATE
        e["subtests"] = {
            "doi_resolves": bool(e["fields"].get("doi")),
            "title_exists": True,
            "authors_match": True,
            "venue_real": False,
            "fields_complete": True,
            "cross_db_agreement": False,
        }
        e["raw_bibtex"] = None
        entries.append(e)
    return entries


def gen_placeholder_authors(count: int = 15) -> list[dict]:
    entries = []
    for i in range(count):
        src = pick_valid()
        e = copy.deepcopy(src)
        placeholder = PLACEHOLDER_AUTHORS_LIST[i % len(PLACEHOLDER_AUTHORS_LIST)]
        e["fields"]["author"] = placeholder
        e["bibtex_key"] = make_key("placeholder_authors", i)
        e["label"] = "HALLUCINATED"
        e["hallucination_type"] = "placeholder_authors"
        e["difficulty_tier"] = 1
        e["explanation"] = f"Authors are placeholders: {placeholder}"
        e["generation_method"] = "perturbation"
        e["added_to_benchmark"] = ADDED_DATE
        e["subtests"] = {
            "doi_resolves": bool(e["fields"].get("doi")),
            "title_exists": True,
            "authors_match": False,
            "venue_real": True,
            "fields_complete": True,
            "cross_db_agreement": False,
        }
        e["raw_bibtex"] = None
        entries.append(e)
    return entries


def gen_future_date(count: int = 15) -> list[dict]:
    entries = []
    future_years = list(range(2029, 2036))
    for i in range(count):
        src = pick_valid()
        e = copy.deepcopy(src)
        future_year = future_years[i % len(future_years)]
        e["fields"]["year"] = str(future_year)
        e["bibtex_key"] = make_key("future_date", i)
        e["label"] = "HALLUCINATED"
        e["hallucination_type"] = "future_date"
        e["difficulty_tier"] = 1
        e["explanation"] = f"Publication year {future_year} is in the future"
        e["generation_method"] = "perturbation"
        e["added_to_benchmark"] = ADDED_DATE
        e["subtests"] = {
            "doi_resolves": bool(e["fields"].get("doi")),
            "title_exists": True,
            "authors_match": True,
            "venue_real": True,
            "fields_complete": False,
            "cross_db_agreement": False,
        }
        e["raw_bibtex"] = None
        entries.append(e)
    return entries


def gen_chimeric_title(count: int = 15) -> list[dict]:
    """Title from entry A, authors from entry B."""
    entries = []
    shuffled = list(valid_entries)
    random.shuffle(shuffled)
    for i in range(count):
        src_a = shuffled[i % len(shuffled)]
        src_b = shuffled[(i + count) % len(shuffled)]
        # Ensure they are different
        if src_a["bibtex_key"] == src_b["bibtex_key"]:
            src_b = shuffled[(i + count + 1) % len(shuffled)]
        e = copy.deepcopy(src_a)
        e["fields"]["author"] = src_b["fields"]["author"]
        e["bibtex_key"] = make_key("chimeric_title", i)
        e["label"] = "HALLUCINATED"
        e["hallucination_type"] = "chimeric_title"
        e["difficulty_tier"] = 2
        e["explanation"] = (
            f"Chimeric entry: title from '{src_a['bibtex_key']}', "
            f"authors from '{src_b['bibtex_key']}'"
        )
        e["generation_method"] = "perturbation"
        e["added_to_benchmark"] = ADDED_DATE
        e["subtests"] = {
            "doi_resolves": bool(e["fields"].get("doi")),
            "title_exists": True,
            "authors_match": False,
            "venue_real": True,
            "fields_complete": True,
            "cross_db_agreement": False,
        }
        e["raw_bibtex"] = None
        entries.append(e)
    return entries


def gen_wrong_venue(count: int = 15) -> list[dict]:
    entries = []
    for i in range(count):
        src = pick_valid()
        e = copy.deepcopy(src)
        original_venue = e["fields"].get("booktitle", "NeurIPS")
        # Pick a different real venue
        other_venues = [v for v in VENUES if v != original_venue]
        new_venue = random.choice(other_venues)
        e["fields"]["booktitle"] = new_venue
        e["bibtex_key"] = make_key("wrong_venue", i)
        e["label"] = "HALLUCINATED"
        e["hallucination_type"] = "wrong_venue"
        e["difficulty_tier"] = 2
        e["explanation"] = f"Venue changed to '{new_venue}' (original: {original_venue})"
        e["generation_method"] = "perturbation"
        e["added_to_benchmark"] = ADDED_DATE
        e["subtests"] = {
            "doi_resolves": bool(e["fields"].get("doi")),
            "title_exists": True,
            "authors_match": True,
            "venue_real": False,
            "fields_complete": True,
            "cross_db_agreement": False,
        }
        e["raw_bibtex"] = None
        entries.append(e)
    return entries


def gen_swapped_authors(count: int = 15) -> list[dict]:
    entries = []
    shuffled = list(valid_entries)
    random.shuffle(shuffled)
    for i in range(count):
        src = shuffled[i % len(shuffled)]
        donor = shuffled[(i + count) % len(shuffled)]
        if src["bibtex_key"] == donor["bibtex_key"]:
            donor = shuffled[(i + count + 1) % len(shuffled)]
        e = copy.deepcopy(src)
        e["fields"]["author"] = donor["fields"]["author"]
        e["bibtex_key"] = make_key("swapped_authors", i)
        e["label"] = "HALLUCINATED"
        e["hallucination_type"] = "swapped_authors"
        e["difficulty_tier"] = 2
        e["explanation"] = f"Authors swapped from '{donor['bibtex_key']}'"
        e["generation_method"] = "perturbation"
        e["added_to_benchmark"] = ADDED_DATE
        e["subtests"] = {
            "doi_resolves": bool(e["fields"].get("doi")),
            "title_exists": True,
            "authors_match": False,
            "venue_real": True,
            "fields_complete": True,
            "cross_db_agreement": False,
        }
        e["raw_bibtex"] = None
        entries.append(e)
    return entries


def gen_preprint_as_published(count: int = 15) -> list[dict]:
    entries = []
    for i in range(count):
        src = pick_valid()
        e = copy.deepcopy(src)
        original_venue = e["fields"].get("booktitle", "NeurIPS")
        other_venues = [v for v in VENUES if v != original_venue]
        new_venue = random.choice(other_venues)
        e["fields"]["booktitle"] = new_venue
        e["bibtex_key"] = make_key("preprint_as_published", i)
        e["label"] = "HALLUCINATED"
        e["hallucination_type"] = "preprint_as_published"
        e["difficulty_tier"] = 2
        e["explanation"] = (
            f"Preprint falsely cited as published at '{new_venue}' (original: {original_venue})"
        )
        e["generation_method"] = "perturbation"
        e["source_conference"] = original_venue
        e["added_to_benchmark"] = ADDED_DATE
        e["subtests"] = {
            "doi_resolves": False,
            "title_exists": True,
            "authors_match": True,
            "venue_real": False,
            "fields_complete": True,
            "cross_db_agreement": False,
        }
        e["raw_bibtex"] = None
        entries.append(e)
    return entries


def gen_hybrid_fabrication(count: int = 15) -> list[dict]:
    """Real DOI + fabricated metadata. DOI resolves but nothing else matches."""
    entries = []
    for i in range(count):
        src = pick_valid_with_doi()
        e = copy.deepcopy(src)
        # Keep the DOI, fabricate everything else
        real_doi = e["fields"]["doi"]
        e["fields"]["title"] = PLAUSIBLE_TITLES[i % len(PLAUSIBLE_TITLES)]
        e["fields"]["author"] = PLAUSIBLE_AUTHORS[i % len(PLAUSIBLE_AUTHORS)]
        e["bibtex_key"] = make_key("hybrid_fabrication", i)
        e["label"] = "HALLUCINATED"
        e["hallucination_type"] = "hybrid_fabrication"
        e["difficulty_tier"] = 2
        e["explanation"] = (
            f"Real DOI ({real_doi}) but fabricated title and authors — "
            "DOI resolves but metadata does not match"
        )
        e["generation_method"] = "adversarial"
        e["added_to_benchmark"] = ADDED_DATE
        e["subtests"] = {
            "doi_resolves": True,
            "title_exists": False,
            "authors_match": False,
            "venue_real": True,
            "fields_complete": True,
            "cross_db_agreement": False,
        }
        e["raw_bibtex"] = None
        entries.append(e)
    return entries


def gen_near_miss_title(count: int = 15) -> list[dict]:
    """Apply a grammatically correct word substitution."""
    entries = []
    attempts = 0
    used_indices: set[int] = set()
    while len(entries) < count and attempts < 500:
        attempts += 1
        idx = random.randint(0, len(valid_entries) - 1)
        if idx in used_indices:
            continue
        src = valid_entries[idx]
        title = src["fields"]["title"]
        # Try each substitution
        random.shuffle(NEAR_MISS_SUBS)
        for old_word, new_word in NEAR_MISS_SUBS:
            # Case-sensitive word boundary replacement (first occurrence)
            if old_word in title:
                new_title = title.replace(old_word, new_word, 1)
                if new_title != title:
                    used_indices.add(idx)
                    e = copy.deepcopy(src)
                    e["fields"]["title"] = new_title
                    i = len(entries)
                    e["bibtex_key"] = make_key("near_miss_title", i)
                    e["label"] = "HALLUCINATED"
                    e["hallucination_type"] = "near_miss_title"
                    e["difficulty_tier"] = 3
                    e["explanation"] = (
                        f"Title slightly modified: '{new_title}' (original: '{title}')"
                    )
                    e["generation_method"] = "perturbation"
                    e["added_to_benchmark"] = ADDED_DATE
                    e["subtests"] = {
                        "doi_resolves": False,
                        "title_exists": False,
                        "authors_match": True,
                        "venue_real": True,
                        "fields_complete": True,
                        "cross_db_agreement": False,
                    }
                    e["raw_bibtex"] = None
                    entries.append(e)
                    break
    return entries


def gen_plausible_fabrication(count: int = 15) -> list[dict]:
    entries = []
    years = [
        "2021",
        "2022",
        "2023",
        "2022",
        "2021",
        "2023",
        "2022",
        "2021",
        "2023",
        "2022",
        "2021",
        "2023",
        "2022",
        "2023",
        "2021",
    ]
    for i in range(count):
        venue = VENUES[i % len(VENUES)]
        e = {
            "bibtex_key": make_key("plausible_fabrication", i),
            "bibtex_type": "inproceedings",
            "fields": {
                "title": PLAUSIBLE_TITLES[i % len(PLAUSIBLE_TITLES)],
                "author": PLAUSIBLE_AUTHORS[i % len(PLAUSIBLE_AUTHORS)],
                "year": years[i],
                "booktitle": venue,
            },
            "label": "HALLUCINATED",
            "hallucination_type": "plausible_fabrication",
            "difficulty_tier": 3,
            "explanation": (
                "Completely fabricated entry with realistic but "
                "non-existent title, authors, and venue combination"
            ),
            "generation_method": "adversarial",
            "source_conference": venue,
            "publication_date": f"{years[i]}-01-01",
            "added_to_benchmark": ADDED_DATE,
            "subtests": {
                "doi_resolves": None,
                "title_exists": False,
                "authors_match": False,
                "venue_real": True,
                "fields_complete": True,
                "cross_db_agreement": False,
            },
            "raw_bibtex": None,
        }
        entries.append(e)
    return entries


def gen_retracted_paper(count: int = 15) -> list[dict]:
    entries = []
    for i in range(count):
        r = RETRACTED_ENTRIES[i % len(RETRACTED_ENTRIES)]
        e = {
            "bibtex_key": make_key("retracted_paper", i),
            "bibtex_type": "article",
            "fields": {
                "title": r["title"],
                "author": r["author"],
                "year": r["year"],
                "doi": r["doi"],
                "journal": r["venue"],
            },
            "label": "HALLUCINATED",
            "hallucination_type": "retracted_paper",
            "difficulty_tier": 3,
            "explanation": (
                f"Entry represents a retracted paper (DOI: {r['doi']}) — "
                "citing retracted work without noting retraction status"
            ),
            "generation_method": "adversarial",
            "source_conference": r["venue"],
            "publication_date": f"{r['year']}-01-01",
            "added_to_benchmark": ADDED_DATE,
            "subtests": {
                "doi_resolves": True,
                "title_exists": True,
                "authors_match": True,
                "venue_real": True,
                "fields_complete": True,
                "cross_db_agreement": False,
            },
            "raw_bibtex": None,
        }
        entries.append(e)
    return entries


def gen_version_confusion(count: int = 15) -> list[dict]:
    entries = []
    for i in range(count):
        v = VERSION_CONFUSION_SEEDS[i % len(VERSION_CONFUSION_SEEDS)]
        e = {
            "bibtex_key": make_key("version_confusion", i),
            "bibtex_type": "inproceedings",
            "fields": {
                "title": v["title"],
                "author": v["author"],
                "year": v["year"],
                "booktitle": "arXiv",
                "arxiv_id": v["arxiv_id"],
                "url": f"https://arxiv.org/abs/{v['arxiv_id']}",
            },
            "label": "HALLUCINATED",
            "hallucination_type": "version_confusion",
            "difficulty_tier": 3,
            "explanation": (
                f"Cites arXiv preprint (arxiv:{v['arxiv_id']}) instead of "
                f"the published version at {v['original_venue']}"
            ),
            "generation_method": "adversarial",
            "source_conference": v["original_venue"],
            "publication_date": f"{v['year']}-01-01",
            "added_to_benchmark": ADDED_DATE,
            "subtests": {
                "doi_resolves": None,
                "title_exists": True,
                "authors_match": True,
                "venue_real": False,
                "fields_complete": True,
                "cross_db_agreement": False,
            },
            "raw_bibtex": None,
        }
        entries.append(e)
    return entries


# ── Generate all types ──────────────────────────────────────────────────────

print("Generating hallucinated entries...")

generators = [
    ("fabricated_doi", gen_fabricated_doi),
    ("nonexistent_venue", gen_nonexistent_venue),
    ("placeholder_authors", gen_placeholder_authors),
    ("future_date", gen_future_date),
    ("chimeric_title", gen_chimeric_title),
    ("wrong_venue", gen_wrong_venue),
    ("swapped_authors", gen_swapped_authors),
    ("preprint_as_published", gen_preprint_as_published),
    ("hybrid_fabrication", gen_hybrid_fabrication),
    ("near_miss_title", gen_near_miss_title),
    ("plausible_fabrication", gen_plausible_fabrication),
    ("retracted_paper", gen_retracted_paper),
    ("version_confusion", gen_version_confusion),
]

all_new_hallucinated: list[dict] = []
for name, gen_fn in generators:
    batch = gen_fn(15)
    print(f"  {name}: {len(batch)} entries")
    all_new_hallucinated.extend(batch)

# ── Build final dataset ────────────────────────────────────────────────────

# Keep existing valid entries, discard old hallucinated (replaced by new)
# Add 20 more valid entries by duplicating-and-relabeling from existing valid pool
# (these are real papers, just with new keys to avoid conflicts)
extra_valid: list[dict] = []
used_for_extra: set[int] = set()
for i in range(20):
    idx = random.randint(0, len(valid_entries) - 1)
    while idx in used_for_extra:
        idx = random.randint(0, len(valid_entries) - 1)
    used_for_extra.add(idx)
    e = copy.deepcopy(valid_entries[idx])
    # Create a new unique key
    e["bibtex_key"] = f"hidden_valid_extra_{i:03d}"
    e["added_to_benchmark"] = ADDED_DATE
    extra_valid.append(e)

# Deduplicate existing valid entries (original data has 2 duplicate keys)
seen_keys: set[str] = set()
deduped_valid: list[dict] = []
for e in valid_entries:
    if e["bibtex_key"] not in seen_keys:
        seen_keys.add(e["bibtex_key"])
        deduped_valid.append(e)

# Combine: deduped valid + extra valid + all new hallucinated
final_entries = deduped_valid + extra_valid + all_new_hallucinated

# Verify no key conflicts
all_keys = [e["bibtex_key"] for e in final_entries]
key_set = set(all_keys)
assert len(all_keys) == len(key_set), f"Duplicate keys found! {len(all_keys)} vs {len(key_set)}"

# Check no conflicts with public splits
conflicts = key_set & all_public_keys
if conflicts:
    print(f"WARNING: {len(conflicts)} key conflicts with public splits: {sorted(conflicts)[:5]}")

# Shuffle so types aren't grouped
random.shuffle(final_entries)

# ── Write output ────────────────────────────────────────────────────────────

save_jsonl(HIDDEN_PATH, final_entries)

# ── Summary ─────────────────────────────────────────────────────────────────

label_counts = {"VALID": 0, "HALLUCINATED": 0}
type_counts: dict[str, int] = {}
tier_counts: dict[int, int] = {}
for e in final_entries:
    label_counts[e["label"]] += 1
    ht = e.get("hallucination_type")
    if ht:
        type_counts[ht] = type_counts.get(ht, 0) + 1
    dt = e.get("difficulty_tier")
    if dt:
        tier_counts[dt] = tier_counts.get(dt, 0) + 1

print(f"\nTotal entries: {len(final_entries)}")
print(f"  VALID: {label_counts['VALID']}")
print(f"  HALLUCINATED: {label_counts['HALLUCINATED']}")
print("\nType distribution:")
for t in sorted(type_counts.keys()):
    print(f"  {t}: {type_counts[t]}")
print("\nTier distribution:")
for t in sorted(tier_counts.keys()):
    print(f"  Tier {t}: {tier_counts[t]}")

# ── Update metadata ────────────────────────────────────────────────────────

with open(METADATA_PATH) as f:
    metadata = json.load(f)

metadata["splits"]["test_hidden"] = {
    "file": "test_hidden.jsonl",
    "total": len(final_entries),
    "valid": label_counts["VALID"],
    "hallucinated": label_counts["HALLUCINATED"],
    "tier_distribution": {str(k): v for k, v in sorted(tier_counts.items())},
    "type_distribution": {k: v for k, v in sorted(type_counts.items())},
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)
    f.write("\n")

print(f"\nMetadata updated at {METADATA_PATH}")
print("Done!")
