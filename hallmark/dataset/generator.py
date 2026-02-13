"""Generate hallucinated BibTeX entries for HALLMARK.

Supports four generation methods:
1. Systematic perturbation of valid entries
2. LLM-generated hallucinations (via API)
3. Real-world hallucinations (manual curation)
4. Adversarial crafting
"""

from __future__ import annotations

import copy
import random
import string
from datetime import date

from hallmark.dataset.schema import (
    BenchmarkEntry,
    DifficultyTier,
    GenerationMethod,
    HallucinationType,
)


def generate_fabricated_doi(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
    """Tier 1: Replace DOI with a non-resolving fabricated DOI."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    fake_prefix = rng.choice(
        [
            "10.9999",
            "10.8888",
            "10.7777",
            "10.6666",
            "10.5432",
            "10.1234",
            "10.4321",
            "10.3141",
            "10.2718",
            "10.1618",
            "10.48550",
            "10.32614",
            "10.15439",
            "10.21033",
            "10.60715",
            "10.47281",
            "10.93105",
            "10.82004",
            "10.55910",
            "10.71336",
        ]
    )
    # Vary suffix patterns to avoid a single learnable format
    suffix_style = rng.choice(["path", "id", "year_id", "conf_id"])
    if suffix_style == "path":
        seg1 = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 6)))
        seg2 = "".join(rng.choices(string.digits, k=rng.randint(4, 7)))
        fake_suffix = f"{seg1}.{seg2}"
    elif suffix_style == "id":
        fake_suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=10))
    elif suffix_style == "year_id":
        year = rng.choice(["2019", "2020", "2021", "2022", "2023", "2024"])
        seq = "".join(rng.choices(string.digits, k=5))
        fake_suffix = f"{year}.{seq}"
    else:  # conf_id
        conf = rng.choice(["conf", "proc", "jour", "art", "pub"])
        seq = "".join(rng.choices(string.digits, k=7))
        fake_suffix = f"{conf}/{seq}"
    new_entry.fields["doi"] = f"{fake_prefix}/{fake_suffix}"
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.FABRICATED_DOI.value
    new_entry.difficulty_tier = DifficultyTier.EASY.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"DOI fabricated: {new_entry.fields['doi']} does not resolve"
    new_entry.subtests = {
        "doi_resolves": False,
        "title_exists": True,
        "authors_match": True,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"fabricated_doi_{new_entry.bibtex_key}"
    return new_entry


def generate_nonexistent_venue(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
    """Tier 1: Replace venue with an invented conference/journal name.

    Note: Always uses booktitle (normalized to inproceedings per P0.2).
    """
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    fake_venues = [
        "International Conference on Advanced AI Systems",
        "Journal of Computational Intelligence and Applications",
        "Workshop on Emerging Methods in Deep Learning",
        "Transactions on Neural Computing Paradigms",
        "Symposium on Frontier Artificial Intelligence Research",
        "Annual Conference on Machine Learning Innovations",
        "IEEE International Conference on Cognitive Computing",
        "Pacific Rim Symposium on Neural Information Systems",
        "European Workshop on Probabilistic Machine Learning",
        "ACM Conference on Automated Reasoning and Learning",
        "International Journal of Adaptive Computation",
        "Workshop on Scalable Representation Learning",
        "Conference on Foundations of Intelligent Systems",
        "Journal of Theoretical and Applied AI Research",
        "International Symposium on Data-Driven Discovery",
        "Transactions on Autonomous Learning Systems",
        "Workshop on Trustworthy AI and Robustness",
        "Conference on Multimodal Learning and Perception",
        "Annual Symposium on Efficient Deep Learning",
        "Journal of Neural Architecture and Optimization",
        "International Conference on Generative Modeling",
        "Workshop on Causal Inference in Machine Learning",
        "Conference on Language Models and Understanding",
        "Symposium on Geometric Deep Learning Methods",
        "Pacific Conference on Knowledge Representation",
        "Journal of Reinforcement Learning and Control",
        "International Workshop on Federated AI Systems",
        "Conference on Bio-Inspired Computing and AI",
        "Transactions on Computer Vision Applications",
        "Annual Workshop on AI Safety and Alignment",
        "Symposium on Graph Neural Network Research",
        "International Conference on Continual Learning",
        "Journal of Explainable Artificial Intelligence",
        "Workshop on Low-Resource Language Processing",
        "Conference on Bayesian Deep Learning Methods",
        "International Symposium on AI for Science",
        "Transactions on Self-Supervised Representation Learning",
        "Workshop on Embodied Intelligence and Robotics",
        "Conference on Privacy-Preserving Machine Learning",
        "Journal of Time Series Analysis and Forecasting",
        "Symposium on Neuro-Symbolic AI Integration",
        "International Conference on Quantum Machine Learning",
        "Workshop on Foundation Models and Adaptation",
        "Conference on Algorithmic Fairness in AI",
        "Annual Conference on Spatial Intelligence",
        "Journal of Statistical Machine Learning Theory",
        "Workshop on Scientific Machine Learning",
        "Symposium on Decision-Making Under Uncertainty",
        "International Conference on Intelligent Automation",
        "Conference on Emergent Communication in AI",
        "IEEE Symposium on Adaptive Neural Systems",
        "Workshop on Memory-Augmented Neural Networks",
        "International Journal of Deep Learning Research",
        "Conference on Structured Prediction Methods",
        "Symposium on Transfer Learning Paradigms",
        "ACM Workshop on Neural Architecture Design",
        "Transactions on Meta-Learning and Adaptation",
        "Conference on Distributed Machine Learning",
        "Journal of Adversarial Machine Learning",
        "International Workshop on Interpretable Models",
        "Symposium on Online Learning Theory",
        "Conference on Attention Mechanisms and Transformers",
        "Workshop on Multi-Agent Learning Systems",
        "Journal of Optimization in Neural Networks",
        "International Conference on Semantic Understanding",
        "Symposium on Probabilistic Reasoning",
        "Conference on Zero-Shot and Few-Shot Learning",
        "Workshop on Neural Information Retrieval",
        "Transactions on Language Model Architectures",
        "International Journal of Representation Learning",
        "Conference on Active Learning Methods",
        "Symposium on Regularization Techniques",
        "Workshop on Curriculum Learning Strategies",
        "Journal of Uncertainty Estimation",
        "International Conference on Model Compression",
        "Conference on Synthetic Data Generation",
        "Symposium on Contrastive Learning Methods",
        "Workshop on Attention and Memory Networks",
        "Transactions on Graph Representation Learning",
        "International Journal of Neural Inference",
        "Conference on Prompt Engineering and Design",
        "Symposium on Test-Time Adaptation Methods",
        "Workshop on Domain Generalization",
        "Journal of Efficient Neural Architectures",
        "International Conference on Multimodal Integration",
        "Conference on Neural Theorem Proving",
        "Symposium on World Models and Planning",
        "Workshop on Latent Variable Models",
        "Transactions on Neural Rendering",
        "International Journal of Metric Learning",
        "Conference on Hierarchical Learning Systems",
        "Symposium on Energy-Based Models",
        "Workshop on Neural Differential Equations",
        "Journal of Sequence Modeling",
        "International Conference on Equivariant Networks",
        "Conference on Disentangled Representations",
        "Symposium on Neural Scaling Laws",
        "Workshop on Sparse Neural Networks",
        "Transactions on Instruction Following Models",
        "International Journal of Context Learning",
        "Conference on Reward Learning and Alignment",
        "Symposium on Compositional Generalization",
    ]
    fake_venue = rng.choice(fake_venues)
    # Always use booktitle (normalized to inproceedings per P0.2)
    new_entry.fields["booktitle"] = fake_venue
    new_entry.fields.pop("journal", None)  # Remove journal if present
    new_entry.bibtex_type = "inproceedings"
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.NONEXISTENT_VENUE.value
    new_entry.difficulty_tier = DifficultyTier.EASY.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Venue fabricated: '{fake_venue}' does not exist"
    has_doi = bool(new_entry.fields.get("doi"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,
        "title_exists": True,
        "authors_match": True,
        "venue_real": False,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"fake_venue_{new_entry.bibtex_key}"
    return new_entry


def generate_placeholder_authors(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
    """Tier 1: Replace authors with generic/fake names."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    fake_author_sets = [
        "Wei Zhang and Priya Patel and James Brown",
        "Maria Santos and Hiroshi Tanaka",
        "Ahmed Hassan and Sarah O'Brien and Raj Gupta",
        "K. Mueller and L. Johansson",
        "Yusuf Ali and Elena Popova and David Lee",
        "R. Kumar and S. Nakamura and T. Fischer",
        "Jing Liu and Carlos Ramirez",
        "Olga Ivanova and Pierre Dubois and Min-Soo Kim",
        "M. Chen and A. Kowalski and B. Okafor",
        "Fatima Al-Rashid and Kenji Yamamoto",
        "Dmitri Volkov and Aisha Mbeki and Luca Rossi",
        "J. Andersen and C. Morales",
        "Ravi Sharma and Mei-Ling Wu and Thomas Schmidt",
        "P. Novak and H. Sato and F. Osei",
        "Soo-Jin Park and Rafael Costa",
        "N. Petersen and Y. Taniguchi and A. Mensah",
        "Ling Chen and Amara Diallo and Viktor Horvat",
        "Q. Wang and D. Fernandez and E. Kimura",
        "Hassan Ibrahim and Sofia Kowalczyk and Jun-Ho Lee",
        "Isabella Romano and Kwame Mensah and Yuki Sato",
        "Lars Bergström and Aisha Ndiaye and Chen Wei",
        "Mateo Garcia and Nadia Petrova and Hiroshi Yamada",
        "Elena Vasquez and Tariq Al-Mansour and Min-Ji Park",
        "Viktor Sokolov and Amina Diop and Takeshi Nakamura",
        "Gabriela Silva and Omar Farah and Yuna Choi",
        "Andreas Schneider and Zainab Ahmed and Kenji Tanaka",
        "Lucia Moretti and Jamal Williams and Hye-Jin Kim",
        "Nikolai Ivanov and Fatou Sow and Akira Suzuki",
        "Carmen Rodriguez and Kwasi Osei and Mei-Ling Huang",
        "Stefan Müller and Amara Koné and Taro Watanabe",
        "Valentina Costa and Ibrahim Kone and Soo-Yeon Lee",
        "Dimitri Popov and Nadine Dubois and Yuki Matsumoto",
        "Rosa Martinez and Adebayo Adeyemi and Ji-Woo Park",
        "Henrik Larsen and Yasmin Hassan and Haruto Nakano",
        "Francesca Bianchi and Mustafa Ali and Hana Taniguchi",
        "Pavel Novotny and Fatima Ndoye and Ryota Kobayashi",
        "Ana Santos and Kwame Asante and Min-Seo Kim",
        "Mikhail Petrov and Amina Traoré and Kaito Yamamoto",
        "Giulia Romano and Youssef Benali and Sora Takahashi",
        "Jan Kowalski and Zara Ibrahim and Ren Watanabe",
        "Marta Fernandez and Abdi Mohamed and Aiko Suzuki",
        "Boris Volkov and Fatoumata Diallo and Riku Nakamura",
        "Cristina Lopez and Tariq Hussein and Yui Tanaka",
        "Andrei Ivanov and Mariama Sow and Haruki Sato",
        "Bianca Silva and Omar Sharif and Ayumi Yamada",
        "Dmitry Kozlov and Aissatou Diop and Sota Kobayashi",
        "Elena Garcia and Adama Toure and Kaede Matsumoto",
        "Igor Sokolov and Ndeye Ndiaye and Yuta Nakano",
        "Laura Martinez and Hassan Ahmed and Rina Taniguchi",
        "Maxim Petrov and Fatou Kone and Sho Watanabe",
        "Nina Rodriguez and Ibrahim Diallo and Kenta Yamamoto",
        "Oleg Volkov and Aminata Seck and Daiki Suzuki",
        "Paula Santos and Yusuf Ibrahim and Hiro Takahashi",
        "Roman Ivanov and Zeynab Hassan and Ryo Nakamura",
        "Sara Lopez and Abdou Diop and Yuki Sato",
        "Tomas Novotny and Fatouma Traore and Takumi Tanaka",
        "Ulyana Petrova and Oumar Ndiaye and Shun Yamada",
        "Vera Sokolova and Mamadou Kone and Kaito Matsumoto",
        "Xavier Costa and Awa Sow and Ren Kobayashi",
        "Yana Ivanova and Cheikh Diallo and Sora Nakano",
        "Zara Fernandez and Boubacar Toure and Aoi Watanabe",
        "Ali Rahman and Ingrid Johansson and Yuto Suzuki",
        "Beatriz Silva and Jamal Ibrahim and Hana Yamamoto",
        "Carlos Garcia and Khadija Hassan and Riku Tanaka",
        "Diana Martinez and Lamine Ndiaye and Saki Sato",
        "Emilio Rodriguez and Maimouna Diop and Yui Nakamura",
        "Fatima Lopez and Ndongo Kone and Haruki Yamada",
        "Gabriel Santos and Oumou Sow and Kaede Kobayashi",
        "Hector Fernandez and Penda Diallo and Yuta Matsumoto",
        "Ibrahim Costa and Ramata Traore and Sota Nakano",
        "Jasmin Novotny and Saliou Ndiaye and Rina Taniguchi",
        "Karim Petrov and Thiam Toure and Kenta Watanabe",
        "Leila Ivanov and Ury Seck and Sho Yamamoto",
        "Malik Sokolov and Viviane Kone and Daiki Suzuki",
        "Noor Rodriguez and Waly Diop and Hiro Takahashi",
        "Oscar Lopez and Xadi Sow and Ryo Nakamura",
        "Petra Silva and Yacouba Ndiaye and Yuki Sato",
        "Quinn Martinez and Zeinab Kone and Takumi Tanaka",
        "Rashid Fernandez and Aida Traore and Shun Yamada",
        "Salma Santos and Binta Diallo and Kaito Matsumoto",
        "Tariq Garcia and Coumba Toure and Ren Kobayashi",
        "Umi Rodriguez and Dieynaba Seck and Sora Nakano",
        "Vera Costa and Elhadj Ndiaye and Aoi Watanabe",
        "Walid Novotny and Fanta Kone and Yuto Suzuki",
        "Yasmin Petrov and Gnima Sow and Hana Yamamoto",
        "Zahra Ivanov and Habib Diop and Riku Tanaka",
        "Amir Sokolov and Ina Diallo and Saki Sato",
        "Basma Lopez and Juma Traore and Yui Nakamura",
        "Cyril Silva and Kadiatou Ndiaye and Haruki Yamada",
        "Dalia Martinez and Lassana Toure and Kaede Kobayashi",
        "Elias Fernandez and Mame Seck and Yuta Matsumoto",
        "Farah Santos and Ndiaga Kone and Sota Nakano",
        "Ghazi Garcia and Oury Sow and Rina Taniguchi",
        "Huda Rodriguez and Papa Ndiaye and Kenta Watanabe",
        "Imran Costa and Qader Diop and Sho Yamamoto",
        "Jana Novotny and Rokhaya Kone and Daiki Suzuki",
        "Kareem Petrov and Safiatou Traore and Hiro Takahashi",
        "Laila Ivanov and Tidiane Diallo and Ryo Nakamura",
    ]
    new_entry.fields["author"] = rng.choice(fake_author_sets)
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.PLACEHOLDER_AUTHORS.value
    new_entry.difficulty_tier = DifficultyTier.EASY.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Authors are placeholders: {new_entry.fields['author']}"
    has_doi = bool(new_entry.fields.get("doi"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,
        "title_exists": True,
        "authors_match": False,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"fake_authors_{new_entry.bibtex_key}"
    return new_entry


def generate_future_date(entry: BenchmarkEntry, rng: random.Random | None = None) -> BenchmarkEntry:
    """Tier 1: Set publication year to the future."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    future_year = str(date.today().year + rng.randint(4, 10))
    new_entry.fields["year"] = future_year
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.FUTURE_DATE.value
    new_entry.difficulty_tier = DifficultyTier.EASY.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Publication year {future_year} is in the future"
    has_doi = bool(new_entry.fields.get("doi"))
    has_identifier = has_doi or bool(new_entry.fields.get("url"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,
        "title_exists": True,
        "authors_match": True,
        "venue_real": True,
        "fields_complete": has_identifier,  # True if DOI/URL present
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"future_{new_entry.bibtex_key}"
    return new_entry


def generate_chimeric_title(
    entry: BenchmarkEntry,
    fake_title: str,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: Keep real author but replace title with a fabricated one."""
    new_entry = _clone_entry(entry)
    new_entry.fields["title"] = fake_title
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.CHIMERIC_TITLE.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Title '{fake_title}' is fabricated; authors are real"
    has_doi = bool(new_entry.fields.get("doi"))
    has_identifier = has_doi or bool(new_entry.fields.get("url"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,  # DOI still resolves to original paper
        "title_exists": False,
        "authors_match": True,
        "venue_real": True,
        "fields_complete": has_identifier,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"chimeric_{new_entry.bibtex_key}"
    return new_entry


def generate_wrong_venue(
    entry: BenchmarkEntry,
    wrong_venue: str,
    wrong_year: str | None = None,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: Correct title/authors but wrong venue or year.

    Note: Always uses booktitle (normalized to inproceedings per P0.2).
    """
    new_entry = _clone_entry(entry)
    # Always use booktitle (normalized to inproceedings per P0.2)
    new_entry.fields["booktitle"] = wrong_venue
    new_entry.fields.pop("journal", None)  # Remove journal if present
    new_entry.bibtex_type = "inproceedings"
    if wrong_year:
        new_entry.fields["year"] = wrong_year
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.WRONG_VENUE.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Venue changed to '{wrong_venue}' (original was different)"
    has_doi = bool(new_entry.fields.get("doi"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,
        "title_exists": True,
        "authors_match": True,
        "venue_real": False,
        "fields_complete": True,
        "cross_db_agreement": True,  # title+authors match across databases; only venue is wrong
    }
    new_entry.bibtex_key = f"wrong_venue_{new_entry.bibtex_key}"
    return new_entry


def generate_swapped_authors(
    entry: BenchmarkEntry,
    donor_entry: BenchmarkEntry,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: Correct title but authors from a different paper."""
    new_entry = _clone_entry(entry)
    new_entry.fields["author"] = donor_entry.fields.get("author", "Unknown Author")
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.AUTHOR_MISMATCH.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Authors swapped from '{donor_entry.bibtex_key}'"
    has_doi = bool(new_entry.fields.get("doi"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,
        "title_exists": True,
        "authors_match": False,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"swapped_{new_entry.bibtex_key}"
    return new_entry


def generate_near_miss_title(
    entry: BenchmarkEntry,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 3: Title off by 1-2 words from the real paper.

    Strategies (all preserve grammatical correctness):
    - synonym: replace a word with a same-POS synonym
    - plural: flip singular/plural on a safe noun
    - spelling: British/American spelling swap
    - abbreviation: expand or contract ML abbreviations (e.g., RL <-> Reinforcement Learning)
    - hyphen: toggle hyphenation (e.g., self-supervised <-> self supervised)
    - article: remove an existing article (fallback)

    The "swap" strategy (swap adjacent words) is intentionally excluded
    because it almost always produces ungrammatical titles.
    """
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    title = new_entry.fields.get("title", "")
    words = title.split()

    # Part-of-speech-safe synonym pairs (noun-noun, adj-adj, prep-prep)
    _noun_synonyms = {
        "learning": "training",
        "training": "learning",
        "network": "architecture",
        "architecture": "network",
        "model": "framework",
        "framework": "model",
        "method": "approach",
        "approach": "method",
        "system": "pipeline",
        "pipeline": "system",
        "analysis": "study",
        "study": "analysis",
        "detection": "recognition",
        "recognition": "detection",
        "generation": "synthesis",
        "synthesis": "generation",
        "estimation": "prediction",
        "prediction": "estimation",
        "classification": "categorization",
        "categorization": "classification",
        "representation": "embedding",
        "embedding": "representation",
        "inference": "reasoning",
        "reasoning": "inference",
        "segmentation": "partitioning",
        "partitioning": "segmentation",
        "regularization": "penalization",
        "penalization": "regularization",
        "search": "exploration",
        "exploration": "search",
        "bounds": "guarantees",
        "guarantees": "bounds",
        "loss": "objective",
        "objective": "loss",
        "task": "problem",
        "problem": "task",
        "survey": "review",
        "review": "survey",
        "score": "metric",
        "metric": "score",
        "bottleneck": "limitation",
        "limitation": "bottleneck",
        "corruption": "perturbation",
        "perturbation": "corruption",
        "design": "construction",
        "construction": "design",
        "planning": "scheduling",
        "scheduling": "planning",
        "attention": "focus",
        "focus": "attention",
        "need": "require",
        "require": "need",
        "information": "knowledge",
        "knowledge": "information",
        "image": "visual",
        "visual": "image",
        "language": "linguistic",
        "linguistic": "language",
        "optimization": "tuning",
        "tuning": "optimization",
        "convergence": "stability",
        "stability": "convergence",
        "transformation": "mapping",
        "mapping": "transformation",
        "dimension": "feature",
        "feature": "dimension",
        "kernel": "filter",
        "filter": "kernel",
        "encoder": "compressor",
        "compressor": "encoder",
        "decoder": "reconstructor",
        "reconstructor": "decoder",
        "mechanism": "scheme",
        "scheme": "mechanism",
        "technique": "strategy",
        "strategy": "technique",
        "evaluation": "assessment",
        "assessment": "evaluation",
        "performance": "accuracy",
        "accuracy": "performance",
        "alignment": "matching",
        "matching": "alignment",
        "augmentation": "enhancement",
        "enhancement": "augmentation",
        "compression": "reduction",
        "reduction": "compression",
        "distillation": "compression",
        "aggregation": "pooling",
        "pooling": "aggregation",
        "interpolation": "blending",
        "blending": "interpolation",
        "retrieval": "extraction",
        "extraction": "retrieval",
        "fusion": "integration",
        "integration": "fusion",
        "decomposition": "factorization",
        "factorization": "decomposition",
        "refinement": "improvement",
        "improvement": "refinement",
    }
    _adj_synonyms = {
        "robust": "resilient",
        "resilient": "robust",
        "efficient": "scalable",
        "scalable": "efficient",
        "optimal": "approximate",
        "approximate": "optimal",
        "deep": "hierarchical",
        "hierarchical": "deep",
        "adversarial": "competitive",
        "competitive": "adversarial",
        "distributed": "decentralized",
        "decentralized": "distributed",
        "adaptive": "dynamic",
        "dynamic": "adaptive",
        "contrastive": "comparative",
        "comparative": "contrastive",
        "deterministic": "stochastic",
        "stochastic": "deterministic",
        "neural": "learned",
        "learned": "neural",
        "causal": "structural",
        "structural": "causal",
        "latent": "hidden",
        "hidden": "latent",
        "exact": "precise",
        "precise": "exact",
        "fast": "rapid",
        "rapid": "fast",
        "unsupervised": "self-supervised",
        "self-supervised": "unsupervised",
        "inverse": "reverse",
        "reverse": "inverse",
        "sparse": "compact",
        "compact": "sparse",
        "dense": "rich",
        "rich": "dense",
        "generative": "probabilistic",
        "probabilistic": "generative",
        "discriminative": "predictive",
        "predictive": "discriminative",
        "continuous": "smooth",
        "smooth": "continuous",
        "discrete": "categorical",
        "categorical": "discrete",
        "local": "spatial",
        "spatial": "local",
        "global": "universal",
        "universal": "global",
        "temporal": "sequential",
        "sequential": "temporal",
        "modular": "compositional",
        "compositional": "modular",
        "differentiable": "smooth",
        "lightweight": "compact",
        "heavy": "complex",
        "complex": "heavy",
        "simple": "basic",
        "basic": "simple",
        "advanced": "sophisticated",
        "sophisticated": "advanced",
        "interpretable": "explainable",
        "explainable": "interpretable",
        "fair": "unbiased",
        "unbiased": "fair",
    }
    _prep_synonyms = {
        "via": "through",
        "through": "via",
        "towards": "for",
        "for": "towards",
        "with": "using",
        "using": "with",
        "across": "over",
        "over": "across",
        "under": "beneath",
        "beneath": "under",
    }
    replacements: dict[str, str] = {}
    replacements.update(_noun_synonyms)
    replacements.update(_adj_synonyms)
    replacements.update(_prep_synonyms)

    # Safe nouns for plural/singular flipping
    safe_plural_roots = {
        "model",
        "network",
        "method",
        "bound",
        "guarantee",
        "constraint",
        "representation",
        "feature",
        "tree",
        "approach",
        "system",
        "attack",
        "image",
        "embedding",
        "distribution",
        "score",
        "prediction",
        "algorithm",
        "graph",
        "function",
        "layer",
        "node",
        "weight",
        "task",
        "objective",
        "problem",
        "gradient",
        "sample",
    }

    # Common ML abbreviations for expansion/contraction
    _abbreviations = {
        "RL": "Reinforcement Learning",
        "NLP": "Natural Language Processing",
        "GAN": "Generative Adversarial Network",
        "CNN": "Convolutional Neural Network",
        "RNN": "Recurrent Neural Network",
        "VAE": "Variational Autoencoder",
        "SGD": "Stochastic Gradient Descent",
        "MLP": "Multi-Layer Perceptron",
        "GNN": "Graph Neural Network",
        "LLM": "Large Language Model",
    }
    # Build reverse map (expansion -> abbreviation)
    _expansions = {v.lower(): k for k, v in _abbreviations.items()}

    new_title = title
    if len(words) >= 3:
        # Choose a mutation strategy (no "swap" -- it breaks grammar)
        strategy = rng.choice(
            ["synonym", "plural", "synonym", "spelling", "abbreviation", "hyphen"]
        )

        if strategy == "plural":
            # Flip plural/singular on a safe noun
            indices = list(range(len(words)))
            rng.shuffle(indices)
            for idx in indices:
                raw = words[idx]
                stripped = raw.rstrip(".,;:!?")
                suffix = raw[len(stripped) :]
                lower = stripped.lower()
                if lower.endswith("s") and lower[:-1] in safe_plural_roots:
                    words[idx] = stripped[:-1] + suffix
                    new_title = " ".join(words)
                    break
                elif lower in safe_plural_roots:
                    words[idx] = stripped + "s" + suffix
                    new_title = " ".join(words)
                    break

        elif strategy == "spelling":
            # British/American spelling swap
            if "ization" in title:
                new_title = title.replace("ization", "isation", 1)
            elif "isation" in title:
                new_title = title.replace("isation", "ization", 1)

        elif strategy == "abbreviation":
            # Expand abbreviation or contract multi-word phrase
            title_lower = title.lower()
            for expansion, abbr in _expansions.items():
                pos = title_lower.find(expansion)
                if pos >= 0:
                    # Contract to abbreviation
                    new_title = title[:pos] + abbr + title[pos + len(expansion) :]
                    break
            if new_title == title:
                for abbr, expansion in _abbreviations.items():
                    if abbr in title:
                        new_title = title.replace(abbr, expansion, 1)
                        break

        elif strategy == "hyphen":
            # Toggle hyphenation (e.g., "self-supervised" <-> "self supervised")
            if "-" in title:
                # Remove a hyphen
                hyphen_pos = title.index("-")
                new_title = title[:hyphen_pos] + " " + title[hyphen_pos + 1 :]
            else:
                # Add a hyphen between common compound modifiers
                _compounds = [
                    ("self ", "self-"),
                    ("semi ", "semi-"),
                    ("multi ", "multi-"),
                    ("cross ", "cross-"),
                    ("non ", "non-"),
                    ("pre ", "pre-"),
                    ("co ", "co-"),
                ]
                for old, new in _compounds:
                    if old in title.lower():
                        pos = title.lower().index(old)
                        new_title = title[:pos] + new + title[pos + len(old) :]
                        break

        if strategy == "synonym" or new_title == title:
            # Synonym substitution (primary strategy and fallback)
            indices = list(range(len(words)))
            rng.shuffle(indices)
            for idx in indices:
                raw = words[idx]
                stripped = raw.rstrip(".,;:!?")
                suffix = raw[len(stripped) :]
                key = stripped.lower()
                if key in replacements:
                    repl = replacements[key]
                    # Preserve capitalization
                    if stripped[0].isupper():
                        repl = repl[0].upper() + repl[1:]
                    words[idx] = repl + suffix
                    new_title = " ".join(words)
                    break

        # Final fallback: remove an article if present
        if new_title == title:
            for idx in range(1, len(words)):
                if words[idx].lower() in {"a", "an", "the"}:
                    words_copy = words[:idx] + words[idx + 1 :]
                    new_title = " ".join(words_copy)
                    break

        # Absolute last resort: flip plural on a long word
        if new_title == title and len(words) >= 2:
            idx = rng.randint(0, len(words) - 1)
            w = words[idx].rstrip(".,;:!?")
            if len(w) >= 4 and w[0].isalpha():
                words[idx] = w + "s" if not w.endswith("s") else w[:-1]
                new_title = " ".join(words)

    new_entry.fields["title"] = new_title
    has_doi = bool(new_entry.fields.get("doi"))
    has_identifier = has_doi or bool(new_entry.fields.get("url"))
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.NEAR_MISS_TITLE.value
    new_entry.difficulty_tier = DifficultyTier.HARD.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Title slightly modified: '{new_title}' (original: '{title}')"
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,  # DOI resolves to original (correct) paper
        "title_exists": False,
        "authors_match": True,
        "venue_real": True,
        "fields_complete": has_identifier,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"near_miss_{new_entry.bibtex_key}"
    return new_entry


def is_preprint_source(entry: BenchmarkEntry) -> bool:
    """Check if an entry looks like an arXiv preprint (suitable for preprint_as_published)."""
    has_eprint = bool(entry.fields.get("eprint"))
    has_conference_doi = bool(entry.fields.get("doi"))
    # A preprint either has an eprint field, or lacks a conference DOI
    # Entries with both DOI and booktitle are conference papers, not preprints
    if has_eprint:
        return True
    return not has_conference_doi and entry.bibtex_type in ("misc", "article")


def generate_preprint_as_published(
    entry: BenchmarkEntry,
    fake_venue: str,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: arXiv preprint cited as if published at a venue.

    Note: source entry should ideally be a genuine preprint (use is_preprint_source()
    to filter). If the source is already a conference paper, the result is effectively
    a wrong_venue entry.

    Per P0.3: Does NOT add eprint/archiveprefix fields (stripped as hallucination-only).
    """
    new_entry = _clone_entry(entry)
    # Add a fake venue
    new_entry.fields["booktitle"] = fake_venue
    new_entry.fields.pop("journal", None)  # Remove journal if present
    new_entry.bibtex_type = "inproceedings"
    # Remove arXiv indicators (P0.3: these are hallucination-only fields)
    new_entry.fields.pop("eprint", None)
    new_entry.fields.pop("archiveprefix", None)
    new_entry.fields.pop("primaryclass", None)
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.PREPRINT_AS_PUBLISHED.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Preprint falsely cited as published at '{fake_venue}'"
    has_identifier = bool(new_entry.fields.get("doi") or new_entry.fields.get("url"))
    new_entry.subtests = {
        "doi_resolves": False,
        "title_exists": True,
        "authors_match": True,
        "venue_real": False,
        "fields_complete": has_identifier,
        "cross_db_agreement": True,  # title+authors match across databases; venue is fabricated
    }
    new_entry.bibtex_key = f"preprint_pub_{new_entry.bibtex_key}"
    return new_entry


def generate_plausible_fabrication(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
    """Tier 3: Fabricate a realistic but non-existent paper at a real prestigious venue."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)

    # Diverse title templates that produce grammatically correct ML paper titles.
    # Each template is a callable that takes the rng and returns a title string.
    _methods = [
        "Contrastive Learning",
        "Self-Supervised Pre-Training",
        "Variational Inference",
        "Reinforcement Learning",
        "Meta-Learning",
        "Knowledge Distillation",
        "Prompt Tuning",
        "Bayesian Optimization",
        "Curriculum Learning",
        "Federated Averaging",
        "Neural Architecture Search",
        "Gradient Descent",
        "Spectral Normalization",
        "Data Augmentation",
        "Domain Adaptation",
        "Transfer Learning",
        "Active Learning",
        "Few-Shot Learning",
        "Zero-Shot Learning",
        "Multi-Task Learning",
        "Adversarial Training",
        "Continual Learning",
        "Online Learning",
        "Causal Inference",
        "Information Bottleneck",
    ]
    _domains = [
        "Vision Transformers",
        "Large Language Models",
        "Graph Neural Networks",
        "Point Cloud Processing",
        "Medical Image Analysis",
        "Autonomous Driving",
        "Molecular Property Prediction",
        "Speech Recognition",
        "Recommender Systems",
        "Time Series Forecasting",
        "Object Detection",
        "Semantic Segmentation",
        "Text Classification",
        "Image Generation",
        "Robot Navigation",
        "Question Answering",
        "Neural Rendering",
        "Protein Structure Prediction",
        "Code Generation",
        "Dialogue Systems",
        "Video Understanding",
        "Multi-Modal Retrieval",
        "Drug Discovery",
        "Traffic Prediction",
        "Anomaly Detection",
    ]
    _properties = [
        "Robust",
        "Efficient",
        "Scalable",
        "Calibrated",
        "Interpretable",
        "Fair",
        "Privacy-Preserving",
        "Communication-Efficient",
        "Sample-Efficient",
        "Parameter-Efficient",
        "Provably Correct",
        "Certifiably Robust",
        "Memory-Efficient",
        "Data-Efficient",
        "Compute-Efficient",
        "Adaptive",
        "Generalizable",
        "Trustworthy",
        "Explainable",
        "Energy-Efficient",
    ]
    _nouns = [
        "Representations",
        "Embeddings",
        "Features",
        "Predictions",
        "Distributions",
        "Policies",
        "Architectures",
        "Objectives",
        "Gradients",
        "Attention Mechanisms",
        "Loss Functions",
        "Activation Functions",
        "Optimization Strategies",
        "Regularization Techniques",
        "Sampling Methods",
        "Inference Procedures",
        "Training Dynamics",
        "Model Weights",
        "Hidden States",
        "Latent Codes",
    ]
    _settings = [
        "Low-Resource Settings",
        "Non-Stationary Environments",
        "High-Dimensional Spaces",
        "Heterogeneous Data",
        "Label-Scarce Regimes",
        "Streaming Data",
        "Multi-Task Settings",
        "Cross-Domain Scenarios",
        "Partial Observability",
        "Noisy Labels",
        "Imbalanced Datasets",
        "Distribution Shift",
        "Out-of-Distribution Detection",
        "Long-Tailed Recognition",
        "Adversarial Perturbations",
        "Missing Data",
        "Limited Supervision",
        "Real-Time Constraints",
        "Cold-Start Problems",
        "Extreme Classification",
    ]

    m, d, p = rng.choice(_methods), rng.choice(_domains), rng.choice(_properties)
    n, s = rng.choice(_nouns), rng.choice(_settings)

    title_template = rng.choice(
        [
            # "Method via Technique for Domain"
            f"{p} {d} via {m}",
            # "On the Property of Method in Domain"
            f"On the Convergence of {m} for {d}",
            # "Toward Property Domain with Method"
            f"Toward {p} {d} with {m}",
            # "Method for Noun in Setting"
            f"{m} for {p} {n} in {s}",
            # "Property Domain: A Method Approach"
            f"{p} {d}: A {m} Approach",
            # "Rethinking Noun: Method for Domain"
            f"Rethinking {n} in {d} via {m}",
            # "Beyond Method: Property Noun for Domain"
            f"Beyond {m}: {p} {n} for {d}",
            # "Method with Property Constraints for Domain"
            f"{m} with {p} Constraints for {d}",
            # "Noun Alignment via Method in Setting"
            f"{n} Alignment via {m} in {s}",
            # "Property Method for Domain under Setting"
            f"{p} {m} for {d} under {s}",
            # "Understanding Noun in Domain through Method"
            f"Understanding {n} in {d} through {m}",
            # "Unifying Method and Technique for Domain"
            f"Unifying {m} and {rng.choice(_methods)} for {d}",
            # Standard NeurIPS-style with colon
            f"{d}: {p} {n} via {m}",
            # Question-style title
            f"When Does {m} Help {d}?",
            # "On Property of Noun for Domain"
            f"On {p} {n} for {d}",
            # Acronym-style
            f"{p} {m} for {d} in {s}",
            # "A Theoretical Analysis of..."
            f"A Theoretical Analysis of {m} for {d}",
            # "Bridging X and Y..."
            f"Bridging {m} and {rng.choice(_methods)} for {d}",
            # "Revisiting..."
            f"Revisiting {m} for {p} {d}",
            # Simple clean pattern
            f"{m} in {s}: Applications to {d}",
        ]
    )
    new_entry.fields["title"] = title_template

    # Diverse pool of 35 fabricated but realistic author combinations
    _first_names = [
        "Wei",
        "Yuki",
        "Sofia",
        "Alex",
        "Nina",
        "Gabriel",
        "Tomoko",
        "Leonardo",
        "Kexin",
        "Jieyu",
        "Pavel",
        "Daphne",
        "Elena",
        "Aditya",
        "Ziqian",
        "Ryuichi",
        "Tamara",
        "Jessica",
        "Ruiqi",
        "Anton",
        "Wenda",
        "Robin",
        "Yuntao",
        "Hanlin",
        "Tiancheng",
        "Georg",
        "Minjoon",
        "Ilia",
        "Fei",
        "Lin",
        "Ekin",
        "Shuang",
        "Yuxin",
        "Zhouhan",
        "Yao",
    ]
    _last_names = [
        "Zhang",
        "Tanaka",
        "Andersson",
        "Wu",
        "Kowalski",
        "Moreira",
        "Watanabe",
        "Ricci",
        "Pei",
        "Chen",
        "Tokmakov",
        "Cornelisse",
        "Vorontsova",
        "Grover",
        "Zhong",
        "Yamamoto",
        "Broderick",
        "Hamrick",
        "Gao",
        "Obukhov",
        "Chu",
        "Rombach",
        "Bai",
        "Goh",
        "Zhao",
        "Martius",
        "Seo",
        "Sucholutsky",
        "Sha",
        "Gui",
        "Cubuk",
        "Li",
        "Sindhwani",
        "Fu",
        "Gu",
    ]

    # Generate 3-5 unique authors
    n_authors = rng.randint(3, 5)
    first_pool = list(_first_names)
    last_pool = list(_last_names)
    rng.shuffle(first_pool)
    rng.shuffle(last_pool)
    authors = [f"{first_pool[i]} {last_pool[i]}" for i in range(n_authors)]
    new_entry.fields["author"] = " and ".join(authors)

    # Keep a real prestigious venue
    real_venues = ["NeurIPS", "ICML", "ICLR", "AAAI", "ACL", "CVPR"]
    # Always use booktitle (normalized to inproceedings per P0.2)
    new_entry.fields["booktitle"] = rng.choice(real_venues)
    new_entry.fields.pop("journal", None)  # Remove journal if present
    new_entry.bibtex_type = "inproceedings"

    # Remove DOI (fabricated paper won't have one)
    new_entry.fields.pop("doi", None)

    has_identifier = bool(new_entry.fields.get("doi") or new_entry.fields.get("url"))
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.PLAUSIBLE_FABRICATION.value
    new_entry.difficulty_tier = DifficultyTier.HARD.value
    new_entry.generation_method = GenerationMethod.ADVERSARIAL.value
    new_entry.explanation = "Completely fabricated paper with plausible metadata at real venue"
    new_entry.subtests = {
        "doi_resolves": False,
        "title_exists": False,
        "authors_match": False,
        "venue_real": True,
        "fields_complete": has_identifier,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"plausible_{new_entry.bibtex_key}"
    return new_entry


def generate_merged_citation(
    entry_a: BenchmarkEntry,
    entry_b: BenchmarkEntry,
    entry_c: BenchmarkEntry | None = None,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: Merge metadata from 2-3 real papers into one BibTeX entry.

    Takes authors from entry_a, title from entry_b, venue from entry_c (or entry_a).
    The resulting entry looks plausible but no single real paper matches all fields.
    """
    rng = rng or random.Random()
    new_entry = _clone_entry(entry_b)  # start from entry_b (title source)

    # Authors from entry_a
    new_entry.fields["author"] = entry_a.fields.get("author", "")

    # Title from entry_b (already cloned)

    # Venue from entry_c if provided, else entry_a
    venue_source = entry_c if entry_c is not None else entry_a
    # Always use booktitle (normalized to inproceedings per P0.2)
    for vf in ("booktitle", "journal"):
        if vf in venue_source.fields:
            new_entry.fields["booktitle"] = venue_source.fields[vf]
            break
    new_entry.fields.pop("journal", None)  # Remove journal if present
    new_entry.bibtex_type = "inproceedings"

    # Year from venue source
    if "year" in venue_source.fields:
        new_entry.fields["year"] = venue_source.fields["year"]

    # Use entry_b's DOI if available (title matches but authors won't)
    if entry_b.doi:
        new_entry.fields["doi"] = entry_b.doi
    else:
        new_entry.fields.pop("doi", None)

    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.MERGED_CITATION.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = (
        "Metadata merged from multiple real papers: "
        f"authors from '{entry_a.bibtex_key}', title from '{entry_b.bibtex_key}'"
    )
    new_entry.subtests = {
        "doi_resolves": entry_b.doi is not None,
        "title_exists": True,
        "authors_match": False,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"merged_{entry_b.bibtex_key}"
    return new_entry


def generate_partial_author_list(
    entry: BenchmarkEntry,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: Cite a real paper with a subset of its authors.

    Keeps first and last author, drops middle co-authors. Common in LLM outputs
    when the model remembers only prominent authors.
    """
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)

    authors_str = entry.fields.get("author", "")
    # BibTeX uses " and " as separator
    authors = [a.strip() for a in authors_str.split(" and ") if a.strip()]

    if len(authors) >= 3:
        # Keep first + last, drop random middle authors
        first = authors[0]
        last = authors[-1]
        middle = authors[1:-1]
        # Keep at most 1 middle author (randomly selected) for some variation
        kept_middle = [rng.choice(middle)] if middle and rng.random() < 0.3 else []
        new_authors = [first, *kept_middle, last]
        new_entry.fields["author"] = " and ".join(new_authors)
    elif len(authors) == 2:
        # Drop one author randomly
        new_entry.fields["author"] = authors[rng.randint(0, 1)]
    else:
        # Single author — can't reduce further, swap initial format instead
        # e.g., "John Smith" -> "J. Smith"
        parts = authors[0].split()
        if len(parts) >= 2 and len(parts[0]) > 1:
            new_entry.fields["author"] = f"{parts[0][0]}. {' '.join(parts[1:])}"

    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.PARTIAL_AUTHOR_LIST.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = (
        f"Partial author list: {len(authors)} authors reduced to "
        f"{len(new_entry.fields.get('author', '').split(' and '))}"
    )
    new_entry.subtests = {
        "doi_resolves": entry.doi is not None,
        "title_exists": True,
        "authors_match": False,  # partial list doesn't fully match
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"partial_authors_{new_entry.bibtex_key}"
    return new_entry


def generate_arxiv_version_mismatch(
    entry: BenchmarkEntry,
    wrong_venue: str,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 3: Mix arXiv preprint metadata with conference publication metadata.

    Takes a real conference paper and creates an entry that mixes preprint and
    publication metadata: the title and authors are real (verifiable), but the
    entry claims the paper was published at a wrong venue with a shifted year.

    This simulates real-world confusion where someone cites the arXiv version
    but attributes it to the wrong conference, or vice versa.

    Note: Per P0.3, does NOT add eprint/archiveprefix fields (stripped as hallucination-only).
    The confusion is signaled by wrong venue + year shift instead.
    """
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)

    # Get current year for shifting
    year = int(entry.fields.get("year", "2020"))

    # Remove arXiv indicators if present (P0.3: hallucination-only fields)
    new_entry.fields.pop("eprint", None)
    new_entry.fields.pop("archiveprefix", None)
    new_entry.fields.pop("primaryclass", None)

    # Claim it was published at a wrong conference venue
    new_entry.fields["booktitle"] = wrong_venue
    new_entry.fields.pop("journal", None)  # Remove journal if present
    new_entry.bibtex_type = "inproceedings"

    # Shift year by ±1 (common confusion between arXiv date and conference date)
    year_shift = rng.choice([-1, 1])
    new_entry.fields["year"] = str(year + year_shift)

    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.ARXIV_VERSION_MISMATCH.value
    new_entry.difficulty_tier = DifficultyTier.HARD.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = (
        f"Real paper cited with wrong venue '{wrong_venue}' and year shifted to "
        f"{year + year_shift}; metadata mixes preprint and publication versions"
    )
    has_doi = bool(new_entry.fields.get("doi"))
    has_identifier = bool(new_entry.fields.get("doi") or new_entry.fields.get("url"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,  # DOI resolves to the real paper; N/A if no DOI
        "title_exists": True,
        "authors_match": True,
        "venue_real": True,
        "fields_complete": has_identifier,
        "cross_db_agreement": False,  # venue/year don't match what DOI resolves to
    }
    new_entry.bibtex_key = f"arxiv_vm_{new_entry.bibtex_key}"
    return new_entry


def generate_hybrid_fabrication(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
    """Tier 2: Keep real DOI but fabricate authors and slightly modify title."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)

    # Keep the DOI from the original entry (it will still resolve)
    # Replace authors with fabricated names
    fake_author_sets = [
        "Michael Zhang and Jennifer Liu and Robert Chen",
        "Anna Petrov and Carlos Martinez and Yuki Nakamura",
        "Thomas Anderson and Sophia Kumar and Daniel Park",
        "Isabella Rossi and Ahmed Ali and Emma Williams",
        "Lucas Brown and Maria Garcia and Kevin Nguyen",
        "Olga Sokolova and James Wright and Mei Lin",
        "Henrik Larsson and Fatima Benali and Takeshi Ito",
        "Priya Chatterjee and Marco Bianchi and Soo-Yeon Choi",
        "David Okonkwo and Laura Fischer and Wei-Ting Hsu",
        "Amir Rezaei and Rachel Thompson and Kenji Morita",
        "Elena Vasquez and Patrick Murphy and Zhi-Yong Xu",
        "Nadia Kowalczyk and Samuel Adjei and Yuki Watanabe",
        "Oscar Herrera and Anya Krishnan and Felix Bauer",
        "Christine Dufour and Jamal Henderson and Hana Suzuki",
        "Dmitri Orlov and Amelia Santos and Jun-Ho Kwon",
    ]
    new_entry.fields["author"] = rng.choice(fake_author_sets)

    # Slightly modify the title (swap 2-3 words)
    title = new_entry.fields.get("title", "")
    words = title.split()
    if len(words) >= 4:
        # Swap 2-3 words with synonyms or related terms
        num_swaps = min(rng.randint(2, 3), len(words))
        swap_words = {
            "learning": "training",
            "neural": "deep",
            "network": "model",
            "efficient": "optimized",
            "robust": "stable",
            "novel": "new",
            "approach": "method",
            "framework": "system",
            "attention": "focus",
            "transformer": "architecture",
        }
        for _ in range(num_swaps):
            idx = rng.randint(0, len(words) - 1)
            word_lower = words[idx].lower().rstrip(".,;:!?")
            if word_lower in swap_words:
                words[idx] = swap_words[word_lower]
            elif rng.random() < 0.5:
                words[idx] = rng.choice(["Enhanced", "Improved", "Advanced", "Novel", "Efficient"])
        new_entry.fields["title"] = " ".join(words)

    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.HYBRID_FABRICATION.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.ADVERSARIAL.value
    new_entry.explanation = "Real DOI with fabricated authors and modified title"
    has_doi = bool(new_entry.fields.get("doi"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,
        "title_exists": False,
        "authors_match": False,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"hybrid_{new_entry.bibtex_key}"
    return new_entry


# --- Batch generation ---


TIER1_GENERATORS = [
    generate_fabricated_doi,
    generate_nonexistent_venue,
    generate_placeholder_authors,
    generate_future_date,
]


def generate_tier1_batch(
    valid_entries: list[BenchmarkEntry],
    count: int,
    seed: int = 42,
) -> list[BenchmarkEntry]:
    """Generate a batch of Tier 1 hallucinated entries from valid entries."""
    rng = random.Random(seed)
    results = []
    for _i in range(count):
        source = rng.choice(valid_entries)
        generator = rng.choice(TIER1_GENERATORS)
        results.append(generator(source, rng))
    return results


def generate_tier2_batch(
    valid_entries: list[BenchmarkEntry],
    count: int,
    seed: int = 42,
) -> list[BenchmarkEntry]:
    """Generate a batch of Tier 2 hallucinated entries."""
    rng = random.Random(seed)
    results = []
    venues = [
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
    ]

    # ML buzzwords for chimeric_title
    ml_buzzwords = [
        "Attention",
        "Transformer",
        "Self-Supervised",
        "Contrastive",
        "Few-Shot",
        "Multi-Modal",
        "Reinforcement",
        "Diffusion",
        "Neural Architecture Search",
        "Meta-Learning",
    ]

    for _i in range(count):
        source = rng.choice(valid_entries)
        method = rng.choice(
            [
                "wrong_venue",
                "swapped_authors",
                "preprint_as_published",
                "chimeric_title",
                "hybrid_fabrication",
                "merged_citation",
                "partial_author_list",
            ]
        )

        if method == "wrong_venue":
            wrong_v = rng.choice(venues)
            results.append(generate_wrong_venue(source, wrong_v, rng=rng))
        elif method == "swapped_authors":
            donor = rng.choice(valid_entries)
            while donor.bibtex_key == source.bibtex_key and len(valid_entries) > 1:
                donor = rng.choice(valid_entries)
            results.append(generate_swapped_authors(source, donor, rng))
        elif method == "preprint_as_published":
            fake_v = rng.choice(venues)
            results.append(generate_preprint_as_published(source, fake_v, rng))
        elif method == "chimeric_title":
            fake_title = f"{rng.choice(ml_buzzwords)} for {rng.choice(['Classification', 'Generation', 'Reasoning'])}"
            results.append(generate_chimeric_title(source, fake_title, rng))
        elif method == "hybrid_fabrication":
            results.append(generate_hybrid_fabrication(source, rng))
        elif method == "merged_citation":
            donor_b = rng.choice(valid_entries)
            while donor_b.bibtex_key == source.bibtex_key and len(valid_entries) > 1:
                donor_b = rng.choice(valid_entries)
            donor_c = rng.choice(valid_entries) if rng.random() < 0.5 else None
            results.append(generate_merged_citation(source, donor_b, donor_c, rng))
        elif method == "partial_author_list":
            results.append(generate_partial_author_list(source, rng))

    return results


def generate_tier3_batch(
    valid_entries: list[BenchmarkEntry],
    count: int,
    seed: int = 42,
) -> list[BenchmarkEntry]:
    """Generate a batch of Tier 3 hallucinated entries."""
    rng = random.Random(seed)
    results = []

    # Conference venues and years for arxiv_version_mismatch
    conferences = [
        ("NeurIPS", "2023"),
        ("ICML", "2023"),
        ("ICLR", "2024"),
        ("AAAI", "2024"),
        ("CVPR", "2023"),
    ]

    for _i in range(count):
        source = rng.choice(valid_entries)
        # Randomly choose between near_miss_title, plausible_fabrication, and arxiv_version_mismatch
        method = rng.choice(["near_miss_title", "plausible_fabrication", "arxiv_version_mismatch"])

        if method == "near_miss_title":
            results.append(generate_near_miss_title(source, rng))
        elif method == "plausible_fabrication":
            results.append(generate_plausible_fabrication(source, rng))
        elif method == "arxiv_version_mismatch":
            conf_venue, _conf_year = rng.choice(conferences)
            results.append(generate_arxiv_version_mismatch(source, conf_venue, rng))

    return results


# --- Helpers ---


def _clone_entry(entry: BenchmarkEntry) -> BenchmarkEntry:
    """Deep clone a BenchmarkEntry."""
    return BenchmarkEntry(
        bibtex_key=entry.bibtex_key,
        bibtex_type=entry.bibtex_type,
        fields=copy.deepcopy(entry.fields),
        label=entry.label,
        hallucination_type=entry.hallucination_type,
        difficulty_tier=entry.difficulty_tier,
        explanation=entry.explanation,
        generation_method=entry.generation_method,
        source_conference=entry.source_conference,
        publication_date=entry.publication_date,
        added_to_benchmark=entry.added_to_benchmark,
        subtests=copy.deepcopy(entry.subtests),
        raw_bibtex=None,
        source=entry.source,
    )
