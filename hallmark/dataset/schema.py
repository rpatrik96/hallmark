"""Core data schema for HALLMARK benchmark entries and predictions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Literal


class HallucinationType(str, Enum):
    """Taxonomy of citation hallucination types, organized by difficulty tier.

    The main taxonomy (11 types) covers empirically-grounded hallucination patterns.
    Stress-test types (3 types) are theoretically-motivated patterns included for
    completeness but evaluated separately from the main benchmark.
    """

    # Tier 1: Easy (detectable by simple API lookup)
    FABRICATED_DOI = "fabricated_doi"
    NONEXISTENT_VENUE = "nonexistent_venue"
    PLACEHOLDER_AUTHORS = "placeholder_authors"
    FUTURE_DATE = "future_date"

    # Tier 2: Medium (requires cross-referencing metadata)
    CHIMERIC_TITLE = "chimeric_title"
    WRONG_VENUE = "wrong_venue"
    AUTHOR_MISMATCH = "swapped_authors"  # Covers swapped and fabricated author names
    PREPRINT_AS_PUBLISHED = "preprint_as_published"
    HYBRID_FABRICATION = "hybrid_fabrication"

    # Tier 3: Hard (requires deep verification / semantic reasoning)
    NEAR_MISS_TITLE = "near_miss_title"
    PLAUSIBLE_FABRICATION = "plausible_fabrication"

    # Stress-test types (theoretically-motivated, evaluated separately)
    MERGED_CITATION = "merged_citation"
    PARTIAL_AUTHOR_LIST = "partial_author_list"
    ARXIV_VERSION_MISMATCH = "arxiv_version_mismatch"


class DifficultyTier(int, Enum):
    """Difficulty tiers for hallucinated entries."""

    EASY = 1
    MEDIUM = 2
    HARD = 3


class GenerationMethod(str, Enum):
    """How a benchmark entry was created."""

    SCRAPED = "scraped"  # Valid entry scraped from proceedings
    LLM_GENERATED = "llm_generated"  # LLM-generated hallucination
    PERTURBATION = "perturbation"  # Systematic perturbation of valid entry
    ADVERSARIAL = "adversarial"  # Hand-crafted to fool specific strategies
    REAL_WORLD = "real_world"  # Actual hallucination from published papers


# Mapping from hallucination type to difficulty tier
HALLUCINATION_TIER_MAP: dict[HallucinationType, DifficultyTier] = {
    HallucinationType.FABRICATED_DOI: DifficultyTier.EASY,
    HallucinationType.NONEXISTENT_VENUE: DifficultyTier.EASY,
    HallucinationType.PLACEHOLDER_AUTHORS: DifficultyTier.EASY,
    HallucinationType.FUTURE_DATE: DifficultyTier.EASY,
    HallucinationType.CHIMERIC_TITLE: DifficultyTier.MEDIUM,
    HallucinationType.WRONG_VENUE: DifficultyTier.MEDIUM,
    HallucinationType.AUTHOR_MISMATCH: DifficultyTier.MEDIUM,
    HallucinationType.PREPRINT_AS_PUBLISHED: DifficultyTier.MEDIUM,
    HallucinationType.HYBRID_FABRICATION: DifficultyTier.MEDIUM,
    HallucinationType.MERGED_CITATION: DifficultyTier.MEDIUM,
    HallucinationType.PARTIAL_AUTHOR_LIST: DifficultyTier.MEDIUM,
    HallucinationType.NEAR_MISS_TITLE: DifficultyTier.HARD,
    HallucinationType.PLAUSIBLE_FABRICATION: DifficultyTier.HARD,
    HallucinationType.ARXIV_VERSION_MISMATCH: DifficultyTier.HARD,
}

# Stress-test types: theoretically-motivated, evaluated in separate split
STRESS_TEST_TYPES: set[HallucinationType] = {
    HallucinationType.MERGED_CITATION,
    HallucinationType.PARTIAL_AUTHOR_LIST,
    HallucinationType.ARXIV_VERSION_MISMATCH,
}

# Main taxonomy types: empirically-grounded, used in primary evaluation
MAIN_TYPES: set[HallucinationType] = set(HallucinationType) - STRESS_TEST_TYPES


# Standard sub-test names
SUBTEST_NAMES = [
    "doi_resolves",
    "title_exists",
    "authors_match",
    "venue_real",
    "fields_complete",
    "cross_db_agreement",
]


# Default sub-test values for verified entries (all checks pass)
VALID_SUBTESTS: dict[str, bool | None] = {
    "doi_resolves": True,
    "title_exists": True,
    "authors_match": True,
    "venue_real": True,
    "fields_complete": True,
    "cross_db_agreement": True,
}


@dataclass
class BenchmarkEntry:
    """A single benchmark entry (valid or hallucinated BibTeX reference).

    Each entry is an atomic test unit with ground truth annotations
    and per-field sub-test labels (HumanEval-inspired multi-criteria).
    """

    # BibTeX content
    bibtex_key: str
    bibtex_type: str  # article, inproceedings, book, misc
    fields: dict[str, str]  # title, author, year, doi, url, booktitle, journal, ...

    # Ground truth
    label: Literal["VALID", "HALLUCINATED"]
    hallucination_type: str | None = None  # from HallucinationType enum
    difficulty_tier: int | None = None  # 1, 2, 3
    explanation: str = ""  # what's wrong (or "valid entry")

    # Metadata
    generation_method: str = "scraped"  # from GenerationMethod enum
    source_conference: str | None = None  # original venue for valid entries
    source: str | None = None  # provenance: where this entry came from
    publication_date: str = ""  # ISO date (YYYY-MM-DD)
    added_to_benchmark: str = ""  # ISO date (YYYY-MM-DD)

    # Sub-test ground truth (HumanEval-inspired)
    # Three-valued: True (verified pass), False (verified fail), None (not applicable)
    # e.g., doi_resolves=None when entry has no DOI field
    subtests: dict[str, bool | None] = field(default_factory=dict)

    # Optional: the raw BibTeX string
    raw_bibtex: str | None = None

    def __post_init__(self) -> None:
        """Validate entry after creation."""
        if self.label == "HALLUCINATED":
            if self.hallucination_type is None:
                raise ValueError("Hallucinated entries must have a hallucination_type")
            if self.difficulty_tier is None:
                raise ValueError("Hallucinated entries must have a difficulty_tier")
        if self.label == "VALID" and self.hallucination_type is not None:
            raise ValueError("Valid entries must not have a hallucination_type")

    @property
    def title(self) -> str:
        return self.fields.get("title", "")

    @property
    def author(self) -> str:
        return self.fields.get("author", "")

    @property
    def year(self) -> str:
        return self.fields.get("year", "")

    @property
    def doi(self) -> str | None:
        return self.fields.get("doi")

    @property
    def venue(self) -> str:
        return self.fields.get("booktitle", "") or self.fields.get("journal", "")

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON output."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON line."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> BenchmarkEntry:
        """Deserialize from dictionary, ignoring unknown fields."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, line: str) -> BenchmarkEntry:
        """Deserialize from JSON line."""
        return cls.from_dict(json.loads(line))

    def to_bibtex(self) -> str:
        """Reconstruct BibTeX string from fields."""
        if self.raw_bibtex:
            return self.raw_bibtex
        lines = [f"@{self.bibtex_type}{{{self.bibtex_key},"]
        for key, value in self.fields.items():
            lines.append(f"  {key} = {{{value}}},")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class Prediction:
    """A tool's prediction for a single benchmark entry.

    Tools produce these when evaluating BibTeX entries.
    """

    bibtex_key: str
    label: Literal["VALID", "HALLUCINATED", "UNCERTAIN"]
    confidence: float  # [0, 1]
    reason: str = ""  # free-text explanation
    subtest_results: dict[str, bool | None] = field(default_factory=dict)
    api_sources_queried: list[str] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    api_calls: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> Prediction:
        return cls(**data)

    @classmethod
    def from_json(cls, line: str) -> Prediction:
        return cls.from_dict(json.loads(line))


@dataclass
class EvaluationResult:
    """Aggregated evaluation result for a tool on a benchmark split."""

    tool_name: str
    split_name: str
    num_entries: int
    num_hallucinated: int
    num_valid: int

    # Primary metrics
    detection_rate: float  # Recall on HALLUCINATED
    false_positive_rate: float  # Valid entries incorrectly flagged
    f1_hallucination: float  # F1 on HALLUCINATED class
    tier_weighted_f1: float  # F1 weighted by difficulty tier

    # Secondary metrics
    detect_at_k: dict[int, float] = field(default_factory=dict)  # k -> fraction detected
    temporal_robustness: float | None = None
    cost_efficiency: float | None = None  # entries per second
    mean_api_calls: float | None = None
    ece: float | None = None  # Expected Calibration Error
    auroc: float | None = None  # Area Under ROC Curve
    auprc: float | None = None  # Area Under Precision-Recall Curve
    num_uncertain: int = 0  # Count of UNCERTAIN predictions

    # Confidence intervals (95% by default, via stratified bootstrap)
    detection_rate_ci: tuple[float, float] | None = None
    f1_hallucination_ci: tuple[float, float] | None = None
    tier_weighted_f1_ci: tuple[float, float] | None = None
    fpr_ci: tuple[float, float] | None = None
    ece_ci: tuple[float, float] | None = None

    # Per-tier breakdown
    per_tier_metrics: dict[int, dict[str, float]] = field(default_factory=dict)

    # Per-type breakdown
    per_type_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> EvaluationResult:
        """Deserialize from dictionary.

        Coerces JSON string keys back to int for per_tier_metrics and detect_at_k.
        """
        data = dict(data)  # shallow copy to avoid mutating caller's dict
        # JSON serializes int keys as strings — coerce them back
        if "per_tier_metrics" in data and data["per_tier_metrics"] is not None:
            data["per_tier_metrics"] = {int(k): v for k, v in data["per_tier_metrics"].items()}
        if "detect_at_k" in data and data["detect_at_k"] is not None:
            data["detect_at_k"] = {int(k): v for k, v in data["detect_at_k"].items()}
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> EvaluationResult:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(text))


def load_entries(path: str | Path) -> list[BenchmarkEntry]:
    """Load benchmark entries from a JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(BenchmarkEntry.from_json(line))
    return entries


def save_entries(entries: list[BenchmarkEntry], path: str | Path) -> None:
    """Save benchmark entries to a JSONL file."""
    from collections import Counter

    keys = [e.bibtex_key for e in entries]
    duplicates = [k for k, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate bibtex_keys found: {duplicates[:10]}")
    with open(path, "w") as f:
        for entry in entries:
            f.write(entry.to_json() + "\n")


def load_predictions(path: str | Path) -> list[Prediction]:
    """Load predictions from a JSONL file."""
    predictions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(Prediction.from_json(line))
    return predictions


def save_predictions(predictions: list[Prediction], path: str | Path) -> None:
    """Save predictions to a JSONL file."""
    with open(path, "w") as f:
        for pred in predictions:
            f.write(pred.to_json() + "\n")
