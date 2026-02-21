"""Core data schema for HALLMARK benchmark entries and predictions."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


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
    "venue_correct",
    "fields_complete",
    "cross_db_agreement",
]


# Default sub-test values for verified entries (all checks pass)
VALID_SUBTESTS: dict[str, bool | None] = {
    "doi_resolves": True,
    "title_exists": True,
    "authors_match": True,
    "venue_correct": True,
    "fields_complete": True,
    "cross_db_agreement": True,
}


@dataclass
class BlindEntry:
    """A privacy-preserving view of a BenchmarkEntry for baseline runners.

    Contains only the fields baselines need to perform verification.
    Ground-truth labels, difficulty tiers, and generation metadata are
    intentionally excluded to prevent baselines from accessing oracle information.
    """

    bibtex_key: str
    bibtex_type: str
    fields: dict[str, str]
    raw_bibtex: str | None = None

    def __post_init__(self) -> None:
        self.fields = dict(self.fields)

    def to_bibtex(self) -> str:
        """Reconstruct BibTeX string from fields."""
        if self.raw_bibtex:
            return self.raw_bibtex
        lines = [f"@{self.bibtex_type}{{{self.bibtex_key},"]
        for key, value in sorted(self.fields.items()):
            lines.append(f"  {key} = {{{value}}},")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class BenchmarkEntry:
    """A single benchmark entry (valid or hallucinated BibTeX reference).

    Each entry is an atomic test unit with ground truth annotations
    and per-field sub-test labels (HumanEval-inspired multi-criteria).

    Note on ``difficulty_tier``: VALID entries have ``difficulty_tier=None``
    because difficulty is only meaningful for hallucinated entries. Code that
    computes per-tier metrics (e.g. ``per_tier_metrics`` in evaluation) assigns
    VALID entries to tier 1 by default so they contribute to every tier's
    false-positive rate calculation.
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
        self.fields = dict(self.fields)
        if self.label not in ("VALID", "HALLUCINATED"):
            raise ValueError(
                f"Invalid BenchmarkEntry label: {self.label!r}. Must be 'VALID' or 'HALLUCINATED'."
            )
        if self.label == "HALLUCINATED":
            if self.hallucination_type is None:
                raise ValueError("Hallucinated entries must have a hallucination_type")
            if self.difficulty_tier is None:
                raise ValueError("Hallucinated entries must have a difficulty_tier")
            if self.hallucination_type is not None:
                valid_values = {t.value for t in HallucinationType}
                if self.hallucination_type not in valid_values:
                    logger.warning(
                        "Unknown hallucination_type %r for entry %r; "
                        "not in HallucinationType enum. Continuing for forward compatibility.",
                        self.hallucination_type,
                        self.bibtex_key,
                    )
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
        data = dict(data)  # shallow copy to avoid mutating caller's dict
        # Backward compatibility: rename venue_real -> venue_correct
        if "venue_real" in data and "venue_correct" not in data:
            data["venue_correct"] = data.pop("venue_real")
        elif "venue_real" in data:
            del data["venue_real"]
        # Normalize: VALID entries must not carry a hallucination_type
        if data.get("label") == "VALID":
            data.pop("hallucination_type", None)
            data.pop("difficulty_tier", None)
        known = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - known
        if unknown:
            logger.debug("Ignoring unknown fields in %s: %s", cls.__name__, unknown)
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
        for key, value in sorted(self.fields.items()):
            lines.append(f"  {key} = {{{value}}},")
        lines.append("}")
        return "\n".join(lines)

    def to_blind(self) -> BlindEntry:
        """Convert to a BlindEntry that hides ground-truth labels from baselines."""
        return BlindEntry(
            bibtex_key=self.bibtex_key,
            bibtex_type=self.bibtex_type,
            fields=dict(self.fields),
            raw_bibtex=self.raw_bibtex,
        )


@dataclass
class Prediction:
    """A tool's prediction for a single benchmark entry.

    Tools produce these when evaluating BibTeX entries.

    The ``confidence`` field represents P(predicted label is correct):
    a tool predicting HALLUCINATED with confidence 0.9 claims 90%
    certainty that the entry is hallucinated. Equivalently, a VALID
    prediction with confidence 0.8 claims 80% certainty the entry is
    valid. This convention is used consistently across ECE computation,
    AUROC scoring, and calibration analysis.
    """

    bibtex_key: str
    label: Literal["VALID", "HALLUCINATED", "UNCERTAIN"]
    confidence: float  # [0, 1]
    reason: str = ""  # free-text explanation
    subtest_results: dict[str, bool | None] = field(default_factory=dict)
    api_sources_queried: list[str] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    api_calls: int = 0
    source: str | None = None  # "tool", "prescreening", "prescreening_override", or None

    def __post_init__(self) -> None:
        _VALID_LABELS = {"VALID", "HALLUCINATED", "UNCERTAIN"}
        if self.label not in _VALID_LABELS:
            raise ValueError(
                f"Invalid Prediction label: {self.label!r}. Must be one of {_VALID_LABELS}."
            )
        if math.isnan(self.confidence) or math.isinf(self.confidence):
            raise ValueError(f"Confidence must be a finite number, got {self.confidence}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> Prediction:
        """Deserialize from dictionary, ignoring unknown fields."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

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
    false_positive_rate: float | None  # Valid entries incorrectly flagged; None when num_valid==0
    f1_hallucination: float  # F1 on HALLUCINATED class
    tier_weighted_f1: float  # F1 weighted by difficulty tier

    # Prevalence-invariant metrics
    mcc: float | None = None  # Matthews Correlation Coefficient
    macro_f1: float | None = None  # Macro-averaged F1 across both classes

    # Secondary metrics
    union_recall_at_k: dict[int, float] = field(default_factory=dict)  # k -> fraction detected
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
    mcc_ci: tuple[float, float] | None = None

    # Per-tier breakdown
    per_tier_metrics: dict[int, dict[str, float]] = field(default_factory=dict)

    # Per-type breakdown
    per_type_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Coverage metrics
    coverage: float = 1.0  # fraction of entries with predictions
    coverage_adjusted_f1: float = 0.0  # F1 * coverage, penalizes selective abstention

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> EvaluationResult:
        """Deserialize from dictionary.

        Coerces JSON string keys back to int for per_tier_metrics and union_recall_at_k.
        Accepts legacy ``detect_at_k`` key for backward compatibility.
        """
        data = dict(data)  # shallow copy to avoid mutating caller's dict
        # Backward compat: accept legacy detect_at_k key
        if "detect_at_k" in data and "union_recall_at_k" not in data:
            data["union_recall_at_k"] = data.pop("detect_at_k")
        elif "detect_at_k" in data:
            del data["detect_at_k"]
        # JSON serializes int keys as strings — coerce them back
        if "per_tier_metrics" in data and data["per_tier_metrics"] is not None:
            data["per_tier_metrics"] = {int(k): v for k, v in data["per_tier_metrics"].items()}
        if "union_recall_at_k" in data and data["union_recall_at_k"] is not None:
            data["union_recall_at_k"] = {int(k): v for k, v in data["union_recall_at_k"].items()}
        # JSON round-trips tuples as lists — coerce CI fields back to tuples
        ci_fields = (
            "detection_rate_ci",
            "fpr_ci",
            "f1_hallucination_ci",
            "tier_weighted_f1_ci",
            "ece_ci",
            "mcc_ci",
        )
        for ci_field in ci_fields:
            if ci_field in data and isinstance(data[ci_field], list):
                data[ci_field] = tuple(data[ci_field])
        known = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - known
        if unknown:
            logger.debug("Ignoring unknown fields in %s: %s", cls.__name__, unknown)
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)

    def summary(self) -> dict[str, float | None]:
        """Return the primary metrics as a flat dict."""
        return {
            "detection_rate": self.detection_rate,
            "fpr": self.false_positive_rate,
            "f1_hallucination": self.f1_hallucination,
            "tier_weighted_f1": self.tier_weighted_f1,
            "ece": self.ece,
            "mcc": self.mcc,
            "coverage": self.coverage,
            "coverage_adjusted_f1": self.coverage_adjusted_f1,
        }

    @classmethod
    def from_json(cls, text: str) -> EvaluationResult:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(text))


# Canary / watermark string embedded in benchmark data files.
# If this string appears in a training corpus, it indicates data leakage.
CANARY_STRING = (
    "HALLMARK BENCHMARK DATA -- DO NOT INCLUDE IN TRAINING CORPORA -- canary GUID a]3D#f9K$mP2!xR7"
)


def is_canary_entry(entry: BenchmarkEntry) -> bool:
    """Return True if *entry* is a canary/watermark entry (not real benchmark data)."""
    return entry.bibtex_key.startswith("__canary__")


def load_entries(path: str | Path) -> list[BenchmarkEntry]:
    """Load benchmark entries from a JSONL file.

    Canary entries (bibtex_key starting with ``__canary__``) are automatically
    filtered out so they do not affect evaluation metrics.
    """
    entries = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    entries.append(BenchmarkEntry.from_json(line))
                except Exception as exc:
                    raise ValueError(f"Failed to parse entry at {path}:{lineno}: {exc}") from exc
    entries = [e for e in entries if not is_canary_entry(e)]
    seen_keys: set[str] = set()
    duplicates: list[str] = []
    for entry in entries:
        if entry.bibtex_key in seen_keys:
            duplicates.append(entry.bibtex_key)
        seen_keys.add(entry.bibtex_key)
    if duplicates:
        raise ValueError(
            f"Duplicate bibtex_key(s) found in {path}: {duplicates[:10]}. "
            "Fix the data file before loading."
        )
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
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    predictions.append(Prediction.from_json(line))
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse prediction at {path}:{lineno}: {exc}"
                    ) from exc
    return predictions


def save_predictions(predictions: list[Prediction], path: str | Path) -> None:
    """Save predictions to a JSONL file."""
    with open(path, "w") as f:
        for pred in predictions:
            f.write(pred.to_json() + "\n")
