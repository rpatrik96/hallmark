"""Shared helpers for the E1 contamination/recall-probe experiment.

Defines the stratified sample (VALID-only, by year, N=150, seed=42) and the
author-name normalization + Jaccard recall scoring used by the recall probe.
"""

from __future__ import annotations

import random
import re
import unicodedata
from pathlib import Path

from hallmark.dataset.schema import BenchmarkEntry, load_entries

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "results/temporal_supplement/temporal_supplement_2024_2025.jsonl"
OUT_DIR = REPO / "results/reviewer_experiments/e1_recall_probe"
SAMPLE_PATH = OUT_DIR / "sample_150.jsonl"
SAMPLE_KEYS_PATH = OUT_DIR / "sample_bibtex_keys.json"

SEED = 42
N_SAMPLE = 150
RECALL_THRESHOLD = 0.5  # Jaccard >= this => "recalled"


def stratified_valid_sample(
    entries: list[BenchmarkEntry], n: int = N_SAMPLE, seed: int = SEED
) -> list[BenchmarkEntry]:
    """Stratified-by-year sample of VALID entries.

    Allocation is proportional to each year's share of the VALID pool, with
    the largest-remainder method to make the per-year counts sum to exactly n.
    Sampling within each year stratum uses an independent seeded RNG.
    """
    valid = [e for e in entries if e.label == "VALID"]
    years = sorted({e.year for e in valid})
    by_year = {y: [e for e in valid if e.year == y] for y in years}
    total = len(valid)

    # Largest-remainder allocation.
    raw = {y: n * len(by_year[y]) / total for y in years}
    base = {y: int(raw[y]) for y in years}
    remainder = n - sum(base.values())
    # Distribute leftover slots to the years with the largest fractional parts.
    order = sorted(years, key=lambda y: raw[y] - base[y], reverse=True)
    for y in order[:remainder]:
        base[y] += 1

    sampled: list[BenchmarkEntry] = []
    for y in years:
        pool = sorted(by_year[y], key=lambda e: e.bibtex_key)  # deterministic order
        rng = random.Random(f"{seed}-{y}")
        k = min(base[y], len(pool))
        sampled.extend(rng.sample(pool, k))
    sampled.sort(key=lambda e: e.bibtex_key)
    return sampled


def load_sample() -> list[BenchmarkEntry]:
    """Load the persisted sample; build + persist it on first call."""
    if SAMPLE_PATH.exists():
        return load_entries(SAMPLE_PATH)
    entries = load_entries(DATA_PATH)
    sample = stratified_valid_sample(entries)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_PATH, "w") as f:
        for e in sample:
            f.write(e.to_json() + "\n")
    import json

    with open(SAMPLE_KEYS_PATH, "w") as f:
        json.dump([e.bibtex_key for e in sample], f, indent=2)
    return sample


# --- Author-name normalization --------------------------------------------

_DBLP_SUFFIX = re.compile(r"\s+\d{3,4}\b")  # DBLP disambiguation, e.g. "Yixuan Li 0002"


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def split_authors(author_field: str) -> list[str]:
    """Split a BibTeX ``author`` field on ' and ' into individual names."""
    if not author_field:
        return []
    parts = re.split(r"\s+and\s+", author_field.strip())
    return [p.strip() for p in parts if p.strip()]


def last_name(name: str) -> str:
    """Extract a normalized last name from a single author string.

    Handles "First Last", "Last, First", DBLP "Name 0001" suffixes, accents,
    and punctuation. Returns lowercase ASCII.
    """
    name = _DBLP_SUFFIX.sub("", name).strip()
    if not name:
        return ""
    if "," in name:
        # "Last, First" -> take the part before the first comma.
        candidate = name.split(",")[0].strip()
    else:
        # "First Middle Last" -> take the final whitespace token.
        candidate = name.split()[-1] if name.split() else name
    candidate = _strip_accents(candidate).lower()
    candidate = re.sub(r"[^a-z]", "", candidate)
    return candidate


def last_name_set(names: list[str]) -> set[str]:
    out = {last_name(n) for n in names}
    out.discard("")
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Canonical venue aliases: maps many surface forms to a single token so that a
# predicted full name matches the DBLP short form (and vice versa). Only used
# for the secondary venue-match signal; recall scoring is author-Jaccard.
_VENUE_ALIASES: dict[str, str] = {
    "neurips": "neurips",
    "nips": "neurips",
    "neuralinformationprocessingsystems": "neurips",
    "advancesinneuralinformationprocessingsystems": "neurips",
    "icml": "icml",
    "internationalconferenceonmachinelearning": "icml",
    "iclr": "iclr",
    "internationalconferenceonlearningrepresentations": "iclr",
    "cvpr": "cvpr",
    "computervisionandpatternrecognition": "cvpr",
    "ieeeconferenceoncomputervisionandpatternrecognition": "cvpr",
    "iccv": "iccv",
    "internationalconferenceoncomputervision": "iccv",
    "eccv": "eccv",
    "europeanconferenceoncomputervision": "eccv",
    "acl": "acl",
    "annualmeetingoftheassociationforcomputationallinguistics": "acl",
    "emnlp": "emnlp",
    "empiricalmethodsinnaturallanguageprocessing": "emnlp",
    "naacl": "naacl",
    "aaai": "aaai",
    "ijcai": "ijcai",
    "kdd": "kdd",
    "sigkdd": "kdd",
    "www": "www",
    "thewebconference": "www",
    "sigir": "sigir",
    "wacv": "wacv",
    "aistats": "aistats",
}


def _venue_canon(v: str) -> str:
    norm = re.sub(r"[^a-z0-9]", "", _strip_accents(v or "").lower())
    return _VENUE_ALIASES.get(norm, norm)


def venue_match(pred_venue: str, true_venue: str) -> bool:
    """Loose venue match via a canonical-alias map plus substring fallback.

    True venues here are short forms like 'NeurIPS', 'ICML', 'CVPR'. We first
    canonicalize both sides through a venue-alias table; if that fails we fall
    back to a normalized substring test in either direction.
    """
    p_raw = re.sub(r"[^a-z0-9]", "", _strip_accents(pred_venue or "").lower())
    t_raw = re.sub(r"[^a-z0-9]", "", _strip_accents(true_venue or "").lower())
    if not p_raw or not t_raw:
        return False
    if _venue_canon(pred_venue) == _venue_canon(true_venue):
        return True
    return p_raw in t_raw or t_raw in p_raw
