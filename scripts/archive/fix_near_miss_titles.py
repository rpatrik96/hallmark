"""Fix near_miss_title entries: apply grammatically correct mutations.

Strategy: parse the original title from the explanation field, then apply a
part-of-speech-safe synonym substitution. Never swap adjacent words (breaks
grammar). Never replace a noun with an adjective.
"""

import json
import re
from pathlib import Path

# Manual overrides for titles that can't be auto-mutated
MANUAL_OVERRIDES: dict[str, str] = {
    "ByPE-VAE: Bayesian Pseudocoresets Exemplar VAE": (
        "ByPE-VAE: Bayesian Pseudocoreset Exemplar VAE"
    ),
    "GraPPa: Grammar-Augmented Pre-Training for Table Semantic Parsing": (
        "GraPPa: Grammar-Augmented Pre-Training for Table Semantic Analysis"
    ),
    "Multi-agent Online Scheduling: MMS Allocations for Indivisible Items": (
        "Multi-agent Online Scheduling: MMS Allocation for Indivisible Items"
    ),
    "Data-Driven Multimodal Patrol Planning for Anti-poaching": (
        "Data-Driven Multimodal Patrol Scheduling for Anti-poaching"
    ),
    # Fix article agreement: "An Exact" -> "An Precise" is wrong
    "An Exact Characterization of the Generalization Error for the Gibbs Algorithm": (
        "A Precise Characterization of the Generalization Error for the Gibbs Algorithm"
    ),
}

# Same-POS synonym pairs: each pair can be swapped without breaking grammar.
# Grouped by part-of-speech category.
NOUN_SYNONYMS: dict[str, str] = {
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
    "optimization": "minimization",
    "minimization": "optimization",
    "inference": "reasoning",
    "reasoning": "inference",
    "attention": "self-attention",
    "self-attention": "attention",
    "segmentation": "partitioning",
    "partitioning": "segmentation",
    "regularization": "penalization",
    "penalization": "regularization",
    "convergence": "stability",
    "stability": "convergence",
    "performance": "accuracy",
    "accuracy": "performance",
    "evaluation": "assessment",
    "assessment": "evaluation",
    "dataset": "benchmark",
    "benchmark": "dataset",
    "algorithm": "procedure",
    "procedure": "algorithm",
    "distribution": "density",
    "density": "distribution",
    "features": "representations",
    "representations": "features",
    "trees": "forests",
    "forests": "trees",
    "bounds": "guarantees",
    "guarantees": "bounds",
    "images": "photographs",
    "loss": "objective",
    "objective": "loss",
    "task": "problem",
    "problem": "task",
    "survey": "review",
    "review": "survey",
    "language": "text",
    "score": "metric",
    "metric": "score",
    "constraint": "restriction",
    "restriction": "constraint",
    "bottleneck": "limitation",
    "limitation": "bottleneck",
    "corruption": "perturbation",
    "perturbation": "corruption",
    "attacks": "perturbations",
    "perturbations": "attacks",
    "design": "construction",
    "construction": "design",
    "search": "exploration",
    "exploration": "search",
    "planning": "scheduling",
    "scheduling": "planning",
    "parsing": "analysis",
    "items": "elements",
    "elements": "items",
    "allocation": "assignment",
    "assignment": "allocation",
    "allocations": "assignments",
    "assignments": "allocations",
}

ADJ_SYNONYMS: dict[str, str] = {
    "robust": "resilient",
    "resilient": "robust",
    "efficient": "scalable",
    "scalable": "efficient",
    "optimal": "approximate",
    "approximate": "optimal",
    "deep": "hierarchical",
    "hierarchical": "deep",
    "generative": "discriminative",
    "discriminative": "generative",
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
    "continuous": "smooth",
    "smooth": "continuous",
    "interpretable": "explainable",
    "explainable": "interpretable",
    "complete": "comprehensive",
    "comprehensive": "thorough",
    "thorough": "comprehensive",
    "novel": "new",
    "new": "novel",
    "limited": "restricted",
    "restricted": "limited",
    "unsupervised": "self-supervised",
    "self-supervised": "unsupervised",
    "inverse": "reverse",
    "reverse": "inverse",
    "fast": "rapid",
    "rapid": "fast",
    "causal": "structural",
    "structural": "causal",
    "latent": "hidden",
    "hidden": "latent",
    "exact": "precise",
    "precise": "exact",
    "biological": "biologically-inspired",
    "bayesian": "probabilistic",
    "probabilistic": "bayesian",
    "indivisible": "atomic",
    "multimodal": "multi-modal",
    "multi-modal": "multimodal",
}

PREP_SYNONYMS: dict[str, str] = {
    "via": "through",
    "through": "via",
    "using": "with",
    "towards": "for",
}

# Merge all synonym dicts
ALL_SYNONYMS: dict[str, str] = {}
ALL_SYNONYMS.update(NOUN_SYNONYMS)
ALL_SYNONYMS.update(ADJ_SYNONYMS)
ALL_SYNONYMS.update(PREP_SYNONYMS)


def apply_synonym_mutation(title: str) -> str | None:
    """Apply a single synonym substitution preserving grammar.

    Returns mutated title or None if no substitution found.
    """
    words = title.split()
    for idx in range(len(words)):
        # Strip trailing punctuation for matching
        raw = words[idx]
        stripped = raw.rstrip(".,;:!?")
        suffix = raw[len(stripped) :]
        key = stripped.lower()

        if key in ALL_SYNONYMS:
            replacement = ALL_SYNONYMS[key]
            # Preserve capitalization
            if stripped[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            if stripped.isupper() and len(stripped) > 1:
                replacement = replacement.upper()
            words[idx] = replacement + suffix
            return " ".join(words)

    return None


def apply_plural_mutation(title: str) -> str | None:
    """Flip singular/plural on a noun. Only applies to safe words."""
    words = title.split()
    # Words that are safe to pluralize/singularize
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
        "example",
        "prediction",
        "sample",
        "parameter",
        "gradient",
        "surface",
        "module",
        "component",
        "dimension",
        "algorithm",
        "graph",
        "function",
        "variable",
        "kernel",
        "layer",
        "node",
        "weight",
        "dataset",
        "task",
        "objective",
        "problem",
        "allocation",
        "item",
        "element",
    }
    for idx in range(len(words)):
        raw = words[idx]
        stripped = raw.rstrip(".,;:!?")
        suffix = raw[len(stripped) :]
        lower = stripped.lower()

        if lower.endswith("s") and lower[:-1] in safe_plural_roots:
            # Singularize
            new_word = stripped[:-1]
            words[idx] = new_word + suffix
            return " ".join(words)
        elif lower in safe_plural_roots:
            # Pluralize
            new_word = stripped + "s"
            words[idx] = new_word + suffix
            return " ".join(words)

    return None


def apply_article_mutation(title: str) -> str | None:
    """Insert or remove an article. Only when grammatically safe."""
    words = title.split()
    articles = {"a", "an", "the"}

    # Try to find and remove an existing article (except at start of title)
    for idx in range(1, len(words)):
        if words[idx].lower() in articles:
            removed = words[:idx] + words[idx + 1 :]
            return " ".join(removed)

    return None


def mutate_title(original_title: str) -> tuple[str, str]:
    """Apply a grammatically safe mutation. Returns (new_title, strategy)."""
    # Check manual overrides first
    if original_title in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[original_title], "manual"

    # Try synonym first (most reliable)
    result = apply_synonym_mutation(original_title)
    if result and result != original_title:
        return result, "synonym"

    # Try plural/singular flip
    result = apply_plural_mutation(original_title)
    if result and result != original_title:
        return result, "plural"

    # Try article removal
    result = apply_article_mutation(original_title)
    if result and result != original_title:
        return result, "article"

    # Last resort: British/American spelling
    if "ization" in original_title:
        return original_title.replace("ization", "isation", 1), "spelling"
    if "isation" in original_title:
        return original_title.replace("isation", "ization", 1), "spelling"

    # If nothing else works, swap "and" with "&" or vice versa
    if " and " in original_title:
        return original_title.replace(" and ", " & ", 1), "conjunction"
    if " & " in original_title:
        return original_title.replace(" & ", " and ", 1), "conjunction"

    # Absolute fallback: just return with a note
    print(f"  WARNING: could not mutate: {original_title!r}")
    return original_title, "none"


def extract_original_title(explanation: str) -> str | None:
    """Extract original title from explanation field."""
    match = re.search(r"\(original: '(.+?)'\)$", explanation)
    if match:
        return match.group(1)
    return None


def fix_file(filepath: Path) -> None:
    with open(filepath) as f:
        entries = [json.loads(line) for line in f]

    nm_count = 0
    for entry in entries:
        if entry.get("hallucination_type") != "near_miss_title":
            continue

        # Get original title from explanation
        original = extract_original_title(entry["explanation"])
        if not original:
            print(f"  WARNING: cannot parse original from: {entry['explanation']!r}")
            continue

        # Apply grammatically safe mutation
        new_title, _strategy = mutate_title(original)

        if new_title == original:
            print(f"  UNCHANGED: {original!r}")
            continue

        entry["fields"]["title"] = new_title
        entry["explanation"] = f"Title slightly modified: '{new_title}' (original: '{original}')"
        nm_count += 1

    with open(filepath, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Fixed {nm_count} near_miss_title entries in {filepath}")


def main() -> None:
    data_dir = Path("data/v1.0")
    fix_file(data_dir / "dev_public.jsonl")
    fix_file(data_dir / "test_public.jsonl")

    # Verify: check all 30 entries are mutated in each file
    for name in ["dev_public.jsonl", "test_public.jsonl"]:
        path = data_dir / name
        with open(path) as f:
            entries = [json.loads(line) for line in f]
        nm = [e for e in entries if e.get("hallucination_type") == "near_miss_title"]

        # Print all for review
        print(f"\n--- {name} near_miss_title entries ---")
        unchanged = 0
        for i, e in enumerate(nm):
            orig = extract_original_title(e["explanation"])
            title = e["fields"]["title"]
            marker = " [SAME!]" if title == orig else ""
            print(f"  {i}: {title!r}{marker}")
            print(f"      original: {orig!r}")
            if title == orig:
                unchanged += 1
        if unchanged:
            print(f"  ERROR: {unchanged} entries unchanged!")
        else:
            print(f"  All {len(nm)} entries successfully mutated.")


if __name__ == "__main__":
    main()
