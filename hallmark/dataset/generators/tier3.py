from __future__ import annotations

import random

from hallmark.dataset.schema import (
    EXPECTED_SUBTESTS,
    BenchmarkEntry,
    DifficultyTier,
    GenerationMethod,
    HallucinationType,
)

from ._helpers import _clone_entry
from ._pools import (
    NEAR_MISS_ABBREVIATIONS,
    NEAR_MISS_ADJ_SYNONYMS,
    NEAR_MISS_NOUN_SYNONYMS,
    NEAR_MISS_PREP_SYNONYMS,
    NEAR_MISS_SAFE_PLURAL_ROOTS,
    PLAUSIBLE_DOMAINS,
    PLAUSIBLE_FIRST_NAMES,
    PLAUSIBLE_LAST_NAMES,
    PLAUSIBLE_METHODS,
    PLAUSIBLE_NOUNS,
    PLAUSIBLE_PROPERTIES,
    PLAUSIBLE_SETTINGS,
    VALID_VENUES,
)
from ._registry import register_generator


@register_generator(
    HallucinationType.NEAR_MISS_TITLE,
    description="Title off by 1-2 words from the real paper",
)
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

    replacements: dict[str, str] = {}
    replacements.update(NEAR_MISS_NOUN_SYNONYMS)
    replacements.update(NEAR_MISS_ADJ_SYNONYMS)
    replacements.update(NEAR_MISS_PREP_SYNONYMS)

    # Build reverse map (expansion -> abbreviation)
    _expansions = {v.lower(): k for k, v in NEAR_MISS_ABBREVIATIONS.items()}

    new_title = title
    if len(words) < 3:
        # Short titles: all normal strategies require >=3 words.
        # Guarantee a change by appending or replacing with a plausible ML word.
        _fallback_words = ["Revisited", "Improved", "Extended", "Revisited", "Unified"]
        fallback = rng.choice(_fallback_words)
        if len(words) == 0:
            new_title = fallback
        elif len(words) == 1:
            # Replace the single word to ensure the title actually changes
            new_title = f"{words[0]} {fallback}"
        else:
            # 2 words: replace the last word with the fallback
            new_title = f"{words[0]} {fallback}"
    elif len(words) >= 3:
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
                if lower.endswith("s") and lower[:-1] in NEAR_MISS_SAFE_PLURAL_ROOTS:
                    words[idx] = stripped[:-1] + suffix
                    new_title = " ".join(words)
                    break
                elif lower in NEAR_MISS_SAFE_PLURAL_ROOTS:
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
                for abbr, expansion in NEAR_MISS_ABBREVIATIONS.items():
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
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.NEAR_MISS_TITLE.value
    new_entry.difficulty_tier = DifficultyTier.HARD.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Title slightly modified: '{new_title}' (original: '{title}')"
    subtests = dict(EXPECTED_SUBTESTS[HallucinationType.NEAR_MISS_TITLE])
    subtests["doi_resolves"] = True if has_doi else None  # DOI resolves to original (correct) paper
    new_entry.subtests = subtests
    new_entry.bibtex_key = f"near_miss_{new_entry.bibtex_key}"
    return new_entry


@register_generator(
    HallucinationType.PLAUSIBLE_FABRICATION,
    description="Fabricate a realistic but non-existent paper at a real prestigious venue",
)
def generate_plausible_fabrication(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
    """Tier 3: Fabricate a realistic but non-existent paper at a real prestigious venue."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)

    m, d, p = (
        rng.choice(PLAUSIBLE_METHODS),
        rng.choice(PLAUSIBLE_DOMAINS),
        rng.choice(PLAUSIBLE_PROPERTIES),
    )
    n, s = rng.choice(PLAUSIBLE_NOUNS), rng.choice(PLAUSIBLE_SETTINGS)

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
            f"Unifying {m} and {rng.choice(PLAUSIBLE_METHODS)} for {d}",
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
            f"Bridging {m} and {rng.choice(PLAUSIBLE_METHODS)} for {d}",
            # "Revisiting..."
            f"Revisiting {m} for {p} {d}",
            # Simple clean pattern
            f"{m} in {s}: Applications to {d}",
        ]
    )
    new_entry.fields["title"] = title_template

    # Generate 3-5 unique authors
    n_authors = rng.randint(3, 5)
    first_pool = list(PLAUSIBLE_FIRST_NAMES)
    last_pool = list(PLAUSIBLE_LAST_NAMES)
    rng.shuffle(first_pool)
    rng.shuffle(last_pool)
    authors = [f"{first_pool[i]} {last_pool[i]}" for i in range(n_authors)]
    new_entry.fields["author"] = " and ".join(authors)

    # Use a venue from the valid set to avoid venue-oracle detectability
    # Always use booktitle (normalized to inproceedings per P0.2)
    new_entry.fields["booktitle"] = rng.choice(VALID_VENUES)
    new_entry.fields.pop("journal", None)  # Remove journal if present
    new_entry.bibtex_type = "inproceedings"

    # Remove DOI (fabricated paper won't have one)
    new_entry.fields.pop("doi", None)

    # DOI was just removed above; URL is stripped by _clone_entry — so no identifier exists.
    has_identifier = False
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.PLAUSIBLE_FABRICATION.value
    new_entry.difficulty_tier = DifficultyTier.HARD.value
    new_entry.generation_method = GenerationMethod.ADVERSARIAL.value
    new_entry.explanation = "Completely fabricated paper with plausible metadata at real venue"
    subtests = dict(EXPECTED_SUBTESTS[HallucinationType.PLAUSIBLE_FABRICATION])
    subtests["fields_complete"] = has_identifier
    new_entry.subtests = subtests
    new_entry.bibtex_key = f"plausible_{new_entry.bibtex_key}"
    return new_entry


@register_generator(
    HallucinationType.ARXIV_VERSION_MISMATCH,
    extra_args=("wrong_venue",),
    description="Mix arXiv preprint metadata with conference publication metadata",
)
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
    subtests = dict(EXPECTED_SUBTESTS[HallucinationType.ARXIV_VERSION_MISMATCH])
    subtests["doi_resolves"] = (
        True if has_doi else None
    )  # DOI resolves to the real paper; N/A if no DOI
    new_entry.subtests = subtests
    new_entry.bibtex_key = f"arxiv_vm_{new_entry.bibtex_key}"
    return new_entry
