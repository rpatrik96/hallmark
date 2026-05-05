"""Parse GPTZero NeurIPS 2025 CSV and create HALLMARK benchmark entries.  [data-collection]

Reads the GPTZero analysis CSV containing hallucinated citations found in NeurIPS 2025
papers and converts them to BenchmarkEntry objects with proper classification.
"""

import csv
import re
from pathlib import Path

from hallmark.contribution.validate_entry import validate_entry
from hallmark.dataset.schema import (
    HALLUCINATION_TIER_MAP,
    BenchmarkEntry,
    GenerationMethod,
    HallucinationType,
)


def parse_citation(citation_text: str) -> dict[str, str]:
    """Extract BibTeX fields from citation text.

    Args:
        citation_text: Raw citation string from CSV

    Returns:
        Dictionary with title, author, year, venue/journal, doi, etc.
    """
    fields = {}

    # Extract authors (usually at the beginning, before title)
    # Patterns: "Name1 and Name2." or "Name1, Name2, and Name3."
    author_match = re.match(r"^([^.]+?\.)(?:\s+(.+))", citation_text)
    if author_match:
        authors_raw = author_match.group(1).rstrip(".")
        fields["author"] = authors_raw
        remaining = citation_text[len(author_match.group(1)) :].strip()
    else:
        # Sometimes authors aren't clearly delimited
        remaining = citation_text
        fields["author"] = "Unknown"

    # Extract title (usually in quotes or italics, or between author and venue)
    # Look for patterns like: Title. Journal/Conference, ...
    title_match = re.search(r'"([^"]+)"', remaining)
    if title_match:
        fields["title"] = title_match.group(1)
    else:
        # Try to find title before venue indicators
        title_match2 = re.search(
            r"^([^.]+)\.\s+(?:In |arXiv|IEEE|Proceedings|Journal|ACM|Nature)",
            remaining,
            re.IGNORECASE,
        )
        if title_match2:
            fields["title"] = title_match2.group(1).strip()
        else:
            # Fallback: take first sentence
            parts = remaining.split(".", 1)
            if parts:
                fields["title"] = parts[0].strip()

    # Extract year
    year_match = re.search(r"\b(19|20)\d{2}\b", citation_text)
    if year_match:
        fields["year"] = year_match.group(0)

    # Extract DOI
    doi_match = re.search(r"doi:\s*([0-9.]+/[^\s,]+)", citation_text, re.IGNORECASE)
    if doi_match:
        fields["doi"] = doi_match.group(1).rstrip(".")

    # Extract arXiv ID
    arxiv_match = re.search(r"arXiv:([0-9.]+(?:v\d+)?)", citation_text)
    if arxiv_match:
        fields["note"] = f"arXiv:{arxiv_match.group(1)}"

    # Detect venue type
    if "arXiv" in citation_text:
        fields["journal"] = "arXiv"
    elif re.search(r"IEEE|ACM|Proceedings|Conference|Workshop", citation_text, re.IGNORECASE):
        # Conference paper
        venue_pattern = (
            r"(?:In )?(?:Proceedings of )?(?:the )?"
            r"([^,]+(?:Conference|Workshop|Symposium|CVPR|ICCV|NeurIPS|ICML|ICLR|ACL|EMNLP|AAAI|IJCAI)[^,]*)"
        )
        venue_match = re.search(venue_pattern, citation_text, re.IGNORECASE)
        if venue_match:
            fields["booktitle"] = venue_match.group(1).strip()
    else:
        # Journal
        journal_match = re.search(
            r"(?:In )?([A-Z][^,]+(?:Journal|Transactions|Review|Science|Nature)[^,]*)",
            citation_text,
        )
        if journal_match:
            fields["journal"] = journal_match.group(1).strip()

    # Extract pages
    pages_match = re.search(r"(?:pp?\.|pages)\s*(\d+[-\u2013]\d+)", citation_text, re.IGNORECASE)
    if pages_match:
        fields["pages"] = pages_match.group(1).replace("\u2013", "--")

    # Extract volume/number
    vol_match = re.search(r"(\d+)\((\d+)\):\d+", citation_text)
    if vol_match:
        fields["volume"] = vol_match.group(1)
        fields["number"] = vol_match.group(2)

    return fields


def classify_hallucination(
    comment: str, fields: dict[str, str]
) -> tuple[HallucinationType, dict[str, bool]]:
    """Classify hallucination type based on GPTZero comment.

    Args:
        comment: Verification comment from CSV
        fields: Parsed BibTeX fields

    Returns:
        Tuple of (hallucination_type, subtests)
    """
    comment_lower = comment.lower()

    # Initialize subtests
    subtests = {
        "doi_resolves": False,
        "title_exists": False,
        "authors_match": False,
        "venue_correct": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }

    # Check for incomplete fields
    if "XXXX" in fields.get("note", "") or "to be updated" in comment_lower:
        subtests["fields_complete"] = False

    # Placeholder authors
    if any(
        phrase in comment_lower
        for phrase in [
            "authors are obviously fabricated",
            "all authors",
            "are fabricated",
            "firstname lastname",
        ]
    ) or any(
        name in fields.get("author", "")
        for name in ["John Doe", "Jane Smith", "Firstname Lastname"]
    ):
        subtests["authors_match"] = False
        return HallucinationType.PLACEHOLDER_AUTHORS, subtests

    # Fabricated DOI
    if any(
        phrase in comment_lower for phrase in ["doi is fake", "doi doesn't exist", "doi are fake"]
    ):
        subtests["doi_resolves"] = False
        return HallucinationType.FABRICATED_DOI, subtests

    # ArXiv ID issues
    if (
        "arxiv id leads to a different article" in comment_lower
        or "arxiv id is incomplete" in comment_lower
    ):
        subtests["doi_resolves"] = False
        if "incomplete" in comment_lower:
            return HallucinationType.PLACEHOLDER_AUTHORS, subtests
        return HallucinationType.FABRICATED_DOI, subtests

    # Chimeric title (real authors, fake title)
    if (
        "title" in comment_lower
        and "don't match" in comment_lower
        and "authors" in comment_lower
        and (
            "authors match" in comment_lower
            or ("authors" in comment_lower and "but" in comment_lower)
        )
    ):
        subtests["title_exists"] = False
        subtests["authors_match"] = True
        return HallucinationType.CHIMERIC_TITLE, subtests

    # Swapped/wrong authors (real title, fake/swapped authors)
    if (
        "title matches" in comment_lower
        and "authors" in comment_lower
        and ("different" in comment_lower or "fabricated" in comment_lower)
    ):
        subtests["title_exists"] = True
        subtests["authors_match"] = False
        return HallucinationType.AUTHOR_MISMATCH, subtests

    # Authors omitted/added
    if "omitted" in comment_lower or "added" in comment_lower:
        subtests["title_exists"] = True
        subtests["authors_match"] = False
        return HallucinationType.AUTHOR_MISMATCH, subtests

    # Wrong venue
    if "published at" in comment_lower and "not" in comment_lower:
        subtests["title_exists"] = True
        return HallucinationType.WRONG_VENUE, subtests

    # Near miss title
    if (
        "similar title" in comment_lower
        or "close" in comment_lower
        or ("title is" in comment_lower and "similar" in comment_lower)
    ):
        subtests["authors_match"] = True
        return HallucinationType.NEAR_MISS_TITLE, subtests

    # Hybrid fabrication (some real, some fake)
    if (
        "match" in comment_lower
        and "but" in comment_lower
        and ("don't" in comment_lower or "doesn't" in comment_lower)
        and ("authors match" in comment_lower or "title matches" in comment_lower)
    ):
        # Mixed real/fake metadata
        if fields.get("doi"):
            subtests["doi_resolves"] = True
        return HallucinationType.HYBRID_FABRICATION, subtests

    # Nonexistent venue
    if "publication doesn't exist" in comment_lower or (
        "journal" in comment_lower and "doesn't exist" in comment_lower
    ):
        subtests["venue_correct"] = False
        return HallucinationType.NONEXISTENT_VENUE, subtests

    # Complete fabrication (no matches at all)
    if (
        all(phrase in comment_lower for phrase in ["no", "match"])
        or "doesn't exist in publication" in comment_lower
    ):
        subtests["title_exists"] = False
        subtests["authors_match"] = False
        return HallucinationType.PLAUSIBLE_FABRICATION, subtests

    # Default to plausible fabrication
    return HallucinationType.PLAUSIBLE_FABRICATION, subtests


def parse_gptzero_csv(csv_path: Path) -> list[BenchmarkEntry]:
    """Parse GPTZero CSV and create benchmark entries.

    Args:
        csv_path: Path to GPTZero CSV file

    Returns:
        List of BenchmarkEntry objects
    """
    entries = []

    with open(csv_path, encoding="utf-8") as f:
        # Use csv.QUOTE_ALL to handle multiline fields
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader, start=1):
            paper_title = row["Published Paper"].strip()
            scan_type = row["GPTZero Scan"].strip().replace("\n", " ")
            citation = row["Example of Verified Hallucination"].strip()
            comment = row["Comment"].strip()

            # Skip empty rows
            if not citation or not comment:
                continue

            # Parse citation into BibTeX fields
            fields = parse_citation(citation)

            # Classify hallucination type
            hallucination_type, subtests = classify_hallucination(comment, fields)

            # Determine BibTeX type
            if "booktitle" in fields:
                bibtex_type = "inproceedings"
            elif "journal" in fields:
                bibtex_type = "article"
            else:
                bibtex_type = "misc"

            # Create explanation
            explanation = (
                f"Real-world hallucination from NeurIPS 2025 accepted paper '{paper_title}'. "
                f"Found by GPTZero analysis (scan type: {scan_type}). "
                f"Comment: {comment}. "
                f"URL: https://gptzero.me/news/neurips/"
            )

            # Create entry
            entry = BenchmarkEntry(
                bibtex_key=f"gptzero_neurips2025_{idx:03d}",
                bibtex_type=bibtex_type,
                fields=fields,
                label="HALLUCINATED",
                hallucination_type=hallucination_type.value,
                difficulty_tier=HALLUCINATION_TIER_MAP[hallucination_type].value,
                explanation=explanation,
                generation_method=GenerationMethod.REAL_WORLD.value,
                source_conference="NeurIPS",
                publication_date="",
                added_to_benchmark="2026-02-13",
                subtests=subtests,
                raw_bibtex=None,
            )

            entries.append(entry)

    return entries


def main():
    """Main entry point."""
    # Paths
    data_dir = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark/data")
    csv_path = data_dir / "Neurips Hallucinations For Public - Main results.csv"
    output_path = data_dir / "v1.0/gptzero_neurips2025.jsonl"
    existing_path = data_dir / "v1.0/real_world_incidents.jsonl"

    # Load existing entries to check for duplicates
    existing_entries = []
    if existing_path.exists():
        with open(existing_path) as f:
            for line in f:
                if line.strip():
                    existing_entries.append(BenchmarkEntry.from_json(line.strip()))

    print("[OBJECTIVE] Parse GPTZero NeurIPS 2025 CSV and create benchmark entries")
    print(f"[DATA] Loaded {len(existing_entries)} existing entries from {existing_path}")

    # Parse CSV
    new_entries = parse_gptzero_csv(csv_path)
    print(f"[DATA] Parsed {len(new_entries)} entries from CSV")

    # Validate and deduplicate
    validated_entries = []
    for entry in new_entries:
        result = validate_entry(entry, existing_entries + validated_entries)

        if not result.valid:
            print(f"[LIMITATION] Entry {entry.bibtex_key} failed validation:")
            for error in result.errors:
                print(f"  - {error}")
        else:
            if result.warnings:
                for warning in result.warnings:
                    print(f"  Warning: {warning}")
            validated_entries.append(entry)

    print(f"[FINDING] {len(validated_entries)} valid entries after validation")

    # Count by type
    type_counts: dict[str, int] = {}
    for entry in validated_entries:
        type_counts[entry.hallucination_type] = type_counts.get(entry.hallucination_type, 0) + 1

    print("[FINDING] Hallucination type distribution:")
    for htype, count in sorted(type_counts.items()):
        print(f"  - {htype}: {count}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in validated_entries:
            f.write(entry.to_json() + "\n")

    print(f"[FINDING] Wrote {len(validated_entries)} entries to {output_path}")


if __name__ == "__main__":
    main()
