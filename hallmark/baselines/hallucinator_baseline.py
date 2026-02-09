"""Baseline wrapper for hallucinator (gianlucasb/hallucinator).

Hallucinator (https://github.com/gianlucasb/hallucinator) detects
potentially hallucinated references in academic PDFs by querying CrossRef,
arXiv, DBLP, Semantic Scholar, ACL Anthology, NeurIPS, Europe PMC, PubMed,
and OpenAlex.

NOTE: This tool is PDF-based, so we generate a temporary BibTeX file,
convert it to a minimal PDF-like format, or invoke its internal reference
checker directly. For best results, use the tool's web UI or CLI with
actual PDF papers.

Requires: Clone https://github.com/gianlucasb/hallucinator
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path

from hallmark.dataset.schema import BenchmarkEntry, Prediction

logger = logging.getLogger(__name__)

# Databases queried by hallucinator
API_SOURCES = [
    "crossref",
    "arxiv",
    "dblp",
    "semantic_scholar",
    "acl_anthology",
    "neurips",
    "europe_pmc",
    "pubmed",
    "openalex",
]


def _create_minimal_pdf(entries: list[BenchmarkEntry], output_path: str) -> None:
    """Create a minimal PDF with a references section from entries.

    Uses reportlab if available, otherwise falls back to a text-based approach.
    """
    try:
        from reportlab.lib.pagesizes import letter  # type: ignore[import-untyped]
        from reportlab.pdfgen import canvas  # type: ignore[import-untyped]

        c = canvas.Canvas(output_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, 750, "References")
        c.setFont("Helvetica", 10)

        y = 720
        for i, entry in enumerate(entries, 1):
            author = entry.fields.get("author", "Unknown")
            title = entry.fields.get("title", "Untitled")
            year = entry.fields.get("year", "")
            venue = entry.fields.get("booktitle", "") or entry.fields.get("journal", "")

            ref_text = f"[{i}] {author}. {title}. {venue}, {year}."
            # Wrap long lines
            while len(ref_text) > 90:
                c.drawString(72, y, ref_text[:90])
                ref_text = "    " + ref_text[90:]
                y -= 14
                if y < 72:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = 750
            c.drawString(72, y, ref_text)
            y -= 18
            if y < 72:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = 750

        c.save()
    except ImportError:
        # Fallback: write a simple text file with .pdf extension
        # hallucinator uses PyMuPDF which can handle some text-like PDFs
        logger.warning("reportlab not available; creating text-based reference file")
        with open(output_path, "w") as f:
            f.write("References\n\n")
            for i, entry in enumerate(entries, 1):
                author = entry.fields.get("author", "Unknown")
                title = entry.fields.get("title", "Untitled")
                year = entry.fields.get("year", "")
                venue = entry.fields.get("booktitle", "") or entry.fields.get("journal", "")
                f.write(f"[{i}] {author}. {title}. {venue}, {year}.\n\n")


def run_hallucinator(
    entries: list[BenchmarkEntry],
    hallucinator_path: str | Path = "hallucinator",
    openalex_key: str | None = None,
    s2_api_key: str | None = None,
) -> list[Prediction]:
    """Run hallucinator on benchmark entries.

    Requires: Clone https://github.com/gianlucasb/hallucinator and
    install dependencies (PyMuPDF, Flask, requests).

    Args:
        entries: Benchmark entries to verify.
        hallucinator_path: Path to hallucinator repository.
        openalex_key: Optional OpenAlex API key.
        s2_api_key: Optional Semantic Scholar API key.

    Returns:
        List of Predictions.
    """
    h_path = Path(hallucinator_path)
    script = h_path / "check_hallucinated_references.py"
    if not script.exists():
        raise FileNotFoundError(
            f"hallucinator not found at {h_path}. "
            "Clone from: https://github.com/gianlucasb/hallucinator"
        )

    # Create a temporary PDF with references
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    output_path = tempfile.mktemp(suffix=".txt")

    try:
        _create_minimal_pdf(entries, pdf_path)

        cmd = ["python3", str(script), pdf_path, "--output", output_path, "--no-color"]
        if openalex_key:
            cmd.extend(["--openalex-key", openalex_key])
        if s2_api_key:
            cmd.extend(["--s2-api-key", s2_api_key])

        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(h_path),
        )
        total_time = time.time() - start

        if result.returncode != 0:
            logger.error(f"hallucinator failed: {result.stderr[:500]}")
            return _fallback_predictions(entries)

        # Parse output
        return _parse_output(entries, result.stdout, output_path, total_time)

    except subprocess.TimeoutExpired:
        logger.error("hallucinator timed out")
        return _fallback_predictions(entries)
    finally:
        Path(pdf_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def _parse_output(
    entries: list[BenchmarkEntry],
    stdout: str,
    output_path: str,
    total_time: float,
) -> list[Prediction]:
    """Parse hallucinator output and match to entries by title."""
    per_entry_time = total_time / max(len(entries), 1)

    # Parse stdout for verification results
    # hallucinator reports: Verified / Author Mismatch / Not Found / Retracted
    not_found_titles: set[str] = set()
    mismatch_titles: set[str] = set()
    retracted_titles: set[str] = set()

    lines = stdout.split("\n")
    current_title = ""
    for line in lines:
        line_stripped = line.strip()
        # Look for title patterns in output
        if "NOT FOUND" in line_stripped.upper() or "not found" in line_stripped:
            if current_title:
                not_found_titles.add(current_title.lower())
        elif "MISMATCH" in line_stripped.upper() or "mismatch" in line_stripped:
            if current_title:
                mismatch_titles.add(current_title.lower())
        elif "RETRACTED" in line_stripped.upper() and current_title:
            retracted_titles.add(current_title.lower())

        # Track current reference being processed
        if line_stripped and not line_stripped.startswith(("=", "-", " ", "✓", "✗", "?")):
            current_title = line_stripped

    # Also try reading output file
    output_file = Path(output_path)
    if output_file.exists():
        try:
            content = output_file.read_text()
            for line in content.split("\n"):
                lower = line.lower()
                if "not found" in lower:
                    # Extract title if possible
                    parts = line.split(":")
                    if len(parts) > 1:
                        not_found_titles.add(parts[0].strip().lower())
        except OSError:
            pass

    # Match entries by title similarity
    predictions = []
    for entry in entries:
        entry_title = entry.fields.get("title", "").lower().strip()

        is_not_found = any(entry_title in t or t in entry_title for t in not_found_titles if t)
        is_mismatch = any(entry_title in t or t in entry_title for t in mismatch_titles if t)
        is_retracted = any(entry_title in t or t in entry_title for t in retracted_titles if t)

        if is_not_found or is_retracted:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="HALLUCINATED",
                    confidence=0.80 if is_not_found else 0.90,
                    reason=f"hallucinator: {'not found' if is_not_found else 'retracted'}",
                    api_sources_queried=API_SOURCES,
                    wall_clock_seconds=per_entry_time,
                    api_calls=len(API_SOURCES),
                )
            )
        elif is_mismatch:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="HALLUCINATED",
                    confidence=0.65,
                    reason="hallucinator: author mismatch",
                    api_sources_queried=API_SOURCES,
                    wall_clock_seconds=per_entry_time,
                    api_calls=len(API_SOURCES),
                )
            )
        else:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.80,
                    reason="hallucinator: verified or no issues found",
                    api_sources_queried=API_SOURCES,
                    wall_clock_seconds=per_entry_time,
                    api_calls=len(API_SOURCES),
                )
            )

    return predictions


def _fallback_predictions(entries: list[BenchmarkEntry]) -> list[Prediction]:
    """Return conservative fallback predictions when hallucinator fails."""
    return [
        Prediction(
            bibtex_key=entry.bibtex_key,
            label="VALID",
            confidence=0.5,
            reason="hallucinator: verification failed, defaulting to VALID",
        )
        for entry in entries
    ]
