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

from hallmark.baselines.common import fallback_predictions
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
        logger.error(
            "hallucinator not found at %s. Clone from: https://github.com/gianlucasb/hallucinator",
            h_path,
        )
        return fallback_predictions(entries, reason="Fallback: hallucinator unavailable")

    # Create a temporary PDF with references
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as outfile:
        output_path = outfile.name

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
            return fallback_predictions(
                entries, reason="hallucinator: verification failed, defaulting to VALID"
            )

        # Parse output
        return _parse_output(entries, result.stdout, output_path, total_time)

    except subprocess.TimeoutExpired:
        logger.error("hallucinator timed out")
        return fallback_predictions(
            entries, reason="hallucinator: verification failed, defaulting to VALID"
        )
    finally:
        Path(pdf_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def _parse_stdout_results(stdout: str) -> tuple[set[str], set[str], set[str]]:
    """Extract not_found, mismatch, and retracted title sets from stdout."""
    not_found_titles: set[str] = set()
    mismatch_titles: set[str] = set()
    retracted_titles: set[str] = set()

    current_title = ""
    for line in stdout.split("\n"):
        line_stripped = line.strip()
        if "NOT FOUND" in line_stripped.upper() or "not found" in line_stripped:
            if current_title:
                not_found_titles.add(current_title.lower())
        elif "MISMATCH" in line_stripped.upper() or "mismatch" in line_stripped:
            if current_title:
                mismatch_titles.add(current_title.lower())
        elif "RETRACTED" in line_stripped.upper() and current_title:
            retracted_titles.add(current_title.lower())

        if line_stripped and not line_stripped.startswith(("=", "-", " ", "✓", "✗", "?")):
            current_title = line_stripped

    return not_found_titles, mismatch_titles, retracted_titles


def _parse_output_file(output_path: str) -> set[str]:
    """Extract not_found titles from output file."""
    not_found: set[str] = set()
    path = Path(output_path)
    if path.exists():
        try:
            content = path.read_text()
            for line in content.split("\n"):
                lower = line.lower()
                if "not found" in lower:
                    parts = line.split(":")
                    if len(parts) > 1:
                        not_found.add(parts[0].strip().lower())
        except OSError:
            pass
    return not_found


def _title_matches(entry_title: str, title_set: set[str]) -> bool:
    """Check if entry title matches any title in the set."""
    return any(entry_title in t or t in entry_title for t in title_set if t)


def _parse_output(
    entries: list[BenchmarkEntry],
    stdout: str,
    output_path: str,
    total_time: float,
) -> list[Prediction]:
    """Parse hallucinator output and match to entries by title."""
    per_entry_time = total_time / max(len(entries), 1)

    not_found, mismatch, retracted = _parse_stdout_results(stdout)
    not_found |= _parse_output_file(output_path)

    predictions = []
    for entry in entries:
        entry_title = entry.fields.get("title", "").lower().strip()

        is_not_found = _title_matches(entry_title, not_found)
        is_mismatch = _title_matches(entry_title, mismatch)
        is_retracted = _title_matches(entry_title, retracted)

        if is_not_found or is_retracted:
            label = "HALLUCINATED"
            confidence = 0.80 if is_not_found else 0.90
            reason = f"hallucinator: {'not found' if is_not_found else 'retracted'}"
        elif is_mismatch:
            label = "HALLUCINATED"
            confidence = 0.65
            reason = "hallucinator: author mismatch"
        else:
            label = "VALID"
            confidence = 0.80
            reason = "hallucinator: verified or no issues found"

        predictions.append(
            Prediction(
                bibtex_key=entry.bibtex_key,
                label=label,  # type: ignore[arg-type]
                confidence=confidence,
                reason=reason,
                api_sources_queried=API_SOURCES,
                wall_clock_seconds=per_entry_time,
                api_calls=len(API_SOURCES),
            )
        )

    return predictions
