#!/usr/bin/env python3
"""Generate hallucinated citations using multiple LLM backends.

Supported backends:
- openai: OpenAI API (GPT-4o, GPT-5.1, etc.)
- ollama: Local Ollama server (Llama 3.1, Mistral, etc.)
- anthropic: Anthropic API (Claude 3.5/4)
- mistral: Mistral API (Mistral Large, etc.)
- gemini: Google Gemini API (Gemini Pro, etc.)
- openrouter: OpenRouter API (100+ models via OpenAI-compatible endpoint)
  Model presets: deepseek-r1, deepseek-v3, qwen, mistral
  Usage: --backend openrouter --model deepseek/deepseek-r1

Two strategies:
1. Naive bibliography: Ask LLM to write bibliographies on topics, verify against CrossRef
2. Targeted type generation: Craft prompts that naturally produce specific hallucination types

All entries are validated before output.
"""

from __future__ import annotations

import argparse
import json as json_mod
import logging
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import date
from pathlib import Path

import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hallmark.contribution.validate_entry import validate_entry
from hallmark.dataset.schema import (
    HALLUCINATION_TIER_MAP,
    BenchmarkEntry,
    GenerationMethod,
    HallucinationType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# CrossRef API configuration
CROSSREF_API = "https://api.crossref.org/works"
CROSSREF_HEADERS = {"User-Agent": "HALLMARK-Generator/1.0 (mailto:patrik.reizinger@gmail.com)"}
RATE_LIMIT_DELAY = 1.0  # seconds between requests

# SSL context for macOS
try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


# ---------------------------------------------------------------------------
# Multi-backend LLM abstraction
# ---------------------------------------------------------------------------


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    name: str
    model: str

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """Generate text from a prompt. Returns the response string."""

    def __repr__(self) -> str:
        return f"{self.name}({self.model})"


class OpenAIBackend(LLMBackend):
    """OpenAI API backend (GPT-4o, GPT-5.1, etc.)."""

    name = "openai"

    def __init__(self, model: str = "gpt-5.1") -> None:
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Export it: export OPENAI_API_KEY='sk-...'")
        import openai

        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


class OllamaBackend(LLMBackend):
    """Ollama local backend (Llama 3.1, Mistral, etc.)."""

    name = "ollama"

    def __init__(
        self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"
    ) -> None:
        self.model = model
        self.base_url = base_url
        # Verify Ollama is running
        try:
            req = urllib.request.Request(f"{base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json_mod.loads(r.read().decode())
                available = [m["name"] for m in data.get("models", [])]
                if model not in available:
                    logger.warning(
                        f"Model '{model}' not found in Ollama. Available: {available}. "
                        f"Pull it with: ollama pull {model}"
                    )
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {base_url}. Start it with: ollama serve\nError: {e}"
            ) from e

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        url = f"{self.base_url}/api/generate"
        payload = json_mod.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
        ).encode()
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json_mod.loads(r.read().decode())
        return data.get("response", "").strip()


class AnthropicBackend(LLMBackend):
    """Anthropic API backend (Claude)."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5-20250929") -> None:
        self.model = model
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Get your key at "
                "https://console.anthropic.com/settings/keys then: "
                "export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


class MistralBackend(LLMBackend):
    """Mistral API backend."""

    name = "mistral"

    def __init__(self, model: str = "mistral-large-latest") -> None:
        self.model = model
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY not set. Get your key at "
                "https://console.mistral.ai/api-keys then: "
                "export MISTRAL_API_KEY='...'"
            )
        self.api_key = api_key

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        url = "https://api.mistral.ai/v1/chat/completions"
        payload = json_mod.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode()
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as r:
            data = json_mod.loads(r.read().decode())
        return data["choices"][0]["message"]["content"].strip()


class GeminiBackend(LLMBackend):
    """Google Gemini API backend."""

    name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        self.model = model
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) not set. Get your key at "
                "https://aistudio.google.com/app/apikey then: "
                "export GEMINI_API_KEY='...'"
            )
        self.api_key = api_key

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = json_mod.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }
        ).encode()
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as r:
            data = json_mod.loads(r.read().decode())
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return parts[0].get("text", "").strip() if parts else ""


class OpenRouterBackend(LLMBackend):
    """OpenRouter API backend (100+ models via OpenAI-compatible endpoint)."""

    name = "openrouter"

    def __init__(self, model: str = "deepseek/deepseek-r1") -> None:
        self.model = model
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Get your key at "
                "https://openrouter.ai/keys then: "
                "export OPENROUTER_API_KEY='sk-or-...'"
            )
        import openai

        self.client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


OPENROUTER_MODEL_PRESETS: dict[str, str] = {
    "deepseek-r1": "deepseek/deepseek-r1",
    "deepseek-v3": "deepseek/deepseek-v3.2",
    "qwen": "qwen/qwen3-235b-a22b-2507",
    "mistral": "mistralai/mistral-large-2512",
    "gemini-flash": "google/gemini-2.5-flash",
}


BACKENDS: dict[str, type[LLMBackend]] = {
    "openai": OpenAIBackend,
    "ollama": OllamaBackend,
    "anthropic": AnthropicBackend,
    "mistral": MistralBackend,
    "gemini": GeminiBackend,
    "openrouter": OpenRouterBackend,
}

# Default models per backend
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-5.1",
    "ollama": "llama3.1:8b",
    "anthropic": "claude-sonnet-4-5-20250929",
    "mistral": "mistral-large-latest",
    "gemini": "gemini-2.0-flash",
    "openrouter": "deepseek/deepseek-r1",
}


def _title_jaccard(title_a: str, title_b: str) -> float:
    """Word-level Jaccard similarity between two titles."""
    words_a = set(title_a.lower().split())
    words_b = set(title_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _extract_lastnames(author_str: str) -> set[str]:
    """Extract last names from a BibTeX-style author string.

    Handles both "Last, First" and "First Last" formats.
    """
    names: set[str] = set()
    for part in author_str.split(" and "):
        part = part.strip()
        if not part:
            continue
        if "," in part:
            # "Last, First" format
            names.add(part.split(",")[0].strip().lower())
        else:
            # "First Last" format — last token is the last name
            tokens = part.split()
            if tokens:
                names.add(tokens[-1].strip().lower())
    return names


def _authors_match_fuzzy(bib_authors: str, cr_authors: list[dict]) -> bool:
    """Fuzzy author match using Jaccard similarity on last names.

    Returns True if >= 50% of last names overlap between the BibTeX
    author string and the CrossRef author list.
    """
    bib_lastnames = _extract_lastnames(bib_authors)
    cr_lastnames = {a.get("family", "").lower() for a in cr_authors if a.get("family")}
    if not bib_lastnames or not cr_lastnames:
        return False
    overlap = len(bib_lastnames & cr_lastnames) / len(bib_lastnames | cr_lastnames)
    return overlap >= 0.5


def verify_title_in_crossref(title: str) -> dict | None:
    """Verify if a title exists in CrossRef API.

    Returns the best match if found and title similarity > 0.5,
    None otherwise. This prevents misclassification from CrossRef
    returning unrelated papers for fuzzy title queries.
    """
    time.sleep(RATE_LIMIT_DELAY)
    try:
        params = {"query.bibliographic": title, "rows": 3}
        response = requests.get(CROSSREF_API, params=params, headers=CROSSREF_HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()

        for item in data["message"]["items"]:
            cr_title = item.get("title", [""])[0] if item.get("title") else ""
            if _title_jaccard(title, cr_title) > 0.5:
                return item
        return None
    except Exception as e:
        logger.warning(f"CrossRef API error for '{title[:50]}...': {e}")
        return None


def verify_doi_in_crossref(doi: str) -> dict | None:
    """Verify if a DOI exists in CrossRef API."""
    time.sleep(RATE_LIMIT_DELAY)
    try:
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url, headers=CROSSREF_HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["message"]
    except Exception:
        return None


def parse_bibtex_entry(bibtex_str: str) -> dict | None:
    """Parse a single BibTeX entry string into a dictionary.

    Simple parser that extracts key fields without full bibtexparser dependency.
    """
    try:
        # Extract entry type and key
        match = re.match(r"@(\w+)\{([^,]+),", bibtex_str, re.IGNORECASE)
        if not match:
            return None

        entry_type = match.group(1).lower()
        key = match.group(2).strip()

        # Extract fields
        fields = {}
        field_pattern = r"(\w+)\s*=\s*[{\"](.*?)[}\"](?:,|\s*\})"
        for match in re.finditer(field_pattern, bibtex_str, re.DOTALL):
            field_name = match.group(1).lower()
            field_value = match.group(2).strip()
            fields[field_name] = field_value

        return {"type": entry_type, "key": key, "fields": fields}
    except Exception as e:
        logger.debug(f"Failed to parse BibTeX: {e}")
        return None


def classify_hallucination(entry_dict: dict, crossref_data: dict | None) -> tuple[str, str, dict]:
    """Classify what type of hallucination this is based on what's wrong.

    Returns: (hallucination_type, explanation, subtests)
    """
    fields = entry_dict["fields"]
    title = fields.get("title", "")
    authors = fields.get("author", "")
    doi = fields.get("doi", "")
    year = fields.get("year", "")
    venue = fields.get("booktitle") or fields.get("journal", "")

    subtests = {
        "doi_resolves": False,
        "title_exists": False,
        "authors_match": False,
        "venue_real": False,
        "fields_complete": bool(title and authors and year),
        "cross_db_agreement": False,
    }

    # Check if completely fabricated (no CrossRef match at all)
    if crossref_data is None:
        # Check for future date
        try:
            year_int = int(year)
            if year_int > date.today().year:
                subtests["fields_complete"] = False
                return (
                    HallucinationType.FUTURE_DATE.value,
                    f"Publication year {year} is in the future",
                    subtests,
                )
        except (ValueError, TypeError):
            pass

        # Check for placeholder authors
        placeholder_patterns = [
            r"\b(john|jane)\s+(doe|smith)\b",
            r"\b(test|example|sample)\s+author\b",
            r"\b(first|second)\s+(last|person)\b",
            r"\balice\s+johnson\b",
            r"\bbob\s+williams\b",
        ]
        authors_lower = authors.lower()
        for pattern in placeholder_patterns:
            if re.search(pattern, authors_lower):
                subtests["authors_match"] = False
                return (
                    HallucinationType.PLACEHOLDER_AUTHORS.value,
                    f"Authors appear to be placeholders: {authors}",
                    subtests,
                )

        # Check for fabricated DOI
        if doi:
            doi_data = verify_doi_in_crossref(doi)
            if doi_data is None:
                return (
                    HallucinationType.FABRICATED_DOI.value,
                    f"DOI {doi} does not resolve",
                    subtests,
                )
            else:
                # DOI exists but title/authors don't match
                subtests["doi_resolves"] = True
                subtests["venue_real"] = True
                return (
                    HallucinationType.HYBRID_FABRICATION.value,
                    f"Real DOI {doi} with fabricated metadata",
                    subtests,
                )

        # Check if venue seems fake
        fake_venue_indicators = [
            "international conference on advanced",
            "journal of computational intelligence",
            "workshop on emerging",
            "symposium on frontier",
            "transactions on neural computing",
        ]
        venue_lower = venue.lower()
        for indicator in fake_venue_indicators:
            if indicator in venue_lower:
                subtests["venue_real"] = False
                return (
                    HallucinationType.NONEXISTENT_VENUE.value,
                    f"Venue '{venue}' appears fabricated",
                    subtests,
                )

        # Default to plausible fabrication for completely made-up papers
        return (
            HallucinationType.PLAUSIBLE_FABRICATION.value,
            "Completely fabricated paper with plausible metadata",
            subtests,
        )

    # We have a CrossRef match - check what's different
    subtests["title_exists"] = True
    subtests["venue_real"] = True

    # Extract CrossRef metadata
    cr_title = crossref_data.get("title", [""])[0] if crossref_data.get("title") else ""
    cr_authors = []
    for author in crossref_data.get("author", []):
        given = author.get("given", "")
        family = author.get("family", "")
        if given and family:
            cr_authors.append(f"{given} {family}")
        elif family:
            cr_authors.append(family)
    cr_authors_str = " and ".join(cr_authors)

    cr_venue = ""
    if "container-title" in crossref_data:
        cr_venue = crossref_data["container-title"][0] if crossref_data["container-title"] else ""

    cr_doi = crossref_data.get("DOI", "")

    # Check for author mismatch using fuzzy last-name matching
    cr_author_list = crossref_data.get("author", [])
    if cr_author_list and not _authors_match_fuzzy(authors, cr_author_list):
        subtests["authors_match"] = False
        if doi and doi.lower() == cr_doi.lower():
            subtests["doi_resolves"] = True
        return (
            HallucinationType.AUTHOR_MISMATCH.value,
            f"Authors don't match CrossRef: '{authors}' vs '{cr_authors_str}'",
            subtests,
        )

    subtests["authors_match"] = True

    # Check for wrong venue (use substring matching since venue names vary)
    if cr_venue and venue:
        venue_lower = venue.lower()
        cr_venue_lower = cr_venue.lower()
        # Check if neither is a substring of the other (accounts for abbreviations)
        if venue_lower not in cr_venue_lower and cr_venue_lower not in venue_lower:
            venue_jaccard = _title_jaccard(venue, cr_venue)
            if venue_jaccard < 0.3:
                if doi and doi.lower() == cr_doi.lower():
                    subtests["doi_resolves"] = True
                return (
                    HallucinationType.WRONG_VENUE.value,
                    f"Venue doesn't match CrossRef: '{venue}' vs '{cr_venue}'",
                    subtests,
                )

    # Check for near-miss title (similar but not exact)
    if cr_title and title.lower() != cr_title.lower():
        # Calculate similarity
        title_words = set(title.lower().split())
        cr_title_words = set(cr_title.lower().split())
        jaccard = len(title_words & cr_title_words) / len(title_words | cr_title_words)

        if jaccard > 0.7:  # Similar but not identical
            return (
                HallucinationType.NEAR_MISS_TITLE.value,
                f"Title slightly different from CrossRef: '{title}' vs '{cr_title}'",
                subtests,
            )
        else:
            # Very different title with same authors
            subtests["title_exists"] = False
            return (
                HallucinationType.CHIMERIC_TITLE.value,
                f"Title fabricated with real authors: '{title}' (real: '{cr_title}')",
                subtests,
            )

    # If we got here, it matched CrossRef well - shouldn't happen for hallucinations
    subtests["cross_db_agreement"] = True
    return (
        HallucinationType.PLAUSIBLE_FABRICATION.value,
        "LLM-generated entry that happens to match real paper",
        subtests,
    )


def generate_naive_bibliography(
    backend: LLMBackend, topic: str, count: int = 10
) -> list[BenchmarkEntry]:
    """Strategy A: Ask LLM to write bibliography, verify against CrossRef.

    Returns list of hallucinated entries found.
    """
    prompt = f"""Write a BibTeX bibliography of {count} recent papers on {topic}.
Include diverse papers from top venues (NeurIPS, ICML, ICLR, CVPR, ACL, AAAI).
Format each entry as standard BibTeX with title, author, year, booktitle/journal.
Include DOIs where applicable.

Output only the BibTeX entries, nothing else."""

    logger.info(f"Generating bibliography for: {topic}")

    try:
        content = backend.generate(prompt, temperature=0.7, max_tokens=2048)
    except Exception as e:
        logger.error(f"LLM API error for {topic}: {e}")
        return []

    # Extract individual BibTeX entries
    entries = []
    entry_pattern = r"@\w+\{[^@]+"
    for match in re.finditer(entry_pattern, content, re.DOTALL):
        bibtex_str = match.group(0).strip()
        if not bibtex_str.endswith("}"):
            bibtex_str += "\n}"

        parsed = parse_bibtex_entry(bibtex_str)
        if parsed is None:
            continue

        # Verify against CrossRef
        title = parsed["fields"].get("title", "")
        if not title:
            continue

        logger.info(f"  Verifying: {title[:60]}...")
        crossref_data = verify_title_in_crossref(title)

        # If no match or significant differences, it's a hallucination
        hall_type, explanation, subtests = classify_hallucination(parsed, crossref_data)

        # Skip if it matched CrossRef perfectly (not a hallucination)
        if subtests.get("cross_db_agreement", False):
            logger.info("    ✓ Valid paper found, skipping")
            continue

        logger.info(f"    ✗ Hallucination detected: {hall_type}")

        # Create BenchmarkEntry
        difficulty_tier = HALLUCINATION_TIER_MAP[HallucinationType(hall_type)].value

        entry = BenchmarkEntry(
            bibtex_key=f"llm_{hall_type}_{parsed['key'][:20]}",
            bibtex_type=parsed["type"],
            fields=parsed["fields"],
            label="HALLUCINATED",
            hallucination_type=hall_type,
            difficulty_tier=difficulty_tier,
            explanation=f"LLM-generated via {backend}. Prompt: 'Bibliography on {topic}'. Verification: {explanation}",
            generation_method=GenerationMethod.LLM_GENERATED.value,
            source_conference=None,
            publication_date="",
            added_to_benchmark=date.today().isoformat(),
            subtests=subtests,
            raw_bibtex=bibtex_str,
        )

        entries.append(entry)

    return entries


def generate_targeted_type(
    backend: LLMBackend, hall_type: HallucinationType, count: int = 5
) -> list[BenchmarkEntry]:
    """Strategy B: Craft prompts that naturally produce specific hallucination types."""

    prompts_by_type = {
        HallucinationType.FABRICATED_DOI: [
            "Write a BibTeX entry for a paper on vision transformers. Include the DOI.",
            "Cite a recent NeurIPS paper on meta-learning with its DOI.",
            "Give me the BibTeX entry with DOI for 'Attention Is All You Need'.",
        ],
        HallucinationType.NONEXISTENT_VENUE: [
            "Write a BibTeX entry for a paper from the International Conference on Advanced Neural Systems.",
            "Cite a paper from the Journal of Deep Learning Applications.",
            "Give me a BibTeX entry from the Workshop on Emerging AI Methods.",
        ],
        HallucinationType.PLACEHOLDER_AUTHORS: [
            "Write a generic BibTeX entry template for a machine learning paper.",
            "Give me an example BibTeX entry for a deep learning paper.",
            "Create a sample BibTeX citation for a paper on reinforcement learning.",
        ],
        HallucinationType.FUTURE_DATE: [
            "What are the most anticipated NeurIPS 2027 papers on transformers?",
            "Cite the groundbreaking ICML 2028 paper on AGI.",
            "Give me BibTeX for papers from ICLR 2029 on quantum machine learning.",
        ],
        HallucinationType.CHIMERIC_TITLE: [
            "Write a BibTeX entry for Geoffrey Hinton's paper on blockchain applications in deep learning.",
            "Cite Yoshua Bengio's work on cryptocurrency optimization.",
            "Give me the BibTeX for Yann LeCun's paper on quantum neural networks.",
        ],
        HallucinationType.WRONG_VENUE: [
            "Cite the paper 'Attention Is All You Need' from AAAI 2017.",
            "Give me the BibTeX for BERT from CVPR 2019.",
            "Write the citation for GPT-3 from ACL 2020.",
        ],
        HallucinationType.AUTHOR_MISMATCH: [
            "Write a BibTeX entry for 'Deep Residual Learning for Image Recognition' by Geoffrey Hinton.",
            "Cite 'BERT: Pre-training of Deep Bidirectional Transformers' by Yann LeCun.",
            "Give me the BibTeX for 'Attention Is All You Need' by Yoshua Bengio.",
        ],
        HallucinationType.PREPRINT_AS_PUBLISHED: [
            "Find an arXiv preprint on diffusion models and cite it as if it was published at CVPR.",
            "Take a recent arXiv paper on large language models and give its ICLR publication details.",
        ],
        HallucinationType.HYBRID_FABRICATION: [
            "Cite DOI 10.1109/CVPR.2016.90 with authors John Smith and Jane Doe.",
            "Give me the BibTeX for DOI 10.5555/3295222.3295349 but change the title slightly.",
        ],
        HallucinationType.NEAR_MISS_TITLE: [
            "Write a BibTeX entry for 'Attention Is Everything You Need'.",
            "Cite the paper 'Deep Residual Networks for Image Recognition'.",
            "Give me the BibTeX for 'BERT: Pre-training Deep Bidirectional Transformers for Language Understanding'.",
        ],
        HallucinationType.PLAUSIBLE_FABRICATION: [
            "Cite the seminal paper on neural architecture search for graph neural networks.",
            "Write a BibTeX entry for the foundational work on few-shot learning in medical imaging.",
            "Give me the citation for the breakthrough paper on interpretable reinforcement learning.",
        ],
        HallucinationType.ARXIV_VERSION_MISMATCH: [
            "Cite arXiv:2104.12345 as if it was published at NeurIPS 2021.",
            "Give me the BibTeX for arXiv:2203.98765 but list it as an ICML 2022 paper.",
        ],
    }

    prompts = prompts_by_type.get(hall_type, [])
    if not prompts:
        logger.warning(f"No prompts defined for {hall_type.value}")
        return []

    entries = []
    for prompt in prompts[:count]:
        logger.info(f"Generating {hall_type.value} with prompt: {prompt[:60]}...")

        try:
            content = backend.generate(prompt, temperature=0.7, max_tokens=512)
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            continue

        # Extract BibTeX entry
        entry_pattern = r"@\w+\{[^@]+"
        match = re.search(entry_pattern, content, re.DOTALL)
        if not match:
            logger.warning("No BibTeX found in response")
            continue

        bibtex_str = match.group(0).strip()
        if not bibtex_str.endswith("}"):
            bibtex_str += "\n}"

        parsed = parse_bibtex_entry(bibtex_str)
        if parsed is None:
            logger.warning("Failed to parse BibTeX")
            continue

        # Verify and classify
        title = parsed["fields"].get("title", "")
        crossref_data = verify_title_in_crossref(title) if title else None

        detected_type, explanation, subtests = classify_hallucination(parsed, crossref_data)

        # Use the intended type if it matches, otherwise use detected type
        final_type = hall_type.value if detected_type == hall_type.value else detected_type
        difficulty_tier = HALLUCINATION_TIER_MAP[HallucinationType(final_type)].value

        logger.info(f"  Detected as: {detected_type} (intended: {hall_type.value})")

        entry = BenchmarkEntry(
            bibtex_key=f"llm_{final_type}_{len(entries)}_{parsed['key'][:15]}",
            bibtex_type=parsed["type"],
            fields=parsed["fields"],
            label="HALLUCINATED",
            hallucination_type=final_type,
            difficulty_tier=difficulty_tier,
            explanation=f"LLM-generated via {backend}. Prompt: '{prompt}'. Verification: {explanation}",
            generation_method=GenerationMethod.LLM_GENERATED.value,
            source_conference=None,
            publication_date="",
            added_to_benchmark=date.today().isoformat(),
            subtests=subtests,
            raw_bibtex=bibtex_str,
        )

        entries.append(entry)

    return entries


def main():
    backends_list = ", ".join(BACKENDS.keys())
    parser = argparse.ArgumentParser(
        description="Generate hallucinated citations using multiple LLM backends"
    )
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS.keys()),
        default="openai",
        help=f"LLM backend to use ({backends_list}) (default: openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: backend-specific, see DEFAULT_MODELS)",
    )
    parser.add_argument(
        "--strategy",
        choices=["naive", "targeted", "both"],
        default="both",
        help="Generation strategy (default: both)",
    )
    parser.add_argument(
        "--output",
        default="data/v1.0/llm_generated.jsonl",
        help="Output JSONL file (default: data/v1.0/llm_generated.jsonl)",
    )
    parser.add_argument(
        "--target-per-type",
        type=int,
        default=10,
        help="Target entries per hallucination type (default: 10)",
    )
    parser.add_argument(
        "--naive-topics",
        type=int,
        default=20,
        help="Number of topics for naive bibliography strategy (default: 20)",
    )

    args = parser.parse_args()

    # Resolve model
    model = args.model or DEFAULT_MODELS[args.backend]

    # Initialize backend
    backend_cls = BACKENDS[args.backend]
    try:
        backend = backend_cls(model=model)
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    logger.info(f"Using backend: {backend}")

    all_entries = []

    # Strategy A: Naive bibliography
    if args.strategy in ("naive", "both"):
        logger.info("=" * 60)
        logger.info("Strategy A: Naive Bibliography Generation")
        logger.info("=" * 60)

        topics = [
            "vision transformers",
            "large language models",
            "graph neural networks",
            "diffusion models for image generation",
            "reinforcement learning from human feedback",
            "meta-learning and few-shot learning",
            "neural architecture search",
            "adversarial robustness in deep learning",
            "explainable AI and interpretability",
            "federated learning",
            "self-supervised learning",
            "multimodal learning",
            "causal inference with machine learning",
            "continual learning and catastrophic forgetting",
            "efficient transformers",
            "protein structure prediction with AI",
            "neural rendering and NeRF",
            "prompt engineering for LLMs",
            "AI safety and alignment",
            "quantum machine learning",
        ]

        for topic in topics[: args.naive_topics]:
            entries = generate_naive_bibliography(backend, topic, count=10)
            all_entries.extend(entries)
            logger.info(f"Found {len(entries)} hallucinations for '{topic}'")

    # Strategy B: Targeted type generation
    if args.strategy in ("targeted", "both"):
        logger.info("=" * 60)
        logger.info("Strategy B: Targeted Type Generation")
        logger.info("=" * 60)

        for hall_type in HallucinationType:
            logger.info(f"\nGenerating {hall_type.value}...")
            entries = generate_targeted_type(backend, hall_type, count=args.target_per_type)
            all_entries.extend(entries)
            logger.info(f"  Generated {len(entries)} entries")

    # Validate all entries
    logger.info("=" * 60)
    logger.info("Validating entries")
    logger.info("=" * 60)

    valid_entries = []
    for entry in all_entries:
        result = validate_entry(entry)
        if result.valid:
            valid_entries.append(entry)
        else:
            logger.warning(f"Entry {entry.bibtex_key} failed validation:")
            for error in result.errors:
                logger.warning(f"  - {error}")

    # Report statistics
    logger.info("=" * 60)
    logger.info("Generation Summary")
    logger.info("=" * 60)
    logger.info(f"Total entries generated: {len(all_entries)}")
    logger.info(f"Valid entries: {len(valid_entries)}")
    logger.info(f"Failed validation: {len(all_entries) - len(valid_entries)}")

    type_counts = defaultdict(int)
    for entry in valid_entries:
        type_counts[entry.hallucination_type] += 1

    logger.info("\nBreakdown by type:")
    for hall_type in HallucinationType:
        count = type_counts[hall_type.value]
        logger.info(f"  {hall_type.value:30s}: {count:3d}")

    # Save to output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for entry in valid_entries:
            f.write(entry.to_json() + "\n")

    logger.info(f"\n✓ Saved {len(valid_entries)} entries to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
