"""Ground truth must not inherit corruption from the indexes it was built against.

OpenAlex serves records carrying the **correct DOI and the correct author list**
under a **wrong title**. Three were reproduced at source:

===========================  ====================  ==========================================
identifier                   really                served as
===========================  ====================  ==========================================
``10.48550/arxiv.2307.16789``  ToolLLM              "Counterfactually Auditable Lifecycle
                                                    Certification for Autonomous Agents"
``10.48550/arxiv.2212.08073``  Constitutional AI    "Affective Coherence Monitoring for
                                                    Transformer-Based Language Models"
``10.48550/arxiv.2106.09685``  LoRA                 "LoRA Fine-Tuning of a 3B Code LLM for
                                                    Algorithmic Efficiency"
===========================  ====================  ==========================================

That is exactly HALLMARK's ``hybrid_fabrication`` signature — a real DOI whose
other metadata does not match the DOI target — produced by a defective index
record rather than by a bad citation. On a corpus of 5,043 real references the
signature came from corruption three times for every one genuine citation error.

**The risk is live, not theoretical.** ``data/v1.2/dev_public.jsonl`` contains
``db9d82ff3f94``, the real Constitutional AI paper, correctly labelled VALID. A
relabelling pass that queries OpenAlex for its DOI — ``relabel_ground_truth.py``
does exactly this kind of resolution — would receive the wrong title with the
right authors and could reasonably conclude the entry is a hybrid fabrication,
converting a correct VALID entry into a false accusation. Nothing else would
notice: the DOI resolves, the authors match, and the title disagreement is the
whole basis of the type.

It is also a shape synthetic perturbation cannot generate. The benchmark's own
generators build hybrid fabrications by corrupting metadata deliberately; an
index serving right authors under a wrong title is a defect of the source, and no
amount of perturbing real entries produces it.

Payloads are pinned from bibtex-updater's recorded fixtures (commit ``8e3c8d1``)
rather than fetched. If OpenAlex repairs these records a fetching test would
start passing for the wrong reason, which is the failure mode this whole area
keeps producing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: arXiv id -> (what the work really is, what OpenAlex served instead).
#: Recorded 2026-09-04. Not fetched at test time, deliberately.
CORRUPT_INDEX_RECORDS: dict[str, tuple[str, str]] = {
    "2307.16789": (
        "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs",
        "Counterfactually Auditable Lifecycle Certification for Autonomous Agents",
    ),
    "2212.08073": (
        "Constitutional AI: Harmlessness from AI Feedback",
        "Affective Coherence Monitoring for Transformer-Based Language Models",
    ),
    "2106.09685": (
        "LoRA: Low-Rank Adaptation of Large Language Models",
        "LoRA Fine-Tuning of a 3B Code LLM for Algorithmic Efficiency",
    ),
}

_SPLITS = (
    "data/v1.2/dev_public.jsonl",
    "data/v1.2/test_public.jsonl",
    "data/v1.2/test_crossdomain.jsonl",
    "data/v1.2/stress_test.jsonl",
    "data/hidden/test_hidden.jsonl",
)


def _entries_touching_corrupt_records() -> list[tuple[str, dict]]:
    """Every split entry whose fields mention one of the three identifiers."""
    found: list[tuple[str, dict]] = []
    for rel in _SPLITS:
        path = _REPO_ROOT / rel
        if not path.is_file():  # data/hidden/ is gitignored
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            blob = json.dumps(entry.get("fields") or {})
            for arxiv_id in CORRUPT_INDEX_RECORDS:
                if arxiv_id in blob:
                    found.append((rel, entry))
                    break
    return found


def _normalise(title: str) -> str:
    return "".join(c for c in title.lower() if c.isalnum())


def test_no_entry_is_labelled_from_the_corrupt_title():
    """An entry citing one of these works must be judged on what it really is.

    If a relabelling pass ever adopts the served title, the entry's own title
    stops matching and it becomes a hybrid fabrication. This asserts the shipped
    labels still describe the real work.
    """
    offenders: list[str] = []
    for rel, entry in _entries_touching_corrupt_records():
        title = _normalise((entry.get("fields") or {}).get("title", ""))
        for _arxiv_id, (_real, served) in CORRUPT_INDEX_RECORDS.items():
            if _normalise(served) in title or title in _normalise(served):
                offenders.append(
                    f"{rel}:{entry.get('bibtex_key')} carries the CORRUPT title "
                    f"{served!r} — the ground truth has absorbed an index defect"
                )
    assert not offenders, "\n  ".join(offenders)


def test_a_correctly_cited_work_is_not_flagged_as_hybrid_fabrication():
    """The specific reversal this guards against.

    ``hybrid_fabrication`` means a real DOI whose other metadata does not match
    the DOI target. A corrupt index record produces that signature from a
    perfectly correct citation, so an entry citing one of these works must not
    carry the type unless something other than the index says so.
    """
    offenders: list[str] = []
    for rel, entry in _entries_touching_corrupt_records():
        if entry.get("hallucination_type") == "hybrid_fabrication":
            offenders.append(
                f"{rel}:{entry.get('bibtex_key')} is typed hybrid_fabrication while "
                "citing a work whose OpenAlex record is known corrupt. Verify against a "
                "source outside the index before trusting that label."
            )
    assert not offenders, "\n  ".join(offenders)


def test_the_known_constitutional_ai_entry_is_still_valid():
    """Pins the one live instance so a relabel cannot flip it silently.

    ``dev_public`` carries the real Constitutional AI paper. Its OpenAlex record
    serves the right authors under the wrong title, which is the exact input that
    would make a resolver conclude the citation is fabricated.
    """
    matches = [
        entry
        for rel, entry in _entries_touching_corrupt_records()
        if "2212.08073" in json.dumps(entry.get("fields") or {})
    ]
    if not matches:
        pytest.skip("Constitutional AI entry not present in the available splits")
    for entry in matches:
        assert entry.get("label") == "VALID", (
            f"{entry.get('bibtex_key')} is a correct citation of a real paper and must "
            f"stay VALID; it is now {entry.get('label')} "
            f"({entry.get('hallucination_type')}). OpenAlex serves this DOI under the "
            "title 'Affective Coherence Monitoring for Transformer-Based Language "
            "Models' with the right author list — if a relabelling pass trusted that, "
            "this is what the damage looks like."
        )


def test_the_register_is_not_empty():
    """Otherwise every assertion above passes vacuously."""
    assert CORRUPT_INDEX_RECORDS
    assert _entries_touching_corrupt_records(), (
        "no split entry cites any of the three records — the guard is inert. "
        "Either the data changed or the identifiers are wrong."
    )
