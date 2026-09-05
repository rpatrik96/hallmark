"""An abstention must not be written as a committed verdict.

``STATUS_TO_LABEL`` collapses two different things onto ``VALID``: entries
bibtex-check checked and cleared, and entries it declined to judge because no
source answered. Scoring the second group conservatively as VALID is the
documented convention; *recording* it as a committed verdict is a reporting bug.

On ``dev_public`` the released aggregate reported ``num_uncertain: 0`` and
``coverage: 1.0`` for a run whose raw output carries 147 ``unconfirmed``
records — the tool answered 87% of the split and the JSON claimed 100%. The
paper's Coverage column says 0.82, so the two artifacts disagreed and only the
one nobody could recompute was wrong.

What is deliberately *not* an abstention here: ``not_found`` and
``partial_match``. bibtex-check sets ``abstained: true`` on ``not_found``, but
that flag means "could not affirmatively verify against a source", which for
this status coincides with the detection — of the 52 such records on
``dev_public``, 51 sit on HALLUCINATED entries. Treating them as abstentions
would surrender 51 correct detections and inflate the abstention count to the
199 that produces the paper's 0.82.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hallmark.baselines.bibtexupdater import (
    ABSTENTION_AS_VALID_ENV,
    ABSTENTION_STATUSES,
    STATUS_TO_LABEL,
    _parse_jsonl_output,
)


def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "btu_raw.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def _labels(tmp_path: Path, records: list[dict]) -> dict[str, str]:
    preds = _parse_jsonl_output(_write(tmp_path, records), 10.0, len(records))
    return {p.bibtex_key: p.label for p in preds}


def test_every_abstention_status_maps_to_valid_in_the_legacy_table():
    """The set is a re-reading of ``STATUS_TO_LABEL``, not a new policy.

    If a status in this set ever stops mapping to VALID the two have diverged
    and one of them is wrong.
    """
    for status in ABSTENTION_STATUSES:
        assert STATUS_TO_LABEL.get(status) == "VALID", (
            f"{status!r} is registered as an abstention but STATUS_TO_LABEL maps it to "
            f"{STATUS_TO_LABEL.get(status)!r}. An abstention is a VALID that carries no "
            "evidence; a status mapping elsewhere is a verdict and does not belong here."
        )


def test_unconfirmed_is_uncertain_not_committed_valid(tmp_path):
    """The 147-record case that made coverage read 1.0."""
    labels = _labels(
        tmp_path,
        [
            {"key": "a", "status": "unconfirmed", "abstained": True, "p_valid": 0.22},
            {"key": "b", "status": "verified", "abstained": False, "p_valid": 0.95},
        ],
    )
    assert labels["a"] == "UNCERTAIN"
    assert labels["b"] == "VALID", "a checked-and-cleared entry is still a committed VALID"


def test_not_found_stays_a_detection_despite_its_abstained_flag(tmp_path):
    """The 51-of-52 case: bibtex-check's ``abstained`` flag is not our signal."""
    labels = _labels(
        tmp_path,
        [{"key": "a", "status": "not_found", "abstained": True, "p_valid": 0.05}],
    )
    assert labels["a"] == "HALLUCINATED", (
        "not_found carries abstained=True but is a verdict: 51 of the 52 such records "
        "on dev_public sit on HALLUCINATED entries. Reading the flag directly would "
        "surrender them."
    )


def test_partial_match_stays_a_detection(tmp_path):
    """The 85 records that separate the 0.82 accounting from the 0.746 one."""
    labels = _labels(
        tmp_path,
        [{"key": "a", "status": "partial_match", "abstained": False, "p_valid": 0.30}],
    )
    assert labels["a"] == "HALLUCINATED"


def test_source_outage_abstention_is_uncertain(tmp_path):
    """``not_found`` reached while sources were throttled is not evidence."""
    labels = _labels(
        tmp_path,
        [
            {
                "key": "a",
                "status": "not_found",
                "coverage_incomplete": True,
                "p_valid": 0.45,
            }
        ],
    )
    assert labels["a"] == "UNCERTAIN", (
        "a lookup that never completed cannot be a detection; it was already "
        "downgraded to VALID, which recorded it as a committed verdict"
    )


def test_legacy_env_reproduces_the_published_mapping(tmp_path, monkeypatch):
    """The escape hatch exists to reproduce a published row, not to bless it."""
    records = [
        {"key": "a", "status": "unconfirmed", "abstained": True, "p_valid": 0.22},
        {"key": "b", "status": "not_found", "coverage_incomplete": True},
    ]
    monkeypatch.setenv(ABSTENTION_AS_VALID_ENV, "1")
    assert set(_labels(tmp_path, records).values()) == {"VALID"}
    monkeypatch.delenv(ABSTENTION_AS_VALID_ENV)
    assert set(_labels(tmp_path, records).values()) == {"UNCERTAIN"}


def test_coverage_now_describes_the_run(tmp_path):
    """End to end: the reported coverage matches the answered fraction.

    This is the assertion the released JSON failed. Ten entries, three of them
    abstentions, must not report coverage 1.0.
    """
    from hallmark.dataset.schema import BenchmarkEntry
    from hallmark.evaluation.metrics import evaluate

    records = (
        [{"key": f"h{i}", "status": "not_found", "p_valid": 0.05} for i in range(4)]
        + [{"key": f"v{i}", "status": "verified", "p_valid": 0.95} for i in range(3)]
        + [
            {"key": f"u{i}", "status": "unconfirmed", "abstained": True, "p_valid": 0.22}
            for i in range(3)
        ]
    )
    preds = _parse_jsonl_output(_write(tmp_path, records), 10.0, len(records))

    entries = [
        BenchmarkEntry(
            bibtex_key=r["key"],
            bibtex_type="article",
            fields={"title": "t", "author": "a", "year": "2020"},
            label="HALLUCINATED" if r["key"].startswith(("h", "u")) else "VALID",
            hallucination_type="plausible_fabrication" if r["key"].startswith(("h", "u")) else None,
            difficulty_tier=3 if r["key"].startswith(("h", "u")) else None,
            source="test",
        )
        for r in records
    ]

    result = evaluate(entries, preds, tool_name="bibtexupdater")
    assert result.num_uncertain == 3, "the three abstentions must be counted as such"
    assert result.coverage == pytest.approx(0.7), (
        f"coverage {result.coverage} — the tool answered 7 of 10 entries; reporting 1.0 "
        "is the defect this test exists for"
    )


def test_the_guard_is_not_vacuous():
    """``ABSTENTION_STATUSES`` must actually name the status that caused this."""
    assert "unconfirmed" in ABSTENTION_STATUSES
    assert "not_found" not in ABSTENTION_STATUSES
    assert "partial_match" not in ABSTENTION_STATUSES


# --- Each abstention says why it abstained; strict_warn_cnv keeps its origin ------


def _preds(tmp_path: Path, records: list[dict]) -> dict[str, object]:
    preds = _parse_jsonl_output(_write(tmp_path, records), 10.0, len(records))
    return {p.bibtex_key: p for p in preds}


def test_strict_warn_cnv_from_a_not_found_stays_a_detection(tmp_path):
    """Under --strict-warn-cnv bibtex-check relabels NOT_FOUND and UNCONFIRMED to
    one status. The record keeps p_valid, 0.35 for a not_found origin and 0.5
    for unconfirmed, so the detection need not be surrendered."""
    preds = _preds(
        tmp_path,
        [{"key": "a", "status": "strict_warn_cnv", "abstained": False, "p_valid": 0.35}],
    )
    assert preds["a"].label == "HALLUCINATED"
    assert "not_found" in preds["a"].reason


def test_strict_warn_cnv_from_an_unconfirmed_is_an_abstention(tmp_path):
    preds = _preds(
        tmp_path,
        [{"key": "a", "status": "strict_warn_cnv", "abstained": False, "p_valid": 0.5}],
    )
    assert preds["a"].label == "UNCERTAIN"


def test_strict_warn_cnv_reached_during_an_outage_is_an_abstention(tmp_path):
    """A not_found origin does not survive coverage_incomplete, exactly as a
    plain not_found does not."""
    preds = _preds(
        tmp_path,
        [
            {
                "key": "a",
                "status": "strict_warn_cnv",
                "p_valid": 0.35,
                "coverage_incomplete": True,
            }
        ],
    )
    assert preds["a"].label == "UNCERTAIN"


@pytest.mark.parametrize(
    "status,phrase",
    [
        ("unconfirmed", "could not be confirmed"),
        ("skipped", "entry type"),
        ("strict_warn_preprint_year", "preprint"),
        ("api_error", "lookup failed"),
        ("network_error", "lookup failed"),
        ("coverage_incomplete", "did not complete"),
    ],
)
def test_each_abstention_reason_names_its_own_cause(tmp_path, status, phrase):
    """'No source answered' described only the outage statuses. bibtex-check's
    unconfirmed means a record was found and a claimed field could not be
    confirmed; skipped means the entry type is not verified at all; the strict
    preprint-year warning is a decision left to the user."""
    preds = _preds(tmp_path, [{"key": "a", "status": status, "p_valid": 0.5}])
    assert preds["a"].label == "UNCERTAIN"
    assert phrase in preds["a"].reason.lower(), preds["a"].reason
    if status in ("unconfirmed", "skipped", "strict_warn_preprint_year"):
        assert "no source answered" not in preds["a"].reason.lower()
