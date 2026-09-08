"""An API failure must not become a verdict about a citation.

``_verify_entries`` used to abort after ``max_consecutive_failures`` and then
write an UNCERTAIN ``[Error fallback]`` prediction for *every remaining entry*,
including ones it had never attempted. Those records were indistinguishable from
model abstentions, counted toward coverage, and survived a resume as completed
work. A sweep of the repo found 9,652 of them; all 180 UNCERTAIN predictions in
one released `test_public` result are of this kind, and that result shipped with
``coverage: 1.0``.

A missing entry is honest and visibly incomplete. An abstention is a claim the
model never made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hallmark.baselines.llm_verifier import _verify_entries
from hallmark.dataset.schema import BlindEntry

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _entries(n: int) -> list[BlindEntry]:
    return [
        BlindEntry(
            bibtex_key=f"k{i}",
            bibtex_type="article",
            fields={"title": f"Paper {i}", "author": "A. Author", "year": "2024"},
        )
        for i in range(n)
    ]


def _always_fails(_prompt: str) -> str:
    raise RuntimeError("simulated API outage")


def _valid_response(_prompt: str) -> str:
    return json.dumps({"label": "VALID", "confidence": 0.9, "reason": "ok"})


class TestUnattemptedEntriesAreNotFabricated:
    def test_abort_records_only_what_was_attempted(self):
        """The run stops; it does not invent a verdict for the rest."""
        entries = _entries(50)
        preds = _verify_entries(
            entries, _always_fails, "test", "model-x", max_consecutive_failures=3
        )
        assert len(preds) == 3, (
            f"expected only the 3 attempted entries to be recorded, got {len(preds)}"
        )
        assert {p.bibtex_key for p in preds} == {"k0", "k1", "k2"}

    def test_no_record_claims_an_entry_was_skipped(self):
        """The 'Skipped after N consecutive API failures' record is gone.

        That string was written into result files as a per-entry reason, which is
        a statement about the pipeline dressed as a statement about the citation.
        """
        preds = _verify_entries(
            _entries(50), _always_fails, "test", "model-x", max_consecutive_failures=3
        )
        assert not [p for p in preds if "Skipped after" in (p.reason or "")]

    def test_coverage_shortfall_is_visible_to_the_caller(self):
        """The gap must show up as missing predictions, not as abstentions."""
        entries = _entries(50)
        preds = _verify_entries(
            entries, _always_fails, "test", "model-x", max_consecutive_failures=3
        )
        answered = {p.bibtex_key for p in preds}
        assert len(entries) - len(answered) == 47, "the shortfall must be countable"


class TestCheckpointResume:
    def test_error_fallbacks_are_retried_by_default(self, tmp_path: Path):
        """A resume must re-attempt a transient failure, not inherit it.

        ``retry_failed`` defaulted to False, so the fabricated records above were
        loaded back as completed work and never retried. Combined with the
        fabrication, one outage could permanently settle an entire split.
        """
        entries = _entries(4)
        ckpt = tmp_path / "ckpt"

        first = _verify_entries(
            entries,
            _always_fails,
            "test",
            "model-x",
            checkpoint_dir=ckpt,
            max_consecutive_failures=2,
        )
        assert all(p.label == "UNCERTAIN" for p in first)

        # Second pass with a working API: the failed entries must be re-attempted.
        second = _verify_entries(
            entries,
            _valid_response,
            "test",
            "model-x",
            checkpoint_dir=ckpt,
            max_consecutive_failures=2,
        )
        assert len(second) == 4, "every entry should now have a prediction"
        assert all(p.label == "VALID" for p in second), (
            "the earlier error fallbacks were inherited instead of retried: "
            f"{[(p.bibtex_key, p.label) for p in second]}"
        )

    def test_clean_predictions_are_not_re_attempted(self, tmp_path: Path):
        """Retrying failures must not mean retrying everything."""
        entries = _entries(3)
        ckpt = tmp_path / "ckpt"
        _verify_entries(entries, _valid_response, "test", "model-x", checkpoint_dir=ckpt)

        calls: list[str] = []

        def _counting(prompt: str) -> str:
            calls.append(prompt)
            return _valid_response(prompt)

        preds = _verify_entries(entries, _counting, "test", "model-x", checkpoint_dir=ckpt)
        assert len(preds) == 3
        assert calls == [], "already-answered entries must not be re-queried"


@pytest.mark.parametrize(
    "path", sorted(_REPO_ROOT.glob("data/*/baseline_results/*.json")), ids=lambda p: p.name
)
def test_no_released_result_is_mostly_error_fallback(path: Path):
    """Released results may not be dominated by abstentions.

    A threshold rather than zero: genuine model abstention is legitimate and
    some released runs carry it. What this catches is the shape of a collapsed
    run -- one committed crossdomain result answers 68 of 500 entries.
    """
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        pytest.skip("not a result object")
    n = data.get("num_entries") or 0
    uncertain = data.get("num_uncertain") or 0
    if n == 0:
        pytest.skip("no entries")
    assert uncertain / n < 0.5, (
        f"{path.name}: {uncertain} of {n} predictions are UNCERTAIN. A run that "
        "abstains on most of a split is not a measurement of the split."
    )
