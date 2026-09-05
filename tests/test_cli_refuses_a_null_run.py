"""``hallmark evaluate --output`` must not write a run that evaluated nothing.

``evaluate()`` logs an error when every prediction is a fallback and then returns
a complete ``EvaluationResult`` with detection rate 0.0 and false-positive rate
0.0. Nothing downstream read ``num_evaluated``, so the CLI wrote that result to
disk like any other, the leaderboard ranked it, and the history log recorded it.
Four pre-screening ablations shipped that way. Refusing at the write is the
cheapest place to stop it; ``--allow-null-run`` exists for looking at the shape
of a failure on purpose.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hallmark import cli
from hallmark.baselines.common import fallback_predictions


@pytest.fixture
def null_baseline(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_run_baseline",
        lambda name, entries, split=None, **kw: fallback_predictions(
            entries, reason="Fallback: tool unavailable"
        ),
    )


def _argv(out: Path, *extra: str) -> list[str]:
    return [
        "evaluate",
        "--split",
        "dev_public",
        "--baseline",
        "doi_only",
        "--max-entries",
        "6",
        "--output",
        str(out),
        *extra,
    ]


def test_a_null_run_is_not_written(null_baseline, tmp_path, capsys):
    out = tmp_path / "null.json"
    rc = cli.main(_argv(out))
    assert rc != 0
    assert not out.exists(), "a run that evaluated nothing must not become a results file"
    assert "--allow-null-run" in capsys.readouterr().err


def test_the_override_writes_it_and_the_file_says_so(null_baseline, tmp_path):
    out = tmp_path / "null.json"
    rc = cli.main(_argv(out, "--allow-null-run"))
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["num_evaluated"] == 0


def test_a_real_run_is_unaffected(monkeypatch, tmp_path):
    from hallmark.dataset.schema import Prediction

    monkeypatch.setattr(
        cli,
        "_run_baseline",
        lambda name, entries, split=None, **kw: [
            Prediction(bibtex_key=e.bibtex_key, label="VALID", confidence=0.9) for e in entries
        ],
    )
    out = tmp_path / "real.json"
    assert cli.main(_argv(out)) == 0
    assert json.loads(out.read_text())["num_evaluated"] == 6
