"""Tests for hallmark.cli."""

from __future__ import annotations

import json

import pytest

from hallmark.cli import _SPLIT_CHOICES, main


def test_stats_dev_public(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["stats", "--split", "dev_public"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "HALLMARK Statistics" in out


def test_list_baselines(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["list-baselines"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Registered Baselines" in out


def test_validate_predictions_nonexistent_file() -> None:
    rc = main(["validate-predictions", "--file", "/nonexistent/path/preds.jsonl"])
    assert rc == 1


def test_validate_predictions_valid_file(tmp_path: pytest.TempPathFactory) -> None:
    pred_file = tmp_path / "preds.jsonl"
    preds = [
        {"bibtex_key": "smith2024foo", "label": "HALLUCINATED", "confidence": 0.9},
        {"bibtex_key": "jones2023bar", "label": "VALID", "confidence": 0.1},
        {"bibtex_key": "lee2022baz", "label": "VALID", "confidence": 0.2},
    ]
    pred_file.write_text("\n".join(json.dumps(p) for p in preds))
    rc = main(["validate-predictions", "--file", str(pred_file)])
    assert rc == 0


def test_validate_predictions_invalid_label(tmp_path: pytest.TempPathFactory) -> None:
    pred_file = tmp_path / "bad_preds.jsonl"
    preds = [
        {"bibtex_key": "smith2024foo", "label": "HALLUCINATED"},
        # "WRONG" is not a valid label
        {"bibtex_key": "jones2023bar", "label": "WRONG"},
    ]
    pred_file.write_text("\n".join(json.dumps(p) for p in preds))
    rc = main(["validate-predictions", "--file", str(pred_file)])
    assert rc == 1


def test_split_choices() -> None:
    assert _SPLIT_CHOICES == ["dev_public", "test_public", "test_hidden", "stress_test"]
