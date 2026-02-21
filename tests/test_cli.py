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


# ── F-20: evaluate and diagnose CLI tests ─────────────────────────────────────


def _make_predictions_file(
    tmp_path: pytest.TempPathFactory, keys: list[str], label: str = "VALID"
) -> str:
    """Write a minimal JSONL predictions file and return its path."""
    pred_file = tmp_path / "preds.jsonl"
    preds = [{"bibtex_key": k, "label": label, "confidence": 0.8} for k in keys]
    pred_file.write_text("\n".join(json.dumps(p) for p in preds))
    return str(pred_file)


class TestEvaluateWithPredictions:
    """F-20a: smoke test for 'evaluate --predictions'."""

    def test_evaluate_with_predictions_exits_zero(self, tmp_path: pytest.TempPathFactory) -> None:
        """evaluate with a predictions file should return exit code 0."""
        from hallmark.dataset.loader import load_split

        entries = load_split("dev_public")
        # Build predictions matching ALL entry keys so coverage = 100%
        keys = [e.bibtex_key for e in entries]
        pred_file = _make_predictions_file(tmp_path, keys, label="VALID")

        rc = main(["evaluate", "--split", "dev_public", "--predictions", pred_file])
        assert rc == 0

    def test_evaluate_with_predictions_shows_metrics(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Output should include standard metric names."""
        from hallmark.dataset.loader import load_split

        entries = load_split("dev_public")
        keys = [e.bibtex_key for e in entries]
        pred_file = _make_predictions_file(tmp_path, keys)

        main(["evaluate", "--split", "dev_public", "--predictions", pred_file])
        out = capsys.readouterr().out
        assert "Detection Rate" in out
        assert "F1" in out

    def test_evaluate_with_subset_predictions(self, tmp_path: pytest.TempPathFactory) -> None:
        """Predictions covering only a subset of entries still returns exit code 0."""
        from hallmark.dataset.loader import load_split

        entries = load_split("dev_public")
        keys = [e.bibtex_key for e in entries[:20]]
        pred_file = _make_predictions_file(tmp_path, keys)

        rc = main(["evaluate", "--split", "dev_public", "--predictions", pred_file])
        assert rc == 0

    def test_evaluate_csv_format(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--format csv should output CSV with header row."""
        from hallmark.dataset.loader import load_split

        entries = load_split("dev_public")
        keys = [e.bibtex_key for e in entries[:10]]
        pred_file = _make_predictions_file(tmp_path, keys)

        rc = main(
            ["evaluate", "--split", "dev_public", "--predictions", pred_file, "--format", "csv"]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "tool" in out
        assert "detection_rate" in out


class TestEvaluateUnavailableBaseline:
    """F-20b: unavailable baseline produces clean error, not a raw traceback."""

    def test_nonexistent_baseline_exits_nonzero(self) -> None:
        rc = main(
            ["evaluate", "--split", "dev_public", "--baseline", "totally_nonexistent_baseline_xyz"]
        )
        assert rc != 0

    def test_nonexistent_baseline_no_traceback(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Error message should not contain a raw Python traceback."""
        main(
            ["evaluate", "--split", "dev_public", "--baseline", "totally_nonexistent_baseline_xyz"]
        )
        captured = capsys.readouterr()
        # The error should appear on stderr (via logging.error), not as a raw traceback
        assert "Traceback" not in captured.out
        assert "Traceback" not in captured.err

    def test_nonexistent_baseline_helpful_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """Error output should hint at the list-baselines command."""
        import logging

        with caplog.at_level(logging.ERROR):
            main(
                [
                    "evaluate",
                    "--split",
                    "dev_public",
                    "--baseline",
                    "totally_nonexistent_baseline_xyz",
                ]
            )
        combined = caplog.text
        assert "list-baselines" in combined or "not available" in combined.lower()


class TestDiagnoseFullFlag:
    """F-20c: 'diagnose --full' shows full reason strings; default truncates at 80 chars."""

    def _make_preds_with_long_reason(
        self, tmp_path: pytest.TempPathFactory, keys: list[str]
    ) -> str:
        long_reason = "A" * 200  # 200-char reason, well beyond the 80-char truncation
        pred_file = tmp_path / "diag_preds.jsonl"
        preds = [
            {"bibtex_key": k, "label": "VALID", "confidence": 0.8, "reason": long_reason}
            for k in keys
        ]
        pred_file.write_text("\n".join(json.dumps(p) for p in preds))
        return str(pred_file)

    def test_diagnose_without_full_truncates(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without --full, reason strings are truncated to 80 chars."""
        from hallmark.dataset.loader import load_split

        entries = load_split("dev_public")
        keys = [e.bibtex_key for e in entries[:5]]
        pred_file = self._make_preds_with_long_reason(tmp_path, keys)

        rc = main(["diagnose", "--split", "dev_public", "--predictions", pred_file])
        assert rc == 0
        out = capsys.readouterr().out
        # The 200-char reason should be truncated to 80
        assert "A" * 200 not in out
        assert "A" * 80 in out

    def test_diagnose_with_full_shows_complete_reason(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With --full, the complete reason string appears in output."""
        from hallmark.dataset.loader import load_split

        entries = load_split("dev_public")
        keys = [e.bibtex_key for e in entries[:5]]
        pred_file = self._make_preds_with_long_reason(tmp_path, keys)

        rc = main(["diagnose", "--split", "dev_public", "--predictions", pred_file, "--full"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "A" * 200 in out
