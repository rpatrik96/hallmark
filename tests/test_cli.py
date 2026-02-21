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


# ── New feature tests ─────────────────────────────────────────────────────────


class TestEvaluateCIFlag:
    """--ci flag triggers bootstrap confidence interval computation."""

    def test_ci_flag_output_contains_interval_markers(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from hallmark.dataset.loader import load_split

        entries = load_split("dev_public")
        keys = [e.bibtex_key for e in entries[:30]]
        pred_file = _make_predictions_file(tmp_path, keys, label="HALLUCINATED")

        rc = main(["evaluate", "--split", "dev_public", "--predictions", pred_file, "--ci"])
        assert rc == 0
        out = capsys.readouterr().out
        # Bootstrap CIs are printed as "95% CI: [lo, hi]"
        assert "95%" in out or "CI" in out


class TestEvaluateFilterTier:
    """--filter-tier restricts entries to a single difficulty tier."""

    def test_filter_tier_reduces_entry_count(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from hallmark.dataset.loader import load_split

        entries = load_split("dev_public")
        all_keys = [e.bibtex_key for e in entries]
        pred_file = _make_predictions_file(tmp_path, all_keys)

        # Without filter
        main(["evaluate", "--split", "dev_public", "--predictions", pred_file])
        out_all = capsys.readouterr().out

        # With --filter-tier 1
        main(
            ["evaluate", "--split", "dev_public", "--predictions", pred_file, "--filter-tier", "1"]
        )
        out_filtered = capsys.readouterr().out

        # Extract entry counts from output (line contains "Entries:")
        def _parse_entries(text: str) -> int:
            for line in text.splitlines():
                if "Entries:" in line:
                    return int(line.split()[-1])
            return -1

        count_all = _parse_entries(out_all)
        count_filtered = _parse_entries(out_filtered)
        assert count_filtered < count_all


class TestEvaluateFormatLatex:
    """--format latex produces booktabs-formatted output."""

    def test_format_latex_contains_toprule(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from hallmark.dataset.loader import load_split

        entries = load_split("dev_public")
        keys = [e.bibtex_key for e in entries[:10]]
        pred_file = _make_predictions_file(tmp_path, keys)

        rc = main(
            [
                "evaluate",
                "--split",
                "dev_public",
                "--predictions",
                pred_file,
                "--format",
                "latex",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        # _format_latex wraps inside a leaderboard table; single-result evaluate
        # just emits the row. Check for LaTeX-style backslash-backslash row end.
        assert "\\\\" in out or "toprule" in out


class TestEvaluateBySource:
    """--by-source shows source-stratified metrics."""

    def test_by_source_output_contains_source_header(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from hallmark.dataset.loader import load_split

        entries = load_split("dev_public")
        keys = [e.bibtex_key for e in entries[:20]]
        pred_file = _make_predictions_file(tmp_path, keys)

        rc = main(
            [
                "evaluate",
                "--split",
                "dev_public",
                "--predictions",
                pred_file,
                "--by-source",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        # The by-source section header or "Source" column label
        assert "Source" in out or "source" in out


class TestStatsGenerationMethod:
    """stats command shows generation method distribution."""

    def test_stats_shows_method_distribution(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["stats", "--split", "dev_public"])
        assert rc == 0
        out = capsys.readouterr().out
        # The _cmd_stats function prints "Generation Method Distribution:" when present
        assert "Generation Method" in out or "method_distribution" in out


class TestContributeInvalidExitsNonzero:
    """contribute exits 1 when the file contains invalid entries."""

    def test_invalid_entries_exit_code_one(self, tmp_path: pytest.TempPathFactory) -> None:
        # Write a JSONL with entries that fail schema validation (missing required fields)
        bad_file = tmp_path / "bad_entries.jsonl"
        bad_entries = [
            {"bibtex_key": "bad1", "label": "HALLUCINATED"},  # missing required fields
            {"bibtex_key": "bad2"},  # almost empty
        ]
        bad_file.write_text("\n".join(json.dumps(e) for e in bad_entries))

        rc = main(["contribute", "--file", str(bad_file), "--contributor", "test"])
        assert rc == 1


class TestDiagnoseGate:
    """--gate exits 1 when misclassifications exist; exit 0 when all correct."""

    def test_gate_exits_one_on_misclassifications(self, tmp_path: pytest.TempPathFactory) -> None:
        from hallmark.dataset.loader import load_split

        entries = load_split("dev_public")
        # Predict VALID for everything — hallucinated entries will be misclassified
        keys = [e.bibtex_key for e in entries[:10]]
        pred_file = _make_predictions_file(tmp_path, keys, label="VALID")

        rc = main(
            [
                "diagnose",
                "--split",
                "dev_public",
                "--predictions",
                pred_file,
                "--gate",
            ]
        )
        # There are hallucinated entries in dev_public, so this must exit 1
        assert rc == 1

    def test_no_gate_exits_zero_despite_misclassifications(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from hallmark.dataset.loader import load_split

        entries = load_split("dev_public")
        keys = [e.bibtex_key for e in entries[:10]]
        pred_file = _make_predictions_file(tmp_path, keys, label="VALID")

        rc = main(["diagnose", "--split", "dev_public", "--predictions", pred_file])
        # Without --gate the command always exits 0
        assert rc == 0


class TestEvaluateSavePredictions:
    """--save-predictions writes predictions to the given path."""

    def test_save_predictions_creates_file(self, tmp_path: pytest.TempPathFactory) -> None:
        from hallmark.dataset.loader import load_split

        entries = load_split("dev_public")
        keys = [e.bibtex_key for e in entries[:10]]
        pred_file = _make_predictions_file(tmp_path, keys)
        save_path = tmp_path / "saved_preds.jsonl"

        rc = main(
            [
                "evaluate",
                "--split",
                "dev_public",
                "--predictions",
                pred_file,
                "--save-predictions",
                str(save_path),
            ]
        )
        assert rc == 0
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestHistoryAppendAllMetrics:
    """history-append writes all expected metric fields to the history file."""

    def test_history_append_contains_new_fields(self, tmp_path: pytest.TempPathFactory) -> None:
        # Create a fake result JSON in a results dir
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        result_file = results_dir / "test_tool_dev_public.json"
        # Write a minimal result JSON with all expected fields
        result_data = {
            "tool_name": "test_tool",
            "split_name": "dev_public",
            "f1_hallucination": 0.5,
            "detection_rate": 0.6,
            "false_positive_rate": 0.2,
            "tier_weighted_f1": 0.4,
            "num_entries": 10,
            "cost_efficiency": None,
            "mcc": 0.3,
            "ece": 0.1,
            "auroc": 0.7,
            "auprc": 0.65,
            "coverage": 1.0,
            "coverage_adjusted_f1": 0.5,
        }
        result_file.write_text(json.dumps(result_data))

        history_file = tmp_path / "history.jsonl"
        rc = main(
            [
                "history-append",
                "--results-dir",
                str(results_dir),
                "--output",
                str(history_file),
            ]
        )
        assert rc == 0
        assert history_file.exists()

        with open(history_file) as f:
            record = json.loads(f.readline())

        for field in ("mcc", "ece", "auroc", "auprc", "coverage", "coverage_adjusted_f1"):
            assert field in record, f"Field '{field}' missing from history record"


class TestLeaderboardCovF1Column:
    """leaderboard text output contains CovF1 column header."""

    def test_leaderboard_text_has_cov_f1(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        result_data = {
            "tool_name": "my_tool",
            "split_name": "dev_public",
            "f1_hallucination": 0.5,
            "detection_rate": 0.6,
            "false_positive_rate": 0.2,
            "tier_weighted_f1": 0.4,
            "mcc": 0.3,
            "coverage_adjusted_f1": 0.5,
        }
        (results_dir / "my_tool_dev_public.json").write_text(json.dumps(result_data))

        rc = main(
            [
                "leaderboard",
                "--split",
                "dev_public",
                "--results-dir",
                str(results_dir),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "CovF1" in out
