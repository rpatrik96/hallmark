"""Tests for hallmark.evaluation.reuse_tracker."""

from __future__ import annotations

import json

from hallmark.evaluation.reuse_tracker import (
    compute_reuse_budget,
    estimate_remaining_budget,
    log_evaluation,
)


class TestEstimateRemainingBudget:
    def test_zero_k_has_budget(self):
        # With ratio=10, k_max = 100/ln(1000) ≈ 14 → remaining = 14
        remaining = estimate_remaining_budget(n=1000, k_current=0, max_budget_ratio=10.0)
        assert remaining > 0

    def test_large_k_exhausted(self):
        remaining = estimate_remaining_budget(n=100, k_current=10000, max_budget_ratio=10.0)
        assert remaining == 0

    def test_n_one_returns_zero(self):
        assert estimate_remaining_budget(n=1, k_current=0) == 0

    def test_budget_decreases_with_k(self):
        r1 = estimate_remaining_budget(n=1000, k_current=0, max_budget_ratio=10.0)
        r2 = estimate_remaining_budget(n=1000, k_current=r1 // 2, max_budget_ratio=10.0)
        assert r2 < r1


class TestComputeReuseBudget:
    def test_no_history_file(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        budgets = compute_reuse_budget(path, split_sizes={"test": 100}, max_budget_ratio=10.0)
        assert "test" in budgets
        assert budgets["test"].k == 0
        assert budgets["test"].remaining_evaluations > 0

    def test_with_history(self, tmp_path):
        path = tmp_path / "history.jsonl"
        records = [
            {"timestamp": "2026-01-01T00:00:00Z", "tool_name": "tool_a", "split_name": "dev"},
            {"timestamp": "2026-01-02T00:00:00Z", "tool_name": "tool_b", "split_name": "dev"},
            {"timestamp": "2026-01-03T00:00:00Z", "tool_name": "tool_a", "split_name": "dev"},
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        budgets = compute_reuse_budget(path, split_sizes={"dev": 500})
        assert budgets["dev"].k == 3
        assert budgets["dev"].unique_tools == 2
        assert budgets["dev"].first_evaluation == "2026-01-01T00:00:00Z"
        assert budgets["dev"].last_evaluation == "2026-01-03T00:00:00Z"

    def test_budget_ratio_computation(self, tmp_path):
        path = tmp_path / "history.jsonl"
        records = [
            {"timestamp": f"2026-01-{i:02d}T00:00:00Z", "tool_name": f"t{i}", "split_name": "s"}
            for i in range(1, 11)
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        budgets = compute_reuse_budget(path, split_sizes={"s": 1000})
        b = budgets["s"]
        assert b.budget_ratio > 0
        assert b.non_adaptive_bound > 0
        assert b.adaptive_bound > 0

    def test_unique_tools_counted(self, tmp_path):
        path = tmp_path / "history.jsonl"
        records = [
            {"timestamp": "2026-01-01T00:00:00Z", "tool_name": "tool_a", "split_name": "s"},
            {"timestamp": "2026-01-02T00:00:00Z", "tool_name": "tool_a", "split_name": "s"},
            {"timestamp": "2026-01-03T00:00:00Z", "tool_name": "tool_b", "split_name": "s"},
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        budgets = compute_reuse_budget(path, split_sizes={"s": 100})
        assert budgets["s"].unique_tools == 2
        assert budgets["s"].k == 3


class TestLogEvaluation:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "history.jsonl"
        log_evaluation(path, "tool_a", "dev_public")
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["tool_name"] == "tool_a"
        assert record["split_name"] == "dev_public"
        assert "timestamp" in record

    def test_appends(self, tmp_path):
        path = tmp_path / "history.jsonl"
        log_evaluation(path, "tool_a", "dev")
        log_evaluation(path, "tool_b", "dev")
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_includes_metrics(self, tmp_path):
        path = tmp_path / "history.jsonl"
        log_evaluation(path, "tool_a", "dev", metrics={"f1": 0.85})
        record = json.loads(path.read_text().strip())
        assert record["metrics"]["f1"] == 0.85

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "history.jsonl"
        log_evaluation(path, "tool_a", "dev")
        assert path.exists()
