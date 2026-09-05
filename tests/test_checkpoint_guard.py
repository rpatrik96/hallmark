"""Tests for the checkpoint guard that refuses to persist an evidence-free run.

The 2026-09-02 incident had two halves. The first was bibtex-check reporting a
wifi outage as ``not_found``; the second, covered here, is that the long-running
evaluation scripts checkpointed the resulting records and then skipped those keys
on every later run, which is what made the loss permanent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from hallmark.baselines.checkpoint_guard import (
    GuardedCheckpointWriter,
    PoisonedBatchError,
    RunHealthTracker,
    assess_run_health,
    is_error_record,
    is_transport_error_record,
    quarantine_error_records,
    refusal_message,
    rejected_path_for,
)

_HAS_OPENAI = importlib.util.find_spec("openai") is not None


def _ok(key: str) -> dict[str, Any]:
    return {
        "bibtex_key": key,
        "label": "VALID",
        "confidence": 0.9,
        "reason": "verified against crossref",
    }


def _api_error(key: str) -> dict[str, Any]:
    return {
        "bibtex_key": key,
        "label": "UNCERTAIN",
        "confidence": 0.5,
        "reason": "[Error fallback] API error: APIConnectionError: Connection error.",
    }


def _parse_error(key: str) -> dict[str, Any]:
    return {
        "bibtex_key": key,
        "label": "UNCERTAIN",
        "confidence": 0.5,
        "reason": "[Error fallback] Parse error: {garbled",
    }


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestRecordClassification:
    def test_verdict_record_is_not_an_error(self) -> None:
        assert not is_error_record(_ok("a"))
        assert not is_transport_error_record(_ok("a"))

    def test_api_error_is_a_transport_failure(self) -> None:
        assert is_error_record(_api_error("a"))
        assert is_transport_error_record(_api_error("a"))

    def test_agentic_error_prefix_is_recognised(self) -> None:
        rec = {"bibtex_key": "a", "reason": "[Agentic error] RateLimitError: 429"}
        assert is_error_record(rec)
        assert is_transport_error_record(rec)

    def test_parse_error_is_unusable_but_not_transport(self) -> None:
        assert is_error_record(_parse_error("a"))
        assert not is_transport_error_record(_parse_error("a"))

    def test_salvaged_record_is_a_verdict(self) -> None:
        """A salvaged record has a real label parsed out of truncated output."""
        rec = {
            "bibtex_key": "a",
            "label": "HALLUCINATED",
            "reason": "[Salvaged] label/confidence extracted from truncated JSON",
        }
        assert not is_error_record(rec)


class TestAssessRunHealth:
    def test_outage_run_trips(self) -> None:
        records = [_api_error(f"k{i}") for i in range(38)] + [_ok(f"g{i}") for i in range(2)]
        health = assess_run_health(records)
        assert health.suspected_transport_failure
        assert health.transport_error == 38
        assert health.no_evidence_share == pytest.approx(0.95)

    def test_normal_run_does_not_trip(self) -> None:
        records = [_ok(f"k{i}") for i in range(38)] + [_api_error(f"e{i}") for i in range(2)]
        health = assess_run_health(records)
        assert not health.suspected_transport_failure

    def test_unusable_answers_count_toward_the_share(self) -> None:
        records = [_parse_error(f"k{i}") for i in range(40)]
        health = assess_run_health(records)
        assert health.suspected_transport_failure
        assert health.transport_error == 0
        assert health.coverage_incomplete == 40

    def test_short_run_is_not_judged(self) -> None:
        health = assess_run_health([_api_error(f"k{i}") for i in range(10)])
        assert not health.suspected_transport_failure

    def test_refusal_message_names_the_cause_and_the_remedy(self) -> None:
        health = assess_run_health([_api_error(f"k{i}") for i in range(40)])
        message = refusal_message(health, Path("/tmp/ckpt.jsonl"))
        assert "40/40" in message
        assert "transport" in message
        assert "retried" in message
        assert "/tmp/ckpt.jsonl" in message


class TestRunHealthTracker:
    def test_tracker_raises_once_the_run_is_poisoned(self, tmp_path: Path) -> None:
        tracker = RunHealthTracker(checkpoint_path=tmp_path / "c.jsonl")
        with pytest.raises(PoisonedBatchError) as excinfo:
            for i in range(40):
                tracker.add(_api_error(f"k{i}"))
        assert excinfo.value.health.suspected_transport_failure

    def test_tracker_stays_quiet_on_a_normal_run(self, tmp_path: Path) -> None:
        tracker = RunHealthTracker(checkpoint_path=tmp_path / "c.jsonl")
        for i in range(40):
            tracker.add(_ok(f"k{i}"))
        assert not tracker.health.suspected_transport_failure
        assert tracker.health.total == 40


class TestGuardedCheckpointWriter:
    def test_verdicts_are_checkpointed(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.jsonl"
        with GuardedCheckpointWriter(path) as writer:
            for i in range(5):
                writer.add(_ok(f"k{i}"))
        assert [r["bibtex_key"] for r in _read(path)] == [f"k{i}" for i in range(5)]
        assert writer.written == 5
        assert not writer.rejected_path.exists()

    def test_error_records_never_reach_the_checkpoint(self, tmp_path: Path) -> None:
        """The core of the incident: a failed call must stay retryable."""
        path = tmp_path / "ckpt.jsonl"
        with GuardedCheckpointWriter(path) as writer:
            writer.add(_ok("good"))
            writer.add(_api_error("bad"))
        keys = {r["bibtex_key"] for r in _read(path)}
        assert keys == {"good"}
        assert "bad" not in keys
        assert [r["bibtex_key"] for r in _read(writer.rejected_path)] == ["bad"]

    def test_poisoned_run_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.jsonl"
        writer = GuardedCheckpointWriter(path)
        with pytest.raises(PoisonedBatchError), writer:
            for i in range(40):
                writer.add(_api_error(f"k{i}"))
        # Nothing usable arrived, so the checkpoint holds nothing and every key
        # is retried by the next run.
        assert _read(path) == []
        assert len(_read(writer.rejected_path)) >= 1

    def test_healthy_prefix_survives_a_later_outage(self, tmp_path: Path) -> None:
        """An outage mid-run keeps the verdicts already earned."""
        path = tmp_path / "ckpt.jsonl"
        writer = GuardedCheckpointWriter(path)
        with pytest.raises(PoisonedBatchError), writer:
            for i in range(30):
                writer.add(_ok(f"g{i}"))
            for i in range(30):
                writer.add(_api_error(f"b{i}"))
        keys = {r["bibtex_key"] for r in _read(path)}
        assert keys == {f"g{i}" for i in range(30)}
        assert not any(k.startswith("b") for k in keys)

    def test_rejected_path_is_a_sidecar_of_the_checkpoint(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.jsonl"
        sidecar = rejected_path_for(path)
        assert sidecar.parent == path.parent
        assert sidecar.name.startswith("ckpt.jsonl.rejected-")
        assert sidecar.name.endswith(".jsonl")


class TestQuarantineErrorRecords:
    def test_only_this_run_s_error_lines_move(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.jsonl"
        records = [_ok("old"), _api_error("older"), _ok("kept"), _api_error("bad")]
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        sidecar = quarantine_error_records(path, {"bad"})

        assert sidecar is not None
        assert [r["bibtex_key"] for r in _read(sidecar)] == ["bad"]
        # Everything else survives unchanged and in order, including an error
        # record from an earlier run that this run did not touch.
        assert [r["bibtex_key"] for r in _read(path)] == ["old", "older", "kept"]

    def test_nothing_to_quarantine_leaves_the_file_alone(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.jsonl"
        original = json.dumps(_ok("a")) + "\n"
        path.write_text(original)
        assert quarantine_error_records(path, {"a"}) is None
        assert path.read_text() == original

    def test_missing_checkpoint_is_safe(self, tmp_path: Path) -> None:
        assert quarantine_error_records(tmp_path / "nope.jsonl", {"a"}) is None


@pytest.mark.skipif(not _HAS_OPENAI, reason="scripts import the openai SDK")
class TestAgenticScriptResume:
    """``_read_done_keys`` must not treat a failed record as done."""

    @staticmethod
    def _load_script() -> Any:
        script = (
            Path(__file__).resolve().parent.parent
            / "scripts"
            / "parallel_agentic_btu_test_public.py"
        )
        spec = importlib.util.spec_from_file_location("parallel_agentic_btu", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_error_records_are_retried_not_skipped(self, tmp_path: Path) -> None:
        module = self._load_script()
        path = tmp_path / "ckpt.jsonl"
        records = [_ok("good"), _api_error("failed")]
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        assert module._read_done_keys(path) == {"good"}

    def test_a_later_verdict_supersedes_an_earlier_failure(self, tmp_path: Path) -> None:
        module = self._load_script()
        path = tmp_path / "ckpt.jsonl"
        records = [_api_error("k"), _ok("k")]
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        assert module._read_done_keys(path) == {"k"}

    def test_a_later_failure_reopens_an_earlier_verdict(self, tmp_path: Path) -> None:
        module = self._load_script()
        path = tmp_path / "ckpt.jsonl"
        records = [_ok("k"), _api_error("k")]
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        assert module._read_done_keys(path) == set()


@pytest.mark.skipif(not _HAS_OPENAI, reason="the script imports the openai SDK")
class TestResumeScriptWiring:
    """End-to-end: the runner refuses an outage instead of checkpointing it."""

    @staticmethod
    def _load_script() -> Any:
        script = (
            Path(__file__).resolve().parent.parent / "scripts" / "parallel_resume_test_public.py"
        )
        spec = importlib.util.spec_from_file_location("parallel_resume_public", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _data_file(tmp_path: Path, n: int) -> Path:
        path = tmp_path / "entries.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "bibtex_key": f"k{i}",
                        "bibtex_type": "article",
                        "fields": {"title": "T", "author": "A", "year": "2024"},
                    }
                )
                for i in range(n)
            )
            + "\n"
        )
        return path

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        responder: Any,
        n: int = 40,
    ) -> tuple[Any, Path]:
        module = self._load_script()
        monkeypatch.setattr(module, "call_one", responder)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        checkpoint_dir = tmp_path / "ckpt"
        monkeypatch.setattr(
            "sys.argv",
            [
                "parallel_resume_test_public.py",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--model",
                "test/model",
                "--jsonl-name",
                "preds.jsonl",
                "--data-file",
                str(self._data_file(tmp_path, n)),
                "--workers",
                "2",
            ],
        )
        return module, checkpoint_dir / "preds.jsonl"

    def test_outage_run_exits_without_checkpointing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _always_fails(_client: Any, _model: str, entry: dict, *_a: Any) -> dict[str, Any]:
            return _api_error(entry["bibtex_key"])

        module, jsonl_path = self._run(monkeypatch, tmp_path, _always_fails)
        with pytest.raises(SystemExit) as excinfo:
            module.main()

        assert "Refusing to checkpoint" in str(excinfo.value)
        # No key was persisted, so every entry is retried by the next run.
        assert _read(jsonl_path) == []

    def test_healthy_run_checkpoints_normally(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _always_succeeds(_client: Any, _model: str, entry: dict, *_a: Any) -> dict[str, Any]:
            return _ok(entry["bibtex_key"])

        module, jsonl_path = self._run(monkeypatch, tmp_path, _always_succeeds)
        module.main()
        assert {r["bibtex_key"] for r in _read(jsonl_path)} == {f"k{i}" for i in range(40)}
