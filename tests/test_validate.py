"""Tests for reference result validation and EvaluationResult roundtrip."""

from __future__ import annotations

import json

import pytest

from hallmark.dataset.schema import EvaluationResult
from hallmark.evaluation.validate import compute_sha256, validate_reference_results

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_result(**overrides: object) -> EvaluationResult:
    """Create a minimal valid EvaluationResult with optional overrides."""
    defaults: dict = {
        "tool_name": "test_tool",
        "split_name": "dev_public",
        "num_entries": 500,
        "num_hallucinated": 50,
        "num_valid": 450,
        "detection_rate": 0.8,
        "false_positive_rate": 0.02,
        "f1_hallucination": 0.75,
        "tier_weighted_f1": 0.70,
        "per_tier_metrics": {1: {"f1": 0.9, "detection_rate": 0.95, "count": 20}},
        "detect_at_k": {1: 0.6, 2: 0.8},
    }
    defaults.update(overrides)
    return EvaluationResult(**defaults)


def _write_result_and_manifest(tmp_path, result, filename="test_tool_dev_public.json"):
    """Write an EvaluationResult and matching manifest to tmp_path."""
    result_path = tmp_path / filename
    result_path.write_text(result.to_json())

    sha = compute_sha256(result_path)
    manifest = {
        "version": "1.0",
        "files": {
            filename: {
                "sha256": sha,
                "baseline": result.tool_name,
                "split": result.split_name,
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return result_path


# ---------------------------------------------------------------------------
# TestComputeSha256
# ---------------------------------------------------------------------------


class TestComputeSha256:
    def test_known_content(self, tmp_path):
        p = tmp_path / "hello.txt"
        p.write_text("hello world")
        digest = compute_sha256(p)
        assert len(digest) == 64
        assert isinstance(digest, str)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_bytes(b"")
        digest = compute_sha256(p)
        # SHA-256 of empty input is a known constant
        assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_deterministic(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"\x00\x01\x02" * 1000)
        assert compute_sha256(p) == compute_sha256(p)


# ---------------------------------------------------------------------------
# TestValidateReferenceResults
# ---------------------------------------------------------------------------


class TestValidateReferenceResults:
    def test_valid_directory(self, tmp_path):
        result = _make_eval_result()
        _write_result_and_manifest(tmp_path, result)

        vr = validate_reference_results(tmp_path)
        assert vr.passed
        assert vr.errors == []

    def test_missing_manifest(self, tmp_path):
        vr = validate_reference_results(tmp_path)
        assert not vr.passed
        assert any("manifest.json not found" in e for e in vr.errors)

    def test_empty_manifest(self, tmp_path):
        (tmp_path / "manifest.json").write_text('{"files": {}}')
        vr = validate_reference_results(tmp_path)
        assert vr.passed
        assert any("no files" in w for w in vr.warnings)

    def test_checksum_mismatch(self, tmp_path):
        result = _make_eval_result()
        filename = "test_tool_dev_public.json"
        result_path = tmp_path / filename
        result_path.write_text(result.to_json())

        manifest = {
            "files": {
                filename: {
                    "sha256": "0" * 64,  # wrong checksum
                    "baseline": "test_tool",
                    "split": "dev_public",
                }
            }
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        vr = validate_reference_results(tmp_path)
        assert not vr.passed
        assert any("checksum mismatch" in e for e in vr.errors)

    def test_missing_file(self, tmp_path):
        manifest = {
            "files": {
                "nonexistent.json": {
                    "sha256": "abc123",
                    "baseline": "test_tool",
                    "split": "dev_public",
                }
            }
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        vr = validate_reference_results(tmp_path)
        assert not vr.passed
        assert any("file not found" in e for e in vr.errors)

    def test_invalid_schema(self, tmp_path):
        filename = "bad.json"
        (tmp_path / filename).write_text('{"not": "an evaluation result"}')
        sha = compute_sha256(tmp_path / filename)

        manifest = {"files": {filename: {"sha256": sha}}}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        vr = validate_reference_results(tmp_path)
        assert not vr.passed
        assert any("failed to deserialize" in e for e in vr.errors)

    def test_strict_rejects_zero_f1(self, tmp_path):
        result = _make_eval_result(f1_hallucination=0.0)
        _write_result_and_manifest(tmp_path, result)

        vr = validate_reference_results(tmp_path, strict=True)
        assert not vr.passed
        assert any("F1=0.0" in e for e in vr.errors)

    def test_non_strict_allows_zero_f1(self, tmp_path):
        result = _make_eval_result(f1_hallucination=0.0)
        _write_result_and_manifest(tmp_path, result)

        vr = validate_reference_results(tmp_path, strict=False)
        assert vr.passed

    def test_metadata_cross_validation(self, tmp_path):
        result = _make_eval_result(num_entries=100)  # doesn't match metadata (500)
        _write_result_and_manifest(tmp_path, result)

        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps({"splits": {"dev_public": {"total": 500}}}))

        vr = validate_reference_results(tmp_path, metadata_path=meta_path)
        assert not vr.passed
        assert any("num_entries" in e for e in vr.errors)


# ---------------------------------------------------------------------------
# TestEvaluationResultRoundtrip
# ---------------------------------------------------------------------------


class TestEvaluationResultRoundtrip:
    def test_to_json_from_json(self):
        original = _make_eval_result()
        restored = EvaluationResult.from_json(original.to_json())
        assert restored.tool_name == original.tool_name
        assert restored.f1_hallucination == original.f1_hallucination
        assert restored.num_entries == original.num_entries

    def test_int_key_coercion_per_tier(self):
        original = _make_eval_result(
            per_tier_metrics={1: {"f1": 0.9}, 2: {"f1": 0.7}},
        )
        json_text = original.to_json()
        # JSON will have string keys "1", "2"
        raw = json.loads(json_text)
        assert "1" in raw["per_tier_metrics"]

        restored = EvaluationResult.from_json(json_text)
        assert 1 in restored.per_tier_metrics
        assert 2 in restored.per_tier_metrics

    def test_int_key_coercion_detect_at_k(self):
        original = _make_eval_result(detect_at_k={1: 0.6, 3: 0.9})
        restored = EvaluationResult.from_json(original.to_json())
        assert 1 in restored.detect_at_k
        assert 3 in restored.detect_at_k
        assert restored.detect_at_k[1] == pytest.approx(0.6)

    def test_from_dict_does_not_mutate_input(self):
        data = json.loads(_make_eval_result().to_json())
        original_keys = set(data["per_tier_metrics"].keys())
        EvaluationResult.from_dict(data)
        # Original dict should be unchanged (string keys)
        assert set(data["per_tier_metrics"].keys()) == original_keys

    def test_optional_fields_roundtrip(self):
        result = _make_eval_result(
            temporal_robustness=0.85,
            cost_efficiency=12.5,
            mean_api_calls=3.2,
        )
        restored = EvaluationResult.from_json(result.to_json())
        assert restored.temporal_robustness == pytest.approx(0.85)
        assert restored.cost_efficiency == pytest.approx(12.5)
        assert restored.mean_api_calls == pytest.approx(3.2)
