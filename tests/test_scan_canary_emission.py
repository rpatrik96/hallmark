"""Tests for scripts/scan_canary_emission.py — the canary-emission CLI scanner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from hallmark.dataset.schema import CANARY_GUID, CANARY_STRING

# scan_canary_emission lives in scripts/; load it by path. Register in
# sys.modules before exec so its @dataclass defs initialize correctly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "scan_canary_emission.py"
_spec = importlib.util.spec_from_file_location("scan_canary_emission", _SCRIPT)
assert _spec is not None and _spec.loader is not None
sce = importlib.util.module_from_spec(_spec)
sys.modules["scan_canary_emission"] = sce
_spec.loader.exec_module(sce)


class TestScanPredictionsFile:
    def test_clean_predictions(self, tmp_path):
        p = tmp_path / "clean.jsonl"
        p.write_text(
            '{"bibtex_key": "a", "label": "VALID", "reason": "looks fine"}\n'
            '{"bibtex_key": "b", "label": "HALLUCINATED", "reason": "fake doi"}\n'
        )
        result = sce.scan_predictions_file(p)
        assert result.clean
        assert result.scanned_units == 2
        assert result.hits == []

    def test_guid_emitted_in_reason(self, tmp_path):
        p = tmp_path / "dirty.jsonl"
        p.write_text(
            f'{{"bibtex_key": "c", "label": "VALID", "reason": "saw {CANARY_GUID} once"}}\n'
        )
        result = sce.scan_predictions_file(p)
        assert not result.clean
        assert len(result.hits) == 1
        assert CANARY_GUID in result.hits[0].markers
        assert "bibtex_key=c" in result.hits[0].location

    def test_malformed_line_scanned_as_text(self, tmp_path):
        p = tmp_path / "broken.jsonl"
        p.write_text(f"this is not json but contains {CANARY_GUID}\n")
        result = sce.scan_predictions_file(p)
        assert not result.clean
        assert CANARY_GUID in result.hits[0].markers


class TestScanTextFile:
    def test_full_sentence_detected(self, tmp_path):
        p = tmp_path / "dump.txt"
        p.write_text(f"model output ... {CANARY_STRING} ... end")
        result = sce.scan_text_file(p)
        assert not result.clean
        assert CANARY_GUID in result.hits[0].markers
        assert CANARY_STRING in result.hits[0].markers

    def test_clean_text(self, tmp_path):
        p = tmp_path / "ok.txt"
        p.write_text("an entirely ordinary log line with no secrets")
        result = sce.scan_text_file(p)
        assert result.clean


class TestMainExitCodes:
    def test_main_clean_exit_zero(self, tmp_path):
        p = tmp_path / "clean.jsonl"
        p.write_text('{"bibtex_key": "a", "label": "VALID", "reason": "ok"}\n')
        assert sce.main([str(p)]) == 0

    def test_main_contaminated_exit_two(self, tmp_path):
        p = tmp_path / "dirty.jsonl"
        p.write_text(f'{{"bibtex_key": "a", "label": "VALID", "reason": "{CANARY_GUID}"}}\n')
        assert sce.main([str(p)]) == 2

    def test_main_text_flag(self, tmp_path):
        p = tmp_path / "dump.txt"
        p.write_text(f"contains {CANARY_GUID}")
        assert sce.main([str(p), "--text"]) == 2
