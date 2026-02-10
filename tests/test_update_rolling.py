"""Tests for the rolling dataset update pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from hallmark.contribution.pool_manager import PoolManager
from hallmark.dataset.loader import _resolve_rolling_path, load_metadata, load_split
from hallmark.dataset.schema import BenchmarkEntry, save_entries
from hallmark.dataset.scraper import _request_with_retry

# --- Helpers ---


def _make_valid_entry(key: str = "test2024valid") -> BenchmarkEntry:
    return BenchmarkEntry(
        bibtex_key=key,
        bibtex_type="inproceedings",
        fields={
            "title": "Test Paper Title",
            "author": "Author One and Author Two",
            "year": "2024",
            "booktitle": "NeurIPS",
            "doi": "10.1234/test.001",
        },
        label="VALID",
        explanation="Valid test entry",
        generation_method="scraped",
        source_conference="NeurIPS",
        publication_date="2024-01-01",
        added_to_benchmark="2026-01-01",
        subtests={
            "doi_resolves": True,
            "title_exists": True,
            "authors_match": True,
            "venue_real": True,
            "fields_complete": True,
            "cross_db_agreement": True,
        },
    )


def _make_hallucinated_entry(key: str = "test2024fake") -> BenchmarkEntry:
    return BenchmarkEntry(
        bibtex_key=key,
        bibtex_type="inproceedings",
        fields={
            "title": "Fake Paper",
            "author": "Nobody Real",
            "year": "2024",
            "booktitle": "FakeConf",
        },
        label="HALLUCINATED",
        hallucination_type="fabricated_doi",
        difficulty_tier=1,
        explanation="Fabricated for testing",
        generation_method="perturbation",
        subtests={
            "doi_resolves": False,
            "title_exists": True,
            "authors_match": True,
            "venue_real": False,
            "fields_complete": True,
            "cross_db_agreement": False,
        },
    )


# --- Tests for _request_with_retry ---


class TestRequestWithRetry:
    def test_success_on_first_try(self) -> None:
        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.raise_for_status = MagicMock()
        mock_client.request.return_value = mock_resp

        result = _request_with_retry(mock_client, "GET", "https://example.com")

        assert result is mock_resp
        assert mock_client.request.call_count == 1

    def test_success_after_transient_failure(self) -> None:
        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.raise_for_status = MagicMock()

        # First call raises, second succeeds
        mock_client.request.side_effect = [
            httpx.TransportError("connection reset"),
            mock_resp,
        ]

        with patch("hallmark.dataset.scraper.time.sleep"):
            result = _request_with_retry(mock_client, "GET", "https://example.com")

        assert result is mock_resp
        assert mock_client.request.call_count == 2

    def test_permanent_failure_returns_none(self) -> None:
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.request.side_effect = httpx.TransportError("always fails")

        with patch("hallmark.dataset.scraper.time.sleep"):
            result = _request_with_retry(mock_client, "GET", "https://example.com", max_retries=2)

        assert result is None
        assert mock_client.request.call_count == 3  # initial + 2 retries

    def test_http_status_error_retries(self) -> None:
        mock_client = MagicMock(spec=httpx.Client)
        mock_resp_ok = MagicMock(spec=httpx.Response)
        mock_resp_ok.raise_for_status = MagicMock()

        mock_resp_503 = MagicMock(spec=httpx.Response)
        mock_resp_503.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=MagicMock()
        )

        mock_client.request.side_effect = [mock_resp_503, mock_resp_ok]

        with patch("hallmark.dataset.scraper.time.sleep"):
            result = _request_with_retry(mock_client, "GET", "https://example.com")

        assert result is mock_resp_ok
        assert mock_client.request.call_count == 2

    def test_kwargs_forwarded_to_client(self) -> None:
        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.raise_for_status = MagicMock()
        mock_client.request.return_value = mock_resp

        _request_with_retry(
            mock_client,
            "GET",
            "https://example.com",
            params={"q": "test"},
            headers={"User-Agent": "test"},
        )

        mock_client.request.assert_called_once_with(
            "GET",
            "https://example.com",
            params={"q": "test"},
            headers={"User-Agent": "test"},
        )


# --- Tests for rolling split creation ---


class TestRollingSplitCreation:
    def test_create_splits_with_rolling_config(self) -> None:
        from scripts.create_splits import ROLLING_SPLIT_CONFIG, create_splits

        valid = [_make_valid_entry(f"valid_{i}") for i in range(300)]
        hallucinated = [_make_hallucinated_entry(f"hall_{i}") for i in range(50)]

        splits = create_splits(valid, hallucinated, seed=42, split_config=ROLLING_SPLIT_CONFIG)

        assert "rolling_test" in splits
        assert len(splits) == 1  # Only one split in rolling mode

    def test_rolling_output_directory_structure(self) -> None:
        from scripts.create_splits import ROLLING_SPLIT_CONFIG, create_splits, update_metadata

        valid = [_make_valid_entry(f"valid_{i}") for i in range(300)]
        hallucinated = [_make_hallucinated_entry(f"hall_{i}") for i in range(50)]
        splits = create_splits(valid, hallucinated, seed=42, split_config=ROLLING_SPLIT_CONFIG)

        with tempfile.TemporaryDirectory() as tmpdir:
            rolling_dir = Path(tmpdir) / "rolling" / "2026-02-10"
            rolling_dir.mkdir(parents=True)

            for split_name, entries in splits.items():
                save_entries(entries, rolling_dir / f"{split_name}.jsonl")

            update_metadata(rolling_dir, splits, rolling=True, seed=42)

            assert (rolling_dir / "rolling_test.jsonl").exists()
            assert (rolling_dir / "metadata.json").exists()

            with open(rolling_dir / "metadata.json") as f:
                meta = json.load(f)
            assert "rolling_metadata" in meta
            assert meta["rolling_metadata"]["seed"] == 42
            assert "pipeline_version" in meta["rolling_metadata"]


# --- Tests for loader with rolling versions ---


class TestLoaderRolling:
    def _setup_rolling_dirs(self, data_dir: Path) -> None:
        """Create test rolling directories with data."""
        for date_str in ["2026-01-15", "2026-02-10"]:
            d = data_dir / "rolling" / date_str
            d.mkdir(parents=True)
            save_entries([_make_valid_entry(f"entry_{date_str}")], d / "rolling_test.jsonl")
            with open(d / "metadata.json", "w") as f:
                json.dump({"splits": {}, "rolling_metadata": {"created": date_str}}, f)

    def test_load_split_rolling_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self._setup_rolling_dirs(data_dir)

            entries = load_split(split="rolling_test", version="rolling", data_dir=data_dir)
            assert len(entries) == 1
            assert entries[0].bibtex_key == "entry_2026-02-10"

    def test_load_split_rolling_specific_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self._setup_rolling_dirs(data_dir)

            entries = load_split(
                split="rolling_test", version="rolling/2026-01-15", data_dir=data_dir
            )
            assert len(entries) == 1
            assert entries[0].bibtex_key == "entry_2026-01-15"

    def test_load_split_rolling_no_dirs_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "rolling").mkdir(parents=True)

            with pytest.raises(FileNotFoundError, match="No rolling splits"):
                load_split(split="rolling_test", version="rolling", data_dir=data_dir)

    def test_load_metadata_rolling_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self._setup_rolling_dirs(data_dir)

            meta = load_metadata(version="rolling", data_dir=data_dir)
            assert meta["rolling_metadata"]["created"] == "2026-02-10"  # type: ignore[index]

    def test_load_metadata_rolling_specific_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self._setup_rolling_dirs(data_dir)

            meta = load_metadata(version="rolling/2026-01-15", data_dir=data_dir)
            assert meta["rolling_metadata"]["created"] == "2026-01-15"  # type: ignore[index]

    def test_resolve_rolling_path_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self._setup_rolling_dirs(data_dir)

            path = _resolve_rolling_path(data_dir, "rolling", "rolling_test")
            assert "2026-02-10" in str(path)
            assert path.name == "rolling_test.jsonl"

    def test_resolve_rolling_path_specific(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self._setup_rolling_dirs(data_dir)

            path = _resolve_rolling_path(data_dir, "rolling/2026-01-15", "rolling_test")
            assert "2026-01-15" in str(path)


# --- Tests for health check in orchestrator ---


class TestHealthChecks:
    def test_min_entries_check(self) -> None:
        from scripts.update_rolling import run_pipeline

        args = MagicMock()
        args.dry_run = False
        args.data_dir = "data"
        args.venues = ["NeurIPS"]
        args.years = [2025]
        args.seed = 42
        args.min_entries = 100
        args.run_baselines = False
        args.verbose = False

        # Mock scraper to return too few entries
        with patch("scripts.update_rolling.scrape_proceedings", return_value=[_make_valid_entry()]):
            exit_code = run_pipeline(args)

        assert exit_code == 1

    def test_dry_run_returns_zero(self) -> None:
        from scripts.update_rolling import run_pipeline

        args = MagicMock()
        args.dry_run = True
        args.data_dir = "data"
        args.venues = ["NeurIPS"]
        args.years = [2025]
        args.seed = 42
        args.min_entries = 50
        args.run_baselines = False
        args.verbose = False

        exit_code = run_pipeline(args)
        assert exit_code == 0

    def test_tier_health_check_fails_on_empty_tier(self) -> None:
        from scripts.update_rolling import run_pipeline

        args = MagicMock()
        args.dry_run = False
        args.data_dir = "data"
        args.venues = ["NeurIPS"]
        args.years = [2025]
        args.seed = 42
        args.min_entries = 1  # low so scraper passes
        args.run_baselines = False
        args.verbose = False

        valid = [_make_valid_entry(f"v_{i}") for i in range(10)]

        with (
            patch("scripts.update_rolling.scrape_proceedings", return_value=valid),
            patch("scripts.update_rolling.generate_tier1_batch", return_value=[]),
            patch(
                "scripts.update_rolling.generate_tier2_batch",
                return_value=[_make_hallucinated_entry()],
            ),
            patch(
                "scripts.update_rolling.generate_tier3_batch",
                return_value=[_make_hallucinated_entry()],
            ),
        ):
            exit_code = run_pipeline(args)

        assert exit_code == 1

    def test_successful_pipeline(self) -> None:
        from scripts.update_rolling import run_pipeline

        valid = [_make_valid_entry(f"v_{i}") for i in range(300)]
        hall_t1 = [_make_hallucinated_entry(f"t1_{i}") for i in range(12)]
        hall_t2 = [_make_hallucinated_entry(f"t2_{i}") for i in range(11)]
        hall_t3 = [_make_hallucinated_entry(f"t3_{i}") for i in range(7)]

        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.dry_run = False
            args.data_dir = tmpdir
            args.venues = ["NeurIPS"]
            args.years = [2025]
            args.seed = 42
            args.min_entries = 10
            args.run_baselines = False
            args.verbose = False

            with (
                patch("scripts.update_rolling.scrape_proceedings", return_value=valid),
                patch("scripts.update_rolling.generate_tier1_batch", return_value=hall_t1),
                patch("scripts.update_rolling.generate_tier2_batch", return_value=hall_t2),
                patch("scripts.update_rolling.generate_tier3_batch", return_value=hall_t3),
            ):
                exit_code = run_pipeline(args)

            assert exit_code == 0
            # Verify output files exist
            rolling_dirs = list(Path(tmpdir).glob("rolling/*/rolling_test.jsonl"))
            assert len(rolling_dirs) == 1
            metadata_files = list(Path(tmpdir).glob("rolling/*/metadata.json"))
            assert len(metadata_files) == 1

    def test_compute_seed_from_date(self) -> None:
        from scripts.update_rolling import compute_seed

        assert compute_seed(123) == 123  # explicit seed passed through
        # Default seed is date-derived
        seed = compute_seed(None)
        assert isinstance(seed, int)
        assert seed > 20000000  # YYYYMMDD format


# --- Tests for pool_manager archive behavior ---


class TestPoolManagerArchive:
    def test_accept_archives_instead_of_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PoolManager(tmpdir)

            # Submit a contribution
            entry = _make_valid_entry("pool_test_entry")
            contrib_path = pm.submit_contribution([entry], "test_contributor")
            assert contrib_path.exists()

            # Accept the contribution
            with patch.object(pm, "load_validated_pool", return_value=[]):
                pm.accept_contribution(contrib_path)

            # Original file should be gone from contributions
            assert not contrib_path.exists()

            # But it should be in the archived directory
            archived = list(pm.archived_dir.glob("*.jsonl"))
            assert len(archived) == 1
            assert archived[0].name == contrib_path.name

    def test_archived_dir_created_on_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PoolManager(tmpdir)
            assert pm.archived_dir.exists()
            assert pm.archived_dir.name == "archived"
