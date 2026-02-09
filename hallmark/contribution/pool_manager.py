"""Manage the ever-expanding ONEBench-style sample pool.

Handles versioned releases, community contributions, and custom sub-benchmarks.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from hallmark.contribution.validate_entry import validate_batch
from hallmark.dataset.schema import BenchmarkEntry, load_entries, save_entries

logger = logging.getLogger(__name__)


class PoolManager:
    """Manages the ever-expanding benchmark sample pool."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.contributions_dir = self.data_dir / "pool" / "contributions"
        self.validated_dir = self.data_dir / "pool" / "validated"
        self.contributions_dir.mkdir(parents=True, exist_ok=True)
        self.validated_dir.mkdir(parents=True, exist_ok=True)

    def submit_contribution(
        self,
        entries: list[BenchmarkEntry],
        contributor: str,
    ) -> Path:
        """Submit entries for review. Returns path to contribution file."""
        today = date.today().isoformat()
        filename = f"{today}_{contributor}.jsonl"
        path = self.contributions_dir / filename
        save_entries(entries, path)
        logger.info(f"Submitted {len(entries)} entries to {path}")
        return path

    def review_contribution(
        self,
        contribution_path: str | Path,
    ) -> dict:
        """Review a pending contribution. Returns validation results."""
        path = Path(contribution_path)
        if not path.exists():
            raise FileNotFoundError(f"Contribution not found: {path}")

        entries = load_entries(path)
        existing = self.load_validated_pool()
        results = validate_batch(entries, existing)

        valid_count = sum(1 for r in results if r.valid)
        invalid_count = len(results) - valid_count

        return {
            "path": str(path),
            "total": len(entries),
            "valid": valid_count,
            "invalid": invalid_count,
            "results": [
                {
                    "key": e.bibtex_key,
                    "valid": r.valid,
                    "errors": r.errors,
                    "warnings": r.warnings,
                }
                for e, r in zip(entries, results, strict=True)
            ],
        }

    def accept_contribution(
        self,
        contribution_path: str | Path,
    ) -> int:
        """Accept valid entries from a contribution into the validated pool."""
        path = Path(contribution_path)
        entries = load_entries(path)
        existing = self.load_validated_pool()
        results = validate_batch(entries, existing)

        valid_entries = [e for e, r in zip(entries, results, strict=True) if r.valid]
        if not valid_entries:
            logger.warning("No valid entries to accept")
            return 0

        # Append to validated pool
        today = date.today().isoformat()
        pool_file = self.validated_dir / f"pool_{today}.jsonl"

        # Append to existing file if same date, else create new
        if pool_file.exists():
            existing_pool = load_entries(pool_file)
            valid_entries = existing_pool + valid_entries

        save_entries(valid_entries, pool_file)

        # Remove from contributions
        path.unlink()
        logger.info(f"Accepted {len(valid_entries)} entries into pool")
        return len(valid_entries)

    def load_validated_pool(self) -> list[BenchmarkEntry]:
        """Load all validated pool entries."""
        entries = []
        for path in sorted(self.validated_dir.glob("*.jsonl")):
            entries.extend(load_entries(path))
        return entries

    def list_contributions(self) -> list[dict]:
        """List pending contributions."""
        results = []
        for path in sorted(self.contributions_dir.glob("*.jsonl")):
            entries = load_entries(path)
            results.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "num_entries": len(entries),
                }
            )
        return results

    def create_sub_benchmark(
        self,
        name: str,
        filter_fn: callable | None = None,  # type: ignore[valid-type]
        tier: int | None = None,
        venue: str | None = None,
    ) -> list[BenchmarkEntry]:
        """Create a custom sub-benchmark from the pool.

        ONEBench-inspired: users can compose custom benchmarks
        from the ever-expanding pool.
        """
        pool = self.load_validated_pool()

        if filter_fn:
            pool = [e for e in pool if filter_fn(e)]

        if tier is not None:
            pool = [e for e in pool if e.difficulty_tier == tier or e.label == "VALID"]

        if venue is not None:
            venue_lower = venue.lower()
            pool = [
                e
                for e in pool
                if e.source_conference and venue_lower in e.source_conference.lower()
            ]

        logger.info(f"Sub-benchmark '{name}': {len(pool)} entries")
        return pool

    def get_pool_stats(self) -> dict:
        """Get statistics about the current pool."""
        pool = self.load_validated_pool()
        contributions = self.list_contributions()

        tier_counts = {1: 0, 2: 0, 3: 0}
        label_counts = {"VALID": 0, "HALLUCINATED": 0}

        for e in pool:
            label_counts[e.label] = label_counts.get(e.label, 0) + 1
            if e.difficulty_tier:
                tier_counts[e.difficulty_tier] = tier_counts.get(e.difficulty_tier, 0) + 1

        return {
            "pool_size": len(pool),
            "pending_contributions": len(contributions),
            "label_distribution": label_counts,
            "tier_distribution": tier_counts,
        }
