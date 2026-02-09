#!/usr/bin/env python3
"""Quarterly pool update script.

Reviews pending contributions and updates the validated pool.

Usage:
    python scripts/update_pool.py --data-dir data/ --auto-accept
"""

from __future__ import annotations

import argparse
import logging

from hallmark.contribution.pool_manager import PoolManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Update benchmark pool")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help="Automatically accept valid contributions",
    )
    parser.add_argument("--stats", action="store_true", help="Show pool statistics")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    manager = PoolManager(args.data_dir)

    if args.stats:
        stats = manager.get_pool_stats()
        print("\nPool Statistics:")
        print(f"  Pool size: {stats['pool_size']}")
        print(f"  Pending contributions: {stats['pending_contributions']}")
        print(f"  Labels: {stats['label_distribution']}")
        print(f"  Tiers: {stats['tier_distribution']}")
        return

    contributions = manager.list_contributions()
    if not contributions:
        logging.info("No pending contributions")
        return

    for contrib in contributions:
        logging.info(f"Reviewing: {contrib['filename']} ({contrib['num_entries']} entries)")
        review = manager.review_contribution(contrib["path"])

        print(f"\n  {contrib['filename']}:")
        print(f"    Valid: {review['valid']}/{review['total']}")

        for r in review["results"]:
            if not r["valid"]:
                print(f"    INVALID {r['key']}: {r['errors']}")

        if args.auto_accept and review["valid"] > 0:
            accepted = manager.accept_contribution(contrib["path"])
            logging.info(f"  Accepted {accepted} entries")
        elif not args.auto_accept:
            logging.info("  Use --auto-accept to accept valid entries")


if __name__ == "__main__":
    main()
