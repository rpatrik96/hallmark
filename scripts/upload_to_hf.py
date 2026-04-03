"""Upload HALLMARK dataset to Hugging Face Hub.

Usage:
    python scripts/upload_to_hf.py --repo-name <username>/hallmark --token <hf_token>

Requires:
    pip install huggingface_hub
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "v1.0"
CROISSANT_PATH = Path(__file__).resolve().parent.parent / "croissant.json"

SPLITS = [
    "dev_public.jsonl",
    "test_public.jsonl",
    "stress_test.jsonl",
]

BLIND_SPLITS = [
    "dev_public_blind.jsonl",
    "test_public_blind.jsonl",
    "stress_test_blind.jsonl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload HALLMARK to Hugging Face Hub")
    parser.add_argument("--repo-name", required=True, help="HF repo (e.g., username/hallmark)")
    parser.add_argument("--token", required=True, help="Hugging Face API token")
    parser.add_argument(
        "--private", action="store_true", help="Create private repo (for pre-publication)"
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as err:
        print("Install huggingface_hub: pip install huggingface_hub")
        raise SystemExit(1) from err

    api = HfApi(token=args.token)

    # Create repo
    api.create_repo(
        repo_id=args.repo_name,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"Repository: https://huggingface.co/datasets/{args.repo_name}")

    # Upload data splits
    for split_file in SPLITS + BLIND_SPLITS:
        path = DATA_DIR / split_file
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"data/{split_file}",
                repo_id=args.repo_name,
                repo_type="dataset",
            )
            print(f"  Uploaded {split_file}")

    # Upload Croissant metadata
    if CROISSANT_PATH.exists():
        api.upload_file(
            path_or_fileobj=str(CROISSANT_PATH),
            path_in_repo="croissant.json",
            repo_id=args.repo_name,
            repo_type="dataset",
        )
        print("  Uploaded croissant.json")

    print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{args.repo_name}")


if __name__ == "__main__":
    main()
