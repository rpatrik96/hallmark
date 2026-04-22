"""SQLite-backed cache for agentic tool calls.

Key: sha256(f"{tool_name}::{sorted_args_json}")
TTL: disabled by default; set CACHE_TTL_DAYS env var to enable expiry.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(".cache") / "agentic_tools.sqlite"

# Schema version bump if we change the table layout.
_SCHEMA_VERSION = 1


def _get_db_path() -> Path:
    """Return the SQLite file path, creating parent dirs as needed."""
    db_path = Path(os.environ.get("AGENTIC_CACHE_PATH", str(_DEFAULT_DB_PATH)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS tool_cache (
            cache_key TEXT PRIMARY KEY,
            result    TEXT NOT NULL,
            stored_at REAL NOT NULL
        )"""
    )
    conn.commit()
    return conn


def cache_key(tool_name: str, args: dict[str, Any]) -> str:
    """Deterministic cache key: sha256 of 'tool_name::canonical_json(args)'."""
    canonical = json.dumps(args, sort_keys=True, ensure_ascii=False)
    raw = f"{tool_name}::{canonical}"
    return hashlib.sha256(raw.encode()).hexdigest()


class AgenticToolCache:
    """Thread-unsafe single-connection SQLite cache for agentic tool results.

    Not designed for concurrent use — each process should own one instance.
    TTL is checked on read: expired entries are treated as misses and evicted.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _get_db_path()
        self._conn = _open_db(self._db_path)
        ttl_env = os.environ.get("CACHE_TTL_DAYS")
        self._ttl_seconds: float | None = float(ttl_env) * 86400 if ttl_env else None

    def get(self, key: str) -> Any | None:
        """Return cached value or None on miss/expiry."""
        row = self._conn.execute(
            "SELECT result, stored_at FROM tool_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            logger.debug("Cache miss: %s", key[:16])
            return None
        result_json, stored_at = row
        if self._ttl_seconds is not None and (time.time() - stored_at) > self._ttl_seconds:
            logger.debug("Cache expired: %s", key[:16])
            self._conn.execute("DELETE FROM tool_cache WHERE cache_key = ?", (key,))
            self._conn.commit()
            return None
        logger.debug("Cache hit: %s", key[:16])
        return json.loads(result_json)

    def set(self, key: str, value: Any) -> None:
        """Store value under key."""
        self._conn.execute(
            "INSERT OR REPLACE INTO tool_cache (cache_key, result, stored_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, ensure_ascii=False), time.time()),
        )
        self._conn.commit()
        logger.debug("Cache stored: %s", key[:16])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> AgenticToolCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
