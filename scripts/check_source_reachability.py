#!/usr/bin/env python3
"""Refuse to start a long run against a source that is not answering.

The ``bibtex-updater`` pre-screening ablation was discarded three times in one
day. Twice the cause was a source outage the run could only discover at the end:
``bibtex-check`` exits 5 and HALLMARK's wrapper raises ``SourceOutageError``,
which is correct -- a source that never answered is not evidence a reference is
absent -- but by then 90 minutes are gone. The last attempt (2026-09-04, 20:49)
reported 244 of 1,119 entries with an incomplete lookup, 230 of them DBLP.

This is the cheap version of that check, run *before* the work: one request per
source, and a non-zero exit if a required one does not answer. Two minutes of
probing against ninety of scoring.

    python scripts/check_source_reachability.py                    # report only
    python scripts/check_source_reachability.py --require dblp,openalex

Exit codes
----------
0   every required source answered
1   at least one required source did not
2   nothing was probed (bad --require, or httpx missing) -- never a pass
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

#: One cheap, cacheable, well-formed query per source. Each asks for a real
#: record rather than hitting a bare root, because several of these return 200
#: from a CDN at the root while the query path behind it is down -- the same
#: shape as the HTTP-202 defect in the DOI check, where the status answered a
#: question nobody asked.
PROBES: dict[str, str] = {
    "dblp": "https://dblp.org/search/publ/api?q=attention+is+all+you+need&format=json&h=1",
    "openalex": "https://api.openalex.org/works?filter=doi:10.48550/arxiv.1706.03762&per-page=1",
    "crossref": "https://api.crossref.org/works/10.1145/3292500.3330701",
    "semanticscholar": "https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.03762?fields=title",
    "arxiv": "http://export.arxiv.org/api/query?id_list=1706.03762&max_results=1",
    "datacite": "https://api.datacite.org/dois/10.48550/arxiv.1706.03762",
}


@dataclass
class Result:
    source: str
    ok: bool
    status: int | None
    elapsed: float
    detail: str


def probe(source: str, url: str, timeout: float, mailto: str | None) -> Result:
    """One request, one retry. A 429 counts as reachable but throttled."""
    import httpx

    headers = {"User-Agent": f"hallmark-source-check (mailto:{mailto})"} if mailto else {}
    last = ""
    for attempt in (1, 2):
        started = time.monotonic()
        try:
            response = httpx.get(url, timeout=timeout, headers=headers, follow_redirects=True)
        except Exception as exc:  # any transport failure is the finding
            last = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
            if attempt == 2:
                return Result(source, False, None, elapsed, last)
            time.sleep(1.0)
            continue
        elapsed = time.monotonic() - started
        if response.status_code == 429:
            # Throttled is not down: the service answered and asked us to slow
            # down, which pacing handles. Reported, not failed.
            return Result(source, True, 429, elapsed, "throttled (429) -- reachable, pace the run")
        if response.status_code < 400:
            return Result(source, True, response.status_code, elapsed, "ok")
        last = f"HTTP {response.status_code}"
        if attempt == 2:
            return Result(source, False, response.status_code, elapsed, last)
        time.sleep(1.0)
    return Result(source, False, None, 0.0, last or "no attempt completed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--require",
        default="",
        help="comma-separated sources that must answer (default: report only, always exit 0)",
    )
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--mailto", default=None, help="contact address for the polite pool")
    args = ap.parse_args()

    try:
        import httpx  # noqa: F401
    except ImportError:
        print("httpx is not installed -- cannot probe, refusing to report a pass", file=sys.stderr)
        return 2

    required = [s.strip() for s in args.require.split(",") if s.strip()]
    unknown = [s for s in required if s not in PROBES]
    if unknown:
        print(
            f"unknown source(s): {', '.join(unknown)}; known: {', '.join(PROBES)}", file=sys.stderr
        )
        return 2

    results = [probe(name, url, args.timeout, args.mailto) for name, url in PROBES.items()]
    width = max(len(r.source) for r in results)
    for r in results:
        mark = "ok  " if r.ok else "DOWN"
        req = " (required)" if r.source in required else ""
        print(f"  {mark}  {r.source:<{width}}  {r.elapsed:5.2f}s  {r.detail}{req}")

    if not required:
        print("\nreport only -- pass --require to gate a run on this")
        return 0

    down = [r.source for r in results if r.source in required and not r.ok]
    if down:
        print(
            f"\nrequired source(s) not answering: {', '.join(down)}. "
            "A run started now would score entries whose lookups never completed, "
            "which is what the exit-5 guard discards at the end. Wait, or drop the "
            "source from the condition and say so in the write-up.",
            file=sys.stderr,
        )
        return 1
    print(f"\nall required sources answered: {', '.join(required)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
