"""E2 tail automation: wait for v4pro first pass, retry empty-response errors,
regenerate the control's result, then write SUMMARY.md.

Designed to run detached after the three --only jobs are launched. Idempotent:
re-running after completion just re-finalizes.

Steps:
  1. Wait until the v4pro checkpoint has 300 entries (first pass complete) OR
     the v4pro process is gone.
  2. Run a retry pass (retry_failed=True) over the SAME 300 subsample with a
     larger completion budget (8192) so reasoning tokens don't truncate the JSON
     on the empty-response entries. This only re-calls [Error fallback] entries.
  3. Recompute the v4pro result_*.json from the (now patched) checkpoint via the
     same evaluate()/by-year logic as the main script.
  4. Run the finalizer to write SUMMARY.md + api_calls_total.json.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path

from hallmark.baselines.llm_verifier import verify_with_openrouter
from hallmark.dataset.schema import Prediction
from hallmark.evaluation.metrics import evaluate

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/reviewer_experiments/e2_latecutoff_control"
CKPT = OUT / "checkpoints"
V4_CKPT = CKPT / "openrouter_deepseek_deepseek-v4-pro.jsonl"
MODEL_ID = "deepseek/deepseek-v4-pro"
ALIAS = "deepseek-v4-pro"


_spec = importlib.util.spec_from_file_location(
    "e2main", str(REPO / "scripts/reviewer_experiments/e2_latecutoff_control.py")
)
_e2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e2)


def count_ckpt() -> int:
    if not V4_CKPT.exists():
        return 0
    return sum(1 for line in V4_CKPT.read_text().splitlines() if line.strip())


def count_errors() -> int:
    if not V4_CKPT.exists():
        return 0
    n = 0
    for line in V4_CKPT.read_text().splitlines():
        if not line.strip():
            continue
        if json.loads(line).get("reason", "").startswith("[Error fallback]"):
            n += 1
    return n


def wait_for_first_pass() -> None:
    """Block until v4pro checkpoint reaches 300 entries."""
    while True:
        n = count_ckpt()
        if n >= _e2.N_TARGET:
            print(f"[tail] first pass complete: {n} entries, {count_errors()} errors")
            return
        print(f"[tail] waiting first pass: {n}/{_e2.N_TARGET} (errs={count_errors()})", flush=True)
        time.sleep(60)


def run_retry_passes(subsample: list, max_passes: int = 3) -> None:
    """Re-attempt [Error fallback] entries with a larger token budget.

    verify_with_openrouter(retry_failed=True) drops error-fallback rows from the
    'completed' set so they get re-called; clean rows are skipped. We bump
    max_completion_tokens to 8192 so V4-Pro's reasoning tokens don't truncate
    the JSON verdict (the empty-response failure mode).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    blind = [e.to_blind() for e in subsample]
    for p in range(1, max_passes + 1):
        errs = count_errors()
        if errs == 0:
            print(f"[tail] no error fallbacks left before pass {p}; done retrying")
            return
        print(f"[tail] retry pass {p}: {errs} error entries to re-attempt", flush=True)
        verify_with_openrouter(
            blind,
            model=MODEL_ID,
            api_key=api_key,
            checkpoint_dir=CKPT,
            log_dir=OUT / "api_logs" / ALIAS,
            retry_failed=True,
            max_completion_tokens=8192,
        )
        print(f"[tail] after retry pass {p}: errs now {count_errors()}", flush=True)
    print(f"[tail] retry passes exhausted; remaining errs={count_errors()}")


def regenerate_result(subsample: list) -> None:
    """Rebuild result_deepseek-v4-pro.json from the patched checkpoint."""
    preds = []
    for line in V4_CKPT.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        preds.append(
            Prediction(
                bibtex_key=d["bibtex_key"],
                label=d["label"],
                confidence=d["confidence"],
                reason=d.get("reason", ""),
                wall_clock_seconds=d.get("wall_clock_seconds", 0.0),
                api_calls=d.get("api_calls", 0),
                api_sources_queried=d.get("api_sources_queried", []),
            )
        )
    # Dedup by bibtex_key keeping last (retry rows are appended after originals).
    by_key = {p.bibtex_key: p for p in preds}
    preds = [by_key[e.bibtex_key] for e in subsample if e.bibtex_key in by_key]

    total_calls = sum(p.api_calls for p in preds)
    res = evaluate(subsample, preds, tool_name=ALIAS, split_name="temporal_2024_2025_n300")

    def subset_eval(year: str) -> dict:
        keys = {
            e.bibtex_key
            for e in subsample
            if (getattr(e, "publication_date", "") or "")[:4] == year
        }
        se = [e for e in subsample if e.bibtex_key in keys]
        sp = [p for p in preds if p.bibtex_key in keys]
        r = evaluate(se, sp, tool_name=ALIAS, split_name=f"y{year}")
        return {
            "n": len(se),
            "fpr": r.false_positive_rate,
            "dr": r.detection_rate,
            "f1": r.f1_hallucination,
        }

    out = {
        "alias": ALIAS,
        "model_id": MODEL_ID,
        "runner": "openrouter",
        "n": len(subsample),
        "api_calls": total_calls,
        "predicted_label_dist": dict(Counter(p.label for p in preds)),
        "overall": res.to_dict(),
        "by_year": {"2024": subset_eval("2024"), "2025": subset_eval("2025")},
    }
    (OUT / f"result_{ALIAS}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(
        f"[tail] regenerated result: FPR={out['overall']['false_positive_rate']} "
        f"DR={out['overall']['detection_rate']} calls={total_calls} "
        f"pred={out['predicted_label_dist']}"
    )


def main() -> None:
    subsample = _e2.build_subsample()
    wait_for_first_pass()
    run_retry_passes(subsample)
    regenerate_result(subsample)
    # Finalize.
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(REPO / "scripts/reviewer_experiments/e2_finalize_summary.py")],
        check=True,
    )
    print("[tail] DONE — SUMMARY.md written")


if __name__ == "__main__":
    main()
