#!/usr/bin/env python3
"""GEPA prompt optimization for the zero-shot LLM verification prompt.

Optimizes VERIFICATION_PROMPT via gepa.optimize_anything (Generalization mode):
  - Task model: claude-haiku-4.5 via the OpenRouter mirror (OPENROUTER_API_KEY)
  - Reflection LM: claude-opus-5 via OpenRouter (same key)
  - Metric: binary per-entry — 1.0 if the predicted label matches the true
    label, else 0.0. UNCERTAIN always scores 0.0 (every true label is
    VALID or HALLUCINATED).
  - dataset: 50 stratified dev_public entries. reflection_minibatch_size is set
    to the full trainset, so every reflection step sees Haiku's own reason
    string plus the true label for all 50 entries in the ASI dict. 50 rather
    than 25 because a 25-entry sample covers only 7-9 of the 14 hallucination
    types and holds ~11 valid entries, too few to read an FPR signal from.
  - valset: 200 stratified dev_public entries, disjoint from the trainset.
  - OBJECTIVE / BACKGROUND are compiled into the reflection prompt template
    (optimize_anything.py:1592) and therefore resent on EVERY reflection call,
    so they are kept short. The low-FPR constraint lives there in plain English.

Stopping — max_metric_calls is the only stop condition; the run ends when the
task-LM budget is spent. Each engine iteration scores the parent prompt on the
minibatch and then the newly proposed child prompt on that SAME minibatch
(reflective_mutation.py:530) — a controlled A/B, so 50 trainset entries cost
100 calls per iteration. A child that beats its parent costs a further 200 for
the full valset eval. So 1800 calls buys ~5 iterations if every proposal is
accepted, or up to 16 if all are rejected; rejected proposals are cheap.
cache_evaluation=True makes the repeated parent scoring a cache hit, since the
minibatch is always the whole trainset. Output is the seed plus one candidate
per acceptance, all written to candidates.jsonl.

Parallelism — max_workers=6 matches the agentic runner's default (the zero-shot
runner uses 8). Haiku replies in ~2-3 s, so 6 workers is roughly 130-180 RPM,
under the 200 RPM cap noted in scripts/parallel_resume_test_public.py. The
openai SDK client is thread-safe and its max_retries=5 exponential backoff
absorbs any 429s.

Candidate prompts carry the literal placeholder <<BIBTEX_ENTRY>> rather than
the str.format() placeholder {bibtex}, so the reflection LM never has to reason
about brace escaping; the winner is converted back to VERIFICATION_PROMPT
format on save.

Resume — GEPA silently continues a previous search whenever run_dir contains a
gepa_state.bin (state.py:669). That is off by default here: any prior state is
renamed to run.archived-N first. Pass --resume to continue instead, remembering
that max_metric_calls is then a LIFETIME budget, since the restored
total_num_evals is what the stopper compares against.

Usage:
    uv run python scripts/gepa_optimize_prompt.py                # full run
    uv run python scripts/gepa_optimize_prompt.py --smoke        # plumbing check
    uv run python scripts/gepa_optimize_prompt.py --resume       # continue a run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hallmark.baselines.llm_verifier import (  # noqa: E402
    VERIFICATION_PROMPT,
    _parse_llm_response,
)
from hallmark.dataset.loader import load_split  # noqa: E402
from hallmark.dataset.schema import BenchmarkEntry  # noqa: E402

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

BIBTEX_MARKER = "<<BIBTEX_ENTRY>>"

# The GEPA seed candidate is our current zero-shot prompt verbatim. format()
# both substitutes the marker and unescapes the {{...}} braces of the JSON
# response example, which is what we want the task model to see.
SEED_PROMPT = VERIFICATION_PROMPT.format(bibtex=BIBTEX_MARKER)

# Resent to the reflection LM on every reflection call — keep short.
OBJECTIVE = """\
Rewrite the citation-verification prompt so that Claude Haiku 4.5 labels each \
BibTeX entry correctly as VALID or HALLUCINATED. Scoring is binary per entry; \
UNCERTAIN always scores 0 (every true label is VALID or HALLUCINATED)."""

BACKGROUND = f"""\
The verifier has no internet access and no tools — parametric knowledge only.

Goal: balanced performance — a high detection rate (catch fabricated entries) \
AND a low false positive rate (flag a genuine citation only for a concrete, \
articulable defect). Scoring is balanced across the two classes: always \
answering VALID or always answering HALLUCINATED scores exactly 0.5, so \
under-flagging loses as much as over-flagging.

Generalization — HARD RULE: the prompt must work on citations it has never \
seen. Never copy author names, author lists, paper titles, or citation keys \
from the evaluation feedback into the prompt — describe failure patterns in \
general terms only. Never state statistics of this dataset (label \
proportions, citation-key formats, or how entries were constructed). \
A prompt containing copied dataset strings automatically scores 0.

Format requirements — a prompt violating any of these scores 0:
1. Contains the literal placeholder {BIBTEX_MARKER} exactly once.
2. Requires a JSON-only reply with exactly the keys "label" \
(VALID/HALLUCINATED/UNCERTAIN), "confidence" (0.0-1.0), \
"predicted_hallucination_type", and "reason".
3. Keeps the 14 hallucination-type strings exactly as spelled in the current \
prompt; predicted_hallucination_type is one of them, or null.
UNCERTAIN is scored as incorrect either way; the prompt must make the model \
commit to VALID or HALLUCINATED."""


def _load_dotenv(repo_root: Path) -> None:
    """Set unset env vars from the repo-root .env file (never overrides)."""
    import os

    env_file = repo_root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def sample_entries(
    train_size: int, val_size: int, seed: int
) -> tuple[list[BenchmarkEntry], list[BenchmarkEntry]]:
    """Class-balanced (50/50) disjoint train/val samples from dev_public."""
    entries = load_split("dev_public")
    rng = random.Random(seed)
    by_label: dict[str, list[BenchmarkEntry]] = {"VALID": [], "HALLUCINATED": []}
    for e in entries:
        by_label[e.label].append(e)
    for pool in by_label.values():
        rng.shuffle(pool)

    def take(n: int) -> list[BenchmarkEntry]:
        # 50/50 by construction: plain per-entry accuracy then equals balanced
        # accuracy, so knowing the label proportions cannot help a prompt.
        n_hall = n // 2
        picked = [by_label["HALLUCINATED"].pop() for _ in range(n_hall)]
        picked += [by_label["VALID"].pop() for _ in range(n - n_hall)]
        rng.shuffle(picked)
        return picked

    return take(train_size), take(val_size)


def build_leak_blocklist(entries: list[BenchmarkEntry]) -> set[str]:
    """Strings from the trainset that must never appear in a candidate prompt.

    Full author names, citation keys, and title 4-grams. Generic tokens that
    legitimately occur in any verification prompt (single common surnames,
    "et al", "and others") are excluded by requiring multi-word specificity.
    """
    generic = {"et al", "and others"}
    block: set[str] = set()
    for e in entries:
        block.add(e.bibtex_key)
        for name in re.split(r"\s+and\s+", e.fields.get("author", "")):
            name = " ".join(name.strip().rstrip(",").split())
            if len(name.split()) >= 2 and name.lower() not in generic:
                block.add(name)
        words = e.fields.get("title", "").split()
        for i in range(len(words) - 3):
            block.add(" ".join(words[i : i + 4]))
    return {b for b in block if len(b) >= 8}


def find_leaks(candidate: str, blocklist: set[str]) -> list[str]:
    """Return blocklisted strings present in the candidate prompt."""
    low = candidate.lower()
    return [b for b in blocklist if b.lower() in low]


def _prepare_run_dir(run_dir: Path, *, resume: bool) -> Path | None:
    """Move a prior run's state aside unless resuming; return where it went.

    GEPA resumes silently whenever ``run_dir/gepa_state.bin`` exists
    (state.py:669) — there is no library-level flag for this. The restored
    state carries ``total_num_evals``, which ``MaxMetricCallsStopper`` compares
    against ``max_metric_calls`` (stop_condition.py:173), so an unnoticed
    resume both continues the old search and spends the new run's budget. The
    prior state is renamed rather than deleted so nothing is lost.
    """
    if resume or not (run_dir / "gepa_state.bin").exists():
        return None
    n = 0
    while (archive := run_dir.with_name(f"{run_dir.name}.archived-{n}")).exists():
        n += 1
    run_dir.rename(archive)
    return archive


class JsonlWriter:
    """Append-only JSONL sink, safe to call from the evaluator worker threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._n = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def write(self, record: dict[str, Any]) -> int:
        with self._lock:
            self._n += 1
            record = {"seq": self._n, **record}
            with open(self.path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            return self._n

    @property
    def count(self) -> int:
        return self._n


class TeeLogger:
    """GEPA's LoggerProtocol: mirror the engine's own log lines to stdout + file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def log(self, message: str) -> None:
        print(f"[gepa] {message}", flush=True)
        with self._lock, open(self.path, "a") as f:
            f.write(message + "\n")


class EventRecorder:
    """GEPACallback: records the loop's decision points to events.jsonl.

    Only the hooks that explain the control flow are implemented; the protocol
    is duck-typed, so unimplemented hooks are simply never called.
    """

    def __init__(self, sink: JsonlWriter) -> None:
        self.sink = sink

    def _emit(self, kind: str, **fields: Any) -> None:
        self.sink.write({"event": kind, **fields})

    def on_minibatch_sampled(self, event: Any) -> None:
        self._emit(
            "minibatch_sampled",
            iteration=event.get("iteration"),
            parent_idx=event.get("parent_idx"),
            minibatch_ids=list(event.get("minibatch_ids") or []),
        )

    def on_proposal_end(self, event: Any) -> None:
        self._emit(
            "proposal_end",
            iteration=event.get("iteration"),
            reflection_prompt=event.get("prompts"),
            raw_lm_output=event.get("raw_lm_outputs"),
            new_instructions=event.get("new_instructions"),
        )

    def on_candidate_accepted(self, event: Any) -> None:
        self._emit(
            "candidate_ACCEPTED",
            iteration=event.get("iteration"),
            new_candidate_idx=event.get("new_candidate_idx"),
            minibatch_score=event.get("new_score"),
            parent_ids=list(event.get("parent_ids") or []),
        )

    def on_candidate_rejected(self, event: Any) -> None:
        self._emit(
            "candidate_REJECTED",
            iteration=event.get("iteration"),
            parent_minibatch_score=event.get("old_score"),
            child_minibatch_score=event.get("new_score"),
            reason=event.get("reason"),
        )

    def on_valset_evaluated(self, event: Any) -> None:
        self._emit(
            "valset_evaluated",
            iteration=event.get("iteration"),
            candidate_idx=event.get("candidate_idx"),
            average_score=event.get("average_score"),
            num_examples=event.get("num_examples_evaluated"),
            is_best=event.get("is_best_program"),
        )

    def on_error(self, event: Any) -> None:
        self._emit("error", **{k: v for k, v in event.items() if k != "state"})


def make_openrouter_lm(
    model: str,
    *,
    temperature: float | None,
    max_completion_tokens: int | None,
    role: str,
    sink: JsonlWriter | None = None,
) -> Any:
    """Return a thread-safe prompt -> response callable for an OpenRouter model.

    Every call is appended to ``sink`` with the full prompt and reply, so a run
    can be replayed offline.
    """
    import os

    import openai

    client = openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        max_retries=5,
        timeout=600.0,
    )

    def call(prompt: str) -> str:
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                seed=42,
                **kwargs,
            )
        except Exception as e:
            if sink is not None:
                sink.write(
                    {
                        "role": role,
                        "model": model,
                        "error": f"{type(e).__name__}: {e}",
                        "elapsed_s": round(time.time() - start, 2),
                        "prompt": prompt,
                    }
                )
            raise
        content = str(resp.choices[0].message.content).strip()
        if sink is not None:
            usage = getattr(resp, "usage", None)
            sink.write(
                {
                    "role": role,
                    "model": model,
                    "elapsed_s": round(time.time() - start, 2),
                    "tokens_in": getattr(usage, "prompt_tokens", None),
                    "tokens_out": getattr(usage, "completion_tokens", None),
                    "prompt": prompt,
                    "response": content,
                }
            )
        return content

    return call


def make_evaluator(
    task_lm: Any,
    sink: JsonlWriter | None = None,
    leak_blocklist: set[str] | None = None,
) -> Any:
    """Binary per-entry evaluator; side_info carries Haiku's reason + the true label.

    Each scored (prompt, entry) pair is appended to ``sink`` with a hash of the
    candidate prompt, so the log shows which prompt version produced which
    verdict on which entry.
    """

    def evaluate(candidate: str, example: BenchmarkEntry) -> tuple[float, dict[str, Any]]:
        # Leak guard: refuse (before any API call) prompts that copy strings
        # from the training data. The all-zero minibatch this produces means a
        # leaking child can never beat its parent, and the ASI message tells
        # the reflection LM exactly why.
        if leak_blocklist is not None and (leaked := find_leaks(candidate, leak_blocklist)):
            return 0.0, {
                "Error": (
                    "The prompt copies strings from the evaluation data "
                    f"(e.g. {leaked[:3]}). Copied author names, titles, or keys "
                    "score 0 automatically — describe failure patterns in "
                    "general terms instead."
                )
            }
        if BIBTEX_MARKER not in candidate:
            return 0.0, {
                "Error": (
                    f"The prompt is missing the literal placeholder {BIBTEX_MARKER}, "
                    "so the BibTeX entry could not be inserted. Every prompt MUST "
                    f"contain {BIBTEX_MARKER} exactly once."
                )
            }

        bibtex = example.to_bibtex()
        prompt = candidate.replace(BIBTEX_MARKER, bibtex)
        try:
            raw = task_lm(prompt)
        except Exception as e:  # transient API failure — not the prompt's fault
            return 0.0, {"Error": f"Transient API error (not caused by the prompt): {e}"}

        pred = _parse_llm_response(raw, example.bibtex_key)
        correct = pred.label == example.label

        true_label = str(example.label)
        if example.label == "HALLUCINATED":
            true_label += f" (hallucination type: {example.hallucination_type})"

        if correct:
            verdict = "CORRECT"
        elif pred.label == "UNCERTAIN":
            verdict = (
                "INCORRECT — the model answered UNCERTAIN, which is always scored "
                "incorrect (or the reply did not parse as JSON; see reason)"
            )
        elif example.label == "VALID":
            verdict = (
                "INCORRECT — FALSE POSITIVE: a genuine citation was flagged as "
                "hallucinated. This is the most damaging error type."
            )
        else:
            verdict = (
                "INCORRECT — MISSED HALLUCINATION: a fabricated citation was accepted as valid."
            )

        side_info: dict[str, Any] = {
            "Input (BibTeX entry)": bibtex,
            "True label": true_label,
            "Model output": {
                "label": pred.label,
                "confidence": pred.confidence,
                "predicted_hallucination_type": pred.predicted_hallucination_type,
                "reason": pred.reason,
            },
            "Verdict": verdict,
        }
        if sink is not None:
            sink.write(
                {
                    "prompt_sha": hashlib.sha256(candidate.encode()).hexdigest()[:12],
                    "bibtex_key": example.bibtex_key,
                    "true_label": example.label,
                    "true_type": example.hallucination_type,
                    "pred_label": pred.label,
                    "pred_type": pred.predicted_hallucination_type,
                    "confidence": pred.confidence,
                    "score": 1.0 if correct else 0.0,
                    "verdict": verdict,
                    "reason": pred.reason,
                    "raw_response": raw,
                }
            )
        return (1.0 if correct else 0.0), side_info

    return evaluate


def to_verification_format(prompt: str) -> str:
    """Convert a marker-style prompt back to VERIFICATION_PROMPT str.format style."""
    return prompt.replace("{", "{{").replace("}", "}}").replace(BIBTEX_MARKER, "{bibtex}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=1800,
        help="task-LM call budget; the only stop condition (see the docstring)",
    )
    parser.add_argument("--task-model", default="anthropic/claude-haiku-4.5")
    parser.add_argument("--reflection-model", default="anthropic/claude-opus-5")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="parallel task-LM calls (agentic runner default; zero-shot runner uses 8)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "results" / "gepa_haiku" / "guarded"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "continue a previous run in --out-dir instead of starting fresh. "
            "NOTE: --max-metric-calls is a LIFETIME budget compared against the "
            "restored total_num_evals, so a resumed run only gets what the "
            "earlier run left unspent."
        ),
    )
    parser.add_argument(
        "--smoke", action="store_true", help="plumbing check (3 train / 4 val, 30 calls)"
    )
    args = parser.parse_args()

    if args.smoke:
        args.train_size, args.val_size = 3, 4
        args.max_metric_calls = 30

    _load_dotenv(REPO_ROOT)

    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        TrackingConfig,
        optimize_anything,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.out_dir / "run"
    archived = _prepare_run_dir(run_dir, resume=args.resume)
    if archived is not None:
        print(f"prior run state moved aside -> {archived}")
    elif args.resume and (run_dir / "gepa_state.bin").exists():
        print("resuming prior run; --max-metric-calls is a LIFETIME budget")

    log_dir = args.out_dir / "logs"
    task_calls = JsonlWriter(log_dir / "task_calls.jsonl")
    reflection_calls = JsonlWriter(log_dir / "reflection_calls.jsonl")
    evaluations = JsonlWriter(log_dir / "evaluations.jsonl")
    events = JsonlWriter(log_dir / "events.jsonl")
    gepa_logger = TeeLogger(log_dir / "gepa.log")

    trainset, valset = sample_entries(args.train_size, args.val_size, args.seed)
    print(
        f"trainset: {len(trainset)} entries "
        f"({sum(e.label == 'HALLUCINATED' for e in trainset)} hallucinated), "
        f"valset: {len(valset)} entries "
        f"({sum(e.label == 'HALLUCINATED' for e in valset)} hallucinated)"
    )
    print(f"logs -> {log_dir}")

    task_lm = make_openrouter_lm(
        args.task_model,
        temperature=0.0,
        max_completion_tokens=1024,
        role="task",
        sink=task_calls,
    )
    # Reflection writes a full replacement prompt: no token cap, provider default temperature.
    reflection_lm = make_openrouter_lm(
        args.reflection_model,
        temperature=None,
        max_completion_tokens=None,
        role="reflection",
        sink=reflection_calls,
    )
    leak_blocklist = build_leak_blocklist(trainset)
    print(f"leak blocklist: {len(leak_blocklist)} protected strings from the trainset")
    evaluator = make_evaluator(task_lm, sink=evaluations, leak_blocklist=leak_blocklist)

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(run_dir),
            seed=args.seed,
            max_metric_calls=args.max_metric_calls,
            parallel=True,
            max_workers=args.max_workers,
            cache_evaluation=True,
            raise_on_exception=False,
        ),
        tracking=TrackingConfig(logger=gepa_logger),
        # GEPACallback declares every hook; EventRecorder implements only the
        # six that matter here, which GEPA dispatches duck-typed via getattr.
        callbacks=[cast(Any, EventRecorder(events))],
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm,
            # 20 of 50: each reflection sees a different random slice of the
            # trainset, so successive proposals reason from different evidence.
            # (At 50-of-50 every reflection saw identical input and proposal
            # diversity depended entirely on LM sampling luck.)
            reflection_minibatch_size=min(20, len(trainset)),
        ),
    )

    result = optimize_anything(
        seed_candidate=SEED_PROMPT,
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        objective=OBJECTIVE,
        background=BACKGROUND,
        config=config,
    )

    candidates_path = args.out_dir / "candidates.jsonl"
    with open(candidates_path, "w") as f:
        for idx, (cand, score) in enumerate(
            zip(result.candidates, result.val_aggregate_scores, strict=True)
        ):
            prompt = cand["current_candidate"] if isinstance(cand, dict) else cand
            f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "val_score": score,
                        "is_best": idx == result.best_idx,
                        "prompt": prompt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    best = result.best_candidate
    best_prompt = best["current_candidate"] if isinstance(best, dict) else best
    (args.out_dir / "best_prompt.txt").write_text(best_prompt)
    (args.out_dir / "best_prompt_verification_format.txt").write_text(
        to_verification_format(best_prompt)
    )

    print(
        f"\n{len(result.candidates)} candidates (seed + accepted proposals), val scores: "
        f"{[round(s, 3) for s in result.val_aggregate_scores]}"
    )
    print(f"seed val score:  {result.val_aggregate_scores[0]:.3f}")
    print(
        f"best val score:  {result.val_aggregate_scores[result.best_idx]:.3f} "
        f"(candidate {result.best_idx})"
    )
    print(f"\nSaved: {candidates_path}")
    print(f"       {args.out_dir / 'best_prompt.txt'}")
    print(f"       {args.out_dir / 'best_prompt_verification_format.txt'}")
    print(
        f"\nLogs in {log_dir}:\n"
        f"  task_calls.jsonl        {task_calls.count} Haiku calls\n"
        f"  reflection_calls.jsonl  {reflection_calls.count} Opus calls\n"
        f"  evaluations.jsonl       {evaluations.count} scored (prompt, entry) pairs\n"
        f"  events.jsonl            {events.count} loop events\n"
        f"  gepa.log                engine log lines"
    )


if __name__ == "__main__":
    main()
