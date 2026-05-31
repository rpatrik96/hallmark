"""Self-consistency analysis of GPT-5.1 from the existing E3 variance runs.

The OpenAI key has insufficient_quota, so GPT-5.1 cannot be re-run for the
self-consistency pilot. Instead we reuse the three independent GPT-5.1 draws
at temperature=1 already in results/reviewer_experiments/e3_variance/ (n=150
dev_public subsample, the exact E3 regime). We compute:

  - per-draw DR/FPR/F1 (re-scored against CURRENT dev_public labels);
  - k=3 majority-vote DR/FPR/F1;
  - across-draw verdict-flip rate (entries whose label is not identical in all
    three draws).

This is the genuine GPT-5.1 self-consistency signal the pilot wanted, already
paid for. Scored with hallmark.evaluation.evaluate.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from hallmark.dataset.schema import Prediction, load_entries
from hallmark.evaluation.metrics import evaluate

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
E3 = REPO / "results/reviewer_experiments/e3_variance"
DEVLABELS = REPO / "data/v1.0/dev_public.jsonl"


def _load_run(run: int) -> dict[str, Prediction]:
    out: dict[str, Prediction] = {}
    for line in (E3 / f"run{run}" / "openai_gpt-5.1.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[d["bibtex_key"]] = Prediction(
            bibtex_key=d["bibtex_key"],
            label=d["label"],
            confidence=d.get("confidence", 0.5),
            reason=d.get("reason", ""),
        )
    return out


def _score(entries: list, preds: list[Prediction], tool: str) -> dict:
    res = evaluate(entries, preds, tool_name=tool, split_name="e3-n150")
    return {
        "detection_rate": res.detection_rate,
        "false_positive_rate": res.false_positive_rate,
        "f1_hallucination": res.f1_hallucination,
        "num_uncertain": res.num_uncertain,
    }


def main() -> None:
    runs = [_load_run(r) for r in (1, 2, 3)]
    keys = sorted(set.intersection(*[set(r) for r in runs]))

    all_entries = {e.bibtex_key: e for e in load_entries(DEVLABELS)}
    entries = [all_entries[k] for k in keys if k in all_entries]
    scored_keys = {e.bibtex_key for e in entries}

    per_draw = []
    for r in runs:
        preds = [r[k] for k in scored_keys]
        per_draw.append(_score(entries, preds, "gpt-5.1"))

    # across-draw verdict-flip rate
    unstable = sum(1 for k in scored_keys if len({r[k].label for r in runs}) > 1)
    flip_rate = unstable / len(scored_keys) if scored_keys else 0.0

    # k=3 majority vote
    mv_preds = []
    for k in scored_keys:
        votes = [r[k].label for r in runs]
        label = Counter(votes).most_common(1)[0][0]
        mv_preds.append(Prediction(bibtex_key=k, label=label, confidence=0.7, reason="maj3"))
    mv = _score(entries, mv_preds, "gpt-5.1-maj3")

    f1s = [d["f1_hallucination"] for d in per_draw]
    fprs = [d["false_positive_rate"] for d in per_draw]
    mean_f1 = sum(f1s) / len(f1s)
    mean_fpr = sum(fprs) / len(fprs)

    out = {
        "source": "E3 reuse (GPT-5.1 temp=1, n=150 dev_public, current labels)",
        "n_scored": len(scored_keys),
        "per_draw": per_draw,
        "majority_vote": mv,
        "across_draw_flip": {
            "unstable_entries": unstable,
            "n_compared": len(scored_keys),
            "rate": flip_rate,
        },
        "f1_range": [min(f1s), max(f1s)],
        "fpr_range": [min(fprs), max(fprs)],
        "maj3_vs_mean_single": {
            "f1_delta": mv["f1_hallucination"] - mean_f1,
            "fpr_delta": mv["false_positive_rate"] - mean_fpr,
        },
    }
    (REPO / "results/ablations/e_decoding/result_gpt51_e3_selfconsistency.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
