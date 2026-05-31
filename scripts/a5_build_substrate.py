#!/usr/bin/env python3
"""A5 inter-annotator-agreement substrate builder.

Builds the natural re-annotation substrate for the inter-annotator kappa study:

  * real-world hallucinated incidents -- entries whose ``generation_method`` is
    ``real_world`` or whose ``source`` is ``gptzero_neurips2025`` / ``real_world``
    (all currently labelled HALLUCINATED), and
  * relabel-recovered real papers -- entries that the systematic ground-truth
    relabel pass flipped HALLUCINATED -> VALID (identified by diffing the
    pre-relabel commit ``f58f779`` against ``HEAD``).

These two pools are the entries whose ground-truth label is most contestable, so
they are exactly where inter-annotator reliability matters. The script emits a
blinded substrate (metadata only -- no gold label, explanation, source, or
generation method) plus a gold key (kept separate so raters cannot see it).

Outputs (under ``results/ablations/a5_kappa/``):
  * ``substrate_blinded.jsonl``   -- one blinded entry per line (what raters see)
  * ``substrate_gold.jsonl``      -- bibtex_key -> gold label + provenance
  * ``substrate_manifest.json``   -- counts + provenance summary
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "v1.0"
OUT = REPO / "results" / "ablations" / "a5_kappa"
PRE_RELABEL_REF = "f58f779"  # last commit before the systematic relabel pass


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_ref(ref: str, rel_path: str) -> dict[str, dict]:
    out = subprocess.run(
        ["git", "show", f"{ref}:{rel_path}"],
        capture_output=True,
        text=True,
        cwd=REPO,
    ).stdout
    d: dict[str, dict] = {}
    for line in out.splitlines():
        if line.strip():
            e = json.loads(line)
            d[e["bibtex_key"]] = e
    return d


def _relabel_recovered_keys() -> set[str]:
    """Keys flipped HALLUCINATED -> VALID by the relabel pass."""
    keys: set[str] = set()
    for rel in ("data/v1.0/dev_public.jsonl", "data/v1.0/test_public.jsonl"):
        old = _load_ref(PRE_RELABEL_REF, rel)
        new = _load_ref("HEAD", rel)
        for k, e in new.items():
            if k in old and old[k]["label"] == "HALLUCINATED" and e["label"] == "VALID":
                keys.add(k)
    return keys


def _is_real_world(e: dict) -> bool:
    return e.get("generation_method") == "real_world" or e.get("source") in (
        "gptzero_neurips2025",
        "real_world",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    relabel_keys = _relabel_recovered_keys()

    blinded: list[dict] = []
    gold: list[dict] = []
    seen: set[str] = set()
    counts = {
        "real_world_hallucinated": 0,
        "relabel_recovered_valid": 0,
    }

    for split in ("dev_public.jsonl", "test_public.jsonl"):
        for e in _load_jsonl(DATA / split):
            key = e["bibtex_key"]
            in_rw = _is_real_world(e)
            in_relabel = key in relabel_keys
            if not (in_rw or in_relabel) or key in seen:
                continue
            seen.add(key)

            if in_relabel:
                pool = "relabel_recovered"
                counts["relabel_recovered_valid"] += 1
            else:
                pool = "real_world_incident"
                counts["real_world_hallucinated"] += 1

            blinded.append(
                {
                    "bibtex_key": key,
                    "bibtex_type": e["bibtex_type"],
                    "fields": e["fields"],
                    "raw_bibtex": e.get("raw_bibtex"),
                }
            )
            gold.append(
                {
                    "bibtex_key": key,
                    "gold_label": e["label"],
                    "gold_hallucination_type": e.get("hallucination_type"),
                    "pool": pool,
                    "split": split.replace("_public.jsonl", ""),
                    "source": e.get("source"),
                    "generation_method": e.get("generation_method"),
                }
            )

    # Deterministic order so the substrate is reproducible.
    blinded.sort(key=lambda r: r["bibtex_key"])
    gold.sort(key=lambda r: r["bibtex_key"])

    (OUT / "substrate_blinded.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in blinded) + "\n"
    )
    (OUT / "substrate_gold.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in gold) + "\n"
    )

    gold_label_dist = {"VALID": 0, "HALLUCINATED": 0}
    for g in gold:
        gold_label_dist[g["gold_label"]] += 1

    manifest = {
        "description": "A5 inter-annotator-agreement substrate: real-world "
        "hallucinated incidents + relabel-recovered real papers.",
        "pre_relabel_ref": PRE_RELABEL_REF,
        "n_entries": len(blinded),
        "pool_counts": counts,
        "gold_label_distribution": gold_label_dist,
        "blinded_fields_exposed": [
            "bibtex_key",
            "bibtex_type",
            "fields",
            "raw_bibtex",
        ],
        "blinded_fields_hidden": [
            "label",
            "hallucination_type",
            "explanation",
            "source",
            "generation_method",
            "difficulty_tier",
        ],
    }
    (OUT / "substrate_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Substrate: {len(blinded)} entries")
    print(f"  real-world hallucinated : {counts['real_world_hallucinated']}")
    print(f"  relabel-recovered valid : {counts['relabel_recovered_valid']}")
    print(f"  gold label dist         : {gold_label_dist}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
