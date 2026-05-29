#!/usr/bin/env python3
"""Patch mislabeled HALLUCINATED entries in dev_public.jsonl back to VALID.

Background
----------
A subset of entries labeled ``HALLUCINATED`` in the v1.0 ``dev_public`` split
are in fact real, correctly-cited papers. The wrong labels were produced by a
flawed automated labeling pass whose failure modes are well characterized:

  * No-op author "corruptions" ("N authors reduced to N" — nothing removed).
  * arXiv DataCite DOIs declared "does not resolve" (they do resolve), plus
    "no title match in CrossRef" for ML-conference papers CrossRef simply does
    not index (NeurIPS/ICLR/ICML rarely register DOIs).
  * ``swapped_authors`` / ``wrong_venue`` verdicts justified by comparison to a
    *different* CrossRef record the auto-matcher retrieved by mistake.
  * Weak-match reclassification ("weak title match sim=X, jacc=0.00") on entries
    whose metadata exactly matches a real, well-known paper.
  * Venue "mismatch" that is only an abbreviation-vs-full-name difference for the
    *same* venue (e.g. "NAACL" vs the full NAACL proceedings title).

Each key below was confirmed by hand against the real publication: title,
authors, year and venue all correspond to a single real paper, and the metadata
in the entry is correct. Genuine fabrications, real-title-with-truly-wrong
authors, chimeric DOIs, and impossible years are NOT in this list — they remain
``HALLUCINATED``.

Behavior
--------
For every confirmed key, this script rewrites the matching record in place:
  * ``label`` -> ``"VALID"``
  * adds ``relabeled_from = "HALLUCINATED"``
  * adds ``relabel_reason = "<short reason>"``
  * adds ``relabeled_by = "mislabel-audit-2026-05-29"``

The original ``hallucination_type`` and ``explanation`` are preserved verbatim
as an audit trail. The script keys off ``bibtex_key`` and is idempotent: a record
already carrying ``relabeled_by == "mislabel-audit-2026-05-29"`` is left untouched,
so re-running changes nothing. Existing key order is preserved; provenance fields
are appended at the end of each patched record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RELABELED_BY = "mislabel-audit-2026-05-29"

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "v1.0" / "dev_public.jsonl"

# bibtex_key -> short relabel_reason. Every entry verified to be a real,
# correctly-cited paper (title + authors + year + venue all correct).
CONFIRMED_MISLABELS: dict[str, str] = {
    # --- Seed cases (provided, re-verified) ---
    "ed071a6dfa34": (
        "Real paper (Gowal et al., 'Improving Robustness using Generated Data', "
        "NeurIPS 2021); flagged on a non-resolving arXiv DOI claim — arXiv DOIs resolve."
    ),
    "d7bd062b5e1b": (
        "Real paper (BLIP-2, Li/Li/Savarese/Hoi, ICML 2023) with the correct authors; "
        "'swapped_authors' came from matching a different CrossRef record (Anjia Cao et al.)."
    ),
    "eaa48be036ab": (
        "Real paper (Barlow Twins, Zbontar et al., ICML 2021) with exact authors; "
        "'plausible_fabrication' was a weak-match (sim=73, jacc=0.00) matcher failure."
    ),
    "a4d318041956": (
        "Real paper (HLGAT person re-ID, Zhang/Zhang/Liu, CVPR 2021); no-op author "
        "injection ('3 authors reduced to 3' — none removed)."
    ),
    # --- No-op author injections (corruption claimed but not applied) ---
    "c337460709f1": (
        "Real paper (Mao/Chakrabarti/Sarkar, 'Consistent Nonparametric Methods for "
        "Network Assisted Covariate Estimation', ICML 2021, PMLR v139); no-op author "
        "injection ('3 authors reduced to 3')."
    ),
    "ec7fc7e09a4f": (
        "Real paper (Zhang/Fu/Zheng, 'UAST: Uncertainty-Aware Siamese Tracking', "
        "ICML 2022, PMLR v162); no-op author injection ('3 authors reduced to 3')."
    ),
    # --- Weak-match reclassification of real papers (exact title/authors/venue/year) ---
    "dccb0bb90563": (
        "Real paper (ViT, 'An Image is Worth 16x16 Words', Dosovitskiy et al., ICLR 2021); "
        "weak-match reclassification (sim=79, jacc=0.08) despite exact authors."
    ),
    "c252e8ceaebd": (
        "Real paper (Peng et al., 'Random Feature Attention', ICLR 2021); weak-match "
        "reclassification (sim=71, jacc=0.00) despite matching authors."
    ),
    # --- arXiv-DOI 'does not resolve' on real papers (correct authors + venue + year) ---
    "b4268fa6464e": (
        "Real paper (Flamingo, Alayrac et al., NeurIPS 2022); flagged on a non-resolving "
        "arXiv DOI claim — arXiv DOIs resolve."
    ),
    "cdfcc07dff9e": (
        "Real paper (Dhariwal & Nichol, 'Diffusion Models Beat GANs on Image Synthesis', "
        "NeurIPS 2021); flagged on a non-resolving arXiv DOI claim — arXiv DOIs resolve."
    ),
    # --- 'wrong_venue' that is an abbreviation-vs-full-name of the SAME venue ---
    "e2c14cf2d74b": (
        "Real paper (Rubin/Herzig/Berant, 'Learning To Retrieve Prompts for In-Context "
        "Learning', NAACL 2022); 'wrong_venue' compared 'NAACL' to the full NAACL "
        "proceedings title — same venue."
    ),
    # --- Missed by the first pass; confirmed real by the adversarial review ---
    "bb8032fd12d8": (
        "Real paper (InstructGPT, Ouyang et al., 'Training Language Models to Follow "
        "Instructions with Human Feedback', NeurIPS 2022) with correct authors in order; "
        "'plausible_fabrication' (sim/jacc weak-match) was an auto-labeler failure."
    ),
}


def patch(data_path: Path, dry_run: bool = False) -> int:
    """Relabel confirmed mislabels in place. Returns the number of records changed."""
    if not data_path.exists():
        sys.exit(f"error: dataset not found at {data_path}")

    out_lines: list[str] = []
    changed = 0
    reverted = 0
    seen_keys: set[str] = set()
    already: list[str] = []

    with data_path.open(encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                out_lines.append(raw.rstrip("\n"))
                continue
            rec = json.loads(stripped)
            key = rec.get("bibtex_key")
            reason = CONFIRMED_MISLABELS.get(key)
            if reason is not None:
                seen_keys.add(key)
                if rec.get("relabeled_by") == RELABELED_BY:
                    # Idempotent: already patched in a prior run.
                    already.append(key)
                else:
                    rec["label"] = "VALID"
                    rec["relabeled_from"] = "HALLUCINATED"
                    rec["relabel_reason"] = reason
                    rec["relabeled_by"] = RELABELED_BY
                    changed += 1
            elif rec.get("relabeled_by") == RELABELED_BY:
                # Relabeled by a prior run of this audit but no longer a confirmed
                # mislabel (rejected by the adversarial review -- e.g. a genuine
                # author-order swap or dropped/duplicated author). Revert it to the
                # original HALLUCINATED label and strip the provenance fields.
                rec["label"] = rec.pop("relabeled_from", "HALLUCINATED")
                rec.pop("relabel_reason", None)
                rec.pop("relabeled_by", None)
                reverted += 1
            out_lines.append(json.dumps(rec, ensure_ascii=False))

    missing = sorted(set(CONFIRMED_MISLABELS) - seen_keys)
    if missing:
        print(f"WARNING: {len(missing)} confirmed key(s) not found in dataset: {missing}")
    if already:
        print(f"Idempotent skip: {len(already)} record(s) already relabeled by this audit.")

    if dry_run:
        print(f"[dry-run] would relabel {changed} record(s); no file written.")
        return changed

    # Write atomically to avoid corrupting the dataset if interrupted.
    tmp = data_path.with_suffix(data_path.suffix + ".tmp")
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    tmp.replace(data_path)
    print(f"Relabeled {changed} record(s) HALLUCINATED -> VALID in {data_path}")
    if reverted:
        print(
            f"Reverted {reverted} record(s) VALID -> HALLUCINATED (rejected by adversarial review)."
        )
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to dev_public.jsonl (default: data/v1.0/dev_public.jsonl).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    args = ap.parse_args()
    patch(args.data_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
