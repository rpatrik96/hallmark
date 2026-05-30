#!/usr/bin/env python3
"""Patch mislabeled HALLUCINATED entries in dev_public.jsonl and test_public.jsonl back to VALID.

Background
----------
A subset of entries labeled ``HALLUCINATED`` in the v1.0 ``dev_public`` and
``test_public`` splits are in fact real, correctly-cited papers. The wrong labels
were produced by a flawed automated labeling pass whose failure modes are well
characterized:

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

Batch 1 — "mislabel-audit-2026-05-29" (dev_public only, 12 entries)
--------------------------------------------------------------------
Initial audit of dev_public; confirmed via title/author/venue/year lookup.
Covered the first 12 mislabeled dev_public entries discovered in that pass.

Batch 2 — "mislabel-audit-2026-05-30-leak-followup" (3 dev + 13 test, 16 entries)
-----------------------------------------------------------------------------------
An independent re-verification of bibtex-check "leaks" (entries the fact-checker
verified as correct citations despite a HALLUCINATED label in HALLMARK v1.0) was
performed on both splits using a two-stage workflow: independent ground-truth
verification followed by adversarial refutation. 16 of the leaks were confirmed as
MISLABEL_CONFIRMED at high confidence; 5 were genuine leaks kept HALLUCINATED.
Failure mode across all 16: the auto-labeller either (a) matched the wrong CrossRef
record by title-fuzz, (b) compared "First Last" author lists against CrossRef's
"Last, First" format and declared swapped_authors, or (c) treated a non-resolving
arXiv DataCite DOI as evidence of fabrication.

Batch 3 — "mislabel-audit-2026-05-30-postfix-followup" (1 dev + 1 test, 2 entries)
-------------------------------------------------------------------------------------
A post-fix re-run after applying bibtex-check FIX 1+2+3+5 surfaced two additional
entries that the patched tool now verifies as correct but HALLMARK still labels
HALLUCINATED. Both were independently confirmed (verify + adversarial refute,
high-conf) as real, correctly-cited papers. Failure mode identical to batch 2:
arXiv DataCite DOIs not indexed by CrossRef caused the auto-labeller to emit
``plausible_fabrication`` for canonical, well-known papers (Imagen, AdaFed).

Batch 4 — "mislabel-audit-2026-05-30-cnv-followup" (1 dev, 1 entry)
---------------------------------------------------------------------
Follow-up to the v1.1.0 bibtex-check CNV venue fix (rpatrik96/bibtexupdater@ea63b7d):
the fix added PLATFORM_MARKERS and _strip_track_decorations, routing arXiv-only
preprint venue strings to NON_COMPARABLE instead of a failed match — lifting
``a0478afc6fb9`` (Classifier-Free Diffusion Guidance, Ho & Salimans) from
``unconfirmed`` (venue) to ``verified``, which exposed its ``swapped_authors``
mislabel. Same arXiv-DOI/CrossRef wrong-record failure mode as batch 2/3 (joins
the FlashAttention/DDPM/Imagen pattern).

Provenance / conflict with prior audit
---------------------------------------
The 2026-05-29 batch's adversarial reviewer **explicitly rejected** relabeling
two entries that appear in batch 2, and one entry that appears in batch 3:

  * ``aaefe29933ae`` — FlashAttention (Tri Dao et al., NeurIPS 2022)
    Prior rejection: "DOI does not resolve" was taken at face value.
    Override rationale: arXiv:2205.14135 ("FlashAttention: Fast and
    Memory-Efficient Exact Attention with IO-Awareness") is the canonical paper.
    The DOI ``10.48550/arXiv.2205.14135`` is a valid arXiv DataCite DOI and
    *does* resolve. The CrossRef non-index of the NeurIPS proceedings is the
    auto-labeller's failure, not the citation's. Independently re-verified
    against the NeurIPS 2022 proceedings page and arXiv. Deliberate override
    informed by new evidence.

  * ``fc4aaf478a08`` — Diffusion Models Beat GANs (Dhariwal & Nichol, NeurIPS 2021)
    Prior rejection: swapped_authors verdict was left unresolved.
    Override rationale: arXiv:2105.05233 ("Diffusion Models Beat GANs on Image
    Synthesis") confirms authors as Prafulla Dhariwal and Alex Nichol in that
    order; the CrossRef mismatch ("Rashmi V and Radhika S K") is a retrieval
    failure — a different paper was matched. Independently re-verified against
    arXiv and the NeurIPS 2021 proceedings. Deliberate override informed by new
    evidence.

  * ``fa77166308b0`` — Imagen (Saharia et al., NeurIPS 2022) — relabeled in batch 3
    Prior rejection: insufficient evidence at the time of the 2026-05-29 audit.
    Override rationale: arXiv:2205.11487 ("Photorealistic Text-to-Image Diffusion
    Models with Deep Language Understanding") confirms the 14-author list (Chitwan
    Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton,
    Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha
    Gontijo Lopes, Tim Salimans, Jonathan Ho, David J. Fleet, Mohammad Norouzi)
    and acceptance at NeurIPS 2022. CrossRef non-indexing of the NeurIPS proceedings
    is the auto-labeller's failure. Independently re-verified on arXiv.
    Deliberate override informed by new evidence.

All three relabelings are deliberate, not oversights. The prior-audit provenance
fields (``hallucination_type``, ``explanation``) are preserved verbatim as the
audit trail.

Behavior
--------
For every confirmed key, this script rewrites the matching record in place:
  * ``label`` -> ``"VALID"``
  * adds ``relabeled_from = "HALLUCINATED"``
  * adds ``relabel_reason = "<short reason>"``
  * adds ``relabeled_by = "<batch tag>"``

The original ``hallucination_type`` and ``explanation`` are preserved verbatim
as an audit trail. The script keys off ``bibtex_key`` and is idempotent: a record
already carrying the expected ``relabeled_by`` tag is left untouched, so re-running
changes nothing. Existing key order is preserved; provenance fields are appended
at the end of each patched record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Batch 1 tag (dev_public only, 12 entries, 2026-05-29 audit)
RELABELED_BY_BATCH1 = "mislabel-audit-2026-05-29"

# Batch 2 tag (3 dev + 13 test, 2026-05-30 leak-followup)
RELABELED_BY_BATCH2 = "mislabel-audit-2026-05-30-leak-followup"

# Batch 3 tag (1 dev + 1 test, 2026-05-30 post-fix re-run)
RELABELED_BY_BATCH3 = "mislabel-audit-2026-05-30-postfix-followup"

# Batch 4 tag (1 dev, 2026-05-30 CNV venue-fix follow-up)
RELABELED_BY_BATCH4 = "mislabel-audit-2026-05-30-cnv-followup"

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "v1.0"

# ---------------------------------------------------------------------------
# Batch 1: dev_public entries (12 keys, tag = RELABELED_BY_BATCH1)
# bibtex_key -> short relabel_reason
# ---------------------------------------------------------------------------
CONFIRMED_MISLABELS_BATCH1: dict[str, str] = {
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

# ---------------------------------------------------------------------------
# Batch 2: dev_public (3) + test_public (13) entries, tag = RELABELED_BY_BATCH2
# Each value is a dict with "reason" (short audit note) and "split" ("dev"|"test").
# ---------------------------------------------------------------------------
CONFIRMED_MISLABELS_BATCH2: dict[str, dict[str, str]] = {
    # --- dev_public (3 entries) ---
    "f36bff1b0e11": {
        "split": "dev",
        "reason": (
            "Real paper (Chen Zhao et al., 'Re2TAL: Rewiring Pretrained Video Backbones "
            "for Reversible Temporal Action Localization', CVPR 2023); auto-labeller "
            "conflated arXiv preprint date with publication year and matched wrong CrossRef record."
        ),
    },
    "fc4aaf478a08": {
        "split": "dev",
        "reason": (
            "Real paper (Prafulla Dhariwal and Alex Nichol, 'Diffusion Models Beat GANs "
            "on Image Synthesis', NeurIPS 2021, arXiv:2105.05233); 'swapped_authors' verdict "
            "was a CrossRef retrieval failure — a different paper matched instead of the canonical one."
        ),
    },
    "aaefe29933ae": {
        "split": "dev",
        "reason": (
            "Real paper (Tri Dao et al., 'FlashAttention: Fast and Memory-Efficient Exact "
            "Attention with IO-Awareness', NeurIPS 2022, arXiv:2205.14135); DOI resolves "
            "correctly; 'does not resolve' was an auto-labeller failure on arXiv DataCite DOIs."
        ),
    },
    # --- test_public (13 entries) ---
    "b6a26b30b14a": {
        "split": "test",
        "reason": (
            "Real paper (Ioana Bica, Ahmed M. Alaa, Mihaela van der Schaar, 'Time Series "
            "Deconfounder', ICML 2020); 'swapped_authors' from CrossRef matching a different paper."
        ),
    },
    "ae61732dac84": {
        "split": "test",
        "reason": (
            "Real paper (Choromanski et al., 'Rethinking Attention with Performers', ICLR 2021); "
            "auto-labeller matched wrong CrossRef record — correct authors and venue confirmed."
        ),
    },
    "bf3addc43203": {
        "split": "test",
        "reason": (
            "Real paper (Xinlei Chen and Kaiming He, 'Exploring Simple Siamese Representation "
            "Learning', CVPR 2021); CrossRef retrieval failure produced author mismatch verdict."
        ),
    },
    "bb59c4bd80e7": {
        "split": "test",
        "reason": (
            "Real paper (Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton, "
            "'A Simple Framework for Contrastive Learning of Visual Representations', ICML 2020); "
            "CrossRef matched wrong record — correct authors and venue independently re-verified."
        ),
    },
    "f46e9569871d": {
        "split": "test",
        "reason": (
            "Real paper (Maithra Raghu et al., 'Do Vision Transformers See Like Convolutional "
            "Neural Networks?', NeurIPS 2021); CrossRef retrieval failure; correct authors "
            "and venue confirmed via arXiv and NeurIPS proceedings."
        ),
    },
    "ff065f4c5765": {
        "split": "test",
        "reason": (
            "Real paper (Rowan Zellers et al., 'MERLOT: Multimodal Neural Script Knowledge "
            "Models', NeurIPS 2021); auto-labeller matched wrong CrossRef record — correct "
            "authors and venue independently confirmed."
        ),
    },
    "c437b8d2fcd4": {
        "split": "test",
        "reason": (
            "Real paper (Jonathan Ho, Ajay Jain, Pieter Abbeel, 'Denoising Diffusion "
            "Probabilistic Models', NeurIPS 2020); 'swapped_authors' verdict came from "
            "CrossRef returning a different paper — correct authors confirmed via arXiv."
        ),
    },
    "b97858a4638f": {
        "split": "test",
        "reason": (
            "Real paper (Pietro Buzzega et al., 'Dark Experience for General Continual "
            "Learning', NeurIPS 2020); CrossRef retrieval failure produced author mismatch; "
            "correct authors and venue confirmed."
        ),
    },
    "b6d85d903dcf": {
        "split": "test",
        "reason": (
            "Real paper (Victor Veitch et al., 'Counterfactual Invariance to Spurious "
            "Correlations in Text Classification', NeurIPS 2021 spotlight); CrossRef matched "
            "wrong record — correct authors and venue confirmed via arXiv."
        ),
    },
    "e7d2f0c1e698": {
        "split": "test",
        "reason": (
            "Real paper (Jason Wei et al., 'Chain-of-Thought Prompting Elicits Reasoning "
            "in Large Language Models', NeurIPS 2022); weak-match reclassification failure — "
            "correct authors and venue confirmed via arXiv and NeurIPS proceedings."
        ),
    },
    "bbd1bcc55e09": {
        "split": "test",
        "reason": (
            "Real paper (Hongbin Pei et al., 'Geom-GCN: Geometric Graph Convolutional "
            "Networks', ICLR 2020); CrossRef retrieval failure produced mismatch verdict; "
            "correct authors and venue confirmed."
        ),
    },
    "b9ce7ba367a4": {
        "split": "test",
        "reason": (
            "Real paper (Takeshi Kojima et al., 'Large Language Models are Zero-Shot "
            "Reasoners', NeurIPS 2022); auto-labeller matched wrong CrossRef record — "
            "correct authors and venue confirmed via arXiv."
        ),
    },
    "f34f26c83153": {
        "split": "test",
        "reason": (
            "Real paper (Xuanyi Dong and Yi Yang, 'NAS-Bench-201: Extending the Scope of "
            "Reproducible Neural Architecture Search', ICLR 2020 spotlight); CrossRef "
            "retrieval failure; correct authors and venue confirmed."
        ),
    },
}


# ---------------------------------------------------------------------------
# Batch 3: dev_public (1) + test_public (1) entries, tag = RELABELED_BY_BATCH3
# Each value is a dict with "reason" and "split" ("dev"|"test").
# ---------------------------------------------------------------------------
CONFIRMED_MISLABELS_BATCH3: dict[str, dict[str, str]] = {
    # --- dev_public (1 entry) ---
    "fa77166308b0": {
        "split": "dev",
        "reason": (
            "Real paper (Chitwan Saharia et al., 'Photorealistic Text-to-Image Diffusion "
            "Models with Deep Language Understanding' [Imagen], NeurIPS 2022, arXiv:2205.11487); "
            "'plausible_fabrication' was a CrossRef non-index failure on the NeurIPS proceedings — "
            "14-author list and venue confirmed via arXiv."
        ),
    },
    # --- test_public (1 entry) ---
    "f746e1c10ae9": {
        "split": "test",
        "reason": (
            "Real paper (Sashank Reddi et al., 'Adaptive Federated Optimization' [AdaFed], "
            "ICLR 2021, arXiv:2003.00295); 'plausible_fabrication' was a CrossRef non-index "
            "failure — correct authors and venue confirmed via arXiv."
        ),
    },
}


# ---------------------------------------------------------------------------
# Batch 4: dev_public (1) entry, tag = RELABELED_BY_BATCH4
# Surfaced by the v1.1.0 bibtex-check CNV venue fix (ea63b7d).
# Each value is a dict with "reason" and "split" ("dev"|"test").
# ---------------------------------------------------------------------------
CONFIRMED_MISLABELS_BATCH4: dict[str, dict[str, str]] = {
    # --- dev_public (1 entry) ---
    "a0478afc6fb9": {
        "split": "dev",
        "reason": (
            "Real paper (Jonathan Ho and Tim Salimans, 'Classifier-Free Diffusion Guidance', "
            "NeurIPS 2022 Workshop on Score-Based Methods, arXiv:2207.12598); "
            "'swapped_authors' tag came from a CrossRef title-fuzz match onto an unrelated "
            "record ('jialiang jiang'); exact 2-author list and venue confirmed via arXiv. "
            "Surfaced by the v1.1.0 bibtex-check CNV venue fix that lifted the entry from "
            "'unconfirmed' (venue) to 'verified'."
        ),
    },
}


def _patch_split(
    data_path: Path,
    mislabels_batch1: dict[str, str] | None,
    mislabels_batch2: dict[str, dict[str, str]] | None,
    mislabels_batch3: dict[str, dict[str, str]] | None,
    mislabels_batch4: dict[str, dict[str, str]] | None,
    split_name: str,
    dry_run: bool = False,
) -> tuple[int, int, int, int, int]:
    """Relabel confirmed mislabels in one jsonl file.

    Returns (changed_batch1, changed_batch2, changed_batch3, changed_batch4, reverted).
    """
    if not data_path.exists():
        sys.exit(f"error: dataset not found at {data_path}")

    out_lines: list[str] = []
    changed_b1 = 0
    changed_b2 = 0
    changed_b3 = 0
    changed_b4 = 0
    reverted = 0
    seen_b1: set[str] = set()
    seen_b2: set[str] = set()
    seen_b3: set[str] = set()
    seen_b4: set[str] = set()
    already_b1: list[str] = []
    already_b2: list[str] = []
    already_b3: list[str] = []
    already_b4: list[str] = []

    b1 = mislabels_batch1 or {}
    b2 = {k: v for k, v in (mislabels_batch2 or {}).items() if v["split"] == split_name}
    b3 = {k: v for k, v in (mislabels_batch3 or {}).items() if v["split"] == split_name}
    b4 = {k: v for k, v in (mislabels_batch4 or {}).items() if v["split"] == split_name}

    with data_path.open(encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                out_lines.append(raw.rstrip("\n"))
                continue
            rec = json.loads(stripped)
            key = rec.get("bibtex_key")

            if key in b1:
                seen_b1.add(key)
                if rec.get("relabeled_by") == RELABELED_BY_BATCH1:
                    already_b1.append(key)
                else:
                    rec["label"] = "VALID"
                    rec["relabeled_from"] = "HALLUCINATED"
                    rec["relabel_reason"] = b1[key]
                    rec["relabeled_by"] = RELABELED_BY_BATCH1
                    changed_b1 += 1
            elif key in b2:
                seen_b2.add(key)
                if rec.get("relabeled_by") == RELABELED_BY_BATCH2:
                    already_b2.append(key)
                else:
                    rec["label"] = "VALID"
                    rec["relabeled_from"] = "HALLUCINATED"
                    rec["relabel_reason"] = b2[key]["reason"]
                    rec["relabeled_by"] = RELABELED_BY_BATCH2
                    changed_b2 += 1
            elif key in b3:
                seen_b3.add(key)
                if rec.get("relabeled_by") == RELABELED_BY_BATCH3:
                    already_b3.append(key)
                else:
                    rec["label"] = "VALID"
                    rec["relabeled_from"] = "HALLUCINATED"
                    rec["relabel_reason"] = b3[key]["reason"]
                    rec["relabeled_by"] = RELABELED_BY_BATCH3
                    changed_b3 += 1
            elif key in b4:
                seen_b4.add(key)
                if rec.get("relabeled_by") == RELABELED_BY_BATCH4:
                    already_b4.append(key)
                else:
                    rec["label"] = "VALID"
                    rec["relabeled_from"] = "HALLUCINATED"
                    rec["relabel_reason"] = b4[key]["reason"]
                    rec["relabeled_by"] = RELABELED_BY_BATCH4
                    changed_b4 += 1
            elif rec.get("relabeled_by") == RELABELED_BY_BATCH1:
                # Relabeled by a prior run of batch1 but no longer confirmed
                # (rejected by adversarial review). Revert.
                rec["label"] = rec.pop("relabeled_from", "HALLUCINATED")
                rec.pop("relabel_reason", None)
                rec.pop("relabeled_by", None)
                reverted += 1

            out_lines.append(json.dumps(rec, ensure_ascii=False))

    # Report missing keys for this split
    missing_b1 = sorted(set(b1) - seen_b1)
    missing_b2 = sorted(set(b2) - seen_b2)
    missing_b3 = sorted(set(b3) - seen_b3)
    missing_b4 = sorted(set(b4) - seen_b4)
    if missing_b1:
        print(f"WARNING [{split_name}]: {len(missing_b1)} batch-1 key(s) not found: {missing_b1}")
    if missing_b2:
        print(f"WARNING [{split_name}]: {len(missing_b2)} batch-2 key(s) not found: {missing_b2}")
    if missing_b3:
        print(f"WARNING [{split_name}]: {len(missing_b3)} batch-3 key(s) not found: {missing_b3}")
    if missing_b4:
        print(f"WARNING [{split_name}]: {len(missing_b4)} batch-4 key(s) not found: {missing_b4}")
    if already_b1:
        print(
            f"Idempotent skip [{split_name}]: {len(already_b1)} batch-1 record(s) already relabeled."
        )
    if already_b2:
        print(
            f"Idempotent skip [{split_name}]: {len(already_b2)} batch-2 record(s) already relabeled."
        )
    if already_b3:
        print(
            f"Idempotent skip [{split_name}]: {len(already_b3)} batch-3 record(s) already relabeled."
        )
    if already_b4:
        print(
            f"Idempotent skip [{split_name}]: {len(already_b4)} batch-4 record(s) already relabeled."
        )

    if dry_run:
        total = changed_b1 + changed_b2 + changed_b3 + changed_b4
        print(
            f"[dry-run] [{split_name}] would relabel {total} record(s) "
            f"(batch1={changed_b1}, batch2={changed_b2}, batch3={changed_b3}, batch4={changed_b4}); no file written."
        )
        return changed_b1, changed_b2, changed_b3, changed_b4, reverted

    tmp = data_path.with_suffix(data_path.suffix + ".tmp")
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    tmp.replace(data_path)
    total = changed_b1 + changed_b2 + changed_b3 + changed_b4
    print(
        f"[{split_name}] Relabeled {total} record(s) HALLUCINATED -> VALID "
        f"(batch1={changed_b1}, batch2={changed_b2}, batch3={changed_b3}, batch4={changed_b4}) in {data_path}"
    )
    if reverted:
        print(
            f"[{split_name}] Reverted {reverted} record(s) VALID -> HALLUCINATED "
            "(rejected by adversarial review)."
        )
    return changed_b1, changed_b2, changed_b3, changed_b4, reverted


def patch(
    dev_path: Path,
    test_path: Path,
    dry_run: bool = False,
) -> int:
    """Relabel confirmed mislabels in dev and test splits. Returns total records changed."""
    b1_dev, b2_dev, b3_dev, b4_dev, _rev_dev = _patch_split(
        dev_path,
        mislabels_batch1=CONFIRMED_MISLABELS_BATCH1,
        mislabels_batch2=CONFIRMED_MISLABELS_BATCH2,
        mislabels_batch3=CONFIRMED_MISLABELS_BATCH3,
        mislabels_batch4=CONFIRMED_MISLABELS_BATCH4,
        split_name="dev",
        dry_run=dry_run,
    )
    _, b2_test, b3_test, b4_test, _rev_test = _patch_split(
        test_path,
        mislabels_batch1=None,  # batch1 was dev-only
        mislabels_batch2=CONFIRMED_MISLABELS_BATCH2,
        mislabels_batch3=CONFIRMED_MISLABELS_BATCH3,
        mislabels_batch4=CONFIRMED_MISLABELS_BATCH4,
        split_name="test",
        dry_run=dry_run,
    )
    total = b1_dev + b2_dev + b3_dev + b4_dev + b2_test + b3_test + b4_test
    print(
        f"\nTotal relabeled: {total} "
        f"(dev batch1={b1_dev}, dev batch2={b2_dev}, dev batch3={b3_dev}, dev batch4={b4_dev}, "
        f"test batch2={b2_test}, test batch3={b3_test}, test batch4={b4_test})"
    )
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing dev_public.jsonl and test_public.jsonl (default: data/v1.0/).",
    )
    ap.add_argument(
        "--dev-path",
        type=Path,
        default=None,
        help="Override path to dev_public.jsonl.",
    )
    ap.add_argument(
        "--test-path",
        type=Path,
        default=None,
        help="Override path to test_public.jsonl.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    # Legacy compat: if --data-path is supplied, treat it as --dev-path only.
    ap.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args()

    if args.data_path is not None:
        # Legacy single-file invocation (batch1 compat).
        dev_path = args.data_path
        test_path = (
            (args.data_dir / "test_public.jsonl") if args.test_path is None else args.test_path
        )
    else:
        dev_path = args.dev_path or (args.data_dir / "dev_public.jsonl")
        test_path = args.test_path or (args.data_dir / "test_public.jsonl")

    patch(dev_path, test_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
