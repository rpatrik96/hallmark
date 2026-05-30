# HALLMARK `dev_public` Mislabel Audit

**Date:** 2026-05-29
**Auditor tag:** `mislabel-audit-2026-05-29`
**Target:** `data/v1.0/dev_public.jsonl` (1119 entries: 486 VALID, 633 HALLUCINATED)
**Patch script:** `scripts/patch_mislabels.py` *(removed — superseded by `scripts/relabel_ground_truth.py`, which re-verifies EVERY hallucinated entry from source evidence rather than a hand-curated allow-list; this doc is retained as the historical record of the original opportunistic pass)*

> **Adversarial-review correction (2026-05-29).** The first pass proposed 16
> relabels. An independent per-entry adversarial review (one skeptic per entry,
> defaulting to "keep HALLUCINATED" unless the paper is provably real with
> fully-correct metadata *including author order*) **rejected 5** of them as
> genuine corruptions the first pass missed, and **added 1** it had overlooked.
> Net: **12 confirmed mislabels** (498 VALID / 621 HALLUCINATED).
> - Rejected → kept HALLUCINATED: `be764c4d9889` (SDEdit — author order swapped + wrong year),
>   `ada7bb27c934` (GVP — 3rd author "Pratham" vs real "Patricia" Suriana),
>   `fa77166308b0` (Imagen — author duplicated/split + reordered),
>   `a24129d1c5e5` (Flamingo 19-author — 12 real authors dropped, 4 fabricated),
>   `aaefe29933ae` (FlashAttention — 5th author dropped/mangled).
> - Added → VALID: `bb8032fd12d8` (InstructGPT, Ouyang et al., NeurIPS 2022).
>
> The section below describes the ORIGINAL 16-entry pass; the 5 rejected entries
> are no longer relabeled (the patch script reverts them and is idempotent).

## Summary

A subset of `HALLUCINATED`-labeled entries are in fact real, correctly-cited
papers. The wrong labels were produced by a flawed automated labeling pass. This
audit identifies **16 confirmed mislabels** (relabeled to `VALID`) and **12
borderline cases** held for human review (left untouched). Confirmation requires
that the entry's **title + authors + year + venue all correspond to a single real
published paper** with correct metadata; the (frequently mangled) `doi` field is
treated as noise. Genuine fabrications, real-title-with-truly-wrong authors,
chimeric DOIs (DOI points to a *different* paper), placeholder text, and
impossible years remain `HALLUCINATED`.

## Mislabel patterns found

1. **No-op author injection.** `hallucination_type = partial_author_list` with an
   explanation "*N authors reduced to N*" (same number) — no author was actually
   removed. The `perturbation_scaleup` pipeline preserved the original real-paper
   metadata. Exactly 3 such entries exist; all 3 are real papers and confirmed.

2. **arXiv DOI "does not resolve" + "no title match in CrossRef."** The
   `[Re-classified]` pass declared `10.48550/arXiv.*` DOIs non-resolving (they do
   resolve via DataCite) and used CrossRef's lack of an ML-conference record
   (NeurIPS/ICLR/ICML rarely register DOIs) as fabrication evidence. Confirmed
   only where title + authors + venue + year match the real paper.

3. **Match against the wrong CrossRef record.** `swapped_authors` / `wrong_venue`
   verdicts justified by comparison to a *different* paper's authors/venue that
   the auto-matcher retrieved by mistake, while the entry's own metadata is the
   correct, well-known paper (e.g. BLIP-2 matched to "Anjia Cao et al.").

4. **Weak-match reclassification.** "*weak title match (sim=X), weak author match
   (jacc=0.00)*" on entries whose title and authors exactly match a famous paper
   the matcher simply failed to retrieve (ViT, Barlow Twins, GVP, RFA).

5. **Abbreviation-vs-full-name venue "mismatch".** `wrong_venue` where the two
   venues are the *same* venue under different names (e.g. "NAACL" vs
   "Proceedings of the 2022 Conference of the North American Chapter ...").

## Confirmed mislabels (relabeled HALLUCINATED -> VALID)

| # | bibtex_key | Title (real paper) | Real venue/year (evidence) | Orig. type | Why the label is wrong |
|---|------------|--------------------|----------------------------|------------|------------------------|
| 1 | `ed071a6dfa34` | Improving Robustness using Generated Data (Gowal, Rebuffi, Wiles, Stimberg, Calian, Mann) | NeurIPS 2021 (arXiv 2104.09425 / 2110.09468) | plausible_fabrication | arXiv DOI claimed non-resolving; arXiv DOIs resolve |
| 2 | `d7bd062b5e1b` | BLIP-2: Bootstrapping Language-Image Pre-training... (Li, Li, Savarese, Hoi) | ICML 2023 (arXiv 2301.12597) | swapped_authors | "swap" derived from a different CrossRef record (Anjia Cao et al.); authors are correct |
| 3 | `eaa48be036ab` | Barlow Twins: Self-Supervised Learning via Redundancy Reduction (Zbontar, Jing, Misra, LeCun, Deny) | ICML 2021 (PMLR v139) | plausible_fabrication | weak-match (sim=73, jacc=0.00) matcher failure; exact authors |
| 4 | `be764c4d9889` | SDEdit: Guided Image Synthesis and Editing with SDEs (Meng, He, Song, Song, Wu, Zhu, Ermon) | ICLR 2022 (arXiv 2108.01073) | wrong_venue | bad CrossRef match ("Stochastic Methods...") |
| 5 | `a4d318041956` | Person Re-ID Using Heterogeneous Local Graph Attention Networks (Zhang, Zhang, Liu) | CVPR 2021 | partial_author_list | no-op injection ("3 authors reduced to 3") |
| 6 | `c337460709f1` | Consistent Nonparametric Methods for Network Assisted Covariate Estimation (Mao, Chakrabarti, Sarkar) | ICML 2021, PMLR v139 (mao21a) | partial_author_list | no-op injection ("3 authors reduced to 3"); verified PMLR |
| 7 | `ec7fc7e09a4f` | UAST: Uncertainty-Aware Siamese Tracking (Zhang, Fu, Zheng) | ICML 2022, PMLR v162 (zhang22g) | partial_author_list | no-op injection ("3 authors reduced to 3"); verified PMLR |
| 8 | `dccb0bb90563` | An Image is Worth 16x16 Words (ViT; Dosovitskiy et al.) | ICLR 2021 (OpenReview YicbFdNTTy) | plausible_fabrication | weak-match (sim=79, jacc=0.08) despite exact authors |
| 9 | `ada7bb27c934` | Learning from Protein Structure with Geometric Vector Perceptrons (Jing, Eismann, Suriana, Townshend, Dror) | ICLR 2021 (arXiv 2009.01411) | plausible_fabrication | weak-match reclassification; matching authors |
| 10 | `c252e8ceaebd` | Random Feature Attention (Peng, Pappas, Yogatama, Schwartz, Smith, Kong) | ICLR 2021 (arXiv 2103.02143) | plausible_fabrication | weak-match (sim=71, jacc=0.00); matching authors |
| 11 | `fa77166308b0` | Photorealistic Text-to-Image Diffusion Models (Imagen; Saharia et al.) | NeurIPS 2022 | plausible_fabrication | arXiv DOI claimed non-resolving; authors/venue/year correct |
| 12 | `a24129d1c5e5` | Flamingo: a Visual Language Model for Few-Shot Learning (Alayrac et al.) | NeurIPS 2022 | plausible_fabrication | arXiv DOI claimed non-resolving; authors/venue/year correct |
| 13 | `b4268fa6464e` | Flamingo: A Visual Language Model for Few-Shot Learning (Alayrac et al.) | NeurIPS 2022 | plausible_fabrication | arXiv DOI claimed non-resolving; authors/venue/year correct |
| 14 | `cdfcc07dff9e` | Diffusion Models Beat GANs on Image Synthesis (Dhariwal, Nichol) | NeurIPS 2021 (proceedings 49ad23d1) | plausible_fabrication | arXiv DOI claimed non-resolving; authors/venue/year correct |
| 15 | `aaefe29933ae` | FlashAttention: Fast and Memory-Efficient Exact Attention (Dao, Fu, Ermon, Rudra, Ré) | NeurIPS 2022 (proceedings 67d57c32) | plausible_fabrication | arXiv DOI claimed non-resolving; authors/venue/year correct |
| 16 | `e2c14cf2d74b` | Learning To Retrieve Prompts for In-Context Learning (Rubin, Herzig, Berant) | NAACL 2022 (aclanthology 2022.naacl-main.191) | wrong_venue | "wrong venue" was "NAACL" vs the full NAACL proceedings name — same venue |

## Needs human review (NOT patched)

These entries name a real paper with correct title/authors, but carry a **venue
and/or year that does not match the real publication** — which could be a
legitimate injected corruption (HALLMARK includes `future_date`,
`preprint_as_published`, `arxiv_version_mismatch`, and genuine `wrong_venue`
types). Because the conservative test (title + authors + **year + venue** all
correct) fails, they are held for a human decision rather than relabeled.

| bibtex_key | Paper | Entry venue/year | Real venue/year | Concern |
|------------|-------|------------------|-----------------|---------|
| `b939e55d7555` | PaLM (Chowdhery et al.) | ICML 2022 | JMLR 2023 (tech report) | Never presented at ICML |
| `ae30af541b9e` | SCAFFOLD (Karimireddy et al.) | ICML 2023 | ICML 2020 | Year off by 3 |
| `f42736ee2643` | Beyond Homophily in GNNs (Zhu et al.) | NeurIPS 2023 | NeurIPS 2020 | Year off by 3 |
| `eee1d119e4bb` | Linformer (Wang et al.) | NeurIPS 2022 | arXiv 2020 only | Never at NeurIPS |
| `c080618bff76` | Concrete Problems in AI Safety (Amodei et al.) | "arXiv ... 2021" | arXiv 2016 | Year wrong (2016 preprint) |
| `db9d82ff3f94` | Constitutional AI (Bai et al.) | ICLR 2023 | Anthropic report, arXiv 2022 | Not an ICLR paper |
| `fdcf8e3071b7` | BYOL (Grill et al.) | NeurIPS 2022 | NeurIPS 2020 | Year off by 2 |
| `c5499dfe46ab` | Understanding Training Regimes in Continual Learning (Mirzadeh et al.) | NeurIPS 2023 | NeurIPS 2020 | Year off by 3 |
| `d46c5fa004ad` | Attention Is All You Need (Vaswani et al.) | NeurIPS 2022 | NeurIPS 2017 | Correct authors, year off by 5 — possible intentional corruption |
| `f887a22f37bf` | Attention Is All You Need (Vaswani et al.) | NeurIPS 2022 | NeurIPS 2017 | Duplicate of above; same concern |
| `e24dfb8c96e7` | Reinforcement Learning: An Introduction (Sutton & Barto) | MIT Press 2022 | MIT Press 2018 (2nd ed) | Correct authors, year slightly off |
| `f9247e4251cf` | Language Models are Few-Shot Learners (GPT-3; Brown et al.) | NeurIPS 2023 | NeurIPS 2020 | Correct authors, year off by 3 |

Note on the `swapped_authors` group (rows for "Attention Is All You Need",
GPT-3, Sutton & Barto): the `[Re-classified]` explanation format lists the
**entry's** author string first and the (mismatched) CrossRef record second. In
these rows the entry author string is the *correct* author list for the famous
paper, so the "author swap" is an auto-verifier artifact — but the **year** is
wrong, so they are held for review rather than relabeled. Other `swapped_authors`
entries (e.g. APE `ab67ecdd4887`, Least-to-Most `db9a596a4d3f`, ViT-Adapter
`d291ca8fd3de`) have genuinely *wrong* first authors and correctly remain
`HALLUCINATED`.

## Method notes

- Verification used model knowledge plus web search against authoritative
  sources (PMLR, NeurIPS/ICLR proceedings, ACL Anthology, arXiv, OpenReview).
- The entry `doi` field is unreliable (often points to a different paper even for
  real entries) and was not used as positive or negative evidence on its own.
- Entries whose DOI resolves to a *different* paper than the stated title
  (chimeras, e.g. `caef38397355`, `b64d4175823c`, `d29bc1a68d33`,
  `cdf7c1798d3d`) remain `HALLUCINATED`.
- The original `scripts/patch_mislabels.py` allow-list pass has been **removed**;
  ground-truth relabeling is now done end-to-end by `scripts/relabel_ground_truth.py`,
  which re-verifies every hallucinated entry from source evidence and is itself
  idempotent (a warm-cache `--apply` on already-correct data is a no-op).
