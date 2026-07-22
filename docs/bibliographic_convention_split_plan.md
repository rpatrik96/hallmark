# Proposal: a bibliographic-convention split

**Status:** plan only — nothing implemented, no data added.
Captured 2026-07-22. Branch: `docs/bibliographic-convention-split`.

## The gap

`docs/v1.1_expansion_brainstorm.md` names **domain-monoculture** as a v1.0
weakness and addresses it *topically* — `test_crossdomain_matched` moves from ML
conferences to PubMed. That covers a shift in subject matter. It does not cover a
shift in **bibliographic convention**, and the benchmark currently has none.

Measured across all three splits (`dev_public` 1119, `test_public` 831,
`test_crossdomain_matched` 452 — 2402 entries):

| feature | entries |
|---|---|
| entry types other than `@inproceedings` / `@article` | **0** |
| `howpublished` field present | **0** |
| `editor` present with no `author` | **0** |
| LaTeX letter-macro accents (`\H`, `\c`, `\k`, `\v`, `\u`, `\r`) | **0** |
| venue naming a publisher book series (LNCS, CCIS, LNICST, IFIP AICT, SCI) | **0** |

Fields present are exactly `author, title, year, booktitle, doi, url, journal`
(plus one `volume`/`number`). `test_crossdomain_matched` is PubMed-matched but
still ASCII `@article`/`@inproceedings`, so it shifts the *subject* while holding
the *conventions* fixed.

**Consequence.** A tool can be perfect on HALLMARK and still fail systematically
on a European systems bibliography, a humanities bibliography, or anything using
`@proceedings`, `@phdthesis`, front matter, or non-ASCII names. The benchmark
cannot see it either way.

This is not hypothetical. Working on `bibtex-updater` (2026-07-22) we found four
defects that HALLMARK provably cannot measure — every one of them is inert on all
2402 entries — yet together they produced a **50% false-positive rate** on a
held-out set of 44 real, DOI-verified, valid references, falling to **2.3%** once
fixed. Chief among them: `latex_to_plain` reduced `Heged{\H u}s` to the surname
key `us` and `Erd{\H o}s` to `os`, so author lists that were character-for-character
correct scored 0.0 similarity. Any tool sharing that normalization bug scores
identically on HALLMARK today.

## What a new split would test

Four axes, none currently represented. Each is a *convention*, not a topic, so
the split is orthogonal to `test_crossdomain_matched`.

1. **Non-ASCII author names in LaTeX escaping** — Hungarian double acute
   (`{\H o}`, `{\H u}`), Polish ogonek/stroke (`{\k e}`, `{\l}`), Turkish and
   Romanian cedilla/breve (`{\c c}`, `{\u a}`), Czech caron (`{\v c}`),
   Scandinavian ring/slash (`{\aa}`, `{\o}`), and the dotless-i form `{\'\i}`.
2. **Publisher-series venues** — papers whose `booktitle` is the conference but
   whose indexed container-title is LNCS/CCIS/LNICST/IFIP AICT/SCI.
3. **Volume-level and non-article entry types** — `@proceedings` (carrying
   `editor`, no `author`, titled after the conference), `@phdthesis`,
   `@techreport`.
4. **Front matter with `howpublished`** — editorials and magazine columns whose
   venue lives in `howpublished` rather than `journal`.

## Candidate material

### Ready, with caveats: the 44-entry held-out set

Built 2026-07-22 as an independent validation set for the `bibtex-updater` fixes
(so it must NOT become the benchmark those fixes are then scored on — see
Contamination). Composition: 14 accented-name entries, 10 Springer/IFIP series
entries, 5 `@proceedings` with `editor`, 5 `@misc` with `howpublished`, 10 ASCII
controls. All 44 are VALID; every DOI re-resolved 200 against Crossref.

It usefully separates two failure modes: 8 of the accented entries have the
diacritics stored **in Crossref itself** (isolating a pure LaTeX-decoding
failure) while 6 are ASCII-folded upstream (isolating a folding failure).

**Not benchmark-ready as-is.** It is all-VALID, so it measures FPR only and
cannot produce a detection rate. It is agent-assembled from public metadata and
has had no second-rater pass.

### High value, but consent-gated: the fabricated given-name cluster

A 192-entry LLM-generated bibliography of one researcher's publication list
yielded **23 substituted given names across 14 papers**, each confirmed against
two of DBLP, Crossref and Semantic Scholar. The type is adversarially hard in a
way the current `swapped_authors` type is not: **21 of 23 preserve the true
name's first letter and its ethno-linguistic register** (Szabolcs→Sándor,
Kristóf→Kadosa, Deepak→Dhruvin, Gerald→Germar, Marwa→Muhammad). A surname-only or
initial-only matcher passes every one. The generator also *shuffled given names
between entries* rather than inventing them — a within-file signal no current
entry exercises.

**Consent gate.** This is a real researcher's publication list, and the entries
are fabricated variants naming real co-authors. Publishing them in a benchmark
attaches invented authorship to identifiable people. This needs the researcher's
explicit permission, and probably anonymization of the surnames, before it can
ship. Do not proceed on this one without that conversation.

### Distinct hard-VALID class: structurally unindexed front matter

The same bibliography contained 15 journal editorials, all verified real, of
which **14 are absent from Crossref entirely** — the journal deposits DOIs for
research articles only. Enumerating every DOI under its ISSN returns 300 records,
all `type: journal-article`; the editorial DOI slots 404.

This is a genuinely different VALID class from anything in v1.0: not "hard to
find" but *structurally never deposited*. It is the cleanest available test of
whether a tool distinguishes "no record exists" from "this is fabricated" — and
it exposes a labelling question the benchmark should answer deliberately (below).

## Design questions to settle first

**1. What does `not_found` mean for an unindexable document?** The harness maps
`not_found → HALLUCINATED` (rescued to VALID only by `coverage_incomplete`),
while `unconfirmed → VALID`. For a real-but-never-deposited editorial, a tool
that abstains is *correct*, yet today it is scored as a false positive. Adding
these entries without deciding this makes the split unpassable by construction.
Options: require `unconfirmed`; add a `coverage_incomplete`-style flag for
structural non-indexing; or score this subset on abstention rather than binary
label.

**2. Does the split measure FPR only, or both rates?** An all-VALID split is
cheap, honest, and directly targets the observed failure — but invites a
degenerate "abstain on everything" baseline. Pairing each convention axis with
matched HALLUCINATED entries costs much more construction effort. Recommend
starting FPR-only and saying so explicitly in the split's description.

**3. Where do HALLUCINATED entries come from?** The existing generators
(`hallmark/generation/`) assume ASCII `@inproceedings`. Fabricating a *plausible*
LNCS venue or a plausible Hungarian given name is a different generator, and the
given-name case needs the register-preserving property above or it is trivially
detectable.

## Contamination and split policy

- **Follow the v1.1 rule: add a new split, do not modify `dev_public`,
  `test_public`, or `test_crossdomain_matched`.** All published numbers stay
  valid.
- **The 44-entry held-out set was used to develop the `bibtex-updater` fixes.**
  If it ships verbatim, `bibtex-updater` is being scored on its own development
  set. Either exclude it and rebuild an independent sample by the same recipe, or
  ship it and label it explicitly as a co-designed split — the README already
  distinguishes co-designed from independent tools, and this must be recorded the
  same way.
- Deduplicate any new DOI against all existing splits before inclusion.

## Suggested sequence

1. Settle the three design questions, especially the `not_found` semantics —
   everything else depends on it.
2. Resolve the consent question for the given-name cluster; drop it if the answer
   is no, the split still stands on axes 1–4.
3. Rebuild an independent convention sample (target ~150–200) by the held-out
   recipe, from Crossref/DBLP/OpenAlex, with a second-rater pass on every entry.
4. Decide FPR-only vs paired, and only then write generators if paired.
5. Re-run the existing baselines on it. **Expect large movement**: this split is
   designed to be orthogonal to what v1.0 measures, so a tool near the top of
   Table 1 may do poorly here. That divergence is the contribution.

## Why this is worth doing

The paper's headline is that recall-optimized verifiers misallocate reviewer
effort. This split extends that argument along a new axis: tools are tuned on a
bibliographic monoculture, and the resulting failures fall on non-anglophone
authors and on document classes outside the ML-conference paper. That is a
measurable fairness claim, not just a coverage gap — and right now no benchmark
can substantiate it.
