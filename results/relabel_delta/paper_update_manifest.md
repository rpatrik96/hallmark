# Paper update manifest — btu 1.2.0 integration + coverage + ablations (REBUILDS lost gen_manifest)

Single source of truth for the btu-1.2.0 / coverage / ablation paper edits. Every cell below
is sourced to a saved file on disk. **Read-only on the paper repo** (`hallmark-paper`,
branch `arxiv-prep`); this manifest is the edit spec the paper-editing agent executes.

Scope of this stage (everything EXCEPT the cascade):
1. bibtex-updater 0.10.0 → **1.2.0** numbers in `tab:results` co-designed row + `tab:codesign`
   + every prose btu mention (abstract / intro / experiments / analysis / appendix / conclusion).
2. The **btu NARRATIVE REFRAME** (btu is no longer highest-DR; it is now a precision-oriented
   co-designed reference that abstains ~18–26% — the clearest illustration of precision-via-abstention).
3. The **Coverage column + conservative/aggressive dual scoring** in `tab:results`, plus the
   risk–coverage selective-prediction appendix figure/para (the abstention contribution).
4. Four **NEW ablation** appendix subsections + tables: A1 prompt-sensitivity, A3 threshold+aggregation,
   A4 input-format/field-LOO, A5 multi-rater κ.
5. The **endpoint-drift reproducibility limitation** paragraph.
6. **A1 prompt-sensitivity scope-honesting** of the post-cutoff FPR magnitude.
7. **stress = 0 VALID after canary removal ⇒ FPR undefined on stress** — verify no stale stress-FPR cell.

**DO NOT TOUCH `tab:cascade` numbers** (experiments.tex L111–125 / takeaway L128). The btu-1.2.0
cascade regen was DEFERRED. Add a one-line note that `tab:cascade` reflects btu 0.10.0 pending a
v1.2.0 regeneration; leave its values.

---

## 0. Provenance of every number used here

| Quantity | Source file (in `hallmark/`) |
|---|---|
| btu 1.2.0 dev/test aggregate (DR/FPR/F1/MCC/TW-F1/ECE) | `data/v1.0/baseline_results/bibtexupdater_{dev,test}_public.json` (already committed v1.2.0; verified below) and `results/relabel_delta/btu_hallmark_full.json` |
| btu 1.2.0 abstention / selective-prediction coverage | `results/ablations/coverage_reporting.json` (`per_tool_coverage_dual.bibtexupdater_v1_2_0`) + `coverage_reporting.md` §(a) |
| Coverage column + conservative/aggressive for all tools | `results/ablations/coverage_reporting.{json,md}` |
| Risk–coverage curves + FPR@90%-coverage | `results/ablations/coverage_reporting.json["risk_coverage"]` |
| A1 prompt sensitivity | `results/ablations/a1_prompt_full/summary.json` |
| A3 threshold + aggregation | `results/ablations/a3_threshold_full/{operating_point_summary,aggregation_table}.csv`, `a3_full_result.json` |
| A4 input-format / field-LOO | `results/ablations/a4_field_loo/summary.json` |
| A5 multi-rater κ | `results/ablations/a5_kappa/kappa_results.json` |
| Old→new Table-2 (relabel) | `results/reviewer_experiments/results_manifest.md` (already applied to the paper) |
| Reframe (Opus-leads) | `results/relabel_delta/reframe_brief.md` (already applied) |

**btu 1.2.0 authoritative aggregate (verified by reading the committed baseline JSONs):**

| Split | DR | FPR | F1 | MCC | TW-F1 | ECE | dFPR |
|---|---|---|---|---|---|---|---|
| dev_public  | **.865** (.8647) | **.092** (.0916) | **.890** (.8904) | **.771** (.7706) | **.908** (.9084) | **.383** (.3829) | — |
| test_public | **.877** (.8767) | **.115** (.1154) | **.901** (.9010) | **.750** (.7498) | **.916** (.9164) | **.399** (.3992) | **+0.024** (.1154−.0916) |

The committed headline aggregate maps abstentions to committed-VALID (so `coverage:1.0`,
`num_uncertain:0` in those JSONs). The *selective-prediction* coverage (where btu's "no record
found" → UNCERTAIN is honoured) is **0.746 dev**, reported in the Coverage column (§1 / §4).
The two views are consistent: the headline DR .865 is what you get when you do NOT let btu abstain
(every "not found"/"partial" counts as a committed VALID, i.e. a missed hallucination), whereas
the selective conservative DR .979 at coverage .746 is what btu achieves *on the 74.6% it commits
to*. State both; they are the headline vs the precision-via-abstention number.

---

## 1. FINAL `tab:results` grid (dev_public) — every cell sourced

`tab:results` is `sections/experiments.tex` L26–63. Two structural changes:
- **Add a `Cov.` (Coverage) column** between TW-F1/ECE and ΔFPR (or after ΔFPR — author's call;
  I place it right before ΔFPR so Calibr.|Robustness grouping stays clean). Full-coverage tools
  print `1.00`; only btu and (on test) DeepSeek-R1 are <1.
- **Add conservative/aggressive scoring** *only where abstention is real* (coverage <1). For
  full-coverage tools cons.=aggr.=the single printed triple, so no second row is needed. For btu
  (and DeepSeek-R1 on the test table) print the aggressive triple in a parenthetical / footnote,
  per coverage_reporting.md §(a) note 1. Recommended: keep `tab:results` showing the **headline
  (committed-VALID) triple** as the primary number and move the cons./aggr. selective triples to
  the Coverage appendix (§4) to avoid a double-decker main table — flag this as an author choice.

The **headline** dev grid (no changes to independent tools — already post-relabel; only the
co-designed btu row changes 0.10.0 → 1.2.0, and a Coverage column is added):

| Block | Tool | DR | FPR | F1 | MCC | TW-F1 | ECE | Cov. | ΔFPR |
|---|---|---|---|---|---|---|---|---|---|
| Citation-db | DOI-only | .268 | .185 | .373 | .099 | .329 | .143 | 1.00 | +0.094 |
| Zero-shot | Gemini 2.5 Pro | .476 | **.050** | .627 | .473 | .609 | .297 | .97 | +0.009 |
| | Claude Opus 4.7 | .752 | .072 | **.830** | **.683** | **.851** | .112 | n/a (drift) | −0.005 |
| | Gemini 2.5 Flash | .500 | .100 | .631 | .429 | .628 | .265 | .99 | +0.006 |
| | Claude Sonnet 4.6 | .781 | .127 | .827 | .652 | .834 | **.066** | n/a (drift) | −0.002 |
| | Llama 4 Maverick | .614 | .146 | .707 | .476 | .709 | .176 | 1.00 | +0.020 |
| | GPT-5.4 (zero-shot) | .767 | .228 | .783 | .538 | .807 | .202 | 1.00 | −0.004 |
| | Mistral Large | .716 | .250 | .742 | .465 | .765 | .229 | .99 | +0.032 |
| | GPT-5.1 (zero-shot) | .837 | .411 | .766 | .442 | .822 | .190 | 1.00 | +0.069 |
| | Qwen3-235B | .860 | .533 | .744 | .358 | .821 | .279 | 1.00 | +0.082 |
| | Qwen3-VL-235B | .860 | .551 | .740 | .342 | .818 | .286 | 1.00 | +0.077 |
| | DeepSeek-R1 | .896 | .623 | .739 | .324 | .825 | .238 | .98 | −0.303 |
| | DeepSeek-V3.2 | **.911** | .702 | .727 | .268 | .821 | .316 | 1.00 | +0.026 |
| Agentic | GPT-5.1 + CrossRef/OpenAlex/arXiv | .967 | .478 | .816 | .558 | .892 | .175 | 1.00 | +0.080 |
| | GPT-5.1 + btu (agentic; optional) | .980 | .470 | .824 | .584 | .900 | .125 | 1.00 | −0.114 |
| | Sonnet 4.6 + btu (agentic; optional) | .990 | .431 | .841 | .630 | .913 | .118 | 1.00 | −0.088 |
| Co-designed | **bibtex-updater (v1.2.0)** | **.865** | **.092** | **.890** | **.771** | **.908** | **.383** | **.75** | **+0.024** |
| | GPT-5.1 + btu (always-call; in prompt) | .843 | .144 | .856 | .698 | .872 | .078 | 1.00 | +0.112 |

**btu row CHANGE (the only `tab:results` data edit):**
- OLD (experiments.tex L57): `bibtex-updater & .969 & .193 & .909 & .794 & .944 & .297 & ... & $+0.144$`
- NEW: `bibtex-updater & .865 & .092 & .890 & .771 & .908 & .383 & .75 & $+0.024$`
  (Coverage `.75` = selective-prediction coverage 0.746; if the table keeps only full-coverage
  semantics, print `.75` in the new Cov. column and footnote that abstentions are committed-VALID
  in the DR/FPR/F1 triple — see §4 note.)
- **The bold marks DO NOT move**: Opus still bold on F1/MCC/TW-F1, Sonnet bold on ECE, Gemini Pro
  bold on FPR, DeepSeek-V3.2 bold on DR. btu is excluded from ranking and **must NOT be bolded** —
  and critically, **btu's FPR .092 is now lower than every independent tool except Gemini Pro
  (.050) and Opus (.072)**, so DO NOT bold it even though it would "win" on FPR; it is excluded
  from ranking. Add one clause to the caption noting btu's FPR/F1 are now precision-competitive
  *because it abstains* (Cov .75), not because it out-discriminates.

**Coverage-column values (full table, from coverage_reporting.json, dev_public):**
DOI-only 1.00; Gemini Pro .967→print .97; Opus n/a(drift); Gemini Flash .988→.99; Sonnet n/a(drift);
Llama4 1.00; GPT-5.4 1.00; Mistral .989→.99; GPT-5.1 1.00; Qwen3-235B .998→1.00; Qwen3-VL .999→1.00;
DeepSeek-R1 .984→.98; DeepSeek-V3.2 1.00; all three agentic 1.00; **btu .746→.75**; always-call 1.00.
(The two Anthropic dev rows are summary-only/drift → Coverage "n/a (drift)", no aggressive cell;
coverage_reporting.md §(a) note + §(b)†.)

**Conservative/aggressive triples (the dual-scoring cells), btu dev_public** (coverage_reporting.json
`bibtexupdater_v1_2_0.dev_public`): coverage **.746**, conservative **DR .979 / FPR .045 / F1 .962**,
aggressive **DR .987 / FPR .140 / F1 .937**. (n_committed 835, n_uncertain 284, n_missing_from_raw 7.)
These are the precision-via-abstention numbers; place them in the Coverage appendix (§4), not the
main table, unless the author wants a double-decker btu row.

**test_public btu (`tab:test_public_full`, appendix L171–190):**
- OLD btu row: `.946/.179/.908 (dev) ... .886/.338/.858 (test) ... dFPR +0.159→+0.144` — these are
  the *relabel* (0.10.0) numbers already in the paper.
- NEW btu 1.2.0: **dev .865/.092/.890, test .877/.115/.901, dFPR +0.024.** This is a *clean*
  cross-split story now (FPR +2.4pp, F1 +1.1pp) — see §2 / FLAG-A. **The selective-prediction
  conservative/aggressive for btu test_public is INCOMPLETE** (raw scored only 500/831;
  coverage_reporting.json `could_not_compute` + `INCOMPLETE`). So for the Coverage *appendix*,
  btu test is held PENDING (GEN workflow `wkp97jqbb`); the **headline btu test_public triple
  .877/.115/.901 IS final** (it comes from the committed baseline JSON, full-N committed-VALID
  convention, not the incomplete selective re-score).

---

## 2. BTU NARRATIVE REFRAME — exact sentences (author voice)

### The reframe in one paragraph (the framing, for orientation — do not paste verbatim)

Under v1.2.0, btu's defaults flipped from recall-maximizing to precision-via-abstention: when no
backing record is found it now *abstains* (routes to UNCERTAIN) rather than guessing, so on the
~75% of dev entries it commits to it reaches DR .979 / FPR .045 / F1 .962, but counted over the
full split — every abstention scored as a missed flag — its headline DR drops to **.865** and FPR
to **.092**. It is therefore **no longer the highest-DR system** (DeepSeek-V3.2 .911, the agentic
harnesses .967–.990, all exceed it). The honest framing is no longer "recall-optimized upper
bound (DR 0.97)" but **"a precision-oriented co-designed reference: lower recall, low FPR (.092),
and ~18–26% abstention — the clearest single illustration of the precision-via-abstention lever
the Coverage column measures."** This *strengthens* two existing threads: (a) it ties the
co-designed tool directly to the selective-prediction contribution, and (b) the cross-split story
gets cleaner — btu's dFPR shrinks from the old +14.4pp (0.10.0) to **+2.4pp** (1.2.0), so the
"rule-based lead is a dev artifact" claim is now carried by the *F1 reversal alone*, not an FPR
blow-up (FLAG-A: re-examine analysis.tex L34/L40, which still describe the +14.4pp FPR jump — that
was the 0.10.0 behaviour; under 1.2.0 the FPR barely moves cross-split).

### Abstract (`sections/abstract.tex`)

- **L6 — REPLACE the "highest DR 0.97" claim.** This is the headline narrative inversion.
  - OLD: *"\texttt{bibtex-updater} reaches the highest detection rate (DR\,0.97) among all systems,
    but we read it only as an upper-bound reference."*
  - NEW: *"Our co-designed \texttt{bibtex-updater} is no longer the highest-recall system: at
    v1.2.0 it abstains when no backing record is found, trading recall for precision---DR\,0.87,
    FPR\,0.09, and ${\sim}20\%$ abstention---so we read it as a precision-oriented reference, not
    a recall ceiling."*
- **L7 — leave** (Opus/Sonnet lead sentence is independent of btu and already correct).
- **L9 — UPDATE the agentic-vs-btu comparison** (the diminishing-returns failure mode).
  - OLD fragment: *"agentic FPR remains 2.5$\times$ higher (0.48 vs.\ 0.19 for the co-designed
    \texttt{bibtex-updater} reference), so F1 still trails by 9.3\,pp"* and *"DR 0.97 vs.\ 0.97"*.
  - NEW: the agentic harness still hits DR ${\approx}0.97$–0.99 but now *exceeds* btu's recall
    (btu DR 0.87), while agentic FPR 0.47–0.48 is **${\sim}5\times$ btu's 0.092**; rephrase to
    *"A matched 5-call budget pushes the agentic harness past \texttt{bibtex-updater}'s recall
    (DR 0.97--0.99 vs.\ 0.87), but at ${\sim}5\times$ its false-positive rate (0.47--0.48 vs.\
    0.09), because the harness flags an entry whenever any one database returns no match whereas
    \texttt{bibtex-updater} abstains."* (Recompute the "F1 trails by 9.3pp": btu F1 .890 vs agentic
    GPT-5.1 .816 — btu now *leads* F1 by ~7.4pp; the old "trails by 9.3pp" is INVERTED, FLAG-B.
    Reframe as "btu's abstention buys it the higher F1 despite lower recall.")
- **L10 — UPDATE the PPV sentence.** OLD groups btu with the "~1 in 10 flagged" cluster (9.3% PPV).
  NEW btu PPV@2% = .865·.02/(.865·.02+.092·.98) = **16.2% ≈ 1 in 6** — btu *joins* the low-FPR
  Opus/Gemini-Pro tier. Add btu to the "~1 in 6" group: *"low-FPR verifiers reach ${\sim}1$ in 6--9
  (Opus~4.7 ${\approx}1$ in 6, Sonnet~4.6 ${\approx}1$ in 9, and the co-designed
  \texttt{bibtex-updater} ${\approx}1$ in 6 by abstaining)"*. The "roughly one true hallucination
  per ten flagged" generic stays for the mid-FPR LLM cluster.

### Intro (`sections/introduction.tex`)

- Intro carries **no btu-specific number** in L18–28 (verified: the contributions list + the DR
  spread "48–91%" + the temporal sentence). The only btu touch is implicit in "co-designed verifier".
  **No edit required** unless the author wants to preview the precision-via-abstention framing; if so,
  one optional clause after the "48--91% DR" sentence: *"a co-designed rule-based verifier anchors
  the precision end by abstaining rather than guessing (\cref{sec:experiments})."* (OPTIONAL.)

### Experiments (`sections/experiments.tex`)

- **L74 — REPLACE "highest raw detection rate (DR 0.969)".**
  - OLD: *"Co-designed tools achieve the highest raw detection rate (bibtex-updater: DR\,=\,0.969),
    at the cost of possible construct-overfitting bias..."*
  - NEW: *"The co-designed \texttt{bibtex-updater} (v1.2.0) abstains when no backing record is
    found, so it trades recall for precision (DR\,0.865, FPR\,0.092) and commits to only
    ${\sim}75\%$ of entries; we read it as a precision-oriented reference rather than a recall
    ceiling, and exclude it from ranking for the construct-overfitting reason."*
- **L78 (takeaway) — UPDATE.** The "Opus leads" clause stays. Recompute the btu-relative numbers:
  the agentic FPR multiplier sentence references btu's FPR — it now compares against btu 0.092
  (was 0.19/0.193). Adjust: *"agentic FPR 0.47--0.48 is ${\sim}5\times$ the co-designed
  \texttt{bibtex-updater}'s 0.092"*. (The DR-0.41→0.47/0.13→0.43 zero-shot→agentic harness numbers
  are LLM-internal and unchanged.)

### Analysis (`sections/analysis.tex`) — FLAG-A is here

- **L34 — REWRITE.** OLD: *"bibtex-updater degrades sharply. Its FPR rises $+14.4$\,pp ... dropping
  F1 by $4.6$\,pp (0.909~$\to$~0.863) and DR by $5.6$\,pp (0.969~$\to$~0.913)."* — this is the
  **0.10.0** cross-split behaviour. Under **1.2.0**, dev→test: DR .865→.877 (**+1.2pp**),
  FPR .092→.115 (**+2.4pp**), F1 .890→.901 (**+1.1pp**). btu no longer "degrades sharply" — it is
  now *cross-split stable*. REWRITE to: *"\texttt{bibtex-updater} is cross-split stable: FPR rises
  only $+2.4$\,pp (0.092~$\to$~0.115) and F1 holds (0.890~$\to$~0.901), because its abstention
  policy refuses to flag entries it cannot back with a record."*
- **L40 (takeaway) — REWRITE the FPR-jump + F1-reversal.** The +14.4pp FPR jump is GONE under 1.2.0.
  The **F1 ordering vs Sonnet** must be recomputed: dev btu F1 .890 vs Sonnet .827 = **+6.3pp**;
  test btu F1 .901 vs Sonnet .866 = **+3.5pp**. **The reversal no longer holds under 1.2.0** — btu
  now F1-*leads* Sonnet on BOTH splits (FLAG-A is load-bearing). REWRITE to: *"On
  \texttt{test\_public} \texttt{bibtex-updater}'s false-positive rate rises only $+2.4$\,pp and its
  F1 lead over Sonnet~4.6 narrows from $6.3$\,pp on \texttt{dev\_public} to $3.5$\,pp on
  \texttt{test\_public} but does not reverse---the v1.2.0 abstention policy removes the cross-split
  FPR blow-up the recall-only configuration showed. We report this as a point-estimate ordering
  (\texttt{bibtex-updater} is summary-only, no paired test)."* **CRITICAL: the old text claims a
  reversal that was a 0.10.0 artifact; under 1.2.0 it does not reverse. This is a real narrative
  change — surface to the author (FLAG-A).** Note this WEAKENS the "rule-based lead is a dev
  artifact" claim; the honest 1.2.0 story is "btu's lead narrows but survives cross-split because
  it abstains." (See also L1099 appendix takeaway + L627 deployment, same reversal claim.)
- **L60 — leave** (always-call btu+GPT-5.1 ECE 0.190→0.078; that variant is unchanged at 0.10.0
  always-call config; note it is the *always-call* path, not the standalone btu — no edit).
- **L14 / L18 — leave** ("high-DR tool such as bibtex-updater" regime prose, cost prose) — but the
  "high-DR tool such as bibtex-updater" example at L14 is now WEAK (btu is mid-DR .865). Soften:
  replace the btu example with a generic "a high-recall tool" or swap in DeepSeek-V3.2/an agentic
  harness as the high-DR exemplar. (FLAG-C, minor.)

### Appendix (`appendix.tex`)

- **L960 — REPLACE "achieves the highest raw detection rate of any evaluated tool".**
  NEW: *"...\texttt{bibtex-updater} (v1.2.0) anchors the precision end of the landscape by abstaining
  when no backing record is found (DR\,0.865, FPR\,0.092, ${\sim}75\%$ coverage)..."*.
- **L972 — `tab:codesign` row UPDATE.** OLD: `bibtex-updater & 0.969 & 0.193 & 0.909 & 0.794 & 0.944
  & 0.297 & 1.00`. NEW: `bibtex-updater & 0.865 & 0.092 & 0.890 & 0.771 & 0.908 & 0.383 & 0.75`.
  (Cov 0.75 = selective coverage. The committed JSON also lets you print 1.00 with the committed-VALID
  convention; pick 0.75 to make the abstention visible and consistent with §4 — flag the choice.)
- **L979 — per-type prose.** btu per-type is summary-only / carried-over (F5; the v1.2.0 per-type
  was not regenerated cell-by-cell). The L979 sentence quotes per-type cells (author_mismatch 0.940,
  near_miss_title 0.927, plausible_fabrication 0.897). These are 0.10.0 per-type; under 1.2.0 the
  abstention changes them but they are NOT recomputed. **Keep the per-type cells + the existing
  caption caveat (L988), add `% TODO(btu-1.2.0 per-type not regenerated)`.** (FLAG-D.)
- **L1028 — relabel/co-design divergence paragraph UPDATE.** OLD: *"Correcting these labels lifts
  \texttt{bibtex-updater}'s reported \texttt{dev\_public} DR from 0.946 to 0.969 and F1 from 0.908
  to 0.909..."* This conflates the *relabel* lift (0.10.0) with the *version* change. REWRITE to
  keep the divergence argument (btu returned VALID on the 52 recovered real papers, diverging from
  the buggy labels — still true and still the strongest anti-circularity evidence) but state the
  current numbers: *"...on those entries \texttt{bibtex-updater}'s database cross-check returned
  VALID, diverging from the then-current labels rather than tracking them. Under v1.2.0 the tool
  abstains on unbacked entries, reporting \texttt{dev\_public} DR\,0.865 / FPR\,0.092 / F1\,0.890;
  the divergence on the recovered papers shows its verdicts rest on external database evidence, not
  the label distribution."*
- **L573 (`tab:ppv` appendix) — btu PPV row UPDATE.** OLD: `bibtex-updater & 0.969 & 0.193 & 9.3\%
  & 20.9\%`. NEW (recomputed from DR .865 / FPR .092): PPV@2% = **16.2%**, PPV@5% =
  .865·.05/(.865·.05+.092·.95) = .043250/(.043250+.0874) = **33.1%**. NEW row:
  `bibtex-updater & 0.865 & 0.092 & 16.2\% & 33.1\%` — and **RE-SORT**: btu now sits between Opus
  (17.6%) and Gemini Pro (16.3%) — i.e. row order becomes Opus, Gemini Pro, **bibtex-updater**,
  Sonnet 4.6, Gemini Flash, ... So btu moves UP from rank 4 to rank 3 (above Sonnet). (FLAG-E.)
- **L625 — deployment PPV prose UPDATE.** OLD: *"${\sim}9\%$ for \texttt{bibtex-updater}"*. NEW:
  *"${\sim}16\%$ for \texttt{bibtex-updater} (now precision-competitive with Opus~4.7 because it
  abstains rather than guesses)"*. Re-order the list so btu sits with the low-FPR tier.
- **L627 — deployment recipe UPDATE.** OLD: *"Pre-2024, recall-prioritized triage: bibtex-updater
  (DR\,=\,0.97 ...)"*. btu is no longer the recall pick (DR .865). REWRITE: either move btu to the
  *precision*-prioritized recipe (DR .865, FPR .092, PPV ~16% at π=2%, ~2–3 orders cheaper) and use
  an agentic harness / DeepSeek-V3.2 as the recall-prioritized exemplar, OR state btu as "the
  cheapest precision-oriented option." Also fix the embedded reversal claim per FLAG-A (lead narrows,
  not reverses, under 1.2.0). (FLAG-A + FLAG-F.)
- **L321 — cross-split prose UPDATE.** OLD: *"the same shift appears for \texttt{bibtex-updater}
  (dev FPR 0.193 $\to$ test FPR ${\sim}0.34$)"*. NEW: *"\texttt{bibtex-updater}'s FPR is
  cross-split stable (dev 0.092 $\to$ test 0.115) under its v1.2.0 abstention policy"* — note this
  now *contrasts* with GPT-5.1's drift rather than mirroring it (FLAG-A consequence).
- **L1099 (`app:codesign` takeaway) — fix the reversal claim** per FLAG-A: *"the rule-based F1 lead
  over Sonnet~4.6 narrows from $6.3$\,pp on \texttt{dev\_public} to $3.5$\,pp on \texttt{test\_public}
  but does not reverse under v1.2.0"*. (Was: collapses + reverses, 0.10.0.)

### Conclusion (`sections/conclusion.tex`)

- **L7 — UPDATE.** OLD: *"A 5-call budget closes GPT-5.1's recall gap to \texttt{bibtex-updater},
  but agentic FPR remains $2.5{\times}$ higher (\cref{tab:results})"*. NEW: agentic now *exceeds*
  btu recall (DR .97–.99 vs .865), FPR ${\sim}5\times$ (0.47 vs 0.092). Rephrase: *"A 5-call budget
  pushes the agentic harness past \texttt{bibtex-updater}'s recall but at ${\sim}5\times$ its FPR."*
- **L9 — UPDATE.** OLD: *"\texttt{bibtex-updater} is cheapest and most temporally stable ... the
  dev-side rule-based F1 lead collapses on \texttt{test\_public}"*. Under 1.2.0 the lead does NOT
  collapse (FLAG-A); btu is still cheapest, now precision-oriented. REWRITE: *"\texttt{bibtex-updater}
  is cheapest, precision-oriented (it abstains rather than over-flags), and cross-split stable; its
  dev-side F1 lead over Sonnet~4.6 narrows but survives on \texttt{test\_public}."*

---

## 3. Four ablation subsections + the coverage figure — placement + content

All four are **appendix subsections**. Recommended order, after `app:codesign` / before/after the
existing `app:statistics` / `app:shortcuts` block (author's call; I suggest a new
`\section{Robustness ablations}` `\label{app:ablations}` grouping A1/A3/A4/A5, then the coverage
material as its own `\subsection` or a sibling `app:coverage`). Frame all four as **robustness
evidence for the precision-ceiling finding** (model-/prompt-/format-/rater-invariance), per
`comparative_expectations.md` §4 — NOT as tuning for best numbers. Do NOT duplicate the 18 already-
settled analyses listed in `comparative_expectations.md` §2 (cutoff-aware, late-cutoff, drift,
thinking-budget, format-tell, co-design bound, cost, recall-probe E1, variance E3, water-filling,
KS, bootstrap, ECE, cascade, cross-split, holdout) — cite those by their existing labels.

### A1 — Prompt-sensitivity sweep (`app:ablation_prompt`)
Source: `a1_prompt_full/summary.json`. n=150 stratified (81 hall / 69 valid), 4 models
(Sonnet 4.6, DeepSeek-V3.2, Gemini 2.5 Flash via OpenRouter; GPT-5.1 via the OpenAI-direct
endpoint, matching the paper's other GPT-5.1 numbers), 4 prompt variants
(default/notaxo/uncertain/terse), T=0, seed 42 (OpenRouter models from the 2026-05-31 cached
snapshot; GPT-5.1 run 2026-06-02).
**Headline finding (two-part):**
1. **Model ranking is prompt-invariant.** Mean pairwise Spearman ρ = 0.90 for F1, DR, and FPR
   across the four prompt variants (Sonnet > GPT-5.1 > DeepSeek > Gemini Flash on F1 at default).
   The only departure from ρ = 1.0 is a 0.001 F1 near-tie between GPT-5.1 (0.794) and Sonnet
   (0.793) under the "uncertain" variant — a tie, not a genuine reordering. So **rankings are robust
   to phrasing, taxonomy-in-prompt, and abstention wording** — this defends the single-prompt design
   (`sec:baselines`). [FLAG-A1: the 3-model panel gave ρ = 1.0 exactly; adding GPT-5.1 softens it to
   ρ = 0.90 through that near-tie (FPR stability actually *rose* from ρ 0.75 → 0.90). Keep
   "prompt-invariant" but cite ρ = 0.90, not 1.0.]
2. **Absolute FPR is wording-sensitive.** GPT-5.1 default FPR 0.580 → 0.212 under the terse variant
   (a **−36.8pp prompt-induced drop**, the largest in the panel), with its UNCERTAIN rate moving
   0% → 19.3% under the "uncertain" variant; DeepSeek-V3.2 0.899 → 0.600 ("uncertain", −29.9pp);
   Sonnet 4.6 0.121 → 0.015 ("uncertain", −10.6pp, UNCERTAIN 2.0% → 8.7%); Gemini Flash 0.130 →
   0.087 (−4.3pp). Pooled verdict-flip rate vs default = 17.4% (all) / 13.6% (strict); GPT-5.1 is
   the most prompt-sensitive model (mean flip 0.258).
**Table:** per-model × per-variant DR/FPR/F1/ECE/UNCERTAIN-rate (16 rows from summary.json
`per_model_per_variant`), plus the Spearman-ρ ranking-stability row and the FPR-decomposition
(`default_fpr`, `min_fpr`, `prompt_induced_fpr_drop`).
**Insert this scope-honesting cross-reference (item 6 of the brief):** in `sec:temporal_robustness`
(and abstract L11 / intro L27 where the "FPR up to ~0.89" post-cutoff magnitude is stated), add one
sentence: *"the *absolute* post-cutoff FPR is partly prompt-induced---A1 (\cref{app:ablation_prompt})
shows up to a 10--37\,pp swing from prompt wording alone---so we treat the FPR *magnitude* as
prompt-conditional and rest the temporal claim on the *cross-regime ranking*, which A1 shows is
prompt-invariant."* (Scopes the FPR magnitude without weakening the ranking-based temporal claim.)

### A3 — Threshold + aggregation ablation (`app:ablation_threshold`)
Source: `a3_threshold_full/{operating_point_summary,aggregation_table}.csv`, `a3_full_result.json`.
Offline re-score of stored confidences on v1.1.1 labels (NO API calls), dev_public.
**Headline finding:** confidences are **quantized**, so threshold tuning buys almost nothing.
default-vs-best-F1 gap (pp): Opus 0.22, Sonnet 0.18, DeepSeek-R1 0.07, DeepSeek-V3 0.0, Mistral 0.0,
Qwen −0.0, GPT-5.4 0.34; **Gemini Flash is the lone exception at 8.22pp** (its best-F1 at thr 0.02
trades FPR up to 0.343 for DR 0.716). AUROC: Sonnet 0.928, Opus 0.906, GPT-5.4 0.834, Mistral 0.744,
Gemini Flash 0.739, DeepSeek-R1 0.741, Qwen 0.695, DeepSeek-V3 0.609. **Conclusion: the fixed-0.5
operating point in `tab:results` is near-optimal; rankings are threshold-robust** (differentiates
from CiteAudit's unjustified hard-coded 0.92, `peer_norms`).
**Aggregation sub-table** (`aggregation_table.csv`, btu/structured-subtest instantiations): the
benchmark's "any-field-miss (k≥1 of 4)" rule (DR .759/FPR .139 field-level; .950/.000 structured
subtests) vs stricter k≥2/k≥3/unanimous and an 8-voter noisy ensemble (majority ≥5/8: DR .819/FPR
.177/F1 .832). Shows the headline detector is the cross-DB-agreement / any-miss rule, and stricter
quorum rules trade DR for FPR monotonically — the chosen rule is the F1-maximizing one.
**Table:** `operating_point_summary.csv` (8 tools × AUROC/default/best-F1) + the aggregation-rule
sweep (14 rows). The full per-threshold curve lives in `a3_full_result.json` (figure-optional).

### A4 — Input-format / field leave-one-out (`app:ablation_format`)
Source: `a4_field_loo/summary.json`. n=150 (81 hall / 69 valid), 2 models (DeepSeek-V3.2 anchor,
Gemini 2.5 Flash), conditions full / structured / LOO-{title,author,venue,year,doi}, T=0, seed 42,
fresh snapshot 2026-05-31. **Dynamic complement to the static W6 format-tell audit.**
**Headline finding: title is the load-bearing field; dropping it spikes FPR.** For Gemini Flash,
LOO-title raises FPR +35.5pp (structured 0.145 → 0.500) and LOO-author +18.8pp; LOO-venue/year/doi
move FPR little. DeepSeek-V3.2 (already near-saturated FPR 0.81 structured) shows the same direction
at smaller magnitude (LOO-title +12.9pp FPR). **Format (full BibTeX vs structured fields) matters
less than field content:** full→structured shifts DR/FPR by ≤7.4pp/14.5pp. **Conclusion: detection
leans on title/author semantics, not a surface format tell** — reinforces W3/W6 construct validity
(detection isn't a single co-designed field or a format artifact).
**Table:** per-model per-condition DR/FPR/F1 (14 rows) + the LOO-Δ-vs-structured block
(`loo_deltas_vs_structured`: dDR, dFPR, n_drop_affected per field).

### A5 — Multi-rater agreement / reliability proxy (`app:ablation_kappa`)
Source: `a5_kappa/kappa_results.json`. **3 independent LLM raters** (Sonnet 4.6, DeepSeek-V3.2,
Gemini 2.5 Pro) on 132 blinded entries (80 real-world-incident HALL + 52 relabel-recovered VALID).
**CRITICAL framing (non-negotiable): this is an *automated multi-rater reliability proxy, NOT human
inter-annotator agreement* — say so explicitly in the first sentence and the caption.** It does not
fulfil the "human κ" gap; it is an LLM-rater reliability check.
**Headline finding:** Fleiss' κ (binary, UNCERTAIN→HALL) = **0.238 ("fair", Landis–Koch)**; pairwise
Cohen's κ: Sonnet–DeepSeek 0.454 (moderate), Sonnet–Gemini Pro 0.266 (fair), DeepSeek–Gemini Pro
0.029 (slight). vs-gold accuracy: DeepSeek 0.871 (κ 0.716 substantial), Sonnet 0.727 (κ 0.360),
Gemini Pro 0.576 (κ 0.026). **Majority-vote vs gold: accuracy 0.720 (κ 0.340).**
**The diagnostic split is the punchline:** majority-vote accuracy is **0.975 on the 80 real-world
HALL** but only **0.327 on the 52 relabel-recovered VALID** — i.e. the LLM raters confidently agree
on genuine hallucinations but systematically over-flag the recovered real papers, **the exact
over-flagging failure mode HALLMARK measures**, and independent corroboration that the relabel
audit corrected a real bias (LLM raters would have repeated it). Use this to *retroactively justify
the relabel* and to motivate the precision/abstention contribution, NOT as a human-κ claim.
**Table:** Fleiss κ + pairwise Cohen κ (3 pairs) + per-rater-vs-gold (κ/accuracy) + the
by-pool majority-vote accuracy (0.975 vs 0.327).

### Coverage / risk–coverage figure + paragraph (`app:coverage`, the selective-prediction contribution)
Source: `coverage_reporting.{md,json}`. This is the **abstention/selective-prediction contribution**
and the home for the btu precision-via-abstention number.
- **Coverage table** = `coverage_reporting.md` §(a) dev + test grids (per-tool Coverage + cons./aggr.
  DR/FPR/F1). Transcribe directly (cons. cells already match `data/v1.0/baseline_results/` to 5e-4).
  **btu dev_public is final** (cov .746, cons .979/.045/.962, aggr .987/.140/.937); **btu test_public
  is PENDING** (raw 500/831; mark INCOMPLETE, hold until GEN `wkp97jqbb` completes — FLAG-G).
- **Risk–coverage figure** = `coverage_reporting.json["risk_coverage"][*]["curve"]` (FPR vs coverage,
  coverage grid 1.00→0.05). Summary number per tool = **FPR@90%-coverage** (md §(b) table):
  Sonnet† 0.120, Opus† 0.095, GPT-5.4 0.222, Mistral 0.241, DeepSeek-R1 0.581, Gemini Flash 0.097,
  Qwen 0.507, DeepSeek-V3 0.699, agentic-btu-Sonnet 0.342, etc. **† = the two Anthropic dev curves
  are appendix-only with a drift caveat** (operating point drift-affected; curve shape + AUROC are
  deterministic re-scores of stored confidences, drift-immune). High-AUROC verifiers (Sonnet/Opus
  AUROC 0.91–0.93) sit at the favourable corner (FPR ~0.10–0.12 @90% coverage).
- **Three honest notes** (`coverage_reporting.md` §(c)) — cite in the caption/text:
  (1) Coverage must sit next to every abstain-excluded metric (prevents gaming: abstain-on-all ⇒
  perfect cons.-precision); the aggressive column closes the loophole. (2) LLM abstention is
  prompt-dependent — cite A1 (Sonnet UNCERTAIN 2.0%→8.7% by wording). (3) **btu vs LLM abstention
  differ in mechanism** (btu = "databases don't know" / data-coverage gap; LLM = "model won't
  commit" / epistemic gap) — same Coverage column, asymmetric meaning; state in one sentence.
- **Lineage to cite:** selective prediction (El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017);
  no peer citation-benchmark reports coverage → ahead-of-field (verify both refs exist before citing,
  per citation-integrity rule; mark `[CITE-CHECK]` if not in `references.bib`).
- **Tie-in:** abstain = route-to-human; connect to the reviewer-bound, FPR-decisive framing in
  `sec:ppv` / `analysis.tex` L14.

---

## 4. Endpoint-drift reproducibility limitation paragraph

Add to `sections/limitations.tex` (after the co-design / pre-screening paragraph at L26) a new
paragraph, `\label{}` optional. Content (author voice, scope-honest):

> *\textbf{Endpoint drift and reproducibility.} Several robustness ablations (A1's OpenRouter
> models, A4, A5) and the
> Anthropic risk--coverage curves were collected against the OpenRouter API on a fresh dated
> snapshot (2026-05-31) that does not reproduce our originally published delta-eval aggregates:
> the OpenRouter Anthropic endpoint drifted since the main run, and the original per-entry Sonnet/
> Opus \texttt{dev\_public} predictions were summary-only (never persisted), so their coverage and
> aggressive-stance cells are not recoverable. We therefore report these two \texttt{dev\_public}
> rows as published point estimates with Coverage marked "n/a (drift)", and present the Anthropic
> risk--coverage curves as appendix-only with a drift caveat---the curve shape and AUROC are
> deterministic re-scores of stored confidences and so are drift-immune, but the operating point is
> not. Within-run deltas (prompt-variant, field-LOO, rater-agreement) are robust to this absolute-
> level drift because they difference conditions measured against the same endpoint on the same day.*

Cross-references: `a1_prompt_full/summary.json.snapshot.policy`, `a4_field_loo/summary.json.endpoint_drift_note`,
`a5_kappa/kappa_results.json.snapshot_note`, `coverage_reporting.json` Anthropic-dev caveats,
`a3_full_result.json.provenance` (mixed persisted-prediction dates).

---

## 5. stress = 0-VALID after canary removal ⇒ FPR undefined — fix locations

**Status: ALREADY CORRECT in the prose** (verified). The two load-bearing statements are right:
- `appendix.tex:200`: *"The \texttt{stress\_test} split contains 122 entries (121 hallucinated,
  1 canary); because no valid entries exist, FPR is undefined and we report only Detection Rate."*
- `sections/experiments.tex:106` (`tab:cascade` caption): *"``---'' marks splits with no
  \textsc{valid} ground-truth entries (\texttt{stress\_test} contains only hallucinated instances)."*
- `tab:cascade` stress rows (experiments.tex L115–116) print `---` for FPR (correct).

**Required check (not an edit unless found):** grep for any *numeric* stress FPR cell or any claim
that stress has 1 valid entry feeding an FPR. The 1 canary is the only "valid" and it is filtered
from metrics (`evaluate()` canary auto-filter, #7503), so post-filter stress has **0 scored VALID**.
- One subtlety to surface (FLAG-H): `tab:stats` / appendix L138 print stress as "122 (1 / 121)" —
  the "1 valid" is the canary. That count is fine in the dataset table, but ensure NO metrics table
  derives an FPR from it. The cascade table and appendix L200 already say FPR undefined → consistent.
  **If any stress-FPR number is found in a metrics cell, replace with `---` and cite L200.** From the
  grep performed, none exists; this item is a verification pass, not an edit.

---

## 6. File:line edit map (grouped by owned section)

Legend: `[DATA]` = table cell number; `[PROSE]` = sentence; `[NEW]` = new block; `[VERIFY]` = check only.

### A. Abstract + Intro
- `sections/abstract.tex:6` [PROSE] — REPLACE "highest DR 0.97 upper-bound" → precision-via-abstention
  (DR 0.87 / FPR 0.09 / ~20% abstention). §2.
- `sections/abstract.tex:9` [PROSE] — agentic-vs-btu: DR 0.97–0.99 *exceeds* btu 0.87; FPR ~5× btu's
  0.09; INVERT the "F1 trails 9.3pp" (btu now F1-leads, FLAG-B). §2.
- `sections/abstract.tex:10` [PROSE] — add btu to the "~1 in 6" PPV tier (btu PPV@2% = 16.2%). §2.
- `sections/abstract.tex:11` [PROSE] — add A1 scope-honesting clause on post-cutoff FPR *magnitude*
  (prompt-induced ±10–30pp; rest temporal claim on prompt-invariant ranking). §3-A1 / item 6.
- `sections/introduction.tex:27` [PROSE] — OPTIONAL: same A1 FPR-magnitude scope clause on the
  temporal sentence ("FPR up to 6×"). §3-A1.
- `sections/introduction.tex` (L18–26) [VERIFY] — no btu number to change; optional precision-anchor
  clause after "48–91% DR". §2.

### B. Experiments
- `sections/experiments.tex:57` [DATA] — `tab:results` btu row 0.10.0→1.2.0:
  `.969/.193/.909/.794/.944/.297` → `.865/.092/.890/.771/.908/.383`; add Cov. `.75`, ΔFPR `+0.024`.
  btu NOT bolded despite low FPR. §1.
- `sections/experiments.tex:20` [PROSE] — add `Cov.` column to header (L28–30) + one caption clause:
  btu's low FPR/F1 reflect abstention (Cov .75), not discrimination; full-coverage tools = 1.00;
  Anthropic dev = "n/a (drift)". §1 / §4.
- `sections/experiments.tex:74` [PROSE] — REPLACE "highest raw detection rate (DR 0.969)" →
  precision-oriented, abstains, DR 0.865/FPR 0.092/~75% coverage. §2.
- `sections/experiments.tex:78` [PROSE] — takeaway: agentic FPR ~5× btu's 0.092 (was 2.5× / 0.19). §2.
- `sections/experiments.tex` cascade block (L101–129) [VERIFY+NEW] — DO NOT change any number.
  Add one footnote/sentence to the `tab:cascade` caption (L106): *"\cref{tab:cascade} reflects
  \texttt{bibtex-updater} v0.10.0 as Stage~1; a v1.2.0 regeneration is deferred to the camera-ready."*

### C. Analysis  (FLAG-A lives here — narrative change)
- `sections/analysis.tex:34` [PROSE] — REWRITE "degrades sharply / FPR +14.4pp" → cross-split stable
  (FPR +2.4pp 0.092→0.115; F1 0.890→0.901; DR 0.865→0.877). §2 / FLAG-A.
- `sections/analysis.tex:40` [PROSE] — REWRITE takeaway: F1 lead vs Sonnet narrows 6.3pp(dev)→3.5pp(test)
  but does NOT reverse under 1.2.0 (was: reverses, 0.10.0). §2 / FLAG-A (surface to author).
- `sections/analysis.tex:14` [PROSE] — soften "high-DR tool such as bibtex-updater" exemplar (btu now
  mid-DR); swap to a generic high-recall tool / agentic harness. §2 / FLAG-C.
- `sections/analysis.tex:18` [VERIFY] — cost prose "btu ~$0.0001–0.001/entry" unchanged (still cheapest).
- `sections/analysis.tex:60` [VERIFY] — always-call btu+GPT-5.1 ECE 0.190→0.078 unchanged (0.10.0
  always-call config; this variant's tab:results row at L59 also UNCHANGED).
- `sections/analysis.tex` `sec:ppv` (L10–12) [VERIFY] — post-relabel PPV prose already applied; ensure
  btu's PPV mention (if any) uses 16% not 9% (cross-check L625 appendix). §2.

### D. Benchmark + Limitations + Appendix
- `sections/benchmark.tex` [VERIFY] — no btu-specific edit here (counts/stats already post-relabel);
  no stress-FPR cell. §5.
- `sections/limitations.tex:26` [VERIFY] — co-design/pre-screening paragraph already correct; no number.
- `sections/limitations.tex` (after L26) [NEW] — endpoint-drift reproducibility paragraph. §4.
- `sections/conclusion.tex:7` [PROSE] — agentic exceeds btu recall, FPR ~5× (was 2.5×). §2.
- `sections/conclusion.tex:9` [PROSE] — btu precision-oriented + cross-split stable; lead narrows not
  collapses. §2 / FLAG-A.
- `appendix.tex:200` [VERIFY] — stress FPR-undefined sentence correct; leave. §5.
- `appendix.tex:321` [PROSE] — btu cross-split FPR now stable 0.092→0.115 (was 0.193→~0.34). §2/FLAG-A.
- `appendix.tex:573` [DATA] — `tab:ppv` btu row → `0.865 & 0.092 & 16.2\% & 33.1\%`; RE-SORT btu to
  rank 3 (above Sonnet). §2 / FLAG-E.
- `appendix.tex:625` [PROSE] — deployment PPV: btu ~16% (precision-competitive). §2.
- `appendix.tex:627` [PROSE] — deployment recipe: move btu to precision-prioritized; fix reversal→narrows.
  §2 / FLAG-A / FLAG-F.
- `appendix.tex:960` [PROSE] — REPLACE "highest raw detection rate". §2.
- `appendix.tex:972` [DATA] — `tab:codesign` row → `0.865 & 0.092 & 0.890 & 0.771 & 0.908 & 0.383 & 0.75`. §1.
- `appendix.tex:979` [VERIFY] — btu per-type carried-over (0.10.0); keep + `% TODO(btu-1.2.0 per-type)`.
  §2 / FLAG-D.
- `appendix.tex:1028` [PROSE] — relabel-divergence paragraph: keep argument, restate 1.2.0 numbers. §2.
- `appendix.tex:1099` [PROSE] — `app:codesign` takeaway: lead narrows not reverses. §2 / FLAG-A.
- `appendix.tex` (new subsections) [NEW] — A1 `app:ablation_prompt`, A3 `app:ablation_threshold`,
  A4 `app:ablation_format`, A5 `app:ablation_kappa`, Coverage/risk-coverage `app:coverage`
  (+ risk-coverage figure). §3.
- `appendix.tex:1143` [VERIFY] — co-designed block header in `tab:test_public_full`; btu row in that
  table (L189) → dev `.865/.092/.890`, test `.877/.115/.901`, dFPR `+0.024` (was .946/.179/.908 dev,
  .886/.338/.858 test, +0.144). §1 / FLAG-A.

---

## 7. Open flags (inconsistent / uncertain — surface to author)

- **FLAG-A (LOAD-BEARING, narrative change):** under btu 1.2.0 the dev→test cross-split FPR jump
  is +2.4pp (not +14.4pp, the 0.10.0 behaviour), and the btu-vs-Sonnet F1 lead **narrows but does
  NOT reverse** (dev +6.3pp → test +3.5pp; btu F1-leads on both splits). The paper currently
  asserts a *reversal* and a *+14.4pp FPR jump* in ≥4 places (analysis L34/L40, appendix L627/L1099,
  L321, tab:test_public_full L189). These were correct for 0.10.0 and are WRONG for 1.2.0. The
  "rule-based lead is a dev artifact" thread is WEAKENED (now "narrows but survives, because btu
  abstains"). **Author must decide the framing** — recommend the cross-split-stable / precision-via-
  abstention story, which is cleaner and ties to the Coverage contribution. *This is the single
  biggest consequence of the version bump; do not silently flip it.*
- **FLAG-B:** abstract L9 "F1 trails by 9.3pp" is INVERTED under 1.2.0 (btu F1 .890 vs agentic .816;
  btu now leads by ~7.4pp). Reframe as abstention buys higher F1 despite lower recall.
- **FLAG-C:** analysis L14 uses btu as the "high-DR" exemplar — now mid-DR (.865). Swap exemplar.
- **FLAG-D:** btu per-type (tab:codesign_pertype, L997+) is 0.10.0 carried-over; v1.2.0 per-type was
  NOT regenerated cell-by-cell. Keep cells + caveat + `% TODO`.
- **FLAG-E:** `tab:ppv` (L573) must RE-SORT — btu moves from rank 4 to rank 3 (PPV@2% 16.2% > Sonnet 11.2%).
- **FLAG-F:** deployment recipe (L627) — btu is no longer the recall pick; reassign to precision.
- **FLAG-G:** btu **test_public** selective-prediction (Coverage appendix cons./aggr.) is INCOMPLETE
  (raw 500/831; coverage_reporting.json `INCOMPLETE`/`could_not_compute`). HOLD that *appendix cell*
  pending GEN `wkp97jqbb`. The **headline** btu test triple .877/.115/.901 (committed baseline JSON,
  full-N) IS final and goes in tab:results/tab:test_public_full now.
- **FLAG-H:** confirm no metrics table derives a stress FPR from the 1 canary (dataset count "1 valid"
  is fine; canary is metric-filtered). From grep, none exists — verification only.
- **Coverage-column convention choice:** btu Cov `.75` (selective) vs `1.00` (committed-VALID, as in
  the baseline JSON). Recommend `.75` so abstention is visible and consistent with §4; flag the choice
  so DR/FPR/F1 (full-N committed-VALID) and Cov (selective) are not read as contradictory — add a
  one-line footnote: "btu's DR/FPR/F1 score abstentions as committed-VALID over the full split;
  Cov.\ reports the fraction it commits to. \cref{app:coverage} reports its selective cons./aggr.\ split."
- **A5 framing guard:** A5 is an LLM-rater reliability *proxy*, NOT human IAA — must be labelled as
  such (the κ gap in `comparative_expectations.md` Rank 1 asked for *human* κ; A5 does not close it,
  it corroborates the relabel). Do not let it read as the human-IAA contribution.
- **Citation check:** El-Yaniv & Wiener 2010 / Geifman & El-Yaniv 2017 (selective prediction) must be
  verified present in `references.bib` before the Coverage section cites them; else `[CITE-CHECK]`.
- **btu version label:** add the v1.2.0 version string somewhere (caption of tab:codesign or
  app:codesign text) so the version bump vs the deferred-0.10.0 cascade is explicit and reproducible.
