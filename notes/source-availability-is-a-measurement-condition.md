# Source availability is a measurement condition, not an incident

The pre-screening ablation was attempted five times across 2026-09-04 and
2026-09-05 and produced two of its four arms. Every failure came from the same
place: a source that did not answer. That is the finding. A retrieval-dependent
verifier cannot be evaluated reproducibly while one of the databases it queries
is intermittent, and the fraction of entries whose lookup completed belongs in
the report beside the detection rate, not in a footnote about infrastructure.

## What was measured

Each attempt ran `bibtex-check` 1.10.1 at 20 requests per minute per service,
with an OpenAlex key, OpenReview credentials, a Semantic Scholar key, and a
polite-pool contact address. `bibtex-check` exits 5 and names the condition when
too many lookups fail, and HALLMARK's wrapper raises `SourceOutageError` rather
than scoring the run, because a source that never answered is not evidence a
reference is absent.

| attempt | arm | outcome | entries with an incomplete lookup |
|---|---|---|---|
| 16:00 | `bibtexupdater` dev | scored | 285 / 1,119 (25.5%) — dblp 275, openalex 26 |
| 20:49 | `bibtexupdater` dev | discarded | 244 / 1,119 (21.8%) — dblp 230, openalex 26, s2 1 |
| 22:42 | `bibtexupdater` dev | discarded | 223 / 1,119 (19.9%) — dblp 205, openalex 26 |
| 23:05 | `bibtexupdater` test | discarded | 163 / 831 (19.6%) — dblp 155, openalex 16, s2 1 |
| 23:46 | `no_prescreening` dev | scored | none |
| 00:55 | `no_prescreening` test | scored | none |

DBLP accounts for 86% to 95% of every failure column. Probing it directly on a
ten-minute cycle gives the shape: answering at 22:11, HTTP 503 at 22:21,
answering at 22:32 and 22:42, HTTP 503 again at 07:52 the next morning. It is not
down, and it is not up. It alternates on a timescale shorter than a single arm
of the ablation, which is why the same command produced 25.5%, 21.8% and 19.9%
incomplete on three runs of identical code against an identical split.

## Why the two surviving arms do not rescue the comparison

The `no_prescreening` arms are clean — no source failures, DR 0.893 / FPR 0.029
on `dev_public` and DR 0.892 / FPR 0.022 on `test_public`. They are also the two
that happened to run after 23:46, when DBLP recovered. Their comparators ran
before it did. So the split between the clean arms and the discarded ones tracks
the clock, not the condition the ablation set out to vary, and pairing them would
attribute a source outage to pre-screening.

## What the condition does to a result

Two independent measurements say the effect is large enough to change a
conclusion. Correcting the source conditions on the same tool and split —
supplying the keys, the credentials and a 20-per-minute pace — moved the
incomplete-lookup fraction from 53.7% to 10.0%, and the run got *faster*: 2.4
seconds per entry against 5.7. Pacing beats retrying, because a starved source
turns into a retry storm that costs more than the wait it was avoiding. And on a
119-entry probe holding the model and prompt fixed, 12% of flags cleared with
arXiv starved against 24% with it answering: the same verifier, the same
entries, half the corrections.

Neither number is about the verifier. Both are about whether the databases
answered, and a benchmark that does not record which is reporting an interaction
as a property.

## What to do about it

Record the condition with every retrieval-dependent result. `bibtex-check`
already names it and the wrapper already refuses to score a run that fails it;
what was missing is that the number never reached the report.

Check before starting rather than after. `scripts/check_source_reachability.py`
probes one real record per source and exits non-zero when a required one does not
answer, which is two minutes against the ninety a discarded arm costs. It treats
HTTP 429 as reachable — the service answered and asked for a slower pace, which
pacing handles — and asks each source for a real record rather than a bare root,
because several serve 200 from a CDN at the root while the query path behind it
is down.

Do not paper over an outage with `HALLMARK_ALLOW_SOURCE_OUTAGE=1` to get a number
tonight. A run scored under a 20% incomplete-lookup rate is a measurement of
DBLP's week.

## Status

The ablation is stopped. Four attempts were discarded by the guard working as
designed; the two arms that completed are recorded above and are not a
comparison. Re-running it requires either DBLP holding for the four hours all
four arms take, or a decision to drop DBLP from the source set and report the
result as an OpenAlex/CrossRef/Semantic Scholar/arXiv condition — which is a
different measurement from the published rows, and would have to say so.

The run's artifacts -- both scored arms, all six logs, and `source_conditions.txt`
-- are at
`~/.claude/projects/-Users-patrik-reizinger-Documents-GitHub-hallmark/prescreening-ablation-2026-09-05/`.
They are not committed: they were produced against a pinned pre-fix worktree and
are a record of the attempt, not a result to cite.

One caveat on the two surviving arms: they were produced from a worktree pinned
before the abstention fix, so they carry `coverage: 1.0` and `num_uncertain: 0`
for a tool that abstains. Anything derived from them needs the coverage
correction applied first.
