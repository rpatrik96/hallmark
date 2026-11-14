# Hall of Hallucinations (real-world, evaluation-only)

Source: Hall of Hallucinations by Andrew Wheeler / VerusCite, https://veruscite.com/hallucinations.
Scraped 2026-11-09 with the site owner's permission; credit and link required when used.

Labels are VerusCite's automated verdicts (no row was human-edited on the site).

| File | Contents |
|---|---|
| `hall_hallucinated_rows.jsonl` | 942 rows with status `hallucination` |
| `hall_hallucinated_rows_flagged.jsonl` | + overlap flags: 1 in VerusCite V1, 0 in HALLMARK, 5 duplicates within Hall (936 kept) |
| `hall_autocheck.jsonl` | free DOI (Crossref, doi.org) and Crossref title-search checks for all 936 rows |
| `hall_hallmark_draft.jsonl` | 932 BenchmarkEntry rows (936 minus 4 human n/u) |

Human check of a seed-0 random 100-row sample (review sheets not shipped here): 96 hallucinated, 2 not, 2 unsure.
Precision 96/98 = 98.0% (Wilson 95% CI 92.9-99.4%).

Types are not human-verified except for 30 rows. Assignment, as in `scripts/parse_gptzero_csv.py`:
keyword rules over VerusCite's notes (736), fallback `plausible_fabrication` when no rule matches (166).
On the sample the notes rule agreed with the human type on 63/74 rows. Report only overall recall on this split.
