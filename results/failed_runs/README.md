# Failed runs, kept as evidence

Four files here report `detection_rate: 0.0`, `false_positive_rate: 0.0` and
`mean_api_calls: 0.0` over a thousand-odd entries. That is not a result. It is
the signature of `fallback_predictions` firing because the external CLI was not
on `PATH`, so the wrapper returned an all-VALID prediction for every entry and
the harness scored it normally.

They matter because of what they were supposed to be. The pre-screening layer
adds local DOI, year and author checks ahead of the external tools, and the
`*_no_prescreening` variants exist to measure what that layer contributes. Four
of the six ablations never ran:

| file | n | DR | API calls |
|---|---|---|---|
| `bibtexupdater_no_prescreening_dev_public.json` | 1,079 | 0.000 | 0.0 |
| `bibtexupdater_no_prescreening_test_public.json` | 849 | 0.000 | 0.0 |
| `harc_no_prescreening_dev_public.json` | 1,079 | 0.000 | 0.0 |
| `harc_no_prescreening_test_public.json` | 849 | 0.000 | 0.0 |

Only `doi_only` produced a real ablation (DR 0.203 without pre-screening against
0.256 with it, on `dev_public`). So the "roughly five points of Tier-1 lift"
attributed to the pre-screening layer rests on one baseline out of three, and
the measurement that would show whether the layer helps or hurts the *co-designed*
tool — the one a reviewer will ask about — is one of the empty files above.

Running the two `bibtexupdater` ablations for real is the load-bearing evidence
for the pre-screening design. It needs `bibtex-check` and `harcx` on `PATH` plus
live API budget.

`tests/test_results_provenance.py` asserts that no result outside this directory
is a null run, so one cannot be mistaken for a measurement again.
