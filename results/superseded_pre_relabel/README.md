# Superseded baseline results

These fifteen files once sat at the top of `results/` under the same names as
files in `data/v1.2/baseline_results/`, and **every one of them disagreed with
its namesake** — on detection rate, false-positive rate, F1 and sometimes on the
number of entries scored. Two directories held two different runs under one set
of names, with nothing comparing them and nothing saying which was current.

`data/v1.2/baseline_results/` is canonical. It is what CI validates, what the
manifest checksums, what the README links, and its entry counts match the
current splits. These copies predate the relabelling passes: most differ from
the canonical numbers by a point or two of detection rate, which is what a
relabel does, and three are worse than stale.

| file | here | canonical | what happened |
|---|---|---|---|
| `bibtexupdater_test_public.json` | DR .130, n=840 | DR .877, n=831 | a broken run, scored against a pre-dedup split |
| `doi_only_test_public.json` | DR .278, n=840 | DR .387, n=831 | same pre-dedup split |
| `harc_dev_public.json` | DR .143, n=1079 | DR .155, n=521 | **the canonical file is the truncated one here** — n=521 matches no split, and `validate-results --strict` fails on it today |

They are kept rather than deleted because they are the provenance of any figure
or table generated before the relabel, and because the `harc` row shows the
divergence does not run one way.

Nothing should read this directory. `tests/test_results_provenance.py` asserts
that no filename appears both here and in a `baseline_results/` directory, so
the collision cannot come back.
