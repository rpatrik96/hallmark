# HOSTING

Release and distribution plan for HALLMARK v1.0.

---

## Distribution channels

| Channel | URL | Format | Role |
|---------|-----|--------|------|
| Anonymous review repo | https://anonymous.4open.science/r/hallmark/ | git | Double-blind review access |
| GitHub (camera-ready) | TBD at camera-ready | git | Canonical public release |
| HuggingFace mirror | https://huggingface.co/datasets/hallmark-neurips2026/HALLMARK | parquet (canonical) + jsonl (mirror) + baseline_results + croissant.json | Dataset hosting |
| Zenodo DOI | planned at camera-ready | archive | Permanent archival + DOI minting |
| PyPI | planned at camera-ready | wheel | `pip install hallmark` |

---

## Croissant metadata

`croissant.json` at the repo root provides [Croissant 1.0](https://github.com/mlcommons/croissant) machine-readable metadata.

Validate locally:

```bash
pip install mlcroissant
mlcroissant validate --jsonld croissant.json
```

The `distribution` block points to parquet files on the HuggingFace mirror; this is the canonical machine-readable target for `mlcroissant` streaming and the NeurIPS Records Generation Test. JSONL versions remain on the same mirror under `jsonl/` for direct human consumption.

RAI fields covered (all required by NeurIPS 2026 D&B):
`rai:dataCollection`, `rai:dataCollectionType`, `rai:personalSensitiveInformation`,
`rai:dataPreprocessing`, `rai:dataSocialImpact`, `rai:dataLimitations`,
`rai:dataBiases`, `rai:dataUseCases`, `rai:hasSyntheticData`, `rai:syntheticDataExplanation`.

---

## License

MIT — https://spdx.org/licenses/MIT.html

Full text: `LICENSE`.

---

## Checksums (SHA-256)

### Parquet (canonical, referenced from `croissant.json`, hosted on HuggingFace)

```
978c676301c343675063caa96a358f014b270d959c3195a5fcb36cb5ebc4ce80  data/dev_public.parquet
a7126002a6f69e26fa169b77c9a983ae4d7b693c2d506ddfa96f7e117864d068  data/test_public.parquet
82652be35d922fca423b071598848918d726959ccdcfa06d0904978e64d3bd71  data/stress_test.parquet
9aa5b854c33c28a8ed1a44965681669bfe2f6bc170fa253454634ff5cbe27956  blind/dev_public_blind.parquet
66e9ae0775747a58fd0315758b34d9d5a079d29f6a61c5f74879d81e2399ff1c  blind/test_public_blind.parquet
d84178e41a6c27729a171d31332caf6fd57bd2765162fc72eb45817743006ab0  blind/stress_test_blind.parquet
```

Paths in this section are HuggingFace-relative (under `https://huggingface.co/datasets/hallmark-neurips2026/HALLMARK/resolve/main/`).

### JSONL mirror (repo `data/v1.0/` and HuggingFace `jsonl/`)

```
f35f5643f12129ccf1cbc5df754329bab8ecf93e3ac703afa63ae975040f97af  data/v1.0/dev_public.jsonl
46b9a23a1e1a7564c52ac490a492090fdc8d87f15bfea4c62bc49f8fd9ce42b7  data/v1.0/dev_public_blind.jsonl
1d9d65873f3c5a016f838ae0665afb52c4f2b29bb719245c2b13189c4783c173  data/v1.0/test_public.jsonl
0461cef293e6f2a1e1448b6fa56de72c0efd18a240cdf426377616afc00649bd  data/v1.0/test_public_blind.jsonl
af3a11606948fea8edccbbd5c512e6efd69c1f9056032d446f01c66950da07d3  data/v1.0/stress_test.jsonl
cf6f829b2ad99c6badd9eb513214e2d811e6b0fe3218a64263aeecb690ea49e3  data/v1.0/stress_test_blind.jsonl
ba1f34270db35047678cc13a5a74056f53eea07c0d16fe52061d7849b65717e7  data/v1.0/metadata.json
f6b5b3fcc7964a8b74be8a30b6c4566582d81f42764f2b1501d89d29ee2dfdbb  data/v1.0/source_mapping.json
3e772d02f71ae74601c419f6bfd6244f13acd329be61e003910cb9eb3c1f03b3  data/v1.0/valid_entry_verification.json
```

Verify the JSONL mirror locally:

```bash
shasum -a 256 --check <(grep "data/v1.0/" HOSTING.md | grep -v "^#")
```

---

## Hidden test split

`test_hidden.jsonl` (453 entries) is intentionally withheld from public distribution to preserve
evaluation integrity. Access procedure will be documented at camera-ready alongside leaderboard
submission instructions.

---

## Versioning

The `version` field in `data/v1.0/metadata.json` tracks the dataset version using semver.
Current version: **1.0**.

Breaking schema changes increment the major version and receive a new `data/vX.Y/` directory.
Backward-compatible additions increment the minor version in-place.

---

## Anonymous review note

Creator and contact metadata in `croissant.json`, `metadata.json`, and HuggingFace dataset
cards are currently set to placeholder values to comply with NeurIPS 2026 double-blind review
policy. All identifying fields will be populated at camera-ready. The anonymous review repository
at https://anonymous.4open.science/r/hallmark/ provides read-only access to reviewers without
revealing author identity.
