# HOSTING

Release and distribution plan for HALLMARK v1.0.

---

## Distribution channels

| Channel | URL | Format | Role |
|---------|-----|--------|------|
| Anonymous review repo | https://anonymous.4open.science/r/hallmark/ | git | Double-blind review access |
| GitHub | https://github.com/rpatrik96/hallmark | git | Canonical public release |
| Companion website | https://rpatrik96.github.io/hallmark/ | static HTML (GitHub Pages, deployed from `site/`) | Interactive results explorer + examples browser |
| HuggingFace mirror | withdrawn with the NeurIPS 2026 submission | — | — |
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

The `distribution` block points to the JSONL files in this repository under `data/v1.2/`; this is the canonical machine-readable target for `mlcroissant` streaming and the NeurIPS Records Generation Test.

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

### JSONL (canonical, referenced from `croissant.json`)

```
0f75d390f95a086aaf39b627369af2bd03a5eb87cd3bf4cea5d5025e203bb09e  data/v1.2/dev_public.jsonl
46b9a23a1e1a7564c52ac490a492090fdc8d87f15bfea4c62bc49f8fd9ce42b7  data/v1.2/dev_public_blind.jsonl
0637ffc80747c36f3e010cd613f16481dcb916f944874122b2e2d3f872429973  data/v1.2/test_public.jsonl
0461cef293e6f2a1e1448b6fa56de72c0efd18a240cdf426377616afc00649bd  data/v1.2/test_public_blind.jsonl
af3a11606948fea8edccbbd5c512e6efd69c1f9056032d446f01c66950da07d3  data/v1.2/stress_test.jsonl
cf6f829b2ad99c6badd9eb513214e2d811e6b0fe3218a64263aeecb690ea49e3  data/v1.2/stress_test_blind.jsonl
ea4f34bc80e844823b9604d880b0b4a7dec63cb1c193014969f6d7fe972751a0  data/v1.2/metadata.json
f6b5b3fcc7964a8b74be8a30b6c4566582d81f42764f2b1501d89d29ee2dfdbb  data/v1.2/source_mapping.json
3e772d02f71ae74601c419f6bfd6244f13acd329be61e003910cb9eb3c1f03b3  data/v1.2/valid_entry_verification.json
```

Verify locally:

```bash
shasum -a 256 --check <(grep -E "^[0-9a-f]{64}  data/v1.2/" HOSTING.md)
```

---

## Hidden test split

`test_hidden.jsonl` (453 entries) is intentionally withheld from public distribution to preserve
evaluation integrity. Access procedure will be documented at camera-ready alongside leaderboard
submission instructions.

---

## Versioning

The `version` field in `data/v1.2/metadata.json` tracks the dataset version using semver.
Current version: **1.0**.

Breaking schema changes increment the major version and receive a new `data/vX.Y/` directory.
Backward-compatible additions increment the minor version in-place.

---

## Anonymous review note

Creator and contact metadata in `croissant.json` and `metadata.json` are set to placeholder values to comply with NeurIPS 2026 double-blind review
policy. All identifying fields will be populated at camera-ready. The anonymous review repository
at https://anonymous.4open.science/r/hallmark/ provides read-only access to reviewers without
revealing author identity.
