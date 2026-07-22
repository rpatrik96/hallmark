# HOSTING

Release and distribution plan for HALLMARK v1.0.

---

## Distribution channels

| Channel | URL | Format | Role |
|---------|-----|--------|------|
| Anonymous review repo | https://anonymous.4open.science/r/hallmark/ | git | Double-blind review access |
| GitHub | https://github.com/rpatrik96/hallmark | git | Canonical public release |
| Companion website | https://rpatrik96.github.io/hallmark/ | static HTML (GitHub Pages, deployed from `site/`) | Interactive results explorer + examples browser |
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
16cf8f28b28606515fadb7ddef489a1bd88ddf9414345b2b46da8e96f52bab4b  data/dev_public.parquet
2f8c6d6c009c1ea6c37f53c4e0abb97c484e0c9a1fe098f06d0c457c7e6e1201  data/test_public.parquet
cb273ad0a04d19320ceed78fee259728179e4886caf2accec660b93f33eab2db  data/stress_test.parquet
b1b4f41ce89a2d43fb67ae961f22070b2e33b88a961c0d68994fe1b363ad9026  blind/dev_public_blind.parquet
8c5c5104afbef57daa5d018c61fa979ee9eb9d740d9ea119f6bed15c21c47d5d  blind/test_public_blind.parquet
3a1e0fdc263fe4c55ca1cc2526b0024eb270bcd6d02d4cc9cd9f8e908c0f9a3e  blind/stress_test_blind.parquet
```

Paths in this section are HuggingFace-relative (under `https://huggingface.co/datasets/hallmark-neurips2026/HALLMARK/resolve/main/`).

### JSONL mirror (repo `data/v1.0/` and HuggingFace `jsonl/`)

```
9b6391c236d2f60d1c33918c13a68d8ac9f53083143eec16620f4559d0188f3e  data/v1.0/dev_public.jsonl
46b9a23a1e1a7564c52ac490a492090fdc8d87f15bfea4c62bc49f8fd9ce42b7  data/v1.0/dev_public_blind.jsonl
ac6da9614e2c36bb4eb553f0bccdff2b73c303ea4a06e373f4bceb067650ba0d  data/v1.0/test_public.jsonl
0461cef293e6f2a1e1448b6fa56de72c0efd18a240cdf426377616afc00649bd  data/v1.0/test_public_blind.jsonl
af3a11606948fea8edccbbd5c512e6efd69c1f9056032d446f01c66950da07d3  data/v1.0/stress_test.jsonl
cf6f829b2ad99c6badd9eb513214e2d811e6b0fe3218a64263aeecb690ea49e3  data/v1.0/stress_test_blind.jsonl
caf4f5597846b4cbca404fb44e2c1550db69ae062288a78f7028f95bd7a3e065  data/v1.0/metadata.json
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
