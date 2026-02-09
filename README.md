# HALLMARK

**HALL**ucination bench**MARK**: A benchmark for evaluating citation hallucination detection tools.

## Installation

```bash
pip install hallmark
```

## Quick Start

```bash
# Evaluate a baseline on the dev split
hallmark evaluate --split dev_public --baseline doi_only

# Show dataset statistics
hallmark stats --split dev_public
```

## Overview

HALLMARK provides:

- **Hallucination taxonomy**: 12 types across 3 difficulty tiers
- **Evaluation framework**: Detection Rate, F1, tier-weighted F1, detect@k
- **Baselines**: DOI-only, bibtex-updater, LLM-based, ensemble
- **Temporal analysis**: Contamination detection via pre/post-cutoff comparison
- **Community contributions**: ONEBench-style ever-expanding sample pool

## Citation

If you use HALLMARK in your research, please cite:

```bibtex
@misc{hallmark2026,
    title={HALLMARK: A Benchmark for Citation Hallucination Detection},
    author={Reizinger, Patrik},
    year={2026},
    url={https://github.com/rpatrik96/hallmark}
}
```
