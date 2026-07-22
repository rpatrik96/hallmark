# Companion website

Interactive companion site for the HALLMARK paper
([arXiv:2607.18360](https://arxiv.org/abs/2607.18360)), served via GitHub Pages
from this directory: <https://rpatrik96.github.io/hallmark/>.

Plain HTML/CSS/JS, no build step, no external dependencies. All charts are
hand-rolled SVG/CSS with tooltips and per-chart table views; light and dark
theme follow the OS setting with a manual toggle.

## Data

`data/site_data.js` is **generated** — do not edit it by hand. It is derived
exclusively from released, tracked artifacts (corpus `metadata.json`,
`data/v1.0/baseline_results/*.json`, the tracked cross-domain and canonical
448-entry temporal supplement metrics, and a seeded sample of
`dev_public.jsonl` entries for the examples browser).

Regenerate after a data or results update:

```bash
python scripts/generate_site_data.py
```

The sampling is seeded (8042), so reruns are deterministic for unchanged
inputs; only the `generated` date stamp changes.

## Deployment

`.github/workflows/pages.yml` deploys this directory to GitHub Pages on every
push to `main` that touches `site/**`. Preview locally with:

```bash
python -m http.server -d site 8000
```
