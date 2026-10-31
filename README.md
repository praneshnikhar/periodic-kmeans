# periodic-kmeans 🔬🧲

**"K-Means rediscovers the periodic table?"** — ML-subject mini project (Session 11 deep-dive).

Cluster the 118 chemical elements on six physical properties only, then grade the clustering
against the periodic table's families with ARI. The honest answer — computed, not hoped — is
**no at family granularity (ARI 0.016)**, weak-but-persistent at the coarse block level
(0.14→0.24), and *the reason why* is the interesting part: families live in electron
configuration, not in bulk properties (the alkali family itself spans Li 7 u → Fr 223 u).

## Run it (offline-ready)

1. Open any notebook in **VS Code**; kernel = `Python 3 (.venv)` (repo's own venv).
2. Notebooks run top-to-bottom with **committed outputs**; data snapshot committed
   (`data/elements.csv`) so grading needs no network.

| # | Notebook | What it does |
|---|---|---|
| 1 | `notebooks/01_data.ipynb` | data audit Labs 1–2 style: coverage, unit heterogeneity (the raw-unit trap readied), missingness-as-physics (Xe!), anti-peek guard |
| 2 | `notebooks/02_kmeans_study.ipynb` | raw-vs-standardized exhibit → elbow + silhouette (3 seeds) → k\*=5 → centroid reading |
| 3 | `notebooks/03_validation_report.ipynb` | ARI battery at k\*, k=18 companion (validation-only), block re-check, H-sensitivity, unlabeled-28, periodic table colored by cluster |

## Repo map

```
periodic-kmeans/
├── context.md                  # project brain: decisions, conventions, session log
├── data/elements.csv           # committed snapshot (offline guarantee)
├── src/{data.py, study.py}     # reusable, tested modules (notebooks stay thin)
├── tests/                      # 9 pytest tests (leakage guard, shapes, physics-missingness, study)
├── notebooks/                  # the deliverable (executed)
├── reports/                    # p_report.md + figures/ + tables/
└── docs/superpowers/           # design spec + implementation plan (agent working record)
```

## Numbers you'll find (do not trust this README over the notebooks)

- silhouette k-selection: **k\* = 5** (mean silhouette 0.359 over 3 seeds)
- ARI vs 18 families @ k\*: **0.016 ± 0.000** · k=18 companion (validation-only): **0.024**
- block battery (s/p/d/f): 0.143 / 0.180 / 0.243 for k = 5/6/7 · H-out sensitivity **Δ +0.0016**
- raw-units exhibit: atomic_weight owns **78.7%** of the raw pairwise-distance mass

## Conventions

Seeds (0,1,2) · median imputation inside pipelines only · chemistry-truth columns never in
features · ARI strictly post-hoc (validation) · language "rediscover/reproduce", no discovery
claims. Full statement in [`reports/p_report.md`](reports/p_report.md).
