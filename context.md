# context.md — periodic-kmeans

**Project:** K-Means Rediscovers the Periodic Table — *ML-subject mini project* (deep-dive on Session 11: K-Means elbow + silhouette).
**Hook:** cluster the 118 chemical elements on physical properties only; afterward check whether the unsupervised clusters line up with the periodic table's families (ARI).
**Deliverable:** 3 instructor-facing notebooks (sir reviews the .ipynb files directly), executed, offline-graded (data snapshot committed).

## Status board

- [x] Design doc approved (`docs/superpowers/specs/2026-10-31-periodic-kmeans-design.md`)
- [x] Implementation plan (`docs/superpowers/plans/2026-10-31-periodic-kmeans.md`)
- [x] Task 1 scaffold + data snapshot + offline tests (4/4 passing)
- [ ] Task 2 — notebook `01_data.ipynb` (fetch, coverage, heterogeneity, missingness-as-physics, leakage guard)
- [ ] Task 3 — `src/study.py` + notebook `02_kmeans_study.ipynb` (raw-units failure → standardization → elbow/silhouette → k* → centroids)
- [ ] Task 4 — notebook `03_validation_report.ipynb` (ARI at k*, k=18 validation-only companion, H-sensitivity, unlabeled-28 table) + `reports/p_report.md` + README + private GitHub repo

## Data facts (recon + tests, 2026-10-31)
- mendeleev `fetch_table("elements")`: 118 elements; snapshot committed at `data/elements.csv` (offline proof).
- Physical feature coverage (rows non-null): weight 118, density 118, radius_rahm 96, radius 90, EN-pauling 85, e-affinity 77.
- Ground truth for validation only: `group_id` present for 90 elements (18 families); 28 f-block/transactinides unlabeled.
- Chemistry nuance the tests taught us: noble gases lack EN **except Xe (2.6)** — it forms compounds. He/Ar have *negative* electron affinity (resist electron capture) — big visual for notebook 01.

## Conventions
- Seeds (0, 1, 2); imputation (median) strictly inside sklearn pipelines — no global pre-impute feeding metrics.
- **Anti-peek rule:** features never include `group_id`/`period`/`block`; ARI computed last, on the 90 labeled elements only; k=18 companion framed validation-only.
- Headline: ARI at **silhouette-chosen k\***; report mean ± std across 3 seeds; PR=no accuracy-talk (unsupervised).
- Language in the report: "rediscover/reproduce" — no discovery claims; ARI reported as-is.

## Environment
| Thing | Value |
|---|---|
| Python | 3.14.2, project venv `.venv` (sklearn/pandas/matplotlib/mendeleev/ipykernel/pytest) |
| Run | `.venv/bin/python ...` · `.venv/bin/pytest -q` |
| Repo | git on `main`; GitHub private remote (created in Task 4) |

## Repo layout
```
periodic-kmeans/
├── context.md            # this file — update every session
├── data/elements.csv     # committed snapshot (offline grading)
├── src/                  # data.py, study.py (Task 3)
├── tests/                # test_data.py (4 passing)
├── notebooks/            # 01_data, 02_kmeans_study, 03_validation_report
├── reports/              # figures/, tables/, p_report.md
└── docs/superpowers/     # design spec + implementation plan
```

## Session log
- **2026-10-31** — brainstormed 13 ideas across the 15 topics; picked "K-Means rediscovers the periodic table" (builder-checked design: grader reads notebooks; k=18 companion sanctioned as validation-only; hydrogen sensitivity flagged call). Scaffold: venv+deps, snapshot committed after mendeleev fetch, 4 data tests passing (one test corrected real chemistry: Xe's EN exists). Next: Task 2 (notebook 01).
