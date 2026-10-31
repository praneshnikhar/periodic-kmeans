# Design: "K-Means Rediscovers the Periodic Table" (ML-subject mini project)

Date: 2026-10-31 · Author: agent + Pranesh · Size: mini (2–3 lab sessions) → short design per process

## Intent & why
A small assessed project anchored to **Session 11 (K-Means: elbow + silhouette)**. The hook:
cluster the 118 chemical elements purely on physical properties and see whether an *unsupervised*
algorithm rediscovers the periodic table's families. Presentation story is unusual, cheap to run,
and gives a rigorous one-algorithm deep-dive (the grader-confirmed format).

## Goals
- End-to-end mini pipeline: data (mendeleev pip package) → EDA → standardization → K-Means with
  elbow + silhouette k-selection → interpretation by centroids → **validation vs periodic families
  (Adjusted Rand Index – ARI) as the LAST step** → notebook report.
- Honest evaluation habits: leakage-free thinking (imputation inside pipelines), stability across seeds.

## Non-goals
No deep learning; no supervised training; no scraping beyond the pip package; no new libraries
beyond sklearn/pandas/matplotlib/mendeleev; not a chemistry report — chemistry only validates.

## Anti-goals (failure even if "working")
- Never tune k by peeking at group labels (cluster first, compare after).
- No "discovery" claims — language: "rediscover/reproduce", with ARI as evidence, never 100% promised.
- No fake completeness: 28 unlabeled f-block/transactinide elements stay honestly reported.

## Constraints
- Python 3.14 venv; scikit-learn/pandas/matplotlib/mendeleev only; runnable in VS Code notebooks.
- Repo conventions copied from exoplanet-hunter (context.md, seeded runs, artifacts to reports/).

## Approaches considered (how to build the study)
- (A) K-Means on raw units — rejected: heterogeneous units (density 8 orders of magnitude) make
  one feature dominate; that failure is *shown* as the motivation for the picked approach.
- (B) K-Means on standardized physical features ← **PICKED** — weight, density, atomic radii
  (classic + Rahm variant), electronegativity, electron affinity; median-impute inside pipelines.
- (C) Extension (optional, builder-freedom): add a chemistry-informed feature block (e.g., block
  one-hot / van-der-Waals radius) and show what changes. Never required.

## Verified data facts (recon on 2026-10-31)
mendeleev `fetch_table("elements")`: 118 rows; non-null coverage — weight 118, density 118,
radius-rahm 96, radius 90, EN-pauling 85, e-affinity 77; group_id present for 90 (18 families),
28 rows unlabeled (f-block etc.). Noble-gas EN is missing *by physics* (a feature, not dirt —
median-impute + coverage table, discussed in notebook 01).

## Decided hows (as decided with Pranesh)
- Folder: `coding2/periodic-kmeans/` (separate repo, private gh remote later).
- Deliverable: lab-style notebook suite (mirroring the exo format): `01_data.ipynb` (fetch+EDA),
  `02_kmeans_study.ipynb` (scale-domination demo → standardization → elbow/silhouette → centroids),
  `03_validation_report.ipynb` (ARI, confusion-style table, unlabeled-element analysis + writeup).
- Metrics: elbow inertia, silhouette (3 seeds), ARI vs families on labeled subset; centroids
  summarized per cluster vs the iconic families (alkalis/halogens/nobles).

## Left to the builder
Figure styling; exact k display range (guide: 2–12); whether radius-rahm or classic radius is the
primary radius column (run both, report); depth of the unlabeled-elements discussion.
