# Periodic K-Means Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the "K-Means Rediscovers the Periodic Table" mini project — 118 elements clustered on physical properties, validated against periodic families (ARI), delivered as 3 instructor-facing notebooks.

**Architecture:** thin reusable `src` modules (data loading w/ offline snapshot, standardization + clustering studies) wrapped by lab-style notebooks; validation-only ARI at silhouette-chosen k plus a labeled-as-validation k=18 companion run; repo conventions copied from exoplanet-hunter.

**Tech Stack:** Python 3.14 venv · scikit-learn · pandas · matplotlib · mendeleev · ipykernel · pytest

**Spec:** `docs/superpowers/specs/2026-10-31-periodic-kmeans-design.md` (moves into the repo at scaffold; read both)

## Global Constraints
- Features NEVER include group_id/period/block (anti-peek rule, spec anti-goals)
- Imputation (median) inside CV/pipeline folds only — never global pre-impute for study metrics
- Reported headline ARI at silhouette-selected k; k=18 run framed validation-only
- All runs seeded (seeds 0,1,2); grader runs offline → data snapshot committed with fetch fallback
- Deliverable = notebooks the instructor reads/executes; artifacts to `reports/`

## Review Focus
- Unit heterogeneity (density spans 8 orders) → Task 3's raw-units demo must show one feature dominating (test: raw-KMeans inertia gap vs standardized)
- Noble-gas NaN electronegativity is physics → Task 2 test pins EN missingness pattern (noble symbols ⊂ EN-null set), not blanket NaN dropping
- Ground-truth leakage → Task 3 test: `feature_names` ∩ {group_id, period, block} = ∅
- ARI scope error → Task 4 test: ARI evaluated on labeled subset only (n=90), never on imputed "labels"
- Offline reproducibility → Task 1 test: fresh-HOME fetch fallback resolves from repo snapshot

---

### Task 1: Scaffold repo + data snapshot

**Files:** Create `periodic-kmeans/{context.md, .gitignore, data/mendeleev_elements.parquet-or-sqlite, src/data.py, tests/test_data.py}`

**Interfaces:** Produces `load_elements() -> pd.DataFrame` (118 rows; feature cols + identity cols + group_id) and `snap_path()` — used by every later task/notebook.

- [ ] **Step 1: Create folder; `python3 -m venv .venv`; install scikit-learn pandas matplotlib mendeleev ipykernel pytest**
- [ ] **Step 2: Write failing test** `tests/test_data.py`: `len(df)==118`; `{"atomic_weight","density","atomic_radius","atomic_radius_rahm","en_pauling","electron_affinity"} ⊆ df.columns`; noble symbols ⊆ EN-null set
- [ ] **Step 3: Run pytest → FAIL (module missing)**
- [ ] **Step 4: Implement `src/data.py`: `fetch_table` → commit snapshot under `data/` → `load_elements()` prefers snapshot, falls back to network (and vice versa on fresh HOME)**
- [ ] **Step 5: pytest → PASS** (includes offline check: `HOME=<fresh dir> pytest` run)
- [ ] **Step 6: context.md (conventions copied from exoplanet), .gitignore, git init, initial commit**

### Task 2: Notebook `01_data.ipynb` (Labs 1–2 style)

**Files:** Create `notebooks/01_data.ipynb` · Test: AST validation script (JSON valid + every cell parses)

**Interfaces:** Consumes `load_elements()`; Produces `reports/tables/feature_coverage.csv` (row counts) referenced by Task 4 notebooks.

- [ ] **Step 1: Author notebook** — intro echo of spec intent; coverage table; heterogeneity exhibit table (min/median/max per feature); missingness-as-physics section (noble gases); leakage guard assert
- [ ] **Step 2: AST-validate all cells; fix if needed; commit**

### Task 3: `src/study.py` + Notebook `02_kmeans_study.ipynb`

**Files:** Create `src/study.py`, `tests/test_study.py`, `notebooks/02_kmeans_study.ipynb` (Labs 11 core)

**Interfaces:** Consumes `load_elements()`; Produces `build_feature_matrix(df, radius_col="atomic_radius_rahm", impute=True) -> (X, feature_names)`; `elbow_silhouette(X, k_range=range(2,13), seeds=(0,1,2)) -> pd.DataFrame` (rows: k, inertia_mean, inertia_std, sil_mean, sil_std); `fit_labels(X, k, seed) -> (labels, centroids, inertia)`.

- [ ] **Step 1: Failing tests** — leakage guard (`feature_names ∩ {group_id, period, block} = ∅`); shape (n=118, k-features); raw-vs-standardized inertia gap documented as `raw_gap_demo()` float > 0
- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement signatures above** (median impute inside Pipeline; KMeans(n_init=10, random_state=seed))
- [ ] **Step 4: pytest → PASS**
- [ ] **Step 5: Author `02_kmeans_study.ipynb`**: raw-units failure exhibit → standardization → elbow + silhouette curves (3 seeds) → silhouette-chosen k* → centroid table/mini heatmap → AST validate → commit

### Task 4: Notebook `03_validation_report.ipynb` + final artifacts

**Files:** Create `notebooks/03_validation_report.ipynb`, `reports/p_report.md`, `README.md`

**Interfaces:** Consumes Task 3 interfaces; Produces `reports/tables/{ari_summary.csv, unlabeled_elements.csv}`, figures to `reports/figures/`.

- [ ] **Step 1: Author notebook 03** — labeled-set ARI at k* (3 seeds mean±std) → **k=18 validation-only companion** + confusion-style crosstab → hydrogen sensitivity cell (ARI with/without H) → unlabeled 28-element table (which cluster they joined) → honest-language wrap-up ("rediscover/reproduce", ARI reported as-is)
- [ ] **Step 2: AST-validate; run notebook end-to-end once locally to persist artifacts; AST-validate again**
- [ ] **Step 3: Write `reports/p_report.md` + `README.md` from notebook outputs; commit**
- [ ] **Step 4: `gh repo create periodic-kmeans --private --source=. --push`; verify remote**

## Provenance notes
- Spec decided-with-user facts: instructor reviews the notebooks (executions committed); k=18 companion sanctioned as validation-only (my flagged call); hydrogen sensitivity = flagged call (default include).
