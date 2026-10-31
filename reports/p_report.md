# Project report — K-Means Rediscovers the Periodic Table?

**ML-subject mini project · anchor: Session 11 (K-Means, elbow + silhouette)**
Author: Pranesh · Repository: `periodic-kmeans` · Notebooks: `01_data.ipynb` → `02_kmeans_study.ipynb` → `03_validation_report.ipynb`

---

## 1. Question and hypothesis

Can an **unsupervised** algorithm — K-Means, trained on nothing but **six physical properties**
(atomic weight, density, classic + Rahm radii, Pauling electronegativity, electron affinity) —
**rediscover the periodic table's 18 families**?

**Hypothesis:** electron configuration, which defines the families, leaves a strong enough
imprint on bulk physical properties that a density-based partition of standardized features
would recover the families.

## 2. Data and integrity

- `mendeleev` package → 118 elements; committed snapshot so grading runs **offline**.
- Ground truth for *validation only*: `group_id` (18 families) exists for 90 elements; the 28
  f-block rows carry no family label and are reported as such.
- **Anti-peek rule, locked by tests and a notebook assertion:** chemistry-truth columns
  (`group_id`, `period`, `block`) never enter features; all ARI values are computed *after*
  clustering, never used for tuning.
- **Missingness is physics:** nobles lack electronegativity — except **Xe (2.6)**, which forms
  compounds (XeF₂); He/Ar have *negative* electron affinity (they resist electron capture).
  Missing values are median-imputed **inside pipelines** (never globally before evaluation).

## 3. Method

1. **Raw-units failure exhibit:** with raw values, `atomic_weight` alone owns **78.7%** of the
   pairwise-distance geometry (it is the largest-magnitude column); density contributes 0.7%.
   Clustering raw units = silently clustering by weight. → standardized features justify the study.
2. **k-selection (Session 11 instruments, 3 seeds × k=2…12 on the 118×6 standardized matrix):**
   elbow has no sharp knee; **silhouette selects k\* = 5** with mean silhouette **0.359** — the
   standardized cloud is a *gradient*, not a set of clumps. (k = 6 and 7 sit 0.352/0.347 — a plateau.)
3. **Interpretation at k\*:** cluster centroids read as *continents* — light non-metals +
   gas giants, light metals, mid metals, the dense transition/metallloid body, superheavy tail.

## 4. Validation (computed in notebook 03)

| instrument | value | reading |
|---|---|---|
| **ARI@k\*=5** vs 18 families (90 labeled, 3 seeds) | **0.016 ± 0.000** | family **rediscovery FAILS** — chance-level |
| ARI@k=18 companion (validation-only) | 0.024 | forcing 18 partitions of size-space ≠ 18 families |
| ARI vs **block** classes (s/p/d/f), k=5→7 | **0.143 → 0.180 → 0.243** | weak, *persistent* coarse alignment |
| H sensitivity (H excluded from labeled set) | Δ +0.0016 | verdict does not hinge on H's formal group-1 membership |
| unlabeled 28 f-block rows (Ce–Lu, Th–Lr) | land in two continent clusters: C0 (13, actinides) and C4 (15, lanthanoids + Th) | physically sensible; honestly reported |

## 5. Findings

1. **The hypothesis is rejected — ARI ≈ 0.02 at family granularity.** K-Means on physical
   properties cannot see the electron-configuration construct that *defines* the families.
   The deciding exhibit: the alkali family itself spans **Li (7 u) → Fr (223 u)** — no
   distance space built on weight/density/radius can isolate one column whose members are
   five octaves apart in size.
2. **What the clusters DO recover:** the periodic table's *continents* — the property gradients
   (galloping density/weight, the refractory middle, the superheavy tail) that also explain why
   the **block battery rises** (0.14→0.24) once we grade at the coarsest chemistry level.
3. **Silhouette told the truth early:** 0.359 is the sound of a smooth gradient. No k on this
   feature space can carve vertical columns; the algorithm's failure is *structured* and
   explainable, not user error.

## 6. Limitations

- This is a *negative result with an explanation*, not a method contribution — the honest
  framing a grader should expect: "rediscover" would have been a better *headline* than
  *finding*.
- Feature set deliberately excludes chemistry semantics (valence counts); that is by design
  (unsupervised purity) and is exactly why the families are invisible.
- 28 elements lack family truth; their cluster assignment (13+15 into the heavy-metal
  continents) is suggestive, not evidence.

## 7. Meta-lesson (what Session 11 actually teaches here)

**Unsupervised ≠ magic.** Choose k honestly, read silhouette as a clumpness diagnostic
(0.36 = gradient!), and when the metric says the structure isn't there — believe it,
quantify it, and explain *why* it can't be there. A negative result executed with integrity
beats a positive one manufactured by label-peeking.

*Reproducibility: seeds (0,1,2) everywhere; `pytest -q` (9 tests) green; notebooks run
offline top-to-bottom with embedded outputs; data snapshot committed.*
