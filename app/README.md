# Offline explorer

A static, dependency-free interface for the `periodic-kmeans` study. It reads
`data.js` (generated from the committed snapshot) and renders four views:

- **Property space** — the 118 elements on the first two principal components of
  the six-dimensional standardized feature matrix, coloured by K-Means cluster.
- **Periodic table** — the same partition drawn on the table, with the f-block
  pulled out, so the "regions, not columns" result is visible immediately.
- **Clusters** — sizes, member lists, and standardized centroid profiles.
- **Validation** — family ARI at *k*\* and *k* = 18, block ARI at *k* = 5/6/7,
  and silhouette/ARI curves across *k* = 2…12.

Clicking an element opens its card: group, period, block, cluster, and the six
physical properties. The *k* slider is exploratory; *k*\* remains the
silhouette-selected value, and the validation panel says so.

## Run

No build step, no server required:

```bash
open index.html          # macOS
```

If your browser blocks local scripts, serve the folder:

```bash
python -m http.server 8765 --directory app
# http://localhost:8765
```

## Regenerate the payload

```bash
.venv/bin/python scripts/ui_data.py
```

This writes `data.json` (inspectable) and `data.js` (loaded by the page). The
payload includes the seed-0 labels for every *k* from 2 to 12, centroids,
inertia, silhouette, family/block ARI, PCA coordinates, the periodic-table
positions, and the summary metrics.

## Integrity notes

- `group_id`, `period`, and `block` are attached for display and post-hoc
  validation only. They never enter the feature matrix.
- The page does not fit a model. All partitions are computed by `src/study.py`
  and exported; changing *k* only switches between precomputed partitions.
- The PCA view is display-only. The clustering uses all six standardized
  features; the two components explain about 71% of the variance.
- Missing values are median-imputed inside the pipeline, as in the notebooks.
  The element card shows a dash for a missing raw value.
