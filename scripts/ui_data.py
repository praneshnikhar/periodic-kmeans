"""ui_data.py — build the payload for the offline periodic-kmeans explorer.

The UI is a pure static page: `app/index.html` reads `app/data.js`, which is
generated here. Nothing is computed in the browser, and no chemistry label
enters the feature matrix. `group_id`, `period`, and `block` are attached to
each element only for display and post-hoc validation.

Usage:
    .venv/bin/python scripts/ui_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_elements  # noqa: E402
from study import (  # noqa: E402
    FEATURES_ALL,
    build_feature_matrix,
    fit_labels,
    raw_share_demo,
)

GROUP_NAMES = {
    1: "Alkali metals",
    2: "Alkaline earth metals",
    3: "Scandium group",
    4: "Titanium group",
    5: "Vanadium group",
    6: "Chromium group",
    7: "Manganese group",
    8: "Iron group",
    9: "Cobalt group",
    10: "Nickel group",
    11: "Copper group",
    12: "Zinc group",
    13: "Boron group",
    14: "Carbon group",
    15: "Pnictogens",
    16: "Chalcogens",
    17: "Halogens",
    18: "Noble gases",
}

# Tableau-10 order, matching the notebook's cluster colours for k = 5.
COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#B07AA1", "#76B7B2",
]
BLOCK_COLORS = {"s": "#4C72B0", "p": "#DD8452", "d": "#55A868", "f": "#C44E52"}

K_RANGE = list(range(2, 13))


def _num(value):
    """JSON-safe float: NaN/inf become None."""
    if value is None:
        return None
    v = float(value)
    return None if not np.isfinite(v) else round(v, 6)


def _family_name(row) -> str:
    if pd.notna(row.group_id):
        return GROUP_NAMES.get(int(row.group_id), f"Group {int(row.group_id)}")
    if 58 <= row.atomic_number <= 71:
        return "Lanthanides"
    if 90 <= row.atomic_number <= 103:
        return "Actinides"
    return "Unlabeled"


def _table_position(row) -> tuple[float, float]:
    """Same layout as notebook 03: main table, then two f-block rows."""
    if pd.notna(row.group_id):
        return float(row.group_id), float(8 - row.period)
    if row.period == 6:
        return float(3 + (row.atomic_number - 57)), -0.7
    return float(3 + (row.atomic_number - 89)), -1.7


def build_payload() -> dict:
    df = load_elements()
    X, feats = build_feature_matrix(df)

    pca = PCA(n_components=2, random_state=0)
    P = pca.fit_transform(X)

    labeled = df.group_id.notna().to_numpy()
    y_family = df.loc[labeled, "group_id"].astype(int).to_numpy()

    k_tables: dict[str, dict] = {}
    for k in K_RANGE:
        labels, centroids, inertia = fit_labels(X, k=k, seed=0)
        k_tables[str(k)] = {
            "k": k,
            "labels": [int(v) for v in labels],
            "sizes": [int(v) for v in np.bincount(labels, minlength=k)],
            "centroids": [[_num(v) for v in row] for row in centroids],
            "inertia": _num(inertia),
            "silhouette": _num(silhouette_score(X, labels)),
            "ari_family": _num(adjusted_rand_score(y_family, labels[labeled])),
            "ari_block": _num(adjusted_rand_score(df.block, labels)),
        }

    # The companion run is validation-only; it is not used to choose k.
    labels18, _, _ = fit_labels(X, k=18, seed=0)
    ari18 = adjusted_rand_score(y_family, labels18[labeled])

    # Hydrogen sensitivity at k* = 5.
    non_h = labeled & (df.symbol != "H").to_numpy()
    labels5, _, _ = fit_labels(X, k=5, seed=0)
    ari_with = adjusted_rand_score(y_family, labels5[labeled])
    ari_without = adjusted_rand_score(
        df.loc[non_h, "group_id"].astype(int), labels5[non_h]
    )

    coverage = []
    for col in list(FEATURES_ALL) + ["group_id", "period", "block"]:
        coverage.append(
            {
                "column": col,
                "role": "feature" if col in FEATURES_ALL else "truth",
                "non_null": int(df[col].notna().sum()),
                "pct": round(float(df[col].notna().mean() * 100), 1),
            }
        )

    elements = []
    for _, r in df.iterrows():
        x, y = _table_position(r)
        elements.append(
            {
                "z": int(r.atomic_number),
                "symbol": r.symbol,
                "name": r["name"],
                "group": int(r.group_id) if pd.notna(r.group_id) else None,
                "group_name": _family_name(r),
                "period": int(r.period),
                "block": r.block,
                "weight": _num(r.atomic_weight),
                "density": _num(r.density),
                "radius": _num(r.atomic_radius),
                "radius_rahm": _num(r.atomic_radius_rahm),
                "en": _num(r.en_pauling),
                "ea": _num(r.electron_affinity),
                "pc1": _num(P[len(elements), 0]),
                "pc2": _num(P[len(elements), 1]),
                "tx": x,
                "ty": y,
            }
        )

    meta = {
        "title": "K-Means and the Periodic Table",
        "subtitle": "An offline explorer for 118 elements, six physical properties, and one negative result",
        "commit": "d773b47",
        "date": "2026-11-01",
        "n_elements": int(len(df)),
        "n_labeled": int(labeled.sum()),
        "n_features": len(feats),
        "features": feats,
        "k_star": 5,
        "silhouette_star": _num(k_tables["5"]["silhouette"]),
        "ari_family_star": _num(ari_with),
        "ari_family_k18": _num(ari18),
        "h_with": _num(ari_with),
        "h_without": _num(ari_without),
        "h_delta": _num(ari_without - ari_with),
        "perm_p_family": 0.0605,
        "perm_p_block": 0.00005,
        "raw_shares": {k: round(float(v) * 100, 2) for k, v in raw_share_demo(df).items()},
        "coverage": coverage,
        "pca_variance": [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_],
        "colors": COLORS,
        "block_colors": BLOCK_COLORS,
        "group_names": {str(k): v for k, v in GROUP_NAMES.items()},
    }

    return {"meta": meta, "elements": elements, "k": k_tables}


def write_payload(out_dir: Path | None = None) -> dict:
    payload = build_payload()
    out_dir = out_dir or (ROOT / "app")
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (out_dir / "data.json").write_text(text, encoding="utf-8")
    (out_dir / "data.js").write_text(
        "/* Generated by scripts/ui_data.py — do not edit by hand. */\n"
        f"window.UI_DATA = {text};\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    payload = write_payload()
    print(
        f"wrote app/data.js and app/data.json "
        f"({payload['meta']['n_elements']} elements, {len(payload['k'])} k values)"
    )
