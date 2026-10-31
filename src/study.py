"""study.py — the K-Means deep dive (Labs 11 core).

Functions used by notebooks 02/03. Everything imputes (median) and standardizes
INSIDE an sklearn Pipeline, so any metric here is leakage-free by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_ALL = [
    "atomic_weight", "density", "atomic_radius", "atomic_radius_rahm",
    "en_pauling", "electron_affinity",
]


def _geometry(df: pd.DataFrame, radius_col: str, impute: bool):
    """Physical feature set (spec approach B): weight, density, radii, EN, electron affinity.
    radius_col: "both" (default) keeps classic AND Rahm radii; a column name uses just that one.
    """
    base = ["atomic_weight", "density", "atomic_radius", "atomic_radius_rahm",
            "en_pauling", "electron_affinity"]
    if radius_col == "both":
        feats = base
    else:
        if radius_col not in base:
            raise KeyError(radius_col)
        feats = [c for c in base if not c.startswith("atomic_radius") or c == radius_col]
    steps: list[tuple[str, object]] = []
    if impute:
        steps.append(("imputer", SimpleImputer(strategy="median")))
        steps.append(("scaler", StandardScaler()))
    pipe = Pipeline(steps)
    X = pipe.fit_transform(df[feats].to_numpy(float))
    return X, feats, pipe


def build_feature_matrix(df: pd.DataFrame, radius_col: str = "both", impute: bool = True):
    """(X standardized float array, feature names list). n=118 rows are never dropped."""
    X, feats, _ = _geometry(df, radius_col, impute)
    return X, feats


def raw_share_demo(df: pd.DataFrame) -> dict[str, float]:
    """How much of the raw-unit pairwise distance each feature owns (imputed, UNscaled).

    This is the 'raw units are a trap' exhibit: on real values the dense-scale
    columns (density, weight) own almost the entire geometry.
    """
    Xi = SimpleImputer(strategy="median").fit_transform(df[FEATURES_ALL].to_numpy(float))
    totals = np.zeros(Xi.shape[1])
    for f in range(Xi.shape[1]):
        col = Xi[:, f:f + 1]
        totals[f] = ((col - col.T) ** 2).sum()
    shares = totals / totals.sum()
    return {c: float(s) for c, s in zip(FEATURES_ALL, shares)}


def elbow_silhouette(X: np.ndarray, k_range=range(2, 13), seeds=(0, 1, 2)) -> pd.DataFrame:
    """Per-k mean±std of inertia and silhouette across seeds (all on standardized X)."""
    rows = []
    for k in k_range:
        inr, sil = [], []
        for seed in seeds:
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
            inr.append(km.inertia_)
            sil.append(silhouette_score(X, km.labels_))
        rows.append({"k": k, "inertia_mean": np.mean(inr), "inertia_std": np.std(inr),
                     "sil_mean": np.mean(sil), "sil_std": np.std(sil)})
    return pd.DataFrame(rows).set_index("k")


def fit_labels(X: np.ndarray, k: int, seed: int):
    """Final descriptive fit on the standardized matrix → (labels, centroids, inertia)."""
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    return km.labels_, km.cluster_centers_, float(km.inertia_)
