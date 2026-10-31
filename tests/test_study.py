import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import load_elements
from study import build_feature_matrix, elbow_silhouette, fit_labels, raw_share_demo


@pytest.fixture(scope="module")
def df():
    return load_elements()


@pytest.fixture(scope="module")
def X_feat(df):
    return build_feature_matrix(df)


def test_leakage_guard(X_feat):
    X, names = X_feat
    assert not (set(names) & {"group_id", "period", "block"})


def test_shape(X_feat):
    X, names = X_feat
    assert X.shape == (118, 6)


def test_max_share_dominates_raw_units(df):
    shares = raw_share_demo(df)
    top = max(shares, key=shares.get)
    assert shares[top] > 0.75  # raw units: ONE column owns ~4/5 of the geometry
    assert shares["density"] < 0.05  # small-magnitude scaled feature: muted, not dominant


def test_elbow_silhouette_frame(X_feat):
    X, _ = X_feat
    tab = elbow_silhouette(X)
    assert list(tab.index) == list(range(2, 13))
    for col in ("inertia_mean", "inertia_std", "sil_mean", "sil_std"):
        assert col in tab.columns

    assert np.isfinite(tab[["inertia_mean", "sil_mean"]].to_numpy()).all()


def test_fit_labels(X_feat):
    X, _ = X_feat
    labels, centroids, inertia = fit_labels(X, k=5, seed=0)
    assert len(labels) == 118
    assert centroids.shape == (5, X.shape[1])
    assert inertia > 0
