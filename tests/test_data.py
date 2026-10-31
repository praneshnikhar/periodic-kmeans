import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import load_elements


def test_shape():
    assert len(load_elements()) == 118


def test_feature_columns_present():
    df = load_elements()
    need = {
        "atomic_weight", "density", "atomic_radius", "atomic_radius_rahm",
        "en_pauling", "electron_affinity",
    }
    assert need <= set(df.columns)


def test_noble_en_missing_is_physics():
    df = load_elements()
    inert = df[df.symbol.isin({"He", "Ne", "Ar", "Kr", "Rn"})]
    assert inert.en_pauling.isna().all(), "truly inert nobles must lack EN"
    # Xe is the exception WITH a value: it forms compounds (XeF2, XeF4...)
    xe = df.loc[df.symbol == "Xe", "en_pauling"].iloc[0]
    assert not pd.isna(xe) and xe > 0


def test_snapshot_offline_load():
    df = load_elements()
    assert (df.atomic_weight > 0).all()
