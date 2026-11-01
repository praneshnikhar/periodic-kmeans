"""Contract tests for the offline explorer payload (app/data.js)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ui_data import FEATURES_ALL, build_payload  # noqa: E402


@pytest.fixture(scope="module")
def payload():
    return build_payload()


def test_payload_shape(payload):
    assert payload["meta"]["n_elements"] == 118
    assert len(payload["elements"]) == 118
    assert set(payload["k"]) == {str(k) for k in range(2, 13)}
    for table in payload["k"].values():
        assert len(table["labels"]) == 118
        assert sum(table["sizes"]) == 118
        assert len(table["centroids"]) == table["k"]
    symbols = [e["symbol"] for e in payload["elements"]]
    assert len(symbols) == len(set(symbols)) == 118


def test_features_never_include_truth(payload):
    assert payload["meta"]["features"] == FEATURES_ALL
    assert not set(payload["meta"]["features"]) & {"group_id", "period", "block"}
    roles = {c["column"]: c["role"] for c in payload["meta"]["coverage"]}
    assert roles["group_id"] == roles["period"] == roles["block"] == "truth"


def test_headline_numbers_match_the_study(payload):
    meta = payload["meta"]
    assert meta["k_star"] == 5
    assert meta["silhouette_star"] == pytest.approx(0.3589, abs=5e-4)
    assert meta["ari_family_star"] == pytest.approx(0.0164, abs=5e-5)
    assert meta["ari_family_k18"] == pytest.approx(0.0240, abs=5e-5)
    assert payload["k"]["5"]["sizes"] == [42, 36, 2, 9, 29]
    assert payload["k"]["5"]["ari_block"] == pytest.approx(0.1432, abs=5e-5)
    assert payload["k"]["6"]["ari_block"] == pytest.approx(0.1801, abs=5e-5)
    assert payload["k"]["7"]["ari_block"] == pytest.approx(0.2426, abs=5e-5)
    assert meta["h_delta"] == pytest.approx(0.0016, abs=5e-5)


def test_periodic_layout_is_complete(payload):
    positions = {(e["tx"], e["ty"]) for e in payload["elements"]}
    assert len(positions) == 118
    f_rows = [e for e in payload["elements"] if e["block"] == "f"]
    assert len(f_rows) == 28
    assert {e["ty"] for e in f_rows} == {-0.7, -1.7}
    main = [e for e in payload["elements"] if e["group"] is not None]
    assert len(main) == 90
    assert all(1 <= e["tx"] <= 18 for e in main)
