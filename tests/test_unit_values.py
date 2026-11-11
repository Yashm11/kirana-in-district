import pandas as pd
import pytest

from src.analytics.a3b_hces_unit_values import summarise_unit_values, weighted_median


def test_weighted_median_follows_weights() -> None:
    values = pd.Series([1.0, 2.0, 10.0])
    assert weighted_median(values, pd.Series([1, 1, 1])) == 2.0
    assert weighted_median(values, pd.Series([1, 1, 10])) == 10.0
    assert pd.isna(weighted_median(pd.Series([], dtype=float), pd.Series([], dtype=float)))


def test_summarise_unit_values() -> None:
    hh = pd.DataFrame(
        {
            "hh_id": ["a", "b", "c"],
            "state": [1, 1, 1],
            "sector": [1, 1, 1],
            "district": [1, 1, 2],
            "hh_size": [2.0, 4.0, 4.0],
            "mult": [1.0, 1.0, 2.0],
            "target_category": ["rice"] * 3,
            "qty_basis": ["g"] * 3,
            "qty_base": [10000.0, 20000.0, 20000.0],
            "value_rs": [400.0, 800.0, 1000.0],
        }
    )
    totals = pd.DataFrame({"state": [1], "sector": [1], "households": [8.0]})
    out = summarise_unit_values(hh, totals).iloc[0]
    assert out["n_buyer_hh"] == 3
    assert out["buyer_share"] == pytest.approx(4 / 8)
    assert out["median_qty_hh"] == 20000.0
    assert out["median_qty_pp"] == 5000.0
    # (400 + 800 + 2 x 1000) / (10000 + 20000 + 2 x 20000) x 100
    assert out["unit_price_per_100"] == pytest.approx(3200 / 70000 * 100)
