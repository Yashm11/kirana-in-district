"""Tests for the A7 pack ladder, pack choice and brand recommendations (tiny hand-made tables)."""

import pandas as pd

from src.analytics.a7_warehouse import (
    brand_recommendations,
    district_category,
    nearest_pack,
    pack_ladder,
)

LADDERS = pd.DataFrame(
    {
        "target_category": ["oil"] * 4 + ["tea"],
        "brand": ["A", "A", "B", "B", "A"],
        "pack_basis": ["ml", "ml", "ml", "ml", "g"],
        "pack_total": [500.0, 1000.0, 1000.0, 5000.0, 250.0],
        "n_products": [1, 1, 1, 1, 1],
        "selling_price": [100.0, 180.0, 900.0, 800.0, 120.0],  # B's 1000 ml is ghee, not oil
        "unit_price": [20.0, 18.0, 90.0, 16.0, 48.0],
    }
)


def test_pack_ladder_rises_with_pack_size() -> None:
    out = pack_ladder(LADDERS)
    oil = out[out["target_category"] == "oil"].sort_values("pack_total")
    assert oil["pack_total"].tolist() == [500.0, 1000.0, 5000.0]
    assert oil["price"].is_monotonic_increasing  # one median unit price, so no 1 L below 500 ml


def test_nearest_pack_uses_log_distance() -> None:
    packs = pd.DataFrame({"pack_total": [100, 1000], "price": [10.0, 100.0]})
    assert nearest_pack(packs, 30.0)["pack_total"] == 100  # 3x from 10, but 3.3x from 100
    assert nearest_pack(packs, 40.0)["pack_total"] == 1000


def test_district_category_gives_poorer_districts_smaller_packs() -> None:
    dist = pd.DataFrame(
        {
            "poly_id": ["d1", "d2"],
            "state": ["S", "S"],
            "district": ["Poor", "Rich"],
            "population_est": [1000.0, 1000.0],
            "spend_pc_oil": [10.0, 80.0],
        }
    )
    out = district_category(dist, pack_ladder(LADDERS), ["oil"], household_size=4.0, cover=0.5)
    poor, rich = out.set_index("district").loc["Poor"], out.set_index("district").loc["Rich"]
    assert poor["pack_total"] < rich["pack_total"]
    assert rich["score"] > poor["score"]
    assert rich["demand_rs_month"] == 80000


def test_brand_recommendations_flags_brands_with_no_fitting_pack() -> None:
    dc = pd.DataFrame(
        {"poly_id": ["d1", "d2"], "target_category": ["oil", "oil"], "target_price": [90.0, 1.0]}
    )
    rec = brand_recommendations(dc, LADDERS, max_ratio=1.6).set_index(["brand", "poly_id"])
    assert rec.loc[("A", "d1"), "pack_total"] == 500.0  # Rs 100 is closest to the Rs 90 target
    assert rec.loc[("A", "d1"), "fits"]
    assert not rec.loc[("A", "d2"), "fits"]  # nothing near Rs 1
