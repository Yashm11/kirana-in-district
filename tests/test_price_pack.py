"""Tests for the A2 calculations on a tiny hand-made ladder."""

import pandas as pd
import pytest

from src.analytics.a2_price_pack import ladder_gaps, pack_premium


@pytest.fixture
def ladders() -> pd.DataFrame:
    """One brand with 100 g / 200 g / 1000 g packs, and one brand with a single pack."""
    rows = [
        # category, subcategory, brand, basis, pack_total, selling_price, unit_price (Rs per 100 g)
        ("biscuits", "Cookies", "A", "g", 100, 20, 20.0),
        ("biscuits", "Cookies", "A", "g", 200, 36, 18.0),
        ("biscuits", "Cookies", "A", "g", 1000, 150, 15.0),
        ("biscuits", "Cookies", "B", "g", 100, 25, 25.0),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "target_category",
            "subcategory",
            "brand",
            "pack_basis",
            "pack_total",
            "selling_price",
            "unit_price",
        ],
    )
    df["mrp"] = df["selling_price"]
    df["n_products"] = 1
    return df


def test_pack_premium_is_smallest_over_largest(ladders: pd.DataFrame) -> None:
    result = pack_premium(ladders, min_packs=2)
    assert list(result["brand"]) == ["A"]  # brand B has one pack size only
    assert result.iloc[0]["pack_premium_x"] == pytest.approx(20 / 15)


def test_ladder_gap_flags_steps_above_ratio(ladders: pd.DataFrame) -> None:
    gaps = ladder_gaps(ladders, min_packs=2, ratio=2.5)
    assert len(gaps) == 2  # 100->200 and 200->1000
    assert list(gaps["size_ratio"]) == [2.0, 5.0]
    assert list(gaps["is_gap"]) == [False, True]  # only the 5x step is above 2.5x
    assert list(gaps["price_jump_rs"]) == [16, 114]
