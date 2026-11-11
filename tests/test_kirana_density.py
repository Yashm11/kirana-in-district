import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from src.analytics.a3c_kirana_density import add_densities, count_by_polygon


def test_count_by_polygon_and_densities() -> None:
    polys = gpd.GeoDataFrame(
        {"poly_id": ["a", "b"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326"
    )
    shops = pd.DataFrame(
        {
            "lon": [0.5, 0.2, 0.9, 1.5, 5.0],
            "lat": [0.5, 0.2, 0.9, 0.5, 5.0],
            "shop": ["general", "kiosk", "supermarket", "bakery", "general"],
        }
    )
    counts = count_by_polygon(shops, polys, ["general", "kiosk"], ["supermarket"]).set_index(
        "poly_id"
    )
    assert counts.loc["a", "n_kirana_like"] == 2
    assert counts.loc["a", "n_modern_trade"] == 1
    assert counts.loc["a", "n_shops_all"] == 3
    assert counts.loc["b", "n_kirana_like"] == 0  # the point at (5, 5) is outside both polygons
    assert counts.loc["b", "n_shops_all"] == 1
    dens = add_densities(counts.reset_index(), pd.Series({"a": 100_000, "b": 50_000})).set_index(
        "poly_id"
    )
    assert dens.loc["a", "kirana_per_100k"] == pytest.approx(2.0)
    assert dens.loc["a", "modern_share"] == pytest.approx(1 / 3)
