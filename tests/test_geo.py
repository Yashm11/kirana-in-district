"""Tests for polygon matching, aggregation and the map figure (tiny hand-made tables)."""

import pandas as pd
import pytest

from src.utils.geo import (
    aggregate,
    choropleth_figure,
    match_to_polygons,
    read_boundary_overrides,
)


def test_match_uses_state_and_spelling_map() -> None:
    polys = pd.DataFrame(
        {
            "poly_id": ["p0", "p1", "p2"],
            "key": ["chamrajnagar", "aurangabad", "aurangabad"],
            "state_code": [29, 10, 27],
        }
    )
    feats = pd.DataFrame(
        {
            "state": ["Karnataka", "Bihar", "Maharastra", "Bihar"],
            "district": ["Chamarajanagar", "Aurangabad", "Aurangabad", "Nowhere"],
            "state_code": [29, 10, 27, 10],
        }
    )
    out = match_to_polygons(feats, polys, source="datameet2011")
    assert out["poly_id"].isna().tolist() == [False, False, False, True]  # "Nowhere" is not drawn
    assert out.loc[out["district"] == "Aurangabad", "poly_id"].tolist() == ["p1", "p2"]


def test_aggregate_weighted_mean_and_cluster_shares() -> None:
    df = pd.DataFrame(
        {
            "poly_id": ["a", "a", "b"],
            "district": ["Big", "Small", "Solo"],
            "population_est": [300.0, 100.0, 50.0],
            "cluster": [2, 0, 1],
            "clean_fuel_pct": [80.0, 40.0, 10.0],
        }
    )
    out = aggregate(df, "poly_id", ["clean_fuel_pct"], "population_est").set_index("poly_id")
    assert out.loc["a", "population_est"] == 400
    assert out.loc["a", "clean_fuel_pct"] == pytest.approx(70.0)  # (80*300 + 40*100) / 400
    assert out.loc["a", "cluster"] == 2  # cluster of the largest member
    assert out.loc["a", "share_cluster_2"] == pytest.approx(0.75)
    assert out.loc["a", "members"] == "Big, Small"


def test_choropleth_figure_has_dropdown_and_grey_base() -> None:
    square = lambda x: {  # noqa: E731
        "type": "Feature",
        "properties": {"poly_id": f"p{x}"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x, 0], [x + 1, 0], [x + 1, 1], [x, 1], [x, 0]]],
        },
    }
    geojson = {"type": "FeatureCollection", "features": [square(0), square(2)]}
    frame = pd.DataFrame(
        {"poly_id": ["p0", "p2"], "hover": ["a", "b"], "cluster": [0.0, 1.0], "pop": [10.0, None]}
    )
    metrics = {
        "Cluster": {"col": "cluster", "kind": "categorical"},
        "Population": {"col": "pop", "kind": "continuous", "fmt": ",.0f"},
    }
    fig = choropleth_figure(geojson, frame, "poly_id", metrics, "hover", "t", n_clusters=2)
    assert len(fig.data) == 2  # grey base layer + one data trace
    assert len(fig.layout.updatemenus[0].buttons) == 2
    single = choropleth_figure(
        geojson, frame, "poly_id", {"Cluster": metrics["Cluster"]}, "hover", "t", 2, dropdown=False
    )
    assert not single.layout.updatemenus


def test_lgd_name_map_targets_exist_and_are_unique() -> None:
    """Every hand-made LGD spelling points at a real polygon, and no polygon is claimed twice."""
    from src.utils.geo import DEFAULT_SOURCE, load_polygons
    from src.utils.paths import get_dir

    if not (get_dir("raw") / "boundaries" / "lgd" / "LGD_Districts.parquet").exists():
        pytest.skip("LGD boundaries not downloaded")
    polys = load_polygons(simplify=None, source=DEFAULT_SOURCE)
    assert polys["poly_id"].is_unique
    overrides = read_boundary_overrides(DEFAULT_SOURCE)
    keys_by_state = set(zip(polys["state_code"], polys["key"], strict=True))
    targets = set(overrides.values())
    assert targets <= {k for _, k in keys_by_state}
    assert len(targets) == len(overrides)
