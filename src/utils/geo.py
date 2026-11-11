"""Boundaries, matching districts to polygons, aggregation, and plotly choropleth figures.

Used by the A6 stage (src/analytics/a6_maps.py) and the Streamlit app.

Boundaries: DataMeet Census-2011 districts (641 polygons, see src/data/download_boundaries.py).
Our data has 706 NFHS-5 districts, many created after 2011, so a district is drawn only when it
matches a polygon by name within its state (plus a small hand-made spelling map,
config/boundary_name_map.csv). Nothing is guessed: unmatched districts are listed in the match
report, and polygons without data are drawn grey.

Maps use plotly's tile-free Choropleth, so no external base map (and no third-party border) is
involved; only the DataMeet polygons are drawn.
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils.names import normalise_name
from src.utils.paths import REPO_ROOT, get_dir

SIMPLIFY_DEG = 0.01  # about 1 km: small files, still recognisable shapes
CLUSTER_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
NO_DATA_GREY = "#dddcd6"

INDICATOR_LABELS = {
    "electricity_pct": "Households with electricity (%)",
    "improved_water_pct": "Improved drinking water (%)",
    "improved_sanitation_pct": "Improved sanitation (%)",
    "clean_fuel_pct": "Clean cooking fuel (%)",
    "women_literate_pct": "Women literate (%)",
    "women_10yr_school_pct": "Women with 10+ years of school (%)",
    "female_ever_school_pct": "Females 6+ who ever attended school (%)",
    "pop_below_15_pct": "Population below age 15 (%)",
    "stunted_pct": "Children under 5 stunted (%)",
    "women_low_bmi_pct": "Women with low BMI (%)",
    "women_overweight_pct": "Women overweight or obese (%)",
    "health_insurance_pct": "Households with health insurance (%)",
}


# ---------------------------------------------------------------- boundaries
def load_polygons(simplify: float | None = SIMPLIFY_DEG) -> gpd.GeoDataFrame:
    """District polygons with modern state codes and a name key.

    Columns: poly_id, poly_district, poly_state, state_code, key, geometry. The 2011 file lists
    Telangana's old districts under Andhra Pradesh and Ladakh under Jammu & Kashmir;
    config/boundary_state_fix.csv gives them their current state code.
    """
    g = gpd.read_file(get_dir("raw") / "boundaries" / "2011_Dist.shp")
    g["poly_district"] = g["DISTRICT"].str.strip()
    g["poly_state"] = g["ST_NM"].str.strip()
    fix = pd.read_csv(REPO_ROOT / "config" / "boundary_state_fix.csv")
    g = g.merge(
        fix[["polygon_district", "polygon_state", "modern_state_code"]],
        left_on=["poly_district", "poly_state"],
        right_on=["polygon_district", "polygon_state"],
        how="left",
    )
    g["state_code"] = g["modern_state_code"].fillna(g["ST_CEN_CD"]).astype(int)
    g["key"] = g["poly_district"].map(normalise_name)
    g["poly_id"] = [f"p{i:03d}" for i in range(len(g))]
    if simplify:  # None keeps full resolution (used when merging districts into states)
        g["geometry"] = g.geometry.simplify(simplify, preserve_topology=True).set_precision(0.001)
    return g[["poly_id", "poly_district", "poly_state", "state_code", "key", "geometry"]]


def to_geojson(gdf: gpd.GeoDataFrame, id_col: str) -> dict:
    """GeoJSON dict whose only property is `id_col` (keeps the file small)."""
    return json.loads(gdf[[id_col, "geometry"]].to_json())


def dissolve_states(simplify: float = SIMPLIFY_DEG) -> gpd.GeoDataFrame:
    """One polygon per modern state code.

    Districts are merged at full resolution first and simplified afterwards; simplifying each
    district first leaves thin gaps between neighbours.
    """
    full = load_polygons(simplify=None)
    s = full.dissolve(by="state_code").reset_index()
    s["geometry"] = s.geometry.simplify(simplify, preserve_topology=True).set_precision(0.001)
    s["state_id"] = s["state_code"].astype(str)
    return s[["state_id", "state_code", "geometry"]]


# ---------------------------------------------------------------- matching
def read_boundary_overrides() -> dict[tuple[str, str], str]:
    """{(NFHS state, normalised NFHS district): normalised polygon district}."""
    m = pd.read_csv(REPO_ROOT / "config" / "boundary_name_map.csv")
    return {
        (r.state, normalise_name(r.source_name)): normalise_name(r.polygon_district)
        for r in m.itertuples()
    }


def match_to_polygons(features: pd.DataFrame, polys: gpd.GeoDataFrame) -> pd.DataFrame:
    """Add `poly_id` (NaN when not drawn) to the features by name within the same state.

    features needs: state, district, state_code. A district matches a polygon when the
    normalised names are equal (after the hand-made spelling map) and the state codes agree.
    """
    overrides = read_boundary_overrides()
    f = features.copy()
    f["key"] = [
        overrides.get((s, normalise_name(d)), normalise_name(d))
        for s, d in zip(f["state"], f["district"], strict=True)
    ]
    f["state_code"] = f["state_code"].astype("Int64")
    cand = polys[["poly_id", "key", "state_code"]].assign(
        state_code=lambda d: d["state_code"].astype("Int64")
    )
    return f.merge(cand, on=["key", "state_code"], how="left").drop(columns="key")


# ---------------------------------------------------------------- aggregation
def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Weighted mean that ignores missing values; NaN if nothing is left."""
    ok = values.notna() & weights.notna()
    return (
        float(np.average(values[ok], weights=weights[ok]))
        if ok.any() and weights[ok].sum() > 0
        else np.nan
    )


def aggregate(group_df: pd.DataFrame, by: str, mean_cols: list[str], weight: str) -> pd.DataFrame:
    """One row per `by`: population sum, weighted means of `mean_cols`, member names.

    Also gives `cluster` (cluster of the largest member) and cluster population shares
    `share_cluster_<k>` when a `cluster` column exists.
    """
    rows = []
    for key, g in group_df.groupby(by):
        row = {by: key, "population_est": g[weight].sum(), "n_districts": len(g)}
        row["members"] = ", ".join(g.sort_values(weight, ascending=False)["district"].head(6))
        if len(g) > 6:
            row["members"] += f" (+{len(g) - 6} more)"
        for c in mean_cols:
            row[c] = weighted_mean(g[c], g[weight])
        if "cluster" in g:
            row["cluster"] = int(g.loc[g[weight].idxmax(), "cluster"])
            share = g.groupby("cluster")[weight].sum() / g[weight].sum()
            for k, v in share.items():
                row[f"share_cluster_{int(k)}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def _discrete_scale(n: int) -> list[list]:
    """Stepped colour scale giving each of `n` integer values its own colour."""
    scale = []
    for i in range(n):
        colour = CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]
        scale += [[i / n, colour], [(i + 1) / n, colour]]
    return scale


def choropleth_figure(
    geojson: dict,
    frame: pd.DataFrame,
    id_col: str,
    metrics: dict[str, dict],
    hover_col: str,
    title: str,
    n_clusters: int = 4,
    dropdown: bool = True,
) -> go.Figure:
    """Interactive choropleth with one data trace and (optionally) a metric dropdown.

    frame: one row per polygon (every polygon, so those without data show grey), with `id_col`,
        `hover_col` (HTML text) and the metric columns.
    metrics: {label: {"col": column, "kind": "continuous" | "categorical", "fmt": ".1f"}}.
    The first metric is shown first.
    """
    labels = list(metrics)
    fig = go.Figure()
    fig.add_trace(  # grey base layer: polygons without data stay visible
        go.Choropleth(
            geojson=geojson, locations=frame[id_col], z=np.zeros(len(frame)),
            featureidkey=f"properties.{id_col}", showscale=False, hoverinfo="skip",
            colorscale=[[0, NO_DATA_GREY], [1, NO_DATA_GREY]],
            marker={"line": {"width": 0.3, "color": "#ffffff"}},
        )
    )  # fmt: skip

    def trace_args(label: str) -> dict:
        spec = metrics[label]
        z = frame[spec["col"]].astype(float)
        if spec["kind"] == "categorical":
            return {
                "z": [z.to_numpy()], "colorscale": [_discrete_scale(n_clusters)],
                "zmin": [-0.5], "zmax": [n_clusters - 0.5],
                "hovertemplate": [f"%{{customdata}}<br>{label}: %{{z:.0f}}<extra></extra>"],
                "colorbar.title.text": [label],
                "colorbar.tickvals": [list(range(n_clusters))],
                "colorbar.ticktext": [[f"Cluster {i}" for i in range(n_clusters)]],
            }  # fmt: skip
        lo, hi = float(z.min()), float(z.max())
        fmt = spec.get("fmt", ".1f")
        return {
            "z": [z.to_numpy()], "colorscale": ["Blues"], "zmin": [lo], "zmax": [hi],
            "hovertemplate": [f"%{{customdata}}<br>{label}: %{{z:,{fmt}}}<extra></extra>"],
            "colorbar.title.text": [label],
            "colorbar.tickvals": [None],
            "colorbar.ticktext": [None],
        }  # fmt: skip

    first = trace_args(labels[0])
    fig.add_trace(
        go.Choropleth(
            geojson=geojson, locations=frame[id_col], featureidkey=f"properties.{id_col}",
            z=first["z"][0], zmin=first["zmin"][0], zmax=first["zmax"][0],
            colorscale=first["colorscale"][0], customdata=frame[hover_col],
            hovertemplate=first["hovertemplate"][0],
            colorbar={"title": {"text": labels[0]}, "thickness": 14},
            marker={"line": {"width": 0.3, "color": "#ffffff"}},
        )
    )  # fmt: skip
    if first["colorscale"][0] != "Blues":
        fig.data[1].colorbar.update(tickvals=first["colorbar.tickvals"][0],
                                    ticktext=first["colorbar.ticktext"][0])  # fmt: skip

    if dropdown and len(labels) > 1:
        buttons = [
            {"label": label, "method": "restyle", "args": [trace_args(label), [1]]}
            for label in labels
        ]
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": buttons,
                    "x": 0.0,
                    "y": 1.12,
                    "xanchor": "left",
                    "yanchor": "top",
                    "showactive": True,
                }
            ]  # fmt: skip
        )
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator",
                    bgcolor="rgba(0,0,0,0)")  # fmt: skip
    fig.update_layout(
        title={"text": title, "x": 0.01, "font": {"size": 16}},
        margin={"l": 0, "r": 0, "t": 90, "b": 30}, height=720,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[{
            "text": "Boundaries: DataMeet (CC BY 2.5 India), Census 2011 vintage. Grey = no "
                    "matching NFHS-5 district. Spend is state-level; clusters are a rough group.",
            "xref": "paper", "yref": "paper", "x": 0, "y": -0.02, "showarrow": False,
            "font": {"size": 10, "color": "#7a7971"}, "xanchor": "left",
        }],
    )  # fmt: skip
    return fig
