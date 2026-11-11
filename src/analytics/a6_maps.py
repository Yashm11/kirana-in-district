"""A6: interactive state and district maps.

Run: uv run python -m src.analytics.a6_maps     (or `make a6`)

Reads : data/processed/district_clusters.parquet (A4), opportunity_index.csv (A5),
        data/raw/boundaries/2011_Dist.shp (src/data/download_boundaries.py)
Writes: data/processed/map_districts.geojson, map_states.geojson   simplified polygons
        data/processed/map_districts.csv, map_states.csv           one row per polygon
        data/processed/map_match_report.csv                       NFHS-5 districts: drawn or not
        reports/maps/state_map.html, district_map.html            standalone interactive maps
        (the HTML pulls plotly.js from a CDN, so it needs internet to open)

Reading the maps
    * The boundary file is Census 2011, so districts created later (many in Telangana, Gujarat,
      Chhattisgarh, ...) have no polygon. A district is drawn only if its name matches a polygon
      within its state; everything else is listed in map_match_report.csv, never guessed.
    * Spend is state-level, so spend maps are uniform inside a state (apart from urban share).
    * Opportunity maps show the relative score of the district's cluster, not a district estimate.
"""

import json

import pandas as pd

from src.utils.geo import (
    INDICATOR_LABELS,
    aggregate,
    choropleth_figure,
    dissolve_states,
    load_polygons,
    match_to_polygons,
    to_geojson,
)
from src.utils.log import get_logger
from src.utils.paths import REPO_ROOT, get_dir, load_config

log = get_logger(__name__)


def pretty(category: str) -> str:
    """'edible_oil_ghee' -> 'Edible oil ghee'."""
    return category.replace("_", " ").capitalize()


def display_state_names(states: pd.Series) -> pd.Series:
    """Apply config/state_display_names.csv (e.g. fix 'Maharastra')."""
    fix = pd.read_csv(REPO_ROOT / "config" / "state_display_names.csv")
    return states.replace(dict(zip(fix["source_name"], fix["display_name"], strict=True)))


def mean_columns(cfg: dict) -> list[str]:
    """Columns averaged (population-weighted) when districts are combined."""
    spend = [f"spend_pc_{c}" for c in cfg["target_categories"]]
    return [*cfg["a3"]["nfhs5_indicators"], *spend, "urban_share"]


# ---------------------------------------------------------------- frames
def build_district_frame(
    polys, feats: pd.DataFrame, opp: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per district polygon, plus the match report for all NFHS-5 districts.

    Returns (frame, report). Polygons without a matching NFHS-5 district keep NaN metrics.
    """
    matched = match_to_polygons(feats, polys)
    drawn = matched.dropna(subset=["poly_id"])
    agg = aggregate(drawn, "poly_id", mean_columns(cfg), "population_est")
    state_names = (
        feats.assign(state=display_state_names(feats["state"]))
        .drop_duplicates("state_code")
        .set_index("state_code")["state"]
    )
    frame = polys[["poly_id", "poly_district", "poly_state", "state_code"]].merge(
        agg, on="poly_id", how="left"
    )
    frame["state_name"] = frame["state_code"].map(state_names).fillna(frame["poly_state"])
    for cat in cfg["target_categories"]:
        scores = opp[opp["category"] == cat].set_index("cluster")["relative_score"]
        frame[f"opp_{cat}"] = frame["cluster"].map(scores)

    def hover(r: pd.Series) -> str:
        head = f"<b>{r['poly_district']}</b>, {r['state_name']}"
        if pd.isna(r.get("cluster")):
            return head + "<br>No matching NFHS-5 district"
        return (
            f"{head}<br>Cluster {int(r['cluster'])} | population "
            f"{r['population_est'] / 1e6:.2f} M<br>Includes: {r['members']}"
        )

    frame["hover"] = frame.apply(hover, axis=1)
    report = matched[
        ["state", "district", "state_code", "poly_id", "population_est", "cluster"]
    ].copy()
    report["drawn"] = report["poly_id"].notna()
    return frame, report


def build_state_frame(state_polys, feats: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per state polygon, aggregated from ALL districts of that state."""
    agg = aggregate(feats, "state_code", mean_columns(cfg), "population_est")
    agg = agg.rename(columns={"cluster": "dominant_cluster"})
    names = (
        feats.assign(state=display_state_names(feats["state"]))
        .drop_duplicates("state_code")
        .set_index("state_code")["state"]
    )
    frame = state_polys[["state_id", "state_code"]].merge(agg, on="state_code", how="left")
    frame["state_name"] = frame["state_code"].map(names)
    # dominant cluster = the one holding most of the state's population
    shares = [c for c in frame.columns if c.startswith("share_cluster_")]
    frame["dominant_cluster"] = (
        frame[shares].fillna(0).idxmax(axis=1).str.extract(r"(\d+)")[0].astype(float)
    )
    frame["hover"] = frame.apply(
        lambda r: (
            f"<b>{r['state_name']}</b><br>{int(r['n_districts'])} districts | population "
            f"{r['population_est'] / 1e6:.1f} M<br>Dominant cluster {int(r['dominant_cluster'])}"
            if pd.notna(r["n_districts"])
            else f"<b>{r['state_code']}</b><br>No data"
        ),
        axis=1,
    )
    return frame


# ---------------------------------------------------------------- metric menus
def district_metrics(cfg: dict) -> dict[str, dict]:
    """Dropdown entries for the district map (label -> column spec)."""
    m: dict[str, dict] = {
        "Cluster": {"col": "cluster", "kind": "categorical"},
        "Estimated population": {"col": "population_est", "kind": "continuous", "fmt": ",.0f"},
        "Urban share of population": {"col": "urban_share", "kind": "continuous", "fmt": ".0%"},
    }
    for col in cfg["a3"]["nfhs5_indicators"]:
        m[INDICATOR_LABELS.get(col, col)] = {"col": col, "kind": "continuous", "fmt": ".1f"}
    for cat in cfg["target_categories"]:
        m[f"Spend per person per month: {pretty(cat)} (₹, state-level)"] = {
            "col": f"spend_pc_{cat}", "kind": "continuous", "fmt": ",.0f"}  # fmt: skip
    for cat in cfg["target_categories"]:
        m[f"Opportunity: {pretty(cat)} (relative score 0-100, from cluster)"] = {
            "col": f"opp_{cat}", "kind": "continuous", "fmt": ".0f"}  # fmt: skip
    return m


def state_metrics(cfg: dict, n_clusters: int) -> dict[str, dict]:
    """Dropdown entries for the state map."""
    m: dict[str, dict] = {
        "Dominant cluster (by population)": {"col": "dominant_cluster", "kind": "categorical"},
        "Estimated population": {"col": "population_est", "kind": "continuous", "fmt": ",.0f"},
        "Urban share of population": {"col": "urban_share", "kind": "continuous", "fmt": ".0%"},
    }
    for k in range(n_clusters):
        m[f"Population share in cluster {k}"] = {
            "col": f"share_cluster_{k}", "kind": "continuous", "fmt": ".0%"}  # fmt: skip
    for col in cfg["a3"]["nfhs5_indicators"]:
        m[INDICATOR_LABELS.get(col, col)] = {"col": col, "kind": "continuous", "fmt": ".1f"}
    for cat in cfg["target_categories"]:
        m[f"Spend per person per month: {pretty(cat)} (₹)"] = {
            "col": f"spend_pc_{cat}", "kind": "continuous", "fmt": ",.0f"}  # fmt: skip
    return m


def main() -> None:
    """Build the map inputs and the standalone HTML maps."""
    cfg = load_config()
    processed = get_dir("processed")
    clusters_path = processed / cfg["files"]["district_clusters"]
    shp = get_dir("raw") / "boundaries" / "2011_Dist.shp"
    if not clusters_path.exists() or not shp.exists():
        print("A6 SKIPPED:")
        if not clusters_path.exists():
            print(f"  {clusters_path.name} not found (run A3 and A4).")
        if not shp.exists():
            print("  boundary file missing: run `uv run python -m src.data.download_boundaries`.")
        return

    feats = pd.read_parquet(clusters_path)
    opp = pd.read_csv(processed / "opportunity_index.csv")
    n_clusters = int(feats["cluster"].nunique())
    polys = load_polygons()
    state_polys = dissolve_states()

    dist_frame, report = build_district_frame(polys, feats, opp, cfg)
    state_frame = build_state_frame(state_polys, feats, cfg)

    (processed / "map_districts.geojson").write_text(json.dumps(to_geojson(polys, "poly_id")))
    (processed / "map_states.geojson").write_text(json.dumps(to_geojson(state_polys, "state_id")))
    dist_frame.to_csv(processed / "map_districts.csv", index=False)
    state_frame.to_csv(processed / "map_states.csv", index=False)
    report.to_csv(processed / "map_match_report.csv", index=False)

    maps_dir = get_dir("figures").parent / "maps"
    maps_dir.mkdir(exist_ok=True)
    state_fig = choropleth_figure(
        to_geojson(state_polys, "state_id"), state_frame, "state_id",
        state_metrics(cfg, n_clusters), "hover", "India by state: pick a metric", n_clusters,
    )  # fmt: skip
    dist_fig = choropleth_figure(
        to_geojson(polys, "poly_id"), dist_frame, "poly_id",
        district_metrics(cfg), "hover", "India by district: pick a metric", n_clusters,
    )  # fmt: skip
    state_fig.write_html(maps_dir / "state_map.html", include_plotlyjs="cdn")
    dist_fig.write_html(maps_dir / "district_map.html", include_plotlyjs="cdn")

    n_poly = len(polys)
    n_with = int(dist_frame["cluster"].notna().sum())
    not_drawn = report[~report["drawn"]]
    print("\n=== A6 summary ===")
    print(
        f"District polygons: {n_poly}; with NFHS-5 data: {n_with}; "
        f"grey (no data): {n_poly - n_with}"
    )
    print(
        f"NFHS-5 districts drawn: {int(report['drawn'].sum())} of {len(report)} "
        f"({not_drawn['population_est'].sum() / report['population_est'].sum():.1%} of estimated "
        "population is in districts not drawn)"
    )
    print(
        "Not drawn, by state:",
        not_drawn.groupby("state").size().sort_values(ascending=False).head(10).to_dict(),
    )
    print(
        f"State polygons: {len(state_polys)}; "
        f"states with data: {int(state_frame['n_districts'].notna().sum())}"
    )
    print(f"Wrote maps to {maps_dir} and tables to {processed}")


if __name__ == "__main__":
    main()
