"""Streamlit viewer for the kirana-in-district results.

Run: uv run streamlit run src/app/app.py     (or `make app`)

It only READS files the pipeline already wrote (data/processed, reports/figures, docs). Run
`make all` first. Nothing here recalculates the analysis.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run` puts this folder on the path, not the repo root, so add the root for `src.*`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.a6_maps import district_metrics, state_metrics  # noqa: E402
from src.utils.geo import choropleth_figure  # noqa: E402
from src.utils.paths import REPO_ROOT, load_config  # noqa: E402

CFG = load_config()
PROCESSED = REPO_ROOT / CFG["paths"]["processed"]
FIGURES = REPO_ROOT / CFG["paths"]["figures"]
SECTIONS = [
    "Overview",
    "Price-pack (A2)",
    "Districts and clusters (A3, A4)",
    "Recommendations (A5)",
    "Maps (A6)",
    "Assumptions",
]

st.set_page_config(page_title="Kirana to district", page_icon="🛒", layout="wide")


@st.cache_data
def read_table(name: str) -> pd.DataFrame | None:
    """Read a CSV or parquet from data/processed, or None if the stage has not run."""
    path = PROCESSED / name
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def need(name: str, stage: str) -> pd.DataFrame | None:
    """Return a table, or show a hint and return None."""
    df = read_table(name)
    if df is None:
        st.info(f"`{name}` not found. Run `make {stage}` first.")
    return df


def show_image(name: str) -> None:
    """Show a chart from reports/figures if it exists."""
    path = FIGURES / name
    if path.exists():
        st.image(str(path))
    else:
        st.info(f"Chart `{name}` not found. Run `make a2` / `make a5`.")


def overview() -> None:
    st.title("Kirana to district")
    st.markdown(
        "**Question:** which pack size and price point should a packaged-goods brand push in which "
        "type of Indian district? Public data and proxies only; there is no company sales data. "
        "_Independent portfolio project, not affiliated with any brand._"
    )
    st.subheader("Pipeline status")
    files = {
        "A1 products_clean.parquet": "products_clean.parquet",
        "A2 price_ladders.csv": "price_ladders.csv",
        "A2 ladder_gaps.csv": "ladder_gaps.csv",
        "A3 district_features.parquet": "district_features.parquet",
        "A4 district_clusters.parquet": "district_clusters.parquet",
        "A5 recommendations.csv": "recommendations.csv",
    }
    rows = []
    for label, name in files.items():
        df = read_table(name)
        rows.append(
            {"output": label, "rows": None if df is None else len(df), "ready": df is not None}
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.warning(
        "Read the numbers as directions, not forecasts: one retailer's online prices, state-level "
        "spend (HCES guidance), weak clusters (silhouette 0.22) and a relative, not rupee, "
        "opportunity score. See the Assumptions page."
    )


def price_pack() -> None:
    st.header("Price-pack analysis (A2)")
    show_image("pack_premium.png")
    by_cat = need("pack_premium_category.csv", "a2")
    if by_cat is not None:
        st.caption(
            "Median pack premium across brand x subcategory ladders "
            "(smallest / largest pack, per 100 g or ml)"
        )
        st.dataframe(by_cat.round(2), hide_index=True)
    ladders = read_table("price_ladders.csv")
    if ladders is None:
        return
    cats = sorted(ladders["target_category"].unique())
    cat = st.selectbox("Category", cats)
    units = sorted(ladders.loc[ladders["target_category"] == cat, "pack_basis"].unique())
    for unit in units:
        show_image(f"price_ladder_{cat}_{unit}.png")
    tab1, tab2, tab3 = st.tabs(["Ladder gaps", "Entry price points", "Discount depth"])
    with tab1:
        gaps = read_table("ladder_gaps.csv")
        if gaps is not None:
            only_gaps = st.checkbox("Only steps above the gap ratio", value=True)
            g = gaps[gaps["target_category"] == cat]
            st.dataframe(g[g["is_gap"]] if only_gaps else g, hide_index=True)
    with tab2:
        entry = read_table("entry_price_points.csv")
        if entry is not None:
            st.dataframe(entry[entry["target_category"] == cat], hide_index=True)
    with tab3:
        disc = read_table("discount_depth.csv")
        if disc is not None:
            st.dataframe(disc[disc["target_category"] == cat].round(1), hide_index=True)


def districts() -> None:
    st.header("Districts and clusters (A3, A4)")
    profiles = need("cluster_profiles.csv", "a4")
    if profiles is not None:
        st.subheader("Cluster profiles (0 = lowest clean-cooking-fuel share)")
        st.dataframe(profiles.round(1), hide_index=True)
        st.bar_chart(profiles.set_index("cluster")["population"], y_label="Estimated population")
    sens = read_table("cluster_sensitivity.csv")
    if sens is not None:
        st.subheader("Do the clusters change if k or features change?")
        st.caption(
            "ARI versus the baseline: 1 = identical grouping. Under 0.7 is marked not stable."
        )
        st.dataframe(sens.round(2), hide_index=True)
    df = read_table("district_clusters.parquet")
    if df is None:
        return
    st.subheader("District explorer")
    c1, c2 = st.columns(2)
    states = c1.multiselect("State", sorted(df["state"].unique()))
    clusters = c2.multiselect("Cluster", sorted(df["cluster"].unique()))
    view = df
    if states:
        view = view[view["state"].isin(states)]
    if clusters:
        view = view[view["cluster"].isin(clusters)]
    show = ["state", "district", "cluster", "population_est", "population_source", "urban_share"]
    show += list(CFG["clustering"]["feature_columns"][:6])
    st.dataframe(view[[c for c in show if c in view.columns]].round(2), hide_index=True)
    st.caption(
        f"{len(view)} of {len(df)} districts. population_est is relative (HCES survey weights)."
    )


def recommendations() -> None:
    st.header("Recommendations (A5)")
    level = st.radio(
        "Spend level",
        ["state (default)", "district (exploratory)"],
        horizontal=True,
        help="The HCES data note advises against district-level spend, "
        "so the district version is exploratory.",
    )
    district = level.startswith("district")
    if district:
        st.warning(
            "Exploratory: district-level spend. Compare with the default before relying on it."
        )
    suffix = "_district_exploratory" if district else ""
    recs = need(f"recommendations{suffix}.csv", "a5-district" if district else "a5")
    if recs is None:
        return
    show_image(f"opportunity_heatmap{suffix}.png")
    c1, c2 = st.columns(2)
    clusters = c1.multiselect("Cluster", sorted(recs["cluster"].unique()))
    top = c2.slider("Top N categories per cluster", 1, 11, 11)
    view = recs[recs["priority_rank"] <= top]
    if clusters:
        view = view[view["cluster"].isin(clusters)]
    cols = ["cluster", "category", "suggested_pack_size", "suggested_price_point",
            "relative_score", "priority_rank", "confidence", "rationale"]  # fmt: skip
    st.dataframe(view[cols].round(0), hide_index=True)
    st.caption(
        "relative_score is 0-100 within a category (100 = biggest gap), not rupees of sales. "
        "Pack size and price point are directions; confidence 'low' means thin evidence."
    )


@st.cache_data
def read_geojson(name: str) -> dict | None:
    """GeoJSON written by A6 (data/processed), or None."""
    path = PROCESSED / name
    return json.loads(path.read_text()) if path.exists() else None


def maps() -> None:
    st.header("Maps (A6)")
    level = st.radio("Level", ["State", "District"], horizontal=True)
    district = level == "District"
    geojson = read_geojson("map_districts.geojson" if district else "map_states.geojson")
    frame = read_table("map_districts.csv" if district else "map_states.csv")
    if geojson is None or frame is None:
        st.info("Map files not found. Run `make a0` (boundary download) and `make a6` first.")
        return
    n_clusters = int(CFG["clustering"].get("n_clusters_hint", 4))
    metrics = district_metrics(CFG) if district else state_metrics(CFG, n_clusters)
    label = st.selectbox("Metric", list(metrics))
    fig = choropleth_figure(
        geojson, frame, "poly_id" if district else "state_id", {label: metrics[label]},
        "hover", f"{level} view: {label}", n_clusters, dropdown=False,
    )  # fmt: skip
    st.plotly_chart(fig, use_container_width=True)
    if "Spend" in label:
        st.caption(
            "Spend is measured at state level (HCES guidance), so districts in a state look alike."
        )
    if "Opportunity" in label:
        st.caption("Each district shows the relative score of its cluster, not its own estimate.")
    if district:
        report = read_table("map_match_report.csv")
        if report is not None:
            missing = report[~report["drawn"]]
            with st.expander(
                f"{len(missing)} NFHS-5 districts are not drawn (no matching 2011 boundary)"
            ):
                st.caption(
                    "Mostly districts created after 2011. They are in every table, "
                    "just not on this map."
                )
                st.dataframe(
                    missing[["state", "district", "population_est", "cluster"]].round(0),
                    hide_index=True,
                )


def assumptions() -> None:
    st.header("Assumptions and simplifications")
    path = REPO_ROOT / "docs" / "assumptions.md"
    st.markdown(path.read_text() if path.exists() else "docs/assumptions.md not found")


PAGES = {
    SECTIONS[0]: overview,
    SECTIONS[1]: price_pack,
    SECTIONS[2]: districts,
    SECTIONS[3]: recommendations,
    SECTIONS[4]: maps,
    SECTIONS[5]: assumptions,
}

choice = st.sidebar.radio("Section", SECTIONS)
PAGES[choice]()
