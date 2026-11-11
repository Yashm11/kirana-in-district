"""A5: whitespace (opportunity) score and recommendations table.

Run: uv run python -m src.analytics.a5_whitespace     (or `make a5`)

Reads : data/processed/district_clusters.parquet (A4), A2 tables (entry_price_points.csv,
        ladder_gaps.csv, pack_premium.csv), config.yaml -> opportunity.spend_columns
Writes: data/processed/opportunity_index.csv   cluster x category scores
        data/processed/recommendations.csv     the final table
        reports/figures/opportunity_heatmap.png

Opportunity index (a RELATIVE score, NOT rupees of sales)
    benchmark_spend_pc = highest cluster spend per capita in that category
    opportunity_index  = (benchmark_spend_pc - cluster_spend_pc) x cluster population
Read it as "how much room this cluster has to grow towards the best cluster, scaled by how
many people live in it". Only the ranking is meaningful; the number has no currency unit.
Spend is state-level (see A3), so clusters differ in spend only through their states' mix and
their urban share. This is a ROUGH signal; see docs/assumptions.md.

recommendations.csv schema
    cluster                 int    cluster label from A4
    category                str    target category
    suggested_pack_size     str    e.g. "100 g" (from A2 ladders)
    suggested_price_point   float  Rs, from the category entry prices (A2)
    opportunity_index       float  relative score above
    relative_score          float  0-100 within the category (100 = biggest gap)
    priority_rank           int    1 = best opportunity within the cluster
    confidence              str    "low" when evidence is thin (few ladders, no direct prices)
    price_basis             str    what the price point is based on
    rationale               str    one sentence naming the evidence used
"""

import argparse

import numpy as np
import pandas as pd

from src.utils.paths import get_dir, load_config


def cluster_category_spend(
    districts: pd.DataFrame, spend_columns: dict[str, str], population_col: str = "population_est"
) -> pd.DataFrame:
    """Population-weighted spend per capita for each cluster x category.

    districts: one row per district with `cluster`, `population_col` and the spend columns.
    spend_columns: {target category: column name holding its spend per capita}.
    Returns columns: cluster, category, spend_per_capita, population.
    """
    rows = []
    for category, col in spend_columns.items():
        d = districts.dropna(subset=[col, population_col])
        for cluster, g in d.groupby("cluster"):
            pop = g[population_col].sum()
            rows.append(
                {
                    "cluster": cluster,
                    "category": category,
                    "spend_per_capita": (g[col] * g[population_col]).sum() / pop,
                    "population": pop,
                }
            )
    return pd.DataFrame(rows)


def opportunity_index(spend: pd.DataFrame) -> pd.DataFrame:
    """Add benchmark, opportunity_index and a 0-100 relative score to a cluster x category table.

    spend: columns cluster, category, spend_per_capita, population.
    Added columns: benchmark_cluster, benchmark_spend_pc, opportunity_index, relative_score
    (relative_score = index / largest index in the same category x 100).
    """
    out = spend.copy()
    best = out.loc[out.groupby("category")["spend_per_capita"].idxmax(), ["category", "cluster"]]
    out = out.merge(
        best.rename(columns={"cluster": "benchmark_cluster"}), on="category", how="left"
    )
    out["benchmark_spend_pc"] = out.groupby("category")["spend_per_capita"].transform("max")
    out["opportunity_index"] = (out["benchmark_spend_pc"] - out["spend_per_capita"]) * out[
        "population"
    ]
    top = out.groupby("category")["opportunity_index"].transform("max")
    out["relative_score"] = (out["opportunity_index"] / top.where(top > 0) * 100).fillna(0)
    return out


NICE_SIZES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500,
              750, 1000, 2000, 5000]  # fmt: skip
LOW_SPEND_RATIO = 0.8  # a cluster below 80% of the benchmark spend counts as "lower spend"


def round_pack_size(x: float) -> float:
    """Round a pack size to the nearest common retail size (in g or ml)."""
    return float(min(NICE_SIZES, key=lambda n: abs(np.log(n) - np.log(x))))


def format_pack(size: float, unit: str) -> str:
    """Human text such as '100 g' or '1 kg'-style sizes kept in base units: '1000 g'."""
    return f"{size:g} {unit}"


def category_evidence(
    entry_points: pd.DataFrame, gaps: pd.DataFrame, ladders: pd.DataFrame
) -> pd.DataFrame:
    """Per category facts from A2 that the rules need.

    Returns one row per category: unit, entry_p25, entry_median (Rs, brands' cheapest price),
    entry_pack (typical smallest pack), gap_pack (a pack inside the typical ladder gap, or NaN),
    n_gaps, n_steps, premium_x.
    """
    rows = []
    for category, ep in entry_points.groupby("target_category"):
        lad = ladders[ladders["target_category"] == category]
        row = {
            "category": category,
            "entry_p25": ep["entry_price_rs"].quantile(0.25),
            "entry_median": ep["entry_price_rs"].median(),
            "unit": np.nan, "entry_pack": np.nan, "gap_pack": np.nan,
            "n_gaps": 0, "n_steps": 0, "premium_x": np.nan, "n_ladders": 0,
        }  # fmt: skip
        if not lad.empty:
            unit = lad["pack_basis"].value_counts().idxmax()  # unit with the most ladders
            lad = lad[lad["pack_basis"] == unit]
            g = gaps[(gaps["target_category"] == category) & (gaps["pack_basis"] == unit)]
            row.update(
                unit=unit,
                entry_pack=round_pack_size(lad["smallest_pack"].median()),
                n_steps=len(g),
                n_ladders=len(lad),
                n_gaps=int(g["is_gap"].sum()),
                premium_x=lad["pack_premium_x"].median(),
            )
            big = g[g["is_gap"]]
            if len(big):
                mid = np.sqrt(big["from_pack"].median() * big["to_pack"].median())
                row["gap_pack"] = round_pack_size(mid)
        rows.append(row)
    return pd.DataFrame(rows)


def price_near_size(
    price_ladders: pd.DataFrame, category: str, unit: str, size: float, quantile: float
) -> tuple[float, str]:
    """Price (Rs) for a pack of `size` g/ml from what brands actually charge.

    Uses the selling prices of packs within 1.5x of `size` in that category (needs 3 or more);
    otherwise unit price x size. Returns (price, short description of the basis).
    """
    p = price_ladders[
        (price_ladders["target_category"] == category) & (price_ladders["pack_basis"] == unit)
    ]
    near = p[p["pack_total"].between(size / 1.5, size * 1.5)]
    if len(near) >= 3:
        return float(near["selling_price"].quantile(quantile)), (
            f"{len(near)} observed packs near {size:g} {unit}"
        )
    if len(p):
        return float(p["unit_price"].quantile(quantile) * size / 100), "unit price x pack size"
    return np.nan, ""


def build_recommendations(
    opportunity: pd.DataFrame,
    entry_points: pd.DataFrame,
    gaps: pd.DataFrame,
    ladders: pd.DataFrame,
    price_ladders: pd.DataFrame,
) -> pd.DataFrame:
    """Turn opportunity scores plus A2 evidence into the recommendations table (schema above).

    Rules (simple on purpose, logged in docs/assumptions.md):
      * A cluster is "lower spend" in a category when its spend per person is under 80% of the
        benchmark cluster's.
      * suggested_price_point: what brands charge for packs near the suggested size (within
        1.5x): the 25th percentile of those prices for lower-spend clusters, the median for the
        rest, so price and pack size agree. With fewer than 3 such packs it is unit price x pack
        size. Categories with no weight/volume ladder use brands' entry prices instead.
      * suggested_pack_size: lower-spend clusters get the typical smallest pack (the "entry
        pack"). Other clusters get a pack inside the typical flagged ladder gap if the category
        has one (the missing rung), otherwise the entry pack.
    `ladders` is A2's pack_premium.csv (one row per brand x subcategory ladder) and
    `price_ladders` is A2's price_ladders.csv (one row per brand x subcategory x pack size).
    `confidence` is "low" when the category has fewer than 4 ladders or no direct price evidence.
    """
    ev = category_evidence(entry_points, gaps, ladders)
    df = opportunity.merge(ev, on="category", how="left")
    df["spend_ratio"] = df["spend_per_capita"] / df["benchmark_spend_pc"]
    low = df["spend_ratio"] < LOW_SPEND_RATIO

    use_gap = ~low & df["gap_pack"].notna()
    size = np.where(use_gap, df["gap_pack"], df["entry_pack"])
    prices, bases, packs = [], [], []
    for is_low, cat, unit, sz, e25, emed in zip(
        low, df["category"], df["unit"], size, df["entry_p25"], df["entry_median"], strict=True
    ):
        if pd.notna(sz) and isinstance(unit, str):
            price, basis = price_near_size(price_ladders, cat, unit, sz, 0.25 if is_low else 0.5)
            packs.append(format_pack(sz, unit))
        else:
            price, basis = np.nan, ""
            packs.append("n/a (no weight/volume ladder)")
        if pd.isna(price):  # no weight/volume evidence: fall back to entry prices
            price, basis = (e25 if is_low else emed), "brands' entry prices (no pack-size ladder)"
        prices.append(round(price))
        bases.append(basis)
    df["suggested_pack_size"] = packs
    df["suggested_price_point"] = prices
    df["price_basis"] = bases
    df["confidence"] = np.where(
        (df["n_ladders"] < 4) | df["price_basis"].str.contains("no pack-size|unit price x"),
        "low",
        "medium",
    )
    df["priority_rank"] = (
        df.groupby("cluster")["relative_score"].rank(ascending=False, method="first").astype(int)
    )

    def why(r: pd.Series) -> str:
        gap_pct = (1 - r["spend_ratio"]) * 100
        parts = [
            "benchmark cluster" if r["cluster"] == r["benchmark_cluster"]
            else f"spend per person is {gap_pct:.0f}% below cluster {int(r['benchmark_cluster'])}"
        ]  # fmt: skip
        if pd.notna(r["premium_x"]):
            parts.append(f"smallest packs cost {r['premium_x']:.1f}x more per 100 {r['unit']}")
        if r["n_steps"]:
            parts.append(f"{int(r['n_gaps'])} of {int(r['n_steps'])} ladder steps jump over 2.5x")
        tier = "lower quartile" if r["spend_ratio"] < LOW_SPEND_RATIO else "median"
        parts.append(f"price = {tier} of {r['price_basis']}")
        return "; ".join(parts)

    df["rationale"] = df.apply(why, axis=1)
    out = df[
        ["cluster", "category", "suggested_pack_size", "suggested_price_point",
         "opportunity_index", "relative_score", "priority_rank", "confidence", "price_basis",
         "rationale"]
    ]  # fmt: skip
    return out.sort_values(["cluster", "priority_rank"]).reset_index(drop=True)


def plot_heatmap(opp: pd.DataFrame, suffix: str = "", note: str = "") -> str:
    """Heatmap of the relative score (0-100) for each cluster x category."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = opp.pivot(index="category", columns="cluster", values="relative_score")
    grid = grid.loc[grid.max(axis=1).sort_values(ascending=False).index]
    top = opp.sort_values("relative_score", ascending=False).iloc[0]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(grid) + 2.2), facecolor="#fcfcfb")
    im = ax.imshow(grid.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(grid.shape[1]), [f"Cluster {c}" for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [c.replace("_", " ").title() for c in grid.index])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > 55 else "#0b0b0b", fontsize=9)  # fmt: skip
    fig.colorbar(im, ax=ax, label="Relative score (100 = largest gap in that category)")
    ax.set_title(
        f"Largest relative whitespace: {top['category'].replace('_', ' ')} in cluster "
        f"{int(top['cluster'])}{note}",
        loc="left", fontweight="bold", fontsize=12,
    )  # fmt: skip
    fig.tight_layout()
    name = f"opportunity_heatmap{suffix}.png"
    fig.savefig(get_dir("figures") / name, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return name


def spend_for_level(
    districts: pd.DataFrame, spend_cols: dict[str, str], level: str
) -> pd.DataFrame:
    """Return the district table with spend columns for the chosen level.

    `state`: the state-level (blended) spend columns, as built by A3.
    `district`: EXPLORATORY. Uses A3's district-level spend where the district had enough
    sampled households, otherwise falls back to the state-level value.
    """
    if level == "state":
        return districts
    d = districts.copy()
    for col in spend_cols.values():
        d[col] = d[f"{col}_district"].fillna(d[col])
    return d


def main() -> None:
    """Run A5 if its inputs exist; otherwise say what is missing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spend-level", choices=["state", "district"], default=None,
        help="state (default, follows the HCES data note) or district (exploratory)",
    )  # fmt: skip
    args = parser.parse_args()
    cfg = load_config()
    level = args.spend_level or cfg["opportunity"]["spend_level"]
    suffix = "" if level == "state" else "_district_exploratory"
    clusters_path = get_dir("processed") / cfg["files"]["district_clusters"]
    spend_cols = cfg["opportunity"]["spend_columns"]
    if not clusters_path.exists() or not spend_cols:
        print("A5 SKIPPED:")
        if not clusters_path.exists():
            print(f"  {clusters_path.name} not found (run A3 and A4).")
        if not spend_cols:
            print("  config.yaml opportunity.spend_columns is empty. Fill it after A3 runs.")
        return

    out = get_dir("processed")
    districts = spend_for_level(pd.read_parquet(clusters_path), spend_cols, level)
    opp = opportunity_index(cluster_category_spend(districts, spend_cols))
    opp.to_csv(out / f"opportunity_index{suffix}.csv", index=False)
    recs = build_recommendations(
        opp,
        pd.read_csv(out / cfg["files"]["entry_price_points"]),
        pd.read_csv(out / cfg["files"]["ladder_gaps"]),
        pd.read_csv(out / cfg["files"]["pack_premium"]),
        pd.read_csv(out / cfg["files"]["price_ladders"]),
    )
    recs_name = cfg["files"]["recommendations"].replace(".csv", f"{suffix}.csv")
    recs.to_csv(out / recs_name, index=False)
    note = "" if level == "state" else " (EXPLORATORY: district-level spend)"
    chart = plot_heatmap(opp, suffix, note)
    print(f"\n=== A5 summary (spend level: {level}) ===")
    print(f"{len(recs)} recommendations -> {recs_name}; chart {chart}.")
    print("opportunity_index is a RELATIVE score, not rupees of sales.")
    cols = ["cluster", "category", "suggested_pack_size", "suggested_price_point",
            "relative_score", "priority_rank", "confidence"]  # fmt: skip
    print(recs.groupby("cluster").head(3)[cols].round(0).to_string(index=False))


if __name__ == "__main__":
    main()
