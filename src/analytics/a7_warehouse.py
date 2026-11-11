"""A7: DuckDB warehouse and district-level scoring.

Run: uv run python -m src.analytics.a7_warehouse     (or `make a7`)

Reads : data/processed/ district_clusters.parquet, map_match_report.csv, kirana_density.csv,
        products_clean.parquet, price_ladders.csv, hces_unit_values.csv
Writes: app_data/kid.duckdb  slim copy without dim_product, committed for the hosted app
        data/warehouse/kid.duckdb with
          dim_district            706 NFHS-5 districts, polygon id, population, cluster, OSM density
          dim_product             cleaned BigBasket listings
          fact_price_ladder       median price per brand x pack size
          fact_unit_values        HCES monthly quantity and unit price by state x sector
          fact_district_category  district x category: spend, demand, score, target price, pack
          rec_brand_district      district x category x brand: the brand's best-fitting pack

How a pack is chosen (same rule for every category)
    target price = household monthly spend in the category x pack_cover_fraction
    category pack = the pack size whose median price is closest to the target (log distance)
    brand pack    = the brand's own pack closest to the target; `fits` is false when it is more
                    than pack_fit_max_ratio away, which means the brand has no suitable pack
Score = percentile rank of monthly demand (spend per person x population) within the category.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.utils.log import get_logger
from src.utils.paths import REPO_ROOT, get_dir, load_config

log = get_logger(__name__)


def pack_ladder(ladders: pd.DataFrame) -> pd.DataFrame:
    """Standard pack sizes per category, priced at the category's median unit price.

    ladders needs: target_category, pack_basis, pack_total, n_products, unit_price (Rs per 100 g
    or ml). Only the category's most common pack unit is used. Pricing every size at one median
    unit price keeps the ladder rising with pack size; a median price per size would mix products
    (oil and ghee) and can put a 1 litre pack below a 500 ml one. The rounding to whole rupees
    mimics shelf prices. Returns target_category, pack_basis, pack_total, price.
    """
    basis = (
        ladders.groupby(["target_category", "pack_basis"])["n_products"]
        .sum()
        .reset_index()
        .sort_values("n_products", ascending=False)
        .drop_duplicates("target_category")[["target_category", "pack_basis"]]
    )
    main = ladders.merge(basis, on=["target_category", "pack_basis"])
    unit = main.groupby("target_category")["unit_price"].median().rename("median_unit_price")
    out = (
        main[["target_category", "pack_basis", "pack_total"]]
        .drop_duplicates()
        .merge(unit, on="target_category")
    )
    out["price"] = (out["median_unit_price"] * out["pack_total"] / 100).round().clip(lower=1)
    return out.drop(columns="median_unit_price").reset_index(drop=True)


def nearest_pack(packs: pd.DataFrame, target: float) -> pd.Series:
    """The row of `packs` (needs `price`) whose price is closest to `target` in log terms."""
    dist = (np.log(packs["price"]) - np.log(target)).abs()
    return packs.loc[dist.idxmin()]


def district_category(
    dist: pd.DataFrame, ladder: pd.DataFrame, categories: list[str], household_size: float,
    cover: float,
) -> pd.DataFrame:  # fmt: skip
    """One row per district x category with demand, score and the category-level pack."""
    rows = []
    for cat in categories:
        col = f"spend_pc_{cat}"
        d = dist[["poly_id", "state", "district", "population_est", col]].rename(
            columns={col: "spend_pc"}
        )
        d = d.assign(target_category=cat)
        d["hh_monthly_spend"] = d["spend_pc"] * household_size
        d["demand_rs_month"] = d["spend_pc"] * d["population_est"]
        d["score"] = d["demand_rs_month"].rank(pct=True) * 100
        d["rank"] = d["demand_rs_month"].rank(ascending=False, method="min").astype(int)
        d["target_price"] = d["hh_monthly_spend"] * cover
        packs = ladder[ladder["target_category"] == cat]
        if packs.empty:
            rows.append(d)
            continue
        best = d["target_price"].map(lambda t, p=packs: nearest_pack(p, t))
        d["pack_total"] = [b["pack_total"] for b in best]
        d["pack_basis"] = [b["pack_basis"] for b in best]
        d["pack_price"] = [b["price"] for b in best]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def brand_recommendations(
    dc: pd.DataFrame, ladders: pd.DataFrame, max_ratio: float
) -> pd.DataFrame:  # fmt: skip
    """For every district x category x brand, the brand's pack closest to the target price."""
    brand_packs = (
        ladders.groupby(["target_category", "brand", "pack_basis", "pack_total"])["selling_price"]
        .median()
        .rename("price")
        .reset_index()
    )
    out = []
    for (cat, brand), packs in brand_packs.groupby(["target_category", "brand"]):
        prices = packs["price"].to_numpy()
        d = dc[dc["target_category"] == cat][["poly_id", "target_price"]]
        idx = np.abs(np.log(prices)[None, :] - np.log(d["target_price"].to_numpy())[:, None])
        pick = idx.argmin(axis=1)
        ratio = prices[pick] / d["target_price"].to_numpy()
        out.append(
            pd.DataFrame(
                {
                    "poly_id": d["poly_id"].to_numpy(),
                    "target_category": cat,
                    "brand": brand,
                    "pack_total": packs["pack_total"].to_numpy()[pick],
                    "pack_basis": packs["pack_basis"].to_numpy()[pick],
                    "pack_price": prices[pick],
                    "price_ratio": ratio,
                    "fits": (ratio <= max_ratio) & (ratio >= 1 / max_ratio),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


APP_TABLES = [
    "dim_district",
    "fact_price_ladder",
    "fact_unit_values",
    "fact_district_category",
    "rec_brand_district",
]


def export_app_db(full: Path, out: Path) -> None:
    """Copy the tables the dashboard needs into a slim database that is safe to publish.

    Leaves out dim_product (the scraped retailer listings with product names and URLs). What
    remains is district-level and brand-level aggregates, which is also all that HCES-derived
    numbers may be published as.
    """
    out.parent.mkdir(exist_ok=True)
    out.unlink(missing_ok=True)
    con = duckdb.connect(str(out))
    con.execute(f"ATTACH '{full}' AS src (READ_ONLY)")
    for t in APP_TABLES:
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM src.{t}")
    con.execute("DETACH src")
    con.close()


def main() -> None:
    cfg = load_config()
    sc = cfg["scoring"]
    processed = get_dir("processed")
    feats = pd.read_parquet(processed / cfg["files"]["district_clusters"])
    match = pd.read_csv(processed / "map_match_report.csv")[["state", "district", "poly_id"]]
    kd = pd.read_csv(processed / "kirana_density.csv")[
        ["poly_id", "n_kirana_like", "n_modern_trade", "kirana_per_100k", "kirana_per_100km2"]
    ]
    dist = feats.merge(match, on=["state", "district"], how="left").merge(
        kd, on="poly_id", how="left"
    )
    assert dist["poly_id"].notna().all(), "every district needs a polygon id"
    products = pd.read_parquet(processed / cfg["files"]["products_clean"])
    ladders = pd.read_csv(processed / cfg["files"]["price_ladders"])
    unit_values = pd.read_csv(processed / "hces_unit_values.csv")

    ladder = pack_ladder(ladders)
    dc = district_category(
        dist, ladder, cfg["target_categories"], sc["household_size"], sc["pack_cover_fraction"]
    )
    rec = brand_recommendations(dc, ladders, sc["pack_fit_max_ratio"])

    wh = get_dir("processed").parent / "warehouse"
    wh.mkdir(exist_ok=True)
    path = wh / "kid.duckdb"
    path.unlink(missing_ok=True)
    con = duckdb.connect(str(path))
    for name, df in {
        "dim_district": dist,
        "dim_product": products.reset_index(names="product_id"),
        "fact_price_ladder": ladders,
        "fact_unit_values": unit_values,
        "fact_district_category": dc,
        "rec_brand_district": rec,
    }.items():
        con.register("tmp", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp")
        con.unregister("tmp")
        log.info("%s: %d rows", name, len(df))
    con.close()
    print(f"\nWrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    slim = REPO_ROOT / "app_data" / "kid.duckdb"
    export_app_db(path, slim)
    print(f"Wrote {slim} ({slim.stat().st_size / 1e6:.1f} MB), the copy the hosted app reads")


if __name__ == "__main__":
    main()
