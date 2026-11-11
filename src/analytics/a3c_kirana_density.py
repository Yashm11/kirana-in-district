"""A3c: how many small shops OpenStreetMap knows about in each district.

Run: uv run python -m src.analytics.a3c_kirana_density     (or `make a3c`)

Reads : data/raw/osm/india.osm.pbf (make a0-osm), district polygons (src/utils/geo.py),
        data/processed/map_districts.csv (population estimate per polygon, from A6)
Writes: data/interim/osm_shops.parquet   every shop point (lon, lat, shop tag)
        data/processed/kirana_density.csv   one row per district polygon

Columns
    n_kirana_like   shops tagged convenience, general, kiosk, grocery, variety_store
    n_modern_trade  supermarkets and department stores
    n_shops_all     every `shop=*` object, a rough measure of how well the district is mapped
    kirana_per_100k, kirana_per_100km2, modern_share
OSM coverage is uneven: a low count can mean few shops or few mappers. Treat the density as a
relative, noisy feature, never as a census of shops (docs/assumptions.md).
"""

import geopandas as gpd
import osmium
import pandas as pd

from src.utils.geo import load_polygons
from src.utils.log import get_logger
from src.utils.paths import get_dir, load_config

log = get_logger(__name__)
AREA_CRS = "EPSG:6933"  # equal-area, so km2 are comparable across the country


def extract_shops(pbf: str) -> pd.DataFrame:
    """Every shop node or way in the extract as a point: columns lon, lat, shop.

    Ways (shop buildings) are placed at the mean of their node locations. Relations are skipped:
    they are rare for shops and need a second pass.
    """
    rows = []
    proc = osmium.FileProcessor(pbf).with_locations().with_filter(osmium.filter.KeyFilter("shop"))
    for obj in proc:
        shop = obj.tags["shop"]
        if obj.is_node():
            rows.append((obj.lon, obj.lat, shop))
        elif obj.is_way():
            locs = [n.location for n in obj.nodes if n.location.valid()]
            if locs:
                rows.append(
                    (
                        sum(p.lon for p in locs) / len(locs),
                        sum(p.lat for p in locs) / len(locs),
                        shop,
                    )
                )
    return pd.DataFrame(rows, columns=["lon", "lat", "shop"])


def count_by_polygon(
    shops: pd.DataFrame, polys: gpd.GeoDataFrame, kirana_tags: list[str], modern_tags: list[str]
) -> pd.DataFrame:
    """Shop counts per polygon (columns poly_id, n_kirana_like, n_modern_trade, n_shops_all, km2).

    `polys` needs `poly_id` and a geometry in lon/lat. Shops outside every polygon are dropped.
    """
    pts = gpd.GeoDataFrame(
        shops, geometry=gpd.points_from_xy(shops["lon"], shops["lat"]), crs=polys.crs
    )
    joined = gpd.sjoin(pts, polys[["poly_id", "geometry"]], how="inner", predicate="within")
    joined["is_kirana"] = joined["shop"].isin(kirana_tags)
    joined["is_modern"] = joined["shop"].isin(modern_tags)
    counts = joined.groupby("poly_id").agg(
        n_kirana_like=("is_kirana", "sum"),
        n_modern_trade=("is_modern", "sum"),
        n_shops_all=("shop", "size"),
    )
    out = polys[["poly_id"]].merge(counts, on="poly_id", how="left").fillna(0)
    out["km2"] = polys.to_crs(AREA_CRS).area.to_numpy() / 1e6
    return out


def add_densities(counts: pd.DataFrame, population: pd.Series) -> pd.DataFrame:
    """Add per-100k-people and per-100-km2 columns; `population` is indexed by poly_id."""
    out = counts.copy()
    pop = out["poly_id"].map(population)
    out["population_est"] = pop
    out["kirana_per_100k"] = out["n_kirana_like"] / pop * 1e5
    out["kirana_per_100km2"] = out["n_kirana_like"] / out["km2"] * 100
    out["modern_share"] = out["n_modern_trade"] / (out["n_kirana_like"] + out["n_modern_trade"])
    return out


def main() -> None:
    """Run the A3c stage."""
    cfg = load_config()
    pbf = get_dir("raw") / "osm" / "india.osm.pbf"
    if not pbf.exists():
        raise SystemExit("Missing data/raw/osm/india.osm.pbf. Run `make a0-osm` first.")
    interim = get_dir("interim") / "osm_shops.parquet"
    if interim.exists():
        shops = pd.read_parquet(interim)
        log.info(
            "reusing %s (%d shops); delete it to re-read the extract", interim.name, len(shops)
        )
    else:
        shops = extract_shops(str(pbf))
        shops.to_parquet(interim, index=False)
        log.info("%d shop points -> %s", len(shops), interim.name)

    polys = load_polygons(simplify=None)
    counts = count_by_polygon(shops, polys, cfg["osm"]["kirana_like"], cfg["osm"]["modern_trade"])
    pop = pd.read_csv(get_dir("processed") / "map_districts.csv").set_index("poly_id")[
        "population_est"
    ]
    out = add_densities(counts, pop)
    path = get_dir("processed") / "kirana_density.csv"
    out.to_csv(path, index=False)
    log.info("%d polygons -> %s", len(out), path.name)
    print(out[["n_kirana_like", "n_modern_trade", "n_shops_all", "kirana_per_100k"]].describe())


if __name__ == "__main__":
    main()
