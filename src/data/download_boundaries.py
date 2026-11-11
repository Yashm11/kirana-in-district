"""Download India district boundaries (Census 2011 vintage) from DataMeet into data/raw/boundaries/.

Run: uv run python -m src.data.download_boundaries

Source: https://github.com/datameet/maps  (folder Districts/Census_2011)
Licence: Creative Commons Attribution 2.5 India. Credit "DataMeet" when showing a map.
Caveats (from DataMeet): not perfect; some northeastern states, Jammu and Kashmir and Jharkhand
may use older boundaries; boundaries are ~2011, so districts created later have no polygon.
The shapefile is four files that GeoPandas reads together (.shp, .shx, .dbf, .prj).
"""

import urllib.request
from pathlib import Path

import geopandas as gpd

from src.utils.log import get_logger
from src.utils.paths import get_dir

log = get_logger(__name__)

BASE = "https://raw.githubusercontent.com/datameet/maps/master/Districts/Census_2011/"
FILES = ["2011_Dist.shp", "2011_Dist.shx", "2011_Dist.dbf", "2011_Dist.prj"]


def boundaries_dir() -> Path:
    """data/raw/boundaries (created if missing)."""
    path = get_dir("raw") / "boundaries"
    path.mkdir(exist_ok=True)
    return path


def main() -> None:
    """Download any missing file, then check that the shapefile opens."""
    out = boundaries_dir()
    for name in FILES:
        target = out / name
        if target.exists() and target.stat().st_size > 0:
            print(f"have      {name}")
            continue
        try:
            urllib.request.urlretrieve(BASE + name, target)
        except OSError as err:
            target.unlink(missing_ok=True)
            raise SystemExit(f"Could not download {BASE + name}: {err}") from err
        print(f"downloaded {name} ({target.stat().st_size / 1e6:.1f} MB)")

    gdf = gpd.read_file(out / "2011_Dist.shp")
    print(f"\nOpened shapefile: {len(gdf)} features, CRS {gdf.crs}")
    print("Columns:", list(gdf.columns))
    print(gdf.drop(columns="geometry").head(3).to_string())


if __name__ == "__main__":
    main()
