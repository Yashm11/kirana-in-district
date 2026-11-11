"""Download the LGD district boundaries into data/raw/boundaries/lgd/.

Run: uv run python -m src.data.download_lgd_boundaries

Source: https://github.com/ramSeraph/indian_admin_boundaries (release "districts"), a GeoParquet
copy of the Local Government Directory (LGD) district layer: 785 polygons, districts created
up to 2023. The repository declares no licence, so the file is not committed (data/raw is
git-ignored) and the map credits the LGD / Survey of India origin.
Caveat: the file draws India's borders as the LGD source does; check the depiction against
Survey of India guidance before publishing a map.
"""

import urllib.request

import geopandas as gpd

from src.utils.log import get_logger
from src.utils.paths import get_dir

log = get_logger(__name__)

URL = (
    "https://github.com/ramSeraph/indian_admin_boundaries/releases/download/"
    "districts/LGD_Districts.parquet"
)


def main() -> None:
    """Download the file if missing, then check that it opens."""
    out = get_dir("raw") / "boundaries" / "lgd"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "LGD_Districts.parquet"
    if target.exists() and target.stat().st_size > 0:
        print("have      LGD_Districts.parquet")
    else:
        try:
            urllib.request.urlretrieve(URL, target)
        except OSError as err:
            target.unlink(missing_ok=True)
            raise SystemExit(f"Could not download {URL}: {err}") from err
        print(f"downloaded LGD_Districts.parquet ({target.stat().st_size / 1e6:.1f} MB)")
    gdf = gpd.read_parquet(target)
    print(f"Opened: {len(gdf)} features, CRS {gdf.crs}")


if __name__ == "__main__":
    main()
