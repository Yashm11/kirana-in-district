"""Download the OpenStreetMap India extract (Geofabrik) into data/raw/osm/.

Run: uv run python -m src.data.download_osm

Source: https://download.geofabrik.de/asia/india.html  (about 1.7 GB, updated daily)
Licence: Open Database Licence (ODbL), credit "(c) OpenStreetMap contributors".
Used for shop counts per district (A3c). The public Overpass API is not used on purpose: it is
meant for small interactive queries, not a whole-country sweep.
"""

import urllib.request
from pathlib import Path

from src.utils.log import get_logger
from src.utils.paths import get_dir

log = get_logger(__name__)

URL = "https://download.geofabrik.de/asia/india-latest.osm.pbf"
USER_AGENT = "kirana-in-district/0.1 (github.com/Yashm11/kirana-in-district)"


def osm_dir() -> Path:
    """data/raw/osm (created if missing)."""
    path = get_dir("raw") / "osm"
    path.mkdir(exist_ok=True)
    return path


def main() -> None:
    """Download the extract unless it is already there."""
    target = osm_dir() / "india.osm.pbf"
    if target.exists() and target.stat().st_size > 0:
        print(f"have      {target.name} ({target.stat().st_size / 1e9:.2f} GB)")
        return
    tmp = target.with_suffix(".part")
    req = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    except OSError as err:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Could not download {URL}: {err}") from err
    tmp.rename(target)
    print(f"downloaded {target.name} ({target.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
