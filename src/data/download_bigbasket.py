"""A0: download the BigBasket products dataset from Kaggle into data/raw/bigbasket/.

Run: uv run python -m src.data.download_bigbasket
Needs a Kaggle token in the KAGGLE_API_TOKEN env var or in ~/.kaggle/access_token.
"""

import shutil
import sys
from pathlib import Path

import kagglehub

from src.utils.log import get_logger
from src.utils.paths import get_dir, load_config

log = get_logger(__name__)

DATASET = "chinmayshanbhag/big-basket-products"


def download() -> Path:
    """Download the dataset (kagglehub caches it), copy files to data/raw/bigbasket/."""
    cfg = load_config()
    out_dir = get_dir("raw") / cfg["files"]["bigbasket_raw_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cache_path = Path(kagglehub.dataset_download(DATASET))
    except Exception as err:  # kagglehub raises several error types for auth problems
        log.error("Download failed: %s", err)
        log.error(
            "What to do: (1) make sure a Kaggle token is set: `export KAGGLE_API_TOKEN=...` "
            "or save it in ~/.kaggle/access_token; (2) open https://www.kaggle.com/datasets/%s "
            "in your browser while logged in and accept any terms shown; (3) re-run.",
            DATASET,
        )
        sys.exit(1)
    print("Path to dataset files:", cache_path)

    # Copy (not move) so the kagglehub cache stays valid
    for src in cache_path.rglob("*"):
        if src.is_file():
            shutil.copy2(src, out_dir / src.name)

    print(f"\nFiles in {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    return out_dir


if __name__ == "__main__":
    download()
