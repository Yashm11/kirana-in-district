"""A0 (optional): save a small India subset of Open Food Facts to data/raw/off_india.parquet.

Run: uv run python -m src.data.download_off --include-off [--yes]

Source: Hugging Face dataset `openfoodfacts/product-database`, file `food.parquet`
(about 8 GB). We do NOT download the whole file: DuckDB reads it remotely and only fetches
the columns and row groups it needs, but that can still move several GB. Hence the flag.
Columns below were checked against the real schema (names inspected on 2026-10-25).
"""

import argparse
import sys

import duckdb

from src.utils.log import get_logger
from src.utils.paths import get_dir, load_config

log = get_logger(__name__)

URL = "https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet"

# product_name is a list of {lang, text} structs, so we take the first entry.
QUERY = f"""
SELECT
    code,
    product_name[1].text AS product_name,
    brands,
    categories,
    quantity,
    product_quantity,
    product_quantity_unit,
    countries_tags,
    stores
FROM read_parquet('{URL}')
WHERE list_contains(countries_tags, 'en:india')
"""


def main() -> None:
    """Warn about size, ask for confirmation, then write the India subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-off", action="store_true", help="actually run the download")
    parser.add_argument("--yes", action="store_true", help="skip the size confirmation prompt")
    args = parser.parse_args()

    if not args.include_off:
        print("Skipping Open Food Facts (optional). Pass --include-off to download.")
        return

    print("WARNING: the source file food.parquet is ~8 GB. DuckDB will read it remotely and may")
    print("transfer a few GB while filtering to India. This can take several minutes.")
    if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        sys.exit(0)

    out = get_dir("raw") / load_config()["files"]["off_india_raw"]
    log.info("Querying remote parquet...")
    duckdb.sql(f"COPY ({QUERY}) TO '{out}' (FORMAT parquet)")
    n = duckdb.sql(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    log.info("Saved %d India products to %s (%.1f MB)", n, out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
