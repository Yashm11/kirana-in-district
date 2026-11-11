"""Sanity checks on A1's output. Runs after A1 (`make a1`) and fails loudly if something is off.

Run: uv run python -m src.analytics.check_products_clean

Hard checks (exit code 1 if any fail): required columns, valid prices, discount range,
unit prices only where a size was parsed, target categories match config.
Soft checks (printed as warnings): thin categories, extreme unit prices, unparsed rows.
"""

import sys

import pandas as pd

from src.utils.log import get_logger
from src.utils.paths import get_dir, load_config

log = get_logger(__name__)

REQUIRED = [
    "product_name", "brand", "target_category", "quantity_raw", "pack_basis", "pack_total",
    "parse_status", "is_combo", "mrp", "selling_price", "discount_pct",
    "price_per_100g", "price_per_100ml", "price_per_piece",
]  # fmt: skip


def hard_checks(df: pd.DataFrame, targets: list[str]) -> list[str]:
    """Return a list of failed-check messages (empty list means all passed)."""
    problems = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if (df["selling_price"] <= 0).any() or df["selling_price"].isna().any():
        problems.append("selling_price has zero, negative or missing values")
    if (df["selling_price"] > df["mrp"]).any():
        problems.append("selling_price is above MRP on some rows")
    if not df["discount_pct"].between(0, 100).all():
        problems.append("discount_pct outside 0-100")
    if df.loc[df["pack_basis"].isna(), ["price_per_100g", "price_per_100ml"]].notna().any().any():
        problems.append("unit price present on a row without a parsed pack size")
    if (
        df.loc[df["is_combo"], ["price_per_100g", "price_per_100ml", "price_per_piece"]]
        .notna()
        .any()
        .any()
    ):
        problems.append("combo rows have a unit price")
    if ((df["price_per_100g"] <= 0) | (df["price_per_100ml"] <= 0)).any():
        problems.append("non-positive price per 100 g/ml")
    unknown = set(df["target_category"].dropna()) - set(targets)
    if unknown:
        problems.append(f"target categories not in config: {sorted(unknown)}")
    if (
        df["product_name"].str.contains(r"\s{2,}").any()
        or df["brand"].ne(df["brand"].str.strip()).any()
    ):
        problems.append("untidy whitespace left in product_name or brand")
    return problems


def soft_checks(df: pd.DataFrame) -> list[str]:
    """Return warnings worth a human look."""
    warnings = []
    in_scope = df[df["target_category"].notna()]
    thin = in_scope.groupby("target_category").size()
    thin = thin[thin < 30]
    if len(thin):
        warnings.append(f"thin categories (<30 rows): {thin.to_dict()}")
    unit = in_scope["price_per_100g"].fillna(in_scope["price_per_100ml"])
    extreme = in_scope[unit > unit.quantile(0.99)]
    if len(extreme):
        names = extreme["product_name"].head(3).str[:40].tolist()
        warnings.append(f"{len(extreme)} rows above the 99th percentile unit price, e.g. {names}")
    bad = df[df["parse_status"] == "unparsed"]
    if len(bad):
        warnings.append(f"{len(bad)} unparsed rows (see data/interim/products_unparsed.csv)")
    in_unparsed = in_scope[in_scope["parse_status"] == "unparsed"]
    if len(in_unparsed):
        warnings.append(f"{len(in_unparsed)} unparsed rows are inside target categories")
    return warnings


def main() -> None:
    """Load A1's output, run the checks, print the result."""
    cfg = load_config()
    path = get_dir("processed") / cfg["files"]["products_clean"]
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `make a1` first.")
    df = pd.read_parquet(path)

    problems = hard_checks(df, cfg["target_categories"])
    warnings = soft_checks(df)
    print(f"\n=== A1 output check ({len(df):,} rows) ===")
    for w in warnings:
        print(f"WARN  {w}")
    for p in problems:
        print(f"FAIL  {p}")
    if problems:
        sys.exit(1)
    print("PASS  all hard checks")


if __name__ == "__main__":
    main()
