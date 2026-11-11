"""A1: clean the raw BigBasket products.

Run: uv run python -m src.analytics.a1_clean_products     (or `make a1`)

Reads : data/raw/bigbasket/BigBasket.csv, config/brand_map.csv, config/category_map.csv
Writes: data/processed/products_clean.parquet   (every row except exact duplicates)
        data/interim/products_unparsed.csv      (rows whose pack size could not be read)

Steps: tidy text -> standardise brands -> drop exact duplicates -> parse pack size ->
compute prices (selling price, MRP, discount %, price per 100 g / 100 ml / per piece) ->
map SubCategory to our target categories -> save -> print a summary.
"""

import pandas as pd

from src.utils.log import get_logger
from src.utils.parsing import (
    PackSize,
    clean_text,
    parse_quantity,
    parse_size_from_name,
    price_per_100,
    price_per_piece,
    standardise_brand,
)
from src.utils.paths import REPO_ROOT, get_dir, load_config

log = get_logger(__name__)

TEXT_COLUMNS = ["ProductName", "Brand", "Quantity", "Category", "SubCategory"]


def load_raw() -> pd.DataFrame:
    """Read the raw BigBasket CSV."""
    cfg = load_config()
    csv_dir = get_dir("raw") / cfg["files"]["bigbasket_raw_dir"]
    csvs = sorted(csv_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV in {csv_dir}. Run `make a0` first.")
    return pd.read_csv(csvs[0])


def tidy_text_and_brands(df: pd.DataFrame) -> pd.DataFrame:
    """Trim/collapse spaces in text columns and standardise brand names."""
    df = df.copy()
    for col in TEXT_COLUMNS:
        df[col] = df[col].map(clean_text)

    brand_map = pd.read_csv(REPO_ROOT / "config" / "brand_map.csv")
    aliases = {
        a.casefold(): b
        for a, b in zip(brand_map["alias"], brand_map["standard_brand"], strict=True)
    }
    df["brand"] = df["Brand"].map(lambda b: standardise_brand(b, aliases))
    return df


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows repeated with the same name, brand, quantity AND prices.

    Rows with the same name/brand/quantity but different prices are kept (they look like
    separate listings) and flagged with `dup_listing`.
    """
    same_all = ["ProductName", "brand", "Quantity", "Price", "DiscountPrice"]
    before = len(df)
    df = df.drop_duplicates(subset=same_all).copy()
    df["dup_listing"] = df.duplicated(["ProductName", "brand", "Quantity"], keep=False)
    return df, before - len(df)


def parse_pack(row: pd.Series) -> PackSize:
    """Parse Quantity; if that fails (and it is not a combo), try the product name."""
    result = parse_quantity(row["Quantity"])
    if result.status == "unparsed":
        from_name = parse_size_from_name(row["ProductName"])
        if from_name.status == "ok":
            return from_name
        return PackSize(None, None, None, None, "unparsed", result.note)
    return result


def add_pack_and_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Add pack-size columns and all price columns."""
    df = df.copy()
    packs = df.apply(parse_pack, axis=1)
    df["pack_basis"] = [p.basis for p in packs]  # g, ml, pc or missing
    df["pack_total"] = [p.total for p in packs]  # total in g / ml / pieces
    df["pack_unit_size"] = [p.unit_size for p in packs]  # size of one unit in a multipack
    df["pack_count"] = [p.count for p in packs]  # units in the multipack
    df["parse_status"] = [p.status for p in packs]  # ok / combo / unparsed
    df["parse_note"] = [p.note for p in packs]
    df["is_combo"] = df["parse_status"] == "combo"

    df["mrp"] = df["Price"]
    df["selling_price"] = df["DiscountPrice"]
    df["discount_pct"] = (df["mrp"] - df["selling_price"]) / df["mrp"] * 100

    per_100 = [
        price_per_100(price, total, basis)
        for price, total, basis in zip(
            df["selling_price"], df["pack_total"], df["pack_basis"], strict=True
        )
    ]
    per_100 = pd.Series(per_100, index=df.index, dtype="float64")
    df["price_per_100g"] = per_100.where(df["pack_basis"] == "g")
    df["price_per_100ml"] = per_100.where(df["pack_basis"] == "ml")
    df["price_per_piece"] = [
        price_per_piece(price, total, basis)
        for price, total, basis in zip(
            df["selling_price"], df["pack_total"], df["pack_basis"], strict=True
        )
    ]
    return df


def map_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Attach target_category using config/category_map.csv (matched on SubCategory)."""
    cmap = pd.read_csv(REPO_ROOT / "config" / "category_map.csv")
    return df.merge(cmap, how="left", left_on="SubCategory", right_on="subcategory").drop(
        columns="subcategory"
    )


def tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to snake_case and put the columns in a readable order."""
    df = df.rename(
        columns={
            "ProductName": "product_name",
            "Quantity": "quantity_raw",
            "Category": "category",
            "SubCategory": "subcategory",
            "Absolute_Url": "url",
            "Brand": "brand_raw",
        }
    )
    cols = [
        "product_name", "brand", "brand_raw", "category", "subcategory", "target_category",
        "quantity_raw", "pack_basis", "pack_total", "pack_unit_size", "pack_count",
        "parse_status", "parse_note", "is_combo",
        "mrp", "selling_price", "discount_pct",
        "price_per_100g", "price_per_100ml", "price_per_piece",
        "dup_listing", "url",
    ]  # fmt: skip
    return df[cols].reset_index(drop=True)


def print_summary(rows_in: int, dupes: int, df: pd.DataFrame) -> None:
    """Short summary: rows in/out, dropped rows and why, % parsed, mapped categories."""
    n = len(df)
    print("\n=== A1 summary ===")
    print(f"Rows in: {rows_in:,}   Rows out: {n:,}")
    print(f"Dropped: {dupes:,} exact duplicate rows (same name, brand, quantity and prices)")
    print(f"Flagged as duplicate listing (different price, kept): {df['dup_listing'].sum():,}")

    print("\nPack size parsing (share of all rows):")
    print(f"  weight (g)   : {(df['pack_basis'] == 'g').mean():6.1%}")
    print(f"  volume (ml)  : {(df['pack_basis'] == 'ml').mean():6.1%}")
    print(f"  pieces (pc)  : {(df['pack_basis'] == 'pc').mean():6.1%}")
    print(f"  combo        : {df['is_combo'].mean():6.1%}  (kept, flagged, no unit price)")
    print(f"  unparsed     : {(df['parse_status'] == 'unparsed').mean():6.1%}  (saved to interim)")
    from_name = (df["parse_note"] == "from_name").sum()
    print(f"  parsed from product name instead of Quantity: {from_name}")
    reasons = df.loc[df["parse_status"] == "unparsed", "parse_note"].value_counts()
    for reason, k in reasons.items():
        print(f"    unparsed because '{reason}': {k}")

    mapped = df[df["target_category"].notna()]
    print(f"\nMapped to a target category: {len(mapped):,} rows ({len(mapped) / n:.1%})")
    table = mapped.groupby("target_category").agg(
        rows=("product_name", "size"),
        brands=("brand", "nunique"),
        pct_parsed=("parse_status", lambda s: (s == "ok").mean() * 100),
        median_per_100g=("price_per_100g", "median"),
        median_per_100ml=("price_per_100ml", "median"),
    )
    print(table.round(1).sort_values("rows", ascending=False).to_string())
    thin = table.index[table["rows"] < 30].tolist()
    if thin:
        print(f"\nThin categories (<30 rows): {', '.join(thin)}")


def main() -> None:
    """Run the whole A1 stage."""
    cfg = load_config()
    df = load_raw()
    rows_in = len(df)

    df = tidy_text_and_brands(df)
    df, dupes = drop_exact_duplicates(df)
    df = add_pack_and_prices(df)
    df = map_categories(df)
    df = tidy_columns(df)

    out = get_dir("processed") / cfg["files"]["products_clean"]
    df.to_parquet(out, index=False)

    bad = df[df["parse_status"] == "unparsed"]
    bad_path = get_dir("interim") / cfg["files"]["products_unparsed"]
    bad[["product_name", "brand", "category", "subcategory", "quantity_raw", "parse_note"]].to_csv(
        bad_path, index=False
    )

    print_summary(rows_in, dupes, df)
    log.info("Saved %s and %s", out, bad_path)


if __name__ == "__main__":
    main()
