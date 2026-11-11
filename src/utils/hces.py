"""Helpers for reading the HCES 2022-23 unit-level files with DuckDB (the files are large).

Everything here follows the layout workbook (data/raw/hces_2022_23/Layout_HCES_2022_23.xlsx):
    Level 03 (HCQ)           one row per household; `b2q2pt1` is the household size (for FDQ)
    Level 05 (Sec 5 and 6)   food items; item code `b5pt1q2`, total consumption value `b5pt1q6`
    Level 06 (Sec 7)         processed food, 7 days; item code `b7pt1q2`, value `b7pt1q4`
    Level 09 (Sec 9-11)      toilet articles, consumables; item code `b9pt1q2`, value `b9pt1q3`
Column meanings come from the column ORDER in each file matching the layout order; the A3 run
prints a sanity check (India population implied by the weights) to catch a wrong mapping.
`mult` is the survey weight in every file. The sample is designed for State/UT estimates, so
spend is only reported at state x rural/urban level (see the "Rider" note, docs/assumptions.md).
"""

import re
from pathlib import Path

import duckdb
import pandas as pd

from src.utils.paths import REPO_ROOT, get_dir, load_config

# level -> (item code column, value column)
LEVEL_COLUMNS: dict[int, tuple[str, str]] = {
    5: ("b5pt1q2", "b5pt1q6"),
    6: ("b7pt1q2", "b7pt1q4"),
    9: ("b9pt1q2", "b9pt1q3"),
}


def hces_raw_dir() -> Path:
    """Folder holding the HCES documents (data/raw/hces_2022_23)."""
    return get_dir("raw") / load_config()["a3"]["hces_dir"]


def find_level_file(level: int) -> Path:
    """Find the unit-level CSV for a level number (file names differ in case and brackets)."""
    pattern = re.compile(rf"^level\s*-\s*0?{level}\b", re.IGNORECASE)
    hits = [p for p in (hces_raw_dir() / "unit_level").glob("*.csv") if pattern.match(p.name)]
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one file for HCES level {level}, found {hits}")
    return hits[0]


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def household_persons() -> pd.DataFrame:
    """Weighted persons and households per state x district x sector.

    Columns: state, district, sector (1 rural, 2 urban), n_households (sample count),
    persons (weighted, = sum(household size x mult) / divisor), households (weighted).
    """
    div = load_config()["hces"]["weight_divisor"]
    f = _sql_path(find_level_file(3))
    return duckdb.sql(
        f"""
        SELECT CAST(state AS INTEGER) AS state, CAST(district AS INTEGER) AS district,
               CAST(sector AS INTEGER) AS sector,
               count(*) AS n_households,
               sum(CAST(b2q2pt1 AS DOUBLE) * CAST(mult AS DOUBLE)) / {div} AS persons,
               sum(CAST(mult AS DOUBLE)) / {div} AS households
        FROM read_csv_auto('{f}')
        GROUP BY 1, 2, 3
        """
    ).df()


def load_item_map() -> pd.DataFrame:
    """config/hces_item_map.csv with level and item code as integers."""
    m = pd.read_csv(REPO_ROOT / "config" / "hces_item_map.csv", dtype={"level": int})
    m["item_code"] = m["item_code"].astype(int)
    return m


def monthly_spend_by_state_sector(item_map: pd.DataFrame) -> pd.DataFrame:
    """Weighted monthly spend (Rs) per target category x state x sector.

    7-day items are scaled by 30/7 to a 30-day month. Returns columns: target_category,
    state, sector, spend_weighted (divided by the same weight divisor as the persons).
    """
    div = load_config()["hces"]["weight_divisor"]
    parts = []
    for level, group in item_map.groupby("level"):
        code_col, value_col = LEVEL_COLUMNS[level]
        f = _sql_path(find_level_file(level))
        con = duckdb.connect()
        con.register("items", group[["item_code", "target_category", "recall_days"]])
        parts.append(
            con.sql(
                f"""
                SELECT i.target_category, CAST(t.state AS INTEGER) AS state,
                       CAST(t.sector AS INTEGER) AS sector,
                       sum(CAST(t.{value_col} AS DOUBLE) * CAST(t.mult AS DOUBLE) * 30.0
                           / i.recall_days) / {div} AS spend_weighted
                FROM read_csv_auto('{f}') t
                JOIN items i ON CAST(t.{code_col} AS INTEGER) = i.item_code
                GROUP BY 1, 2, 3
                """
            ).df()
        )
    out = pd.concat(parts, ignore_index=True)
    return out.groupby(["target_category", "state", "sector"], as_index=False)[
        "spend_weighted"
    ].sum()


def monthly_spend_by_district(item_map: pd.DataFrame) -> pd.DataFrame:
    """EXPLORATORY: weighted monthly spend (Rs) per target category x state x district.

    Same as `monthly_spend_by_state_sector` but grouped by district, rural and urban pooled.
    The HCES data note advises against district-level estimates; use only with a sample-size cut.
    """
    div = load_config()["hces"]["weight_divisor"]
    parts = []
    for level, group in item_map.groupby("level"):
        code_col, value_col = LEVEL_COLUMNS[level]
        f = _sql_path(find_level_file(level))
        con = duckdb.connect()
        con.register("items", group[["item_code", "target_category", "recall_days"]])
        parts.append(
            con.sql(
                f"""
                SELECT i.target_category, CAST(t.state AS INTEGER) AS state,
                       CAST(t.district AS INTEGER) AS district,
                       sum(CAST(t.{value_col} AS DOUBLE) * CAST(t.mult AS DOUBLE) * 30.0
                           / i.recall_days) / {div} AS spend_weighted
                FROM read_csv_auto('{f}') t
                JOIN items i ON CAST(t.{code_col} AS INTEGER) = i.item_code
                GROUP BY 1, 2, 3
                """
            ).df()
        )
    out = pd.concat(parts, ignore_index=True)
    return out.groupby(["target_category", "state", "district"], as_index=False)[
        "spend_weighted"
    ].sum()


def household_category_quantities(item_map: pd.DataFrame) -> pd.DataFrame:
    """One row per household x target category with quantity (g or ml) and value, per 30 days.

    Only items with a `qty_factor` in the item map count (biscuits, noodles and chips have value
    but no quantity in the data, tea cups are counted in numbers). Quantity is reported in kg,
    litres or grams per the item (factor converts to g or ml) and scaled to 30 days like spend.
    Columns: hh_id, state, sector, district, hh_size, mult (weight), target_category, qty_basis,
    qty_base, value_rs. Both quantity and value come from the same rows, so unit prices agree.
    """
    qty_levels = {5: ("b5pt1q2", "b5pt1q5", "b5pt1q6"), 6: ("b7pt1q2", "b7pt1q3", "b7pt1q4")}
    key = "fsu, sector, b1q1pt7, b1q1pt10, b1q1pt11, b1q1pt12"
    hh_file = _sql_path(find_level_file(3))
    items_all = item_map.dropna(subset=["qty_factor"])
    parts = []
    for level, (code_col, qty_col, value_col) in qty_levels.items():
        items = items_all[items_all["level"] == level]
        if items.empty:
            continue
        f = _sql_path(find_level_file(level))
        con = duckdb.connect()
        con.register(
            "items",
            items[["item_code", "target_category", "recall_days", "qty_basis", "qty_factor"]],
        )
        parts.append(
            con.sql(
                f"""
                WITH hh AS (
                    SELECT {key}, CAST(b2q2pt1 AS DOUBLE) AS hh_size,
                           CAST(state AS INTEGER) AS state, CAST(district AS INTEGER) AS district
                    FROM read_csv_auto('{hh_file}')
                )
                SELECT concat_ws('-', t.fsu, t.sector, t.b1q1pt7, t.b1q1pt10, t.b1q1pt11,
                                 t.b1q1pt12) AS hh_id,
                       hh.state, CAST(t.sector AS INTEGER) AS sector, hh.district, hh.hh_size,
                       CAST(t.mult AS DOUBLE) AS mult, i.target_category, i.qty_basis,
                       sum(CAST(t.{qty_col} AS DOUBLE) * i.qty_factor * 30.0 / i.recall_days)
                           AS qty_base,
                       sum(CAST(t.{value_col} AS DOUBLE) * 30.0 / i.recall_days) AS value_rs
                FROM read_csv_auto('{f}') t
                JOIN items i ON CAST(t.{code_col} AS INTEGER) = i.item_code
                JOIN hh ON t.fsu = hh.fsu AND t.sector = hh.sector
                    AND t.b1q1pt7 IS NOT DISTINCT FROM hh.b1q1pt7
                    AND t.b1q1pt10 IS NOT DISTINCT FROM hh.b1q1pt10
                    AND t.b1q1pt11 IS NOT DISTINCT FROM hh.b1q1pt11
                    AND t.b1q1pt12 IS NOT DISTINCT FROM hh.b1q1pt12
                WHERE CAST(t.{qty_col} AS DOUBLE) > 0
                GROUP BY ALL
                """
            ).df()
        )
    out = pd.concat(parts, ignore_index=True)
    keys = ["hh_id", "state", "sector", "district", "hh_size", "mult", "target_category"]
    return out.groupby(keys + ["qty_basis"], as_index=False)[["qty_base", "value_rs"]].sum()
