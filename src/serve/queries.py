"""Read-only questions we can ask the warehouse (data/warehouse/kid.duckdb).

Every function takes an open DuckDB connection and returns a DataFrame or a plain dict, so the
dashboard and the language-model tools (Phase 4) answer from exactly the same code. Rows always
carry `poly_id`, the key the map uses to highlight districts.
"""

import os
import re
from pathlib import Path

import duckdb
import pandas as pd

from src.utils.names import normalise_name
from src.utils.paths import REPO_ROOT, get_dir

CATEGORY_WORDS = {
    "biscuits": ["biscuit", "biscuits", "cookies"],
    "namkeen_snacks": ["namkeen", "snack", "snacks", "bhujia", "chips"],
    "noodles_pasta": ["noodle", "noodles", "pasta"],
    "tea": ["tea", "chai"],
    "edible_oil_ghee": ["oil", "ghee", "edible oil"],
    "atta_flour": ["atta", "flour"],
    "rice": ["rice"],
    "spices_masala": ["spice", "spices", "masala"],
    "laundry_detergent": ["detergent", "laundry", "washing powder"],
    "personal_wash": ["soap", "bodywash", "body wash", "handwash", "personal wash"],
    "shampoo": ["shampoo"],
}


# what people type -> the state name as it is spelled in the warehouse (NFHS-5 spellings)
STATE_ALIASES = {
    "delhi": "NCT of Delhi",
    "new delhi": "NCT of Delhi",
    "maharashtra": "Maharastra",
    "orissa": "Odisha",
    "pondicherry": "Puducherry",
    "uttaranchal": "Uttarakhand",
    "jammu and kashmir": "Jammu & Kashmir",
    "j&k": "Jammu & Kashmir",
    "andaman and nicobar islands": "Andaman & Nicobar Islands",
    "andaman and nicobar": "Andaman & Nicobar Islands",
    "dadra and nagar haveli": "Dadra and Nagar Haveli & Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli & Daman and Diu",
}


def resolve_states(con: duckdb.DuckDBPyConnection, names: list[str] | None) -> list[str]:
    """Warehouse spellings for user-typed state names; names that match nothing are dropped."""
    known = {normalise_name(s): s for s in states(con)}
    known.update({normalise_name(a): v for a, v in STATE_ALIASES.items()})
    out = [known.get(normalise_name(n)) for n in names or []]
    return list(dict.fromkeys(s for s in out if s))


def warehouse_path() -> Path:
    """$KID_DB if set, else the full warehouse if built here, else the slim committed copy."""
    if os.environ.get("KID_DB"):
        return Path(os.environ["KID_DB"])
    full = get_dir("processed").parent / "warehouse" / "kid.duckdb"
    slim = REPO_ROOT / "app_data" / "kid.duckdb"
    return full if full.exists() else slim


def connect(path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open the warehouse read-only."""
    path = path or warehouse_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make a7` first.")
    return duckdb.connect(str(path), read_only=True)


def categories(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT target_category FROM rec_brand_district ORDER BY 1"
        ).fetchall()
    ]


def brands(con: duckdb.DuckDBPyConnection, category: str | None = None) -> list[str]:
    """Brands in the price ladders, optionally only those sold in `category`."""
    sql = "SELECT DISTINCT brand FROM fact_price_ladder"
    args: list = []
    if category:
        sql += " WHERE target_category = ?"
        args.append(category)
    return [r[0] for r in con.execute(sql + " ORDER BY 1", args).fetchall()]


def brand_categories(con: duckdb.DuckDBPyConnection, brand: str) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT target_category FROM fact_price_ladder WHERE brand = ? ORDER BY 1",
        [brand],
    ).fetchall()
    return [r[0] for r in rows]


def states(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [
        r[0] for r in con.execute("SELECT DISTINCT state FROM dim_district ORDER BY 1").fetchall()
    ]


def pack_plan(
    con: duckdb.DuckDBPyConnection, category: str, brand: str | None = None
) -> pd.DataFrame:
    """One row per district: the pack to send, its price, and how well it fits.

    With a brand it is that brand's own best-fitting pack; without one, the category's standard
    pack at the price a household can spend. Columns: poly_id, state, district, population,
    score, target_price, pack_total, pack_basis, pack_price, price_ratio, fits.
    """
    if brand:
        sql = """
            SELECT d.poly_id, d.state, d.district, d.population_est AS population, d.score,
                   d.target_price, r.pack_total, r.pack_basis, r.pack_price, r.price_ratio, r.fits
            FROM fact_district_category d
            JOIN rec_brand_district r USING (poly_id, target_category)
            WHERE d.target_category = ? AND r.brand = ?
        """
        args = [category, brand]
    else:
        sql = """
            SELECT poly_id, state, district, population_est AS population, score, target_price,
                   pack_total, pack_basis, pack_price,
                   pack_price / target_price AS price_ratio,
                   pack_price / target_price BETWEEN 1 / 1.6 AND 1.6 AS fits
            FROM fact_district_category WHERE target_category = ?
        """
        args = [category]
    df = con.execute(sql, args).df()
    df["pack_label"] = [
        pack_label(t, b) for t, b in zip(df["pack_total"], df["pack_basis"], strict=True)
    ]
    return df


def select_districts(
    plan: pd.DataFrame,
    states: list[str] | None = None,
    top_n: int | None = None,
    pack: str | None = None,
) -> set[str]:
    """poly_ids matching ALL given filters: states, one pack label, then the top N by demand score
    within what is left (so "top 5 in Bihar" is Bihar's five).

    No filter at all returns an empty set, which the map reads as "highlight nothing special".
    """
    scope = plan
    if states:
        scope = scope[scope["state"].isin(states)]
    if pack:
        scope = scope[scope["pack_label"] == pack]
    if top_n:
        scope = scope.nlargest(int(top_n), "score")
    return set(scope["poly_id"]) if (states or pack or top_n) else set()


def find_district(
    con: duckdb.DuckDBPyConnection, name: str, state: str | None = None
) -> str | None:
    """poly_id of the district whose normalised name equals `name` (state narrows ties)."""
    df = con.execute("SELECT poly_id, state, district FROM dim_district").df()
    key = normalise_name(name)
    hit = df[df["district"].map(normalise_name) == key]
    if state:
        hit = hit[hit["state"].map(normalise_name) == normalise_name(state)]
    return hit["poly_id"].iloc[0] if len(hit) else None


def pack_label(total: float | None, basis: str | None) -> str:
    """'140 g', '1 L', '2 kg' from a size in g or ml."""
    if total is None or pd.isna(total):
        return "no pack"
    if total >= 1000:
        return f"{total / 1000:g} {'kg' if basis == 'g' else 'L'}"
    return f"{total:g} {basis}"


def pack_summary(plan: pd.DataFrame) -> pd.DataFrame:
    """Per pack size: how many districts, how many people, price, and share of population."""
    g = (
        plan.groupby(["pack_total", "pack_label"], dropna=False)
        .agg(
            districts=("poly_id", "count"),
            population=("population", "sum"),
            price=("pack_price", "first"),
            fit_share=("fits", "mean"),
        )
        .reset_index()
        .sort_values("pack_total")
    )
    g["population_share"] = g["population"] / g["population"].sum()
    return g


def district_why(
    plan: pd.DataFrame, con: duckdb.DuckDBPyConnection, poly_id: str, category: str
) -> dict:
    """Everything behind one district's recommendation, as numbers and a sentence."""
    row = plan[plan["poly_id"] == poly_id]
    if row.empty:
        return {}
    r = row.iloc[0]
    d = (
        con.execute(
            """SELECT fd.spend_pc, fd.hh_monthly_spend, fd.demand_rs_month, fd."rank",
                  dd.urban_share, dd.cluster, dd.kirana_per_100k, dd.n_kirana_like
           FROM fact_district_category fd JOIN dim_district dd USING (poly_id)
           WHERE poly_id = ? AND target_category = ?""",
            [poly_id, category],
        )
        .df()
        .iloc[0]
    )
    ratio = r["price_ratio"]
    fit = (
        "a good fit"
        if r["fits"]
        else ("too expensive for the budget" if ratio > 1 else "far cheaper than the budget allows")
    )
    text = (
        f"A household in {r['district']} spends about Rs {d['hh_monthly_spend']:.0f} a month on "
        f"{category.replace('_', ' ')} (Rs {d['spend_pc']:.0f} per person). A pack that costs "
        f"about a week of that is Rs {r['target_price']:.0f}, so the closest pack is "
        f"{r['pack_label']} at Rs {r['pack_price']:.0f}: {fit}."
    )
    return {**d.to_dict(), **r.to_dict(), "text": text}


def parse_query(con: duckdb.DuckDBPyConnection, text: str) -> dict:
    """Pull a brand, category, states and a 'top N' out of free text (no language model).

    Returns {brand, category, states, top_n}; anything not found is None / []. It is the
    fallback the language-model layer is checked against.
    """
    t = " " + re.sub(r"[^a-z0-9& ]+", " ", text.lower()) + " "
    brand = next(
        (b for b in sorted(brands(con), key=len, reverse=True) if f" {b.lower()} " in t), None
    )
    category = next(
        (c for c, words in CATEGORY_WORDS.items() if any(f" {w} " in t for w in words)), None
    )
    if brand and not category:
        cats = brand_categories(con, brand)
        category = cats[0] if cats else None
    known = {normalise_name(s): s for s in states(con)}
    known.update({normalise_name(a): v for a, v in STATE_ALIASES.items()})
    padded = " " + normalise_name(text) + " "
    found = sorted(
        set(s for k, s in sorted(known.items(), key=lambda kv: -len(kv[0])) if f" {k} " in padded)
    )  # noqa: E501
    m = re.search(r"top\s+(\d+)", t)
    return {
        "brand": brand,
        "category": category,
        "states": found,
        "top_n": int(m[1]) if m else None,
    }
