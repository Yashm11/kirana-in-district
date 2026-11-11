"""The tools a model (or the rules fallback) can call. Plain functions over the warehouse.

Each tool returns a dict that is safe to show to a model: numbers, names and short text, plus a
`highlight` list of poly_ids the map should light up. Nothing here writes to the warehouse.
"""

import duckdb
import pandas as pd

from src.serve import queries as q

MAX_NAMES = 25  # district names listed back to the model; the map still highlights all of them

TOOL_SPECS = [
    {
        "name": "pack_plan",
        "description": (
            "Which pack size, at what price, should go to which districts, for a product "
            "category and optionally one brand. Optionally restrict to states or the top N "
            "districts by demand. Returns the packs with their price and how many districts "
            "and people they cover, and highlights the chosen districts on the map."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "one of the categories listed by list_options",
                },  # noqa: E501
                "brand": {"type": "string", "description": "brand name, optional"},
                "states": {"type": "array", "items": {"type": "string"}},
                "top_n": {
                    "type": "integer",
                    "description": "only the N districts with highest demand",
                },  # noqa: E501
            },
            "required": ["category"],
        },
    },
    {
        "name": "explain_district",
        "description": "Why one district gets the pack it gets: budget, target price, chosen pack.",
        "parameters": {
            "type": "object",
            "properties": {
                "district": {"type": "string"},
                "category": {"type": "string"},
                "brand": {"type": "string"},
                "state": {"type": "string", "description": "optional, to disambiguate"},
            },
            "required": ["district", "category"],
        },
    },
    {
        "name": "list_options",
        "description": "The product categories, and the brands sold in a category, that can be asked about.",  # noqa: E501
        "parameters": {
            "type": "object",
            "properties": {"category": {"type": "string", "description": "optional"}},
        },
    },
]


def pack_plan(
    con: duckdb.DuckDBPyConnection,
    category: str,
    brand: str | None = None,
    states: list[str] | None = None,
    top_n: int | None = None,
) -> dict:
    if category not in q.categories(con):
        return {"error": f"unknown category {category!r}", "categories": q.categories(con)}
    if brand and brand not in q.brands(con, category):
        return {"error": f"{brand!r} has no {category} listings", "brands": q.brands(con, category)}
    plan = q.pack_plan(con, category, brand)
    states = q.resolve_states(con, states)
    chosen = q.select_districts(plan, states, top_n)
    scope = plan[plan["poly_id"].isin(chosen)] if chosen else plan
    summary = q.pack_summary(scope)
    names = scope.sort_values("score", ascending=False)
    examples = {
        lab: [f"{r.district} ({r.state})" for r in g.head(5).itertuples()]
        for lab, g in names.groupby("pack_label")
    }
    return {
        "category": category,
        "brand": brand,
        "scope": "selected districts" if chosen else "all 706 districts",
        "n_districts": len(scope),
        "packs": [
            {
                "pack": r.pack_label,
                "price_rs": float(r.price),
                "districts": int(r.districts),
                "share_of_people": round(float(r.population_share), 3),
                "examples": examples.get(r.pack_label, []),
            }
            for r in summary.itertuples()
        ],
        "weak_fit_share": round(float((~scope["fits"]).mean()), 3),
        "districts": [f"{r.district} ({r.state})" for r in names.head(MAX_NAMES).itertuples()],
        "highlight": sorted(chosen),
    }


def explain_district(
    con: duckdb.DuckDBPyConnection,
    district: str,
    category: str,
    brand: str | None = None,
    state: str | None = None,
) -> dict:
    pid = q.find_district(con, district, state)
    if pid is None:
        return {"error": f"no district called {district!r}"}
    if category not in q.categories(con):
        return {"error": f"unknown category {category!r}", "categories": q.categories(con)}
    plan = q.pack_plan(con, category, brand)
    why = q.district_why(plan, con, pid, category)
    keep = ["text", "pack_label", "pack_price", "target_price", "fits", "score", "rank"]
    return {**{k: _plain(why[k]) for k in keep}, "highlight": [pid]}


def list_options(con: duckdb.DuckDBPyConnection, category: str | None = None) -> dict:
    if category:
        return {"category": category, "brands": q.brands(con, category)}
    return {"categories": q.categories(con)}


def _plain(v):
    """numpy scalars to Python, so results are JSON-serialisable."""
    return v.item() if hasattr(v, "item") else v


def call(con: duckdb.DuckDBPyConnection, name: str, args: dict) -> dict:
    """Run a tool by name; a bad call comes back as {"error": ...}, never an exception."""
    fns = {
        "pack_plan": pack_plan,
        "explain_district": explain_district,
        "list_options": list_options,
    }  # noqa: E501
    if name not in fns:
        return {"error": f"unknown tool {name!r}", "tools": list(fns)}
    try:
        return fns[name](con, **args)
    except TypeError as err:
        return {"error": f"bad arguments for {name}: {err}"}


def to_frame(result: dict) -> pd.DataFrame:
    """The packs of a pack_plan result as a table."""
    return pd.DataFrame(result.get("packs", []))
