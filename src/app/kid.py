"""K.I.D dashboard: which pack, at what price, goes to which districts.

Run: uv run streamlit run src/app/kid.py     (or `make kid`)

Reads the DuckDB warehouse built by `make a7` and the district polygons from `make a6`. The
question box understands a brand, a category, states and "top N" without a language model; the
language-model layer (Phase 4) plugs into the same place.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.serve import assistant  # noqa: E402
from src.serve import queries as q  # noqa: E402
from src.serve.mapfig import pack_map  # noqa: E402
from src.utils.paths import get_dir  # noqa: E402

st.set_page_config(page_title="K.I.D: kirana in district", page_icon="🛒", layout="wide")


@st.cache_resource
def warehouse():
    return q.connect()


@st.cache_data
def geojson() -> dict:
    return json.loads((get_dir("processed") / "map_districts.geojson").read_text())


@st.cache_data
def plan(category: str, brand: str | None) -> pd.DataFrame:
    return q.pack_plan(warehouse(), category, brand)


def apply_question() -> None:
    """Answer the question box and fill the pickers from the answer (runs before the widgets)."""
    ans = assistant.ask(
        warehouse(), st.session_state.get("question", ""), st.session_state.get("backend", "rules")
    )
    st.session_state["answer"] = ans
    if ans.category:
        st.session_state["category"] = ans.category
    st.session_state["brand"] = ans.brand or "(all brands)"
    st.session_state["states"] = q.resolve_states(warehouse(), ans.states)
    st.session_state["top_n"] = ans.top_n or 0


def main() -> None:
    try:
        con = warehouse()
    except FileNotFoundError as err:
        st.error(str(err))
        return
    st.title("K.I.D: kirana in district")
    st.caption(
        "Which pack size, at what price, should go to which district. Built on household "
        "spend (HCES 2022-23), NFHS-5 district profiles, BigBasket price ladders and OSM shops."
    )
    backends = {"Rules (no model)": "rules"}
    if assistant.ollama_available():
        backends[f"Local model ({assistant.DEFAULT_MODEL})"] = f"ollama:{assistant.DEFAULT_MODEL}"
    if len(backends) > 1:
        st.radio("Answered by", list(backends), horizontal=True, key="backend_label")
    st.session_state["backend"] = backends.get(st.session_state.get("backend_label"), "rules")
    st.text_input(
        "Ask a question",
        key="question",
        placeholder="Maggi: which pack should go to Bihar and Kerala?   or   top 20 tea districts",
        on_change=apply_question,
    )

    if "answer" in st.session_state:
        st.info(st.session_state["answer"].text)
    cats = q.categories(con)
    st.session_state.setdefault("category", cats[0])
    c1, c2, c3 = st.columns(3)
    category = c1.selectbox(
        "Category", cats, key="category", format_func=lambda c: c.replace("_", " ")
    )
    brand_opts = ["(all brands)", *q.brands(con, category)]
    if st.session_state.get("brand") not in brand_opts:
        st.session_state["brand"] = "(all brands)"
    brand = c2.selectbox("Brand", brand_opts, key="brand")
    brand = None if brand == "(all brands)" else brand
    p = plan(category, brand)
    packs = ["(any)", *p.sort_values("pack_total")["pack_label"].drop_duplicates().tolist()]
    pack_pick = c3.selectbox("Highlight pack", packs)
    d1, d2 = st.columns([3, 1])
    state_pick = d1.multiselect("Highlight states", q.states(con), key="states")
    top_n = d2.number_input("Top N districts by demand", 0, 200, key="top_n", help="0 = off")

    highlight = q.select_districts(
        p, state_pick, int(top_n) or None, None if pack_pick == "(any)" else pack_pick
    )

    left, right = st.columns([3, 2])
    label = f"{brand or 'Standard'} packs for {category.replace('_', ' ')}"
    with left:
        st.plotly_chart(pack_map(geojson(), p, highlight, label), use_container_width=True)
        if highlight:
            st.caption(f"{len(highlight)} districts highlighted")
    with right:
        st.subheader("Pack and price card")
        summary = q.pack_summary(p if not highlight else p[p["poly_id"].isin(highlight)])
        st.dataframe(
            summary.assign(population_share=(summary["population_share"] * 100).round(1)).rename(
                columns={
                    "pack_label": "pack",
                    "price": "price (Rs)",
                    "population_share": "% of people",
                }
            )[["pack", "price (Rs)", "districts", "% of people"]],
            hide_index=True,
            use_container_width=True,
        )
        weak = int((~p["fits"]).sum())
        if weak > len(p) / 2:
            st.warning(
                f"For {weak} of {len(p)} districts no listed pack is close to a week's budget; "
                "smaller packs are missing from the catalogue (see assumptions 52)."
            )

    st.subheader("Why this pack?")
    scope = p[p["poly_id"].isin(highlight)] if highlight else p
    names = scope.sort_values("score", ascending=False)
    pick = st.selectbox(
        "District",
        names["poly_id"],
        format_func=lambda i: "{}, {}".format(
            *names.set_index("poly_id").loc[i, ["district", "state"]]
        ),
    )
    why = q.district_why(p, con, pick, category)
    st.write(why["text"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Demand rank in category", f"{int(why['rank'])} of {len(p)}")
    m2.metric("Urban share", f"{why['urban_share']:.0%}")
    m3.metric("Mapped kirana-like shops (OSM)", f"{int(why['n_kirana_like'])}")
    m4.metric("District cluster", int(why["cluster"]))


main()
