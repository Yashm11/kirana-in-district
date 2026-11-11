"""Tests for the warehouse queries, the pack map and the dashboard (skipped without a warehouse)."""

import pandas as pd
import pytest

from src.serve import queries as q
from src.serve.mapfig import pack_colours, pack_map
from src.utils.paths import REPO_ROOT, get_dir

HAS_WAREHOUSE = (get_dir("processed").parent / "warehouse" / "kid.duckdb").exists()
needs_warehouse = pytest.mark.skipif(not HAS_WAREHOUSE, reason="run `make a7` first")


def test_pack_label_switches_to_kg_and_litres() -> None:
    assert q.pack_label(140, "g") == "140 g"
    assert q.pack_label(2000, "g") == "2 kg"
    assert q.pack_label(1000, "ml") == "1 L"
    assert q.pack_label(float("nan"), "g") == "no pack"


def test_pack_colours_run_light_to_dark_over_the_packs_present() -> None:
    c = pack_colours(["a", "b", "c"])
    assert len(set(c.values())) == 3
    assert pack_colours(["only"])["only"]


def test_pack_map_highlight_dims_the_rest() -> None:
    plan = pd.DataFrame(
        {
            "poly_id": ["d1", "d2"],
            "state": ["S", "S"],
            "district": ["A", "B"],
            "pack_total": [100.0, 100.0],
            "pack_label": ["100 g", "100 g"],
            "pack_price": [10.0, 10.0],
            "target_price": [9.0, 9.0],
            "score": [50.0, 60.0],
            "fits": [True, True],
        }
    )
    gj = {"features": [{"properties": {"poly_id": i}} for i in ("d1", "d2", "d3")]}
    fig = pack_map(gj, plan, {"d1"})
    opac = [t.marker.opacity for t in fig.data]
    assert min(opac) < 1 and 1 in opac  # a dimmed trace and a full-strength one
    assert list(fig.data[0].locations) == ["d3"]  # polygon without a plan row is grey


@needs_warehouse
def test_parse_query_finds_brand_category_states_and_top_n() -> None:
    con = q.connect()
    out = q.parse_query(con, "Maggi: which pack for Bihar and Kerala, top 5?")
    assert out == {
        "brand": "Maggi",
        "category": "noodles_pasta",
        "states": ["Bihar", "Kerala"],
        "top_n": 5,
    }


@needs_warehouse
def test_pack_plan_covers_every_district_once() -> None:
    con = q.connect()
    p = q.pack_plan(con, "noodles_pasta", "Maggi")
    assert p["poly_id"].is_unique and len(p) == 706
    assert q.pack_summary(p)["districts"].sum() == 706
    assert "spends about" in q.district_why(p, con, p["poly_id"].iloc[0], "noodles_pasta")["text"]


@needs_warehouse
def test_dashboard_runs_and_answers_a_question() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "src/app/kid.py"), default_timeout=60).run()
    assert not at.exception
    at.text_input(key="question").set_value("Maggi packs for Bihar").run()
    assert not at.exception
    assert at.selectbox(key="category").value == "noodles_pasta"
    assert at.selectbox(key="brand").value == "Maggi"
    assert at.multiselect(key="states").value == ["Bihar"]


def test_ollama_backend_loops_through_tool_calls(monkeypatch) -> None:
    """A fake model first asks for pack_plan, then answers from the tool result."""
    from src.serve import assistant

    steps = iter(
        [
            {"role": "assistant", "tool_calls": [{"function": {"name": "pack_plan", "arguments": {
                "category": "noodles_pasta", "brand": "Maggi", "states": ["Maharashtra"]}}}]},
            {"role": "assistant", "content": "Send the 140 g pack."},
        ]
    )  # fmt: skip
    monkeypatch.setattr(assistant, "_chat", lambda model, messages, timeout=120: next(steps))
    ans = assistant.ask(q.connect(), "Maggi in Maharashtra?", "ollama:fake")
    assert ans.text == "Send the 140 g pack."
    assert ans.states == ["Maharashtra"] and ans.brand == "Maggi"
    assert len(ans.highlight) > 20  # Maharashtra's districts, from the alias-resolved state


@needs_warehouse
def test_state_aliases_resolve_to_warehouse_spelling() -> None:
    con = q.connect()
    assert q.resolve_states(con, ["Delhi", "maharashtra", "Orissa", "Atlantis"]) == [
        "NCT of Delhi",
        "Maharastra",
        "Odisha",
    ]


@needs_warehouse
def test_top_n_ranks_inside_the_selected_state() -> None:
    p = q.pack_plan(q.connect(), "tea")
    chosen = q.select_districts(p, ["Bihar"], 5)
    assert len(chosen) == 5
    assert set(p[p["poly_id"].isin(chosen)]["state"]) == {"Bihar"}


@needs_warehouse
def test_rules_backend_passes_every_easy_eval_question() -> None:
    from src.serve.eval import run

    res = run("rules")
    assert res[res["difficulty"] == "easy"]["passed"].all(), res[~res["passed"]]
    assert len(res) >= 30
