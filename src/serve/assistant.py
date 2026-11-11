"""Answer a free-text question with the warehouse tools.

Two backends share the tools in src/serve/tools.py and return the same `Answer`:
  rules   parses brand, category, states and "top N" with plain string matching; no model, so
          it also runs on the hosted site and in tests.
  ollama  a local model (default qwen2.5:7b) that chooses which tool to call. Free, needs
          Ollama running on this machine: `ollama pull qwen2.5:7b`.
The model never sees raw tables or writes SQL, only tool results, so it cannot invent a pack or
a price that the warehouse did not return.
"""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import duckdb

from src.serve import queries as q
from src.serve import tools
from src.utils.names import normalise_name

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
MAX_STEPS = 4

SYSTEM_PROMPT = (
    "You help a consumer-goods team decide which pack size and price point to send to which "
    "Indian districts. Answer only from tool results; never guess a price, pack or district. "
    "Call pack_plan for 'which pack to which districts' questions and explain_district for "
    "'why' questions about one district. If a brand or category is unclear, call list_options. "
    "Keep the answer to a few sentences and name the packs, prices and places returned."
)


@dataclass
class Answer:
    text: str
    highlight: set[str] = field(default_factory=set)
    category: str | None = None
    brand: str | None = None
    states: list[str] = field(default_factory=list)
    top_n: int | None = None
    calls: list[dict] = field(default_factory=list)
    backend: str = "rules"


def _find_district_in_text(con: duckdb.DuckDBPyConnection, text: str) -> tuple[str, str] | None:
    """(district, state) of a district named in the text, longest name first."""
    df = con.execute("SELECT district, state FROM dim_district").df()
    t = " " + normalise_name(text) + " "
    rows = sorted(df.itertuples(), key=lambda r: len(r.district), reverse=True)
    for r in rows:
        if f" {normalise_name(r.district)} " in t:
            return r.district, r.state
    return None


def format_pack_plan(r: dict) -> str:
    """One readable paragraph for a pack_plan result."""
    if "error" in r:
        return r["error"]
    who = r["brand"] or "Standard"
    head = f"{who} packs for {r['category'].replace('_', ' ')}, {r['scope']} ({r['n_districts']}):"
    lines = []
    for p in r["packs"]:
        n = p["districts"]
        share = p["share_of_people"]
        line = (
            f"{p['pack']} at Rs {p['price_rs']:.0f} for {n} district{'s' if n != 1 else ''} "
            f"({'<1%' if 0 < share < 0.01 else f'{share:.0%}'} of people)"
        )
        if p["examples"] and n < r["n_districts"]:
            line += ", e.g. " + ", ".join(p["examples"][:3])
        lines.append(line)
    text = head + " " + "; ".join(lines) + "."
    if r["n_districts"] <= tools.MAX_NAMES:
        text += " Districts: " + ", ".join(r["districts"]) + "."
    if r["weak_fit_share"] > 0.5:
        text += (
            f" Caution: for {r['weak_fit_share']:.0%} of these districts no listed pack is close "
            "to a week's budget, so smaller packs may be missing from the catalogue."
        )
    return text


def ask_rules(con: duckdb.DuckDBPyConnection, text: str) -> Answer:
    p = q.parse_query(con, text)
    wants_why = bool(re.search(r"\bwhy\b|\bexplain\b", text.lower()))
    hit = _find_district_in_text(con, text) if wants_why else None
    if hit and p["category"]:
        call = {"name": "explain_district", "args": {
            "district": hit[0], "category": p["category"], "brand": p["brand"], "state": hit[1],
        }}  # fmt: skip
        r = tools.call(con, call["name"], call["args"])
        return Answer(
            r.get("text", r.get("error", "")), set(r.get("highlight", [])), p["category"],
            p["brand"], [hit[1]], None, [call],
        )  # fmt: skip
    if not p["category"]:
        cats = ", ".join(c.replace("_", " ") for c in q.categories(con))
        return Answer(f"I could not tell which product you mean. I know: {cats}.")
    call = {"name": "pack_plan", "args": {
        "category": p["category"], "brand": p["brand"], "states": p["states"], "top_n": p["top_n"],
    }}  # fmt: skip
    r = tools.call(con, call["name"], call["args"])
    return Answer(
        format_pack_plan(r), set(r.get("highlight", [])), p["category"], p["brand"],
        p["states"], p["top_n"], [call],
    )  # fmt: skip


def _chat(model: str, messages: list[dict], timeout: float = 120) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "tools": [{"type": "function", "function": s} for s in tools.TOOL_SPECS],
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["message"]


def ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """True when the local Ollama server is up and has `model` pulled."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=1.5) as resp:
            names = [m["name"] for m in json.load(resp)["models"]]
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        return False
    return any(n == model or n.startswith(model + ":") for n in names)


def ask_ollama(con: duckdb.DuckDBPyConnection, text: str, model: str = DEFAULT_MODEL) -> Answer:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    ans = Answer("", backend=f"ollama:{model}")
    for _ in range(MAX_STEPS):
        msg = _chat(model, messages)
        messages.append(msg)
        if not msg.get("tool_calls"):
            ans.text = msg.get("content", "").strip()
            return ans
        for tc in msg["tool_calls"]:
            name, args = tc["function"]["name"], tc["function"].get("arguments") or {}
            result = tools.call(con, name, args)
            ans.calls.append({"name": name, "args": args})
            if result.get("highlight"):
                ans.highlight = set(result["highlight"])
            if name == "pack_plan" and "error" not in result:
                ans.category, ans.brand = args.get("category"), args.get("brand")
                ans.states, ans.top_n = args.get("states") or [], args.get("top_n")
            shown = {k: v for k, v in result.items() if k != "highlight"}
            messages.append({"role": "tool", "content": json.dumps(shown, default=str)})
    ans.text = ans.text or "I could not finish that question; try naming a product and a place."
    return ans


def ask(con: duckdb.DuckDBPyConnection, text: str, backend: str = "rules") -> Answer:
    """Answer `text` with the chosen backend ("rules" or "ollama[:model]")."""
    if backend.startswith("ollama"):
        model = backend.split(":", 1)[1] if ":" in backend else DEFAULT_MODEL
        return ask_ollama(con, text, model)
    return ask_rules(con, text)
