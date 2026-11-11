"""Score a backend on config/eval_questions.csv.

Run: uv run python -m src.serve.eval [rules|ollama[:model]]     (or `make eval`)

A question passes when the first tool the backend calls, and the arguments that matter for it,
equal the expected ones. "none" means the backend must not call a tool (the question is outside
the warehouse). Hard questions need synonyms, regions, typos or a decision that string matching
cannot make; they are where a model is supposed to add value, so they are reported separately.
"""

import sys

import pandas as pd

from src.serve import assistant, queries
from src.utils.paths import REPO_ROOT


def _split(v) -> set[str]:
    return set() if pd.isna(v) else set(str(v).split("|"))


def judge(row: pd.Series, ans: assistant.Answer) -> tuple[bool, str]:
    """(passed, reason) for one question."""
    called = ans.calls[0] if ans.calls else None
    tool = called["name"] if called else "none"
    if tool != row["tool"]:
        return False, f"called {tool}, expected {row['tool']}"
    if tool == "none":
        return True, ""
    args = called["args"]
    checks = {
        "category": (_split(row["category"]), {args.get("category")} - {None}),
        "brand": (_split(row["brand"]), {args.get("brand")} - {None}),
    }
    if tool == "pack_plan":
        checks["states"] = (_split(row["states"]), set(args.get("states") or []))
        want_n = None if pd.isna(row["top_n"]) else int(row["top_n"])
        if (args.get("top_n") or None) != want_n:
            return False, f"top_n {args.get('top_n')} != {want_n}"
    else:
        checks["district"] = (_split(row["district"]), {args.get("district")} - {None})
    for name, (want, got) in checks.items():
        ok = (got <= want and bool(got)) if name == "category" and len(want) > 1 else want == got
        if not ok:
            return False, f"{name} {sorted(got)} != {sorted(want)}"
    return True, ""


def run(backend: str = "rules") -> pd.DataFrame:
    con = queries.connect()
    qs = pd.read_csv(REPO_ROOT / "config" / "eval_questions.csv")
    out = []
    for row in qs.itertuples(index=False):
        r = pd.Series(row._asdict())
        ans = assistant.ask(con, r["question"], backend)
        ok, why = judge(r, ans)
        out.append({"id": r["id"], "difficulty": r["difficulty"], "passed": ok, "why": why,
                    "question": r["question"]})  # fmt: skip
    return pd.DataFrame(out)


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "rules"
    res = run(backend)
    print(res[~res["passed"]].to_string(index=False))
    print(f"\nbackend {backend}")
    print(res.groupby("difficulty")["passed"].agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
