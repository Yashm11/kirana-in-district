"""Build the HCES district-code -> district-name table from the Vol I PDF (Appendix I).

Run: uv run python -m src.data.extract_hces_lookups

HCES unit-level files only carry numeric `state` and `district` codes. Appendix I of the
"Instructions to Field Staff, Vol I" lists every NSS region with its districts and codes.
We parse that appendix and check the result against the (state, district) pairs that actually
occur in the household file, so any parsing gap is reported rather than hidden.

Reads : data/raw/hces_2022_23/HCES2022_Vol_I.pdf, tabulation_state_code.xlsx,
        unit_level/LEVEL - 01*.csv
Writes: data/interim/hces_districts.csv  (state_code, state_name, nss_region, district_code,
        district_name)
"""

import re

import duckdb
import pandas as pd
import pypdf

from src.utils.hces import find_level_file, hces_raw_dir
from src.utils.log import get_logger
from src.utils.paths import get_dir

log = get_logger(__name__)

DISTRICT_END = re.compile(r"\((\d{2})\)\s*$")
SERIAL = re.compile(r"(?<![\d.])(\d{1,3})\s*\.\s+")  # "45.  Name" or "237 . Name"
CODE_ONLY = re.compile(r"^\s*\((\d{2})\)\s*$")  # district code wrapped onto its own line
REGION = re.compile(r"(?<![\d.])(\d{3})(?![\d.])\s+[A-Za-z]")


def appendix_lines(pdf_path: str) -> list[str]:
    """Text lines of Appendix I (list of NSS regions), up to the next appendix."""
    reader = pypdf.PdfReader(pdf_path)
    lines: list[str] = []
    inside = False
    for page in reader.pages:
        text = page.extract_text() or ""
        if "LIST OF NSS REGIONS" in text.upper():
            inside = True
        elif inside and "APPENDIX" in text.upper() and "NSS REGIONS" not in text.upper():
            break
        if inside:
            lines += text.splitlines()
    return lines


def parse_districts(lines: list[str]) -> pd.DataFrame:
    """Turn appendix lines into rows of (nss_region, district_code, district_name)."""
    rows = []
    region: int | None = None
    pending: str | None = None  # name seen on a line whose "(code)" wrapped to the next line
    for line in lines:
        only = CODE_ONLY.match(line)
        if only and pending and region is not None:
            rows.append(
                {
                    "nss_region": region,
                    "district_code": int(only.group(1)),
                    "district_name": pending,
                }
            )
            pending = None
            continue
        end = DISTRICT_END.search(line)
        head = line[: end.start()] if end else line
        parts = list(SERIAL.finditer(head))
        if not parts:
            continue
        found = REGION.search(head[: parts[-1].start()])
        if found:
            region = int(found.group(1))
        name = head[parts[-1].end() :].strip()
        if end and region is not None and name:
            rows.append(
                {"nss_region": region, "district_code": int(end.group(1)), "district_name": name}
            )
            pending = None
        elif not end and name:
            pending = name
    return pd.DataFrame(rows)


def main() -> None:
    """Parse the appendix, validate against the household file and save."""
    raw = hces_raw_dir()
    parsed = parse_districts(appendix_lines(str(raw / "HCES2022_Vol_I.pdf")))
    parsed["state_code"] = parsed["nss_region"] // 10
    states = pd.read_excel(raw / "tabulation_state_code.xlsx", dtype=str, engine="calamine")
    states = states.assign(state_code=states["st"].astype(int)).rename(
        columns={"stn": "state_name"}
    )
    parsed = parsed.merge(states[["state_code", "state_name"]], on="state_code", how="left")

    level1 = find_level_file(1)
    used = duckdb.sql(
        f"SELECT DISTINCT CAST(state AS INTEGER) AS state_code, "
        f"CAST(district AS INTEGER) AS district_code "
        f"FROM read_csv_auto('{level1}')"
    ).df()
    check = used.merge(
        parsed[["state_code", "district_code", "district_name"]].drop_duplicates(
            ["state_code", "district_code"]
        ),
        how="left",
        on=["state_code", "district_code"],
    )
    missing = check[check["district_name"].isna()]
    dupes = parsed.duplicated(["state_code", "district_code"]).sum()
    out = get_dir("interim") / "hces_districts.csv"
    parsed[["state_code", "state_name", "nss_region", "district_code", "district_name"]].to_csv(
        out, index=False
    )

    print("\n=== HCES district lookup ===")
    print(f"Parsed {len(parsed)} district rows from Appendix I ({dupes} repeated state+code)")
    print(f"District codes in household file: {len(used)}; named: {len(used) - len(missing)}")
    for r in missing.itertuples():
        print(f"  NO NAME  state {r.state_code} district {r.district_code}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
