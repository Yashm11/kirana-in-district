"""Check that the hand-downloaded files exist in data/raw/<source>/ and say what to do if not.

Run: uv run python -m src.data.check_manual_files
Nothing here is downloaded automatically: each source has its own login or terms.
"""

from src.utils.log import get_logger
from src.utils.paths import get_dir

log = get_logger(__name__)

# folder name in data/raw/ -> (what it is, where to get it, what to save)
SOURCES: dict[str, tuple[str, str, str]] = {
    "nfhs5": (
        "NFHS-5 (2019-21) district fact-sheet indicators",
        "https://rchiips.org/nfhs/ (district fact sheets) or a district-level CSV compilation "
        "from data.gov.in / Kaggle. Pick one and note it in data/README.md.",
        "the district-level CSV(s), e.g. data/raw/nfhs5/nfhs5_districts.csv",
    ),
    "hces_2022_23": (
        "HCES 2022-23 household consumption microdata (MoSPI)",
        "https://microdata.gov.in (free registration/login), search 'Household Consumption "
        "Expenditure Survey 2022-23'.",
        "the unit-level data files and layout/documentation into data/raw/hces_2022_23/",
    ),
    "census_2011": (
        "Census 2011 district population",
        "https://censusindia.gov.in (Primary Census Abstract, district level) or the "
        "data.gov.in copy.",
        "the district population file (xlsx or csv) into data/raw/census_2011/",
    ),
}
DATA_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet", ".dta", ".txt", ".zip", ".tsv", ".pdf"}


def main() -> None:
    """Report found/missing for each manual source."""
    raw = get_dir("raw")
    missing = 0
    for folder, (what, where, save) in SOURCES.items():
        path = raw / folder
        path.mkdir(exist_ok=True)  # so you can drop files straight in
        files = [f for f in path.iterdir() if f.suffix.lower() in DATA_SUFFIXES]
        if files:
            print(f"[OK]      {folder}: {len(files)} file(s): {', '.join(f.name for f in files)}")
            continue
        missing += 1
        print(f"[MISSING] {folder}: {what}")
        print(f"          Get it from: {where}")
        print(f"          Save: {save}\n")
    print(f"\n{len(SOURCES) - missing} of {len(SOURCES)} manual sources present.")


if __name__ == "__main__":
    main()
