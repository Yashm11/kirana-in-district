"""A3: build one row per district from NFHS-5, HCES 2022-23 (weights and spend) and Census 2011.

Run: uv run python -m src.analytics.a3_district_features     (or `make a3`)

Reads : data/raw/nfhs5/nfhs5_districts.xls            district indicators (the table spine)
        data/raw/hces_2022_23/unit_level/*.csv         household weights and item spend
        data/interim/hces_districts.csv                district code -> name (extract_hces_lookups)
        data/raw/census_2011/census2011_district_pca.xlsx   cross-check only
        config/config.yaml, hces_item_map.csv, state_name_map.csv, district_name_map.csv
Writes: data/processed/district_features.parquet       one row per NFHS-5 district
        data/processed/hces_state_spend.csv            spend per person by state x sector x category
        data/interim/district_name_map_template.csv    unmatched district names + suggestions

How each column is made
    NFHS-5 indicators   from the district workbook (short names in config a3.nfhs5_indicators).
                        "*" (small sample, suppressed by NFHS) becomes missing.
    spend_pc_<category> monthly Rs per person, STATE-level HCES spend for rural and urban,
                        blended with the district's urban population share. The HCES "Rider" note
                        says estimates below State/UT level are unreliable, so districts in one
                        state differ only through their urban share.
    population          HCES survey weights summed per district (persons = sum(household size x
                        mult) / divisor). Districts that do not match an HCES district by name
                        share the rest of their state's HCES population equally (flagged
                        `population_imputed`). The HCES total is about 13% below official
                        projections, so read it as a relative size, not a census count.
    Census 2011         used only to cross-check the population estimate (boundaries differ).

District names are matched to HCES names inside each state (src/utils/names.py). Every unmatched
name is printed and written, with suggestions, to the template file for your review.
"""

import numpy as np
import pandas as pd

from src.utils.hces import (
    household_persons,
    load_item_map,
    monthly_spend_by_district,
    monthly_spend_by_state_sector,
)
from src.utils.log import get_logger
from src.utils.names import match_names, normalise_name, suggest_matches
from src.utils.paths import REPO_ROOT, get_dir, load_config

log = get_logger(__name__)


# ---------------------------------------------------------------- loaders
def input_files_present() -> dict[str, bool]:
    """Report whether each manual source folder contains at least one file."""
    cfg = load_config()["a3"]
    raw = get_dir("raw")
    return {
        key: any(p.is_file() for p in (raw / cfg[key]).glob("*"))
        for key in ("nfhs5_dir", "census_dir", "hces_dir")
    }


def load_nfhs5() -> pd.DataFrame:
    """NFHS-5 district indicators.

    Returns: state, district, nfhs_households_surveyed, and one column per entry of
    config a3.nfhs5_indicators (values as numbers, "*" -> NaN). Raises if an indicator
    fragment matches zero or several columns.
    """
    cfg = load_config()["a3"]
    path = next((get_dir("raw") / cfg["nfhs5_dir"]).glob("nfhs5_districts.*"))
    raw = pd.read_excel(path, sheet_name="Sheet1", engine="calamine")
    raw.columns = [" ".join(str(c).split()) for c in raw.columns]

    out = pd.DataFrame(
        {
            "state": raw["State/UT"].str.strip(),
            "district": raw["District Names"].str.strip(),
            "nfhs_households_surveyed": raw["Number of Households surveyed"],
        }
    )
    for short, fragment in cfg["nfhs5_indicators"].items():
        hits = [c for c in raw.columns[2:] if fragment.lower() in c.lower()]
        if len(hits) != 1:
            raise ValueError(f"NFHS-5 indicator '{short}': {len(hits)} columns match {fragment!r}")
        out[short] = pd.to_numeric(raw[hits[0]], errors="coerce")
    return drop_copied_rows(out, ["nfhs_households_surveyed", *cfg["nfhs5_indicators"]])


def drop_copied_rows(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Drop rows whose values exactly repeat an earlier row (a copy error in the workbook).

    Example: Mizoram "Chandel" is an exact copy of Manipur's Chandel (847 households and
    identical indicators), so its values are not Mizoram's. Each drop is printed.
    """
    dup = df.duplicated(value_cols, keep="first")
    for r in df[dup].itertuples():
        print(f"DROPPED NFHS row {r.state} / {r.district}: identical values to an earlier district")
    return df[~dup].reset_index(drop=True)


def load_census_2011() -> pd.DataFrame:
    """Census 2011 district population (cross-check only).

    Returns: name, population_2011 (Total rows of the district level).
    """
    path = next((get_dir("raw") / load_config()["a3"]["census_dir"]).glob("*.xls*"))
    c = pd.read_excel(
        path,
        sheet_name="Sheet1",
        usecols=["State", "District", "Level", "Name", "TRU", "TOT_P"],
        engine="calamine",
    )
    c = c[(c["Level"] == "DISTRICT") & (c["TRU"] == "Total")]
    return pd.DataFrame(
        {"name": c["Name"].str.strip(), "population_2011": c["TOT_P"].astype(float)}
    )


def load_hces_geography() -> tuple[pd.DataFrame, pd.DataFrame]:
    """HCES district names and weighted persons.

    Returns (districts, states):
      districts: state_code, state_name, district_code, district_name, persons, urban_persons,
                 n_households
      states:    state_code, state_name, persons, urban_persons
    """
    names_path = get_dir("interim") / "hces_districts.csv"
    if not names_path.exists():
        from src.data.extract_hces_lookups import main as extract

        extract()
    names = pd.read_csv(names_path)
    hp = household_persons().rename(columns={"state": "state_code", "district": "district_code"})
    hp["urban_persons"] = hp["persons"].where(hp["sector"] == 2, 0.0)
    by_dist = hp.groupby(["state_code", "district_code"], as_index=False).agg(
        persons=("persons", "sum"),
        urban_persons=("urban_persons", "sum"),
        n_households=("n_households", "sum"),
    )
    districts = by_dist.merge(
        names[["state_code", "state_name", "district_code", "district_name"]],
        on=["state_code", "district_code"],
        how="left",
    )
    states = districts.groupby(["state_code", "state_name"], as_index=False)[
        ["persons", "urban_persons"]
    ].sum()
    return districts, states


# ---------------------------------------------------------------- matching
def read_overrides(filename: str, state: bool) -> dict:
    """Hand-made name map from config/<filename>. District map is keyed by (state, source)."""
    path = REPO_ROOT / "config" / filename
    if not path.exists():
        return {}
    m = pd.read_csv(path)
    if state:
        return {(r.state, normalise_name(r.source_name)): r.standard_name for r in m.itertuples()}
    return {normalise_name(r.source_name): r.standard_name for r in m.itertuples()}


def match_states(nfhs_states: list[str], hces_states: list[str]) -> dict[str, str]:
    """NFHS state name -> HCES state name; raises if any state stays unmatched."""
    overrides = read_overrides("state_name_map.csv", state=False)
    matched, unmatched = match_names(nfhs_states, hces_states, overrides)
    if unmatched:
        raise ValueError(f"Unmatched states (add to config/state_name_map.csv): {unmatched}")
    return matched


def match_districts(
    nfhs: pd.DataFrame, hces_districts: pd.DataFrame, state_map: dict[str, str]
) -> pd.DataFrame:
    """Add `hces_state_code`, `hces_district_code` to the NFHS table by matching names per state.

    Prints every unmatched NFHS district and writes a suggestions template.
    """
    overrides = read_overrides("district_name_map.csv", state=True)
    out = nfhs.assign(hces_state_name=nfhs["state"].map(state_map))
    out["hces_state_code"] = np.nan
    out["hces_district_code"] = np.nan
    unmatched_rows: list[tuple[str, str]] = []
    reference_all: list[str] = []

    for state, part in out.groupby("hces_state_name"):
        ref = hces_districts[hces_districts["state_name"] == state]
        ref_names = ref["district_name"].tolist()
        reference_all += ref_names
        ov = {k[1]: v for k, v in overrides.items() if state_map.get(k[0]) == state}
        matched, unmatched = match_names(part["district"].tolist(), ref_names, ov)
        code_of = ref.set_index("district_name")[["state_code", "district_code"]]
        for idx, name in part["district"].items():
            if name in matched:
                out.loc[idx, "hces_state_code"] = code_of.loc[matched[name], "state_code"]
                out.loc[idx, "hces_district_code"] = code_of.loc[matched[name], "district_code"]
        unmatched_rows += [(nfhs.loc[part.index[0], "state"], u) for u in unmatched]

    print(f"\nNFHS-5 districts without an HCES district of the same name: {len(unmatched_rows)}")
    for state, name in unmatched_rows:
        print(f"  UNMATCHED  {state} / {name}")
    if unmatched_rows:
        template = suggest_matches([u for _, u in unmatched_rows], reference_all, cutoff=0.75)
        template.insert(0, "state", [s for s, _ in unmatched_rows])
        path = get_dir("interim") / "district_name_map_template.csv"
        template.to_csv(path, index=False)
        print(f"  Suggestions in {path}. Copy confirmed rows (fill standard_name) into")
        print("  config/district_name_map.csv (columns: state, source_name, standard_name).")
    return out


# ---------------------------------------------------------------- spend and population
def state_spend_per_capita(states: pd.DataFrame) -> pd.DataFrame:
    """Monthly Rs per person by state x sector x category, from HCES weights.

    Returns columns: state_code, sector (1 rural, 2 urban), target_category, spend_pc.
    """
    spend = monthly_spend_by_state_sector(load_item_map())
    hp = household_persons().rename(columns={"state": "state_code"})
    persons = hp.groupby(["state_code", "sector"], as_index=False)["persons"].sum()
    out = spend.rename(columns={"state": "state_code"}).merge(
        persons, on=["state_code", "sector"], how="left"
    )
    out["spend_pc"] = out["spend_weighted"] / out["persons"]
    return out[["state_code", "sector", "target_category", "spend_pc"]]


def split_shared_population(df: pd.DataFrame) -> pd.DataFrame:
    """NFHS districts that map to the same HCES district split its population equally.

    This covers districts created after the HCES frame (a child shares its parent's HCES row)
    and HCES districts that NFHS has split in two. Needs state_code, hces_district_code,
    population, urban_persons. Adds `n_sharing` (1 when the district is not shared). The
    urban share is unchanged because both urban persons and population are divided.
    """
    out = df.copy()
    n = out.groupby(["state_code", "hces_district_code"])["district"].transform("size")
    out["n_sharing"] = np.where(out["hces_district_code"].notna(), n, 1).astype(int)
    out["population"] = out["population"] / out["n_sharing"]
    out["urban_persons"] = out["urban_persons"] / out["n_sharing"]
    return out


def district_spend_per_capita(hces_d: pd.DataFrame, min_households: int) -> pd.DataFrame:
    """EXPLORATORY monthly Rs per person by HCES district (rural and urban pooled).

    Districts with fewer than `min_households` sampled households are left empty (NaN).
    Returns columns: state_code, hces_district_code, then spend_pc_<category>_district.
    """
    spend = monthly_spend_by_district(load_item_map()).rename(
        columns={"state": "state_code", "district": "hces_district_code"}
    )
    d = hces_d.rename(columns={"district_code": "hces_district_code"})[
        ["state_code", "hces_district_code", "persons", "n_households"]
    ]
    spend = spend.merge(d, on=["state_code", "hces_district_code"], how="inner")
    spend["spend_pc"] = spend["spend_weighted"] / spend["persons"]
    spend.loc[spend["n_households"] < min_households, "spend_pc"] = np.nan
    wide = spend.pivot_table(
        index=["state_code", "hces_district_code"], columns="target_category", values="spend_pc"
    )
    wide.columns = [f"spend_pc_{c}_district" for c in wide.columns]
    return wide.reset_index()


def impute_population(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing district population with the state's unallocated HCES population.

    df needs: state_code, population (NaN when unmatched), state_persons (HCES state total).
    For each state, the persons not covered by matched districts are shared equally between
    the unmatched districts. If the leftover is under 10% of the median matched district (or
    negative), the median matched district is used instead.
    Adds `population_imputed` (bool).
    """
    out = df.copy()
    out["population_imputed"] = out["population"].isna()
    for _, idx in out.groupby("state_code").groups.items():
        part = out.loc[idx]
        missing = part["population"].isna()
        if not missing.any():
            continue
        left = part["state_persons"].iloc[0] - part["population"].sum()
        share = left / missing.sum()
        median = part["population"].median()
        # a negligible leftover (everything already matched) would give a near-zero district
        if not np.isfinite(share) or share < 0.1 * median:
            share = median
        out.loc[part.index[missing], "population"] = share
    return out


def cross_check_population(features: pd.DataFrame, census: pd.DataFrame) -> dict[str, float]:
    """Compare HCES-weight population with Census 2011 for districts with unique, equal names."""
    f = features[~features["population_imputed"]].assign(
        key=lambda d: d["district"].map(normalise_name)
    )
    c = census.assign(key=lambda d: d["name"].map(normalise_name))
    f = f[~f["key"].duplicated(keep=False)]
    c = c[~c["key"].duplicated(keep=False)]
    m = f.merge(c, on="key")
    if len(m) < 10:
        return {"n_compared": float(len(m))}
    ratio = m["population_est"] / m["population_2011"]
    return {
        "n_compared": float(len(m)),
        "median_ratio_hces_over_census2011": float(ratio.median()),
        "rank_correlation": float(
            m["population_est"].corr(m["population_2011"], method="spearman")
        ),
        "share_within_0.5x_to_2x": float(ratio.between(0.5, 2).mean()),
    }


# ---------------------------------------------------------------- assemble
def build_district_features() -> pd.DataFrame:
    """Join NFHS-5, HCES population/urban share and HCES state spend (see module docstring)."""
    cfg = load_config()
    nfhs = load_nfhs5()
    hces_d, hces_s = load_hces_geography()
    state_map = match_states(sorted(nfhs["state"].unique()), sorted(hces_s["state_name"].unique()))
    df = match_districts(nfhs, hces_d, state_map)

    # state code for every district (even unmatched ones) from the state name
    names_to_code = hces_s.set_index("state_name")["state_code"]
    df["state_code"] = df["hces_state_name"].map(names_to_code)
    df = df.merge(
        hces_d[["state_code", "district_code", "persons", "urban_persons", "n_households"]].rename(
            columns={"district_code": "hces_district_code", "n_households": "hces_n_households"}
        ),
        on=["state_code", "hces_district_code"],
        how="left",
    ).rename(columns={"persons": "population"})
    df = df.merge(
        hces_s[["state_code", "persons", "urban_persons"]].rename(
            columns={"persons": "state_persons", "urban_persons": "state_urban_persons"}
        ),
        on="state_code",
        how="left",
    )
    df["hces_matched"] = df["population"].notna()
    df = split_shared_population(df)

    # urban share: the district's own, else its state's
    own = df["urban_persons"] / df["population"]
    state_share = df["state_urban_persons"] / df["state_persons"]
    df["urban_share"] = own.fillna(state_share)
    df = impute_population(df)
    df["population_source"] = np.select(
        [df["population_imputed"], df["n_sharing"] > 1],
        ["imputed", "hces_split_between_districts"],
        default="hces",
    )

    # EXPLORATORY district-level spend (NaN where the district has too few sampled households)
    dspend = district_spend_per_capita(hces_d, cfg["hces"]["district_min_households"])
    df = df.merge(dspend, on=["state_code", "hces_district_code"], how="left")
    df["district_spend_reliable"] = (
        df["hces_n_households"] >= cfg["hces"]["district_min_households"]
    )

    # blended state-level spend
    spend = state_spend_per_capita(hces_s)
    wide = spend.pivot_table(
        index="state_code", columns=["target_category", "sector"], values="spend_pc"
    )
    for category in cfg["target_categories"]:
        rural = df["state_code"].map(wide[(category, 1)]) if (category, 1) in wide else np.nan
        urban = df["state_code"].map(wide[(category, 2)]) if (category, 2) in wide else np.nan
        df[f"spend_pc_{category}"] = rural * (1 - df["urban_share"]) + urban * df["urban_share"]

    spend.to_csv(get_dir("processed") / "hces_state_spend.csv", index=False)
    keep = [
        "state", "district", "state_code", "hces_state_code", "hces_district_code", "hces_matched",
        "population", "population_imputed", "population_source", "n_sharing", "urban_share",
        "nfhs_households_surveyed", "hces_n_households", "district_spend_reliable",
        *cfg["a3"]["nfhs5_indicators"],
        *[f"spend_pc_{c}" for c in cfg["target_categories"]],
        *[f"spend_pc_{c}_district" for c in cfg["target_categories"]],
    ]  # fmt: skip
    return df[keep].rename(columns={"population": "population_est"}), hces_s


def print_summary(df: pd.DataFrame, hces_s: pd.DataFrame, check: dict[str, float]) -> None:
    """Short summary of the A3 result."""
    cfg = load_config()
    print("\n=== A3 summary ===")
    n_matched = int(df["hces_matched"].sum())
    print(f"Districts (NFHS-5 rows): {len(df)}   matched to an HCES district: {n_matched}")
    print(f"Population source: {df['population_source'].value_counts().to_dict()}")
    n_rel = int(df["district_spend_reliable"].sum())
    print(f"Exploratory district spend available for {n_rel} districts (>= min households)")
    print(f"HCES-weight population, India: {hces_s['persons'].sum() / 1e9:.2f} billion")
    print(f"Population cross-check vs Census 2011 (by unique district name): {check}")
    inds = list(cfg["a3"]["nfhs5_indicators"])
    miss = df[inds].isna().mean().mul(100).round(1)
    print("\nNFHS-5 missing share (%) per indicator:")
    print(miss.to_string())
    print(f"Districts with any missing indicator: {int(df[inds].isna().any(axis=1).sum())}")
    spend_cols = [f"spend_pc_{c}" for c in cfg["target_categories"]]
    print("\nSpend per person per month (Rs), district median / min / max:")
    print(df[spend_cols].describe().loc[["50%", "min", "max"]].round(1).T.to_string())


def main() -> None:
    """Run A3 if all three inputs are present; otherwise say what is missing and stop."""
    present = input_files_present()
    if not all(present.values()):
        missing = [k for k, ok in present.items() if not ok]
        print(f"A3 SKIPPED: missing input folders/files: {missing}")
        print("Run `uv run python -m src.data.check_manual_files` for download instructions.")
        return
    df, hces_s = build_district_features()
    out = get_dir("processed") / load_config()["files"]["district_features"]
    df.to_parquet(out, index=False)
    print_summary(df, hces_s, cross_check_population(df, load_census_2011()))
    log.info("Saved %d districts to %s", len(df), out)


if __name__ == "__main__":
    main()
