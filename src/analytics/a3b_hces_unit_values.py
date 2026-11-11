"""A3b: what households actually buy, in quantity: unit prices and monthly quantity per household.

Run: uv run python -m src.analytics.a3b_hces_unit_values     (or `make a3b`)

Reads : HCES unit-level files (data/raw/hces_2022_23), config/hces_item_map.csv
Writes: data/processed/hces_unit_values.csv

One row per target category x state x sector (1 rural, 2 urban). Only categories where HCES
records a quantity are covered (rice, atta, edible oil and ghee, spices, tea leaf, namkeen).
Biscuits, noodles, chips and the personal-care categories only have a value, so they are absent.

Columns
    buyer_share            weighted share of households that bought it in the recall period
    median_qty_hh          median monthly quantity (g or ml) among buying households
    median_qty_pp          the same per person (household size divided out)
    unit_price_per_100     rupees per 100 g or ml actually paid = sum(value) / sum(quantity) x 100
    n_buyer_hh             sampled buying households (small numbers mean a shaky estimate)
The sample is designed for State/UT estimates, so nothing here goes below state x sector.
"""

import numpy as np
import pandas as pd

from src.utils.hces import household_category_quantities, household_persons, load_item_map
from src.utils.log import get_logger
from src.utils.paths import get_dir, load_config

log = get_logger(__name__)


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    """Median of `values` where each value counts `weights` times (NaN if there is nothing)."""
    order = np.argsort(values.to_numpy())
    v = values.to_numpy()[order]
    cum = np.cumsum(weights.to_numpy()[order])
    if len(v) == 0 or cum[-1] <= 0:
        return float("nan")
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


def summarise_unit_values(hh: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Aggregate household rows to category x state x sector (columns in the module docstring).

    `hh` is the output of `household_category_quantities`; `totals` has weighted households per
    state x sector (columns state, sector, households) and gives the buyer-share denominator.
    """
    hh = hh.assign(qty_pp=hh["qty_base"] / hh["hh_size"])
    rows = []
    for (cat, basis, state, sector), g in hh.groupby(
        ["target_category", "qty_basis", "state", "sector"]
    ):
        rows.append(
            {
                "target_category": cat,
                "qty_basis": basis,
                "state": state,
                "sector": sector,
                "n_buyer_hh": len(g),
                "buyers_weighted": g["mult"].sum(),
                "median_qty_hh": weighted_median(g["qty_base"], g["mult"]),
                "median_qty_pp": weighted_median(g["qty_pp"], g["mult"]),
                "unit_price_per_100": (g["value_rs"] * g["mult"]).sum()
                / (g["qty_base"] * g["mult"]).sum()
                * 100,
            }
        )
    out = pd.DataFrame(rows).merge(totals, on=["state", "sector"], how="left")
    out["buyer_share"] = out["buyers_weighted"] / out["households"]
    return out.drop(columns=["buyers_weighted", "households"]).rename(
        columns={"state": "state_code"}
    )


def main() -> None:
    """Run the A3b stage."""
    item_map = load_item_map()
    hh = household_category_quantities(item_map)
    div = load_config()["hces"]["weight_divisor"]
    hp = household_persons()
    totals = hp.groupby(["state", "sector"], as_index=False)["households"].sum()
    totals["households"] *= div  # same scale as `mult`, which is not divided above
    out = summarise_unit_values(hh, totals)
    path = get_dir("processed") / "hces_unit_values.csv"
    out.to_csv(path, index=False)
    log.info("%d rows -> %s", len(out), path.name)
    national = (
        out.groupby("target_category")
        .agg(
            unit_price_per_100=("unit_price_per_100", "median"),
            median_qty_hh=("median_qty_hh", "median"),
            buyer_share=("buyer_share", "median"),
        )
        .round(2)
    )
    print(national.to_string())


if __name__ == "__main__":
    main()
