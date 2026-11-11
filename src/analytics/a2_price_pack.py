"""A2: price-pack analysis (price ladders, pack premium, entry price points, ladder gaps).

Run: uv run python -m src.analytics.a2_price_pack     (or `make a2`)

Reads : data/processed/products_clean.parquet
Writes: data/processed/price_ladders.csv, pack_premium.csv, pack_premium_category.csv,
        entry_price_points.csv, ladder_gaps.csv, discount_depth.csv
        reports/figures/price_ladder_<category>[_<g|ml>].png, pack_premium.png

Key terms
    ladder      the pack sizes one brand sells in one SubCategory (one unit: g or ml)
    pack premium  price per 100 g (or ml) of the SMALLEST pack / that of the LARGEST pack.
                  2.0 means the smallest pack costs twice as much per 100 g.
    entry price   the cheapest rupee price a brand has in a category
    ladder gap    a step between neighbouring pack sizes bigger than `ladder_gap_ratio`
All prices are the selling price (DiscountPrice in the raw data). Combos are excluded.
"""

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator, ScalarFormatter  # noqa: E402

from src.utils.log import get_logger  # noqa: E402
from src.utils.paths import get_dir, load_config  # noqa: E402

log = get_logger(__name__)

LADDER_KEYS = ["target_category", "subcategory", "brand", "pack_basis"]
UNIT_NAME = {"g": "g", "ml": "ml"}
ROUND_TICKS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

# Chart style: validated categorical palette (fixed order), recessive grid, text in ink colours
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]  # second cue besides colour
INK, INK_2, GRID, SURFACE, GREY = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#b5b4ae"


# ---------------------------------------------------------------- data prep
def load_products() -> pd.DataFrame:
    """Read products_clean.parquet (written by A1)."""
    cfg = load_config()
    path = get_dir("processed") / cfg["files"]["products_clean"]
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `make a1` first.")
    return pd.read_parquet(path)


def build_ladders(df: pd.DataFrame) -> pd.DataFrame:
    """One row per brand x subcategory x unit x pack size, with median prices.

    Only weight (g) and volume (ml) packs in a target category, combos excluded.
    Products of one brand and subcategory with the same pack size are collapsed to their median.
    """
    d = df[
        df["target_category"].notna()
        & (df["parse_status"] == "ok")
        & df["pack_basis"].isin(UNIT_NAME)
    ].copy()
    d["unit_price"] = d["price_per_100g"].fillna(d["price_per_100ml"])  # Rs per 100 g or 100 ml
    ladders = (
        d.groupby([*LADDER_KEYS, "pack_total"])
        .agg(
            n_products=("product_name", "size"),
            selling_price=("selling_price", "median"),
            mrp=("mrp", "median"),
            unit_price=("unit_price", "median"),
        )
        .reset_index()
        .sort_values([*LADDER_KEYS, "pack_total"])
    )
    return ladders


def multi_size_ladders(ladders: pd.DataFrame, min_packs: int) -> pd.DataFrame:
    """Keep only ladders with at least `min_packs` distinct pack sizes."""
    n = ladders.groupby(LADDER_KEYS)["pack_total"].transform("nunique")
    return ladders[n >= min_packs]


# ---------------------------------------------------------------- analyses
def pack_premium(ladders: pd.DataFrame, min_packs: int) -> pd.DataFrame:
    """Per brand x subcategory x unit ladder: price per 100 of smallest pack / largest pack."""
    rows = []
    for keys, g in multi_size_ladders(ladders, min_packs).groupby(LADDER_KEYS):
        g = g.sort_values("pack_total")
        small, large = g.iloc[0], g.iloc[-1]
        rows.append(
            {
                **dict(zip(LADDER_KEYS, keys, strict=True)),
                "n_pack_sizes": len(g),
                "smallest_pack": small["pack_total"],
                "largest_pack": large["pack_total"],
                "smallest_price": small["selling_price"],
                "largest_price": large["selling_price"],
                "smallest_per_100": small["unit_price"],
                "largest_per_100": large["unit_price"],
                "pack_premium_x": small["unit_price"] / large["unit_price"],
            }
        )
    return pd.DataFrame(rows)


def pack_premium_by_category(premium: pd.DataFrame) -> pd.DataFrame:
    """Median pack premium across ladders (brand x subcategory) for each category x unit."""
    out = (
        premium.groupby(["target_category", "pack_basis"])
        .agg(
            n_ladders=("brand", "size"),
            n_brands=("brand", "nunique"),
            median_premium_x=("pack_premium_x", "median"),
            min_premium_x=("pack_premium_x", "min"),
            max_premium_x=("pack_premium_x", "max"),
        )
        .reset_index()
        .sort_values("median_premium_x", ascending=False)
    )
    return out


def entry_price_points(df: pd.DataFrame) -> pd.DataFrame:
    """Cheapest rupee price per brand x category (any pack type, combos excluded)."""
    d = df[df["target_category"].notna() & ~df["is_combo"]]
    cheapest = d.loc[d.groupby(["target_category", "brand"])["selling_price"].idxmin()]
    counts = d.groupby(["target_category", "brand"]).size().rename("n_products")
    out = cheapest.merge(counts.reset_index(), on=["target_category", "brand"])
    out = out[
        ["target_category", "brand", "n_products", "selling_price", "quantity_raw", "product_name"]
    ].rename(columns={"selling_price": "entry_price_rs", "quantity_raw": "entry_pack"})
    return out.sort_values(["target_category", "entry_price_rs"])


def ladder_gaps(ladders: pd.DataFrame, min_packs: int, ratio: float) -> pd.DataFrame:
    """Every step between neighbouring pack sizes, flagged when the size jump > `ratio`.

    size_ratio  = bigger pack / smaller pack (e.g. 100 g -> 500 g is 5.0)
    price_jump_rs = extra rupees a shopper must find to step up one rung.
    """
    rows = []
    for keys, g in multi_size_ladders(ladders, min_packs).groupby(LADDER_KEYS):
        g = g.sort_values("pack_total").reset_index(drop=True)
        for i in range(1, len(g)):
            lo, hi = g.iloc[i - 1], g.iloc[i]
            rows.append(
                {
                    **dict(zip(LADDER_KEYS, keys, strict=True)),
                    "from_pack": lo["pack_total"],
                    "to_pack": hi["pack_total"],
                    "size_ratio": hi["pack_total"] / lo["pack_total"],
                    "from_price": lo["selling_price"],
                    "to_price": hi["selling_price"],
                    "price_jump_rs": hi["selling_price"] - lo["selling_price"],
                    "price_ratio": hi["selling_price"] / lo["selling_price"],
                    "is_gap": hi["pack_total"] / lo["pack_total"] > ratio,
                }
            )
    return pd.DataFrame(rows)


def discount_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Discount % (MRP vs selling price) per brand x category, combos excluded."""
    d = df[df["target_category"].notna() & ~df["is_combo"]]
    return (
        d.groupby(["target_category", "brand"])
        .agg(
            n_products=("product_name", "size"),
            mean_discount_pct=("discount_pct", "mean"),
            median_discount_pct=("discount_pct", "median"),
            max_discount_pct=("discount_pct", "max"),
        )
        .reset_index()
        .sort_values(["target_category", "median_discount_pct"], ascending=[True, False])
    )


# ---------------------------------------------------------------- charts
def style_axes(ax: plt.Axes) -> None:
    """Recessive grid and spines so the data stands out."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def rupee(x: float, _pos: int | None = None) -> str:
    """Axis tick label like ₹1,200."""
    return f"₹{x:,.0f}"


def plot_price_ladder(
    ladders: pd.DataFrame, premium: pd.DataFrame, category: str, basis: str, max_brands: int
) -> str | None:
    """Price-ladder chart for one category and unit. Returns the file name, or None if no ladder."""
    d = ladders[(ladders["target_category"] == category) & (ladders["pack_basis"] == basis)]
    prem = premium[(premium["target_category"] == category) & (premium["pack_basis"] == basis)]
    if prem.empty:
        return None
    unit = UNIT_NAME[basis]

    # One line per ladder (brand x subcategory); highlight those with the most pack sizes
    d = d.assign(label=d["brand"] + " · " + d["subcategory"])
    prem = prem.assign(label=prem["brand"] + " · " + prem["subcategory"])
    top = prem.sort_values(["n_pack_sizes", "label"], ascending=[False, True])["label"].tolist()
    highlight = top[:max_brands]

    med = prem["pack_premium_x"].median()
    if med > 1.05:
        message = (
            f"{category.replace('_', ' ').title()}: the smallest pack costs "
            f"{med:.1f}x more per 100 {unit} than the largest"
        )
    else:
        message = (
            f"{category.replace('_', ' ').title()}: bigger packs give little saving per 100 {unit}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
    panels = [("selling_price", "Selling price (₹)"), ("unit_price", f"Price per 100 {unit} (₹)")]
    for ax, (col, ylabel) in zip(axes, panels, strict=True):
        style_axes(ax)
        for lab in top[max_brands:]:  # grey context lines
            b = d[d["label"] == lab]
            ax.plot(b["pack_total"], b[col], color=GREY, linewidth=1, alpha=0.7, zorder=1)
        for i, lab in enumerate(highlight):
            b = d[d["label"] == lab]
            ax.plot(
                b["pack_total"], b[col], color=SERIES_COLOURS[i], marker=MARKERS[i],
                markersize=7, linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=lab, zorder=3,
            )  # fmt: skip
        ax.set_xscale("log")
        # a few round tick values (not every pack size) so labels do not collide
        lo, hi = d["pack_total"].min(), d["pack_total"].max()
        ticks = [t for t in ROUND_TICKS if lo / 1.3 <= t <= hi * 1.3]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(rupee))
        ax.set_xlabel(f"Pack size ({unit}, log scale)", color=INK_2)
        ax.set_ylabel(ylabel, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    if len(top) > max_brands:
        labels.append(f"Other ladders ({len(top) - max_brands}, grey)")
        handles.append(plt.Line2D([], [], color=GREY, linewidth=1))
    axes[1].legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(message, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(
        0.01, 0.9, f"Each line = one brand in one subcategory with 2+ pack sizes in BigBasket data "
        f"({len(top)} lines). Prices are selling prices after discount.", fontsize=9, color=INK_2,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    name = f"price_ladder_{category}_{unit}.png"
    fig.savefig(get_dir("figures") / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return name


def plot_pack_premium(by_cat: pd.DataFrame) -> str:
    """Horizontal bar chart of the median pack premium per category."""
    d = by_cat.sort_values("median_premium_x").copy()
    d["label"] = (
        d["target_category"].str.replace("_", " ").str.title() + " (" + d["pack_basis"] + ")"
    )
    n_pay = int((by_cat["median_premium_x"] > 1).sum())
    top = by_cat.iloc[0]
    how_many = "all " + str(n_pay) if n_pay == len(by_cat) else f"{n_pay} of {len(by_cat)}"
    message = (
        f"Small packs cost more per 100 g/ml in {how_many} categories; "
        f"most in {top['target_category'].replace('_', ' ')} ({top['median_premium_x']:.1f}x)"
    )
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(d) + 2), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(False, axis="y")
    bars = ax.barh(d["label"], d["median_premium_x"], color=SERIES_COLOURS[0], height=0.6)
    ax.axvline(1, color=INK_2, linewidth=1, linestyle="--")
    ax.text(1, len(d) - 0.35, " 1x = same ₹ per 100 g/ml", color=INK_2, fontsize=8, va="bottom")
    for bar, (_, row) in zip(bars, d.iterrows(), strict=True):
        ax.text(
            bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
            f"{row['median_premium_x']:.1f}x  (n={row['n_ladders']} ladders)",
            va="center", fontsize=9, color=INK,
        )  # fmt: skip
    ax.set_xlim(0, d["median_premium_x"].max() * 1.35)
    ax.set_xlabel(
        "Pack premium: ₹ per 100 g/ml of smallest pack ÷ largest pack (median of ladders, x)",
        color=INK_2,
    )
    fig.suptitle(message, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    name = "pack_premium.png"
    fig.savefig(get_dir("figures") / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return name


# ---------------------------------------------------------------- run
def print_summary(
    df: pd.DataFrame, ladders: pd.DataFrame, by_cat: pd.DataFrame, entry: pd.DataFrame,
    gaps: pd.DataFrame, disc: pd.DataFrame, charts: list[str], ratio: float,
) -> None:  # fmt: skip
    """Short summary of what A2 found."""
    in_scope = df[df["target_category"].notna()]
    n_ladder_rows = ladders["n_products"].sum()
    print("\n=== A2 summary ===")
    print(f"Rows in (target categories): {len(in_scope):,}")
    print(
        f"Used for ladders: {n_ladder_rows:,} (g/ml packs, parsed, not combo). "
        f"Excluded: {(in_scope['is_combo']).sum()} combos, "
        f"{(in_scope['pack_basis'] == 'pc').sum()} count packs, "
        f"{(in_scope['parse_status'] == 'unparsed').sum()} unparsed"
    )
    print(f"Ladders with 2+ pack sizes: {len(by_cat)} category/unit groups\n")
    print("Pack premium by category (median across brand x subcategory ladders):")
    print(by_cat.round(2).to_string(index=False))
    print("\nEntry price (median of brands' cheapest ₹ price) by category:")
    print(
        entry.groupby("target_category")["entry_price_rs"].agg(["count", "min", "median"]).round(0)
    )
    n_gaps = int(gaps["is_gap"].sum())
    print(f"\nLadder steps: {len(gaps)}; flagged gaps (size jump > {ratio}x): {n_gaps}")
    print(gaps.groupby("target_category")["is_gap"].agg(["sum", "count"]).T.to_string())
    print("\nMedian discount % by category:")
    print(disc.groupby("target_category")["median_discount_pct"].median().round(1).to_string())
    print(f"\nCharts written: {len(charts)} -> reports/figures/")


def main() -> None:
    """Run the whole A2 stage."""
    cfg = load_config()
    th = cfg["thresholds"]
    out_dir = get_dir("processed")
    files = cfg["files"]

    df = load_products()
    ladders = build_ladders(df)
    premium = pack_premium(ladders, th["min_packs_per_ladder"])
    by_cat = pack_premium_by_category(premium)
    entry = entry_price_points(df)
    gaps = ladder_gaps(ladders, th["min_packs_per_ladder"], th["ladder_gap_ratio"])
    disc = discount_depth(df)

    ladders.to_csv(out_dir / files["price_ladders"], index=False)
    premium.to_csv(out_dir / files["pack_premium"], index=False)
    by_cat.to_csv(out_dir / files["pack_premium_category"], index=False)
    entry.to_csv(out_dir / files["entry_price_points"], index=False)
    gaps.to_csv(out_dir / files["ladder_gaps"], index=False)
    disc.to_csv(out_dir / files["discount_depth"], index=False)

    charts = []
    for category, basis in by_cat[["target_category", "pack_basis"]].itertuples(index=False):
        name = plot_price_ladder(ladders, premium, category, basis, th["max_highlighted_brands"])
        if name:
            charts.append(name)
    charts.append(plot_pack_premium(by_cat))

    print_summary(df, ladders, by_cat, entry, gaps, disc, charts, th["ladder_gap_ratio"])


if __name__ == "__main__":
    main()
