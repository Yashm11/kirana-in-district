"""The pack map: districts coloured by the pack size they should get, with a highlighted subset.

Plain plotly Choropleth traces on the LGD polygons, no tiles and no base map. One trace per
(pack size, highlighted or not), so the legend lists the packs and a click hides one.
"""

import pandas as pd
import plotly.graph_objects as go

from src.utils.geo import NO_DATA_GREY

# ordered light to dark: small packs to large packs (blue, readable on light and dark themes)
PACK_COLOURS = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b", "#041f4a"]
DIM_OPACITY = 0.22


def pack_colours(labels_in_size_order: list[str]) -> dict[str, str]:
    """Spread the palette over the packs present, smallest pack = lightest."""
    n = len(labels_in_size_order)
    idx = [round(i * (len(PACK_COLOURS) - 1) / max(n - 1, 1)) for i in range(n)]
    return {lab: PACK_COLOURS[i] for lab, i in zip(labels_in_size_order, idx, strict=True)}


def hover(row: pd.Series) -> str:
    fit = "good fit" if row["fits"] else "weak fit"
    return (
        f"<b>{row['district']}</b>, {row['state']}<br>Pack: {row['pack_label']} at "
        f"Rs {row['pack_price']:.0f} ({fit})<br>Budget for a pack: Rs {row['target_price']:.0f}"
        f"<br>Demand score: {row['score']:.0f}/100<extra></extra>"
    )


def pack_map(
    geojson: dict, plan: pd.DataFrame, highlight: set[str] | None = None, title: str = ""
) -> go.Figure:
    """Figure for one pack plan. `highlight` is a set of poly_ids; None or empty means all."""
    plan = plan.dropna(subset=["pack_total"]).copy()
    plan["hover"] = plan.apply(hover, axis=1)
    order = plan.sort_values("pack_total")["pack_label"].drop_duplicates().tolist()
    colour = pack_colours(order)
    dim = bool(highlight)
    fig = go.Figure()
    all_ids = [f["properties"]["poly_id"] for f in geojson["features"]]
    nodata = [i for i in all_ids if i not in set(plan["poly_id"])]
    fig.add_trace(
        go.Choropleth(
            geojson=geojson, locations=nodata, z=[0] * len(nodata),
            featureidkey="properties.poly_id", showscale=False, hoverinfo="skip",
            colorscale=[[0, NO_DATA_GREY], [1, NO_DATA_GREY]], showlegend=False,
            marker={"line": {"width": 0.2, "color": "#ffffff"}, "opacity": 0.6 if dim else 1},
        )
    )  # fmt: skip
    seen: set[str] = set()
    for label in order:
        sub = plan[plan["pack_label"] == label]
        for is_hl in (False, True):
            part = sub[sub["poly_id"].isin(highlight)] if is_hl else sub
            if dim and not is_hl:
                part = sub[~sub["poly_id"].isin(highlight)]
            elif not dim and is_hl:
                continue
            if part.empty:
                continue
            fig.add_trace(
                go.Choropleth(
                    geojson=geojson, locations=part["poly_id"], z=[0] * len(part),
                    featureidkey="properties.poly_id", customdata=part["hover"],
                    hovertemplate="%{customdata}", showscale=False, name=label,
                    legendgroup=label, showlegend=label not in seen,
                    colorscale=[[0, colour[label]], [1, colour[label]]],
                    marker={
                        "line": {
                            "width": 1.4 if is_hl else 0.25,
                            "color": "#111" if is_hl else "#fff",
                        },
                        "opacity": 1 if (is_hl or not dim) else DIM_OPACITY,
                    },
                )
            )  # fmt: skip
            seen.add(label)
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    fig.update_layout(
        title={"text": title, "x": 0.01, "font": {"size": 15}},
        margin={"l": 0, "r": 0, "t": 40, "b": 0}, height=640,
        paper_bgcolor="rgba(0,0,0,0)", legend={"title": {"text": "Pack to send"}},
    )  # fmt: skip
    return fig
