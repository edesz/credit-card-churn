#!/usr/bin/env python3


"""Define helper functions for creating visualizations."""

from typing import Dict, List

import altair as alt
import pandas as pd

_ = alt.data_transformers.disable_max_rows()
_ = alt.renderers.set_embed_options(actions=False)


def customize_altair_chart(
    chart: alt.Chart,
    labelAngle: int = 0,
    labelFontSize: int = 15,
    titleFontSize: int = 18,
    title_width: int = 400,
) -> alt.Chart:
    """Customize altair chart."""
    chart = (
        chart.configure_axis(
            ticks=False,
            labelFontSize=labelFontSize,
            titleFontSize=titleFontSize,
            labelAngle=labelAngle,
            grid=False,
            domain=False,
        )
        .configure_legend(
            titleLimit=title_width,
            labelFontSize=labelFontSize,
            titleFontSize=titleFontSize,
        )
        .configure_view(stroke=None)
    )
    return chart


def plot_grouped_overlapping_altair_histogram(
    df: pd.DataFrame,
    num_bins: int,
    xvar: str,
    xtitle: str,
    ytitle: str,
    color_by_col: str,
    legend_title: str,
    ptitle: alt.TitleParams,
    scale_params: Dict[str, List] = dict(
        domain=[True, False], range=["red", "lightgrey"]
    ),
    fig_size: Dict[str, int] = dict(width=700, height=350),
) -> alt.Chart:
    """."""
    chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.70)
        .encode(
            alt.X(xvar, bin=alt.Bin(maxbins=num_bins), title=xtitle),
            alt.Y("count()", title=ytitle),
            alt.Color(
                color_by_col,
                title=legend_title,
                scale=alt.Scale(**scale_params),
            ),
        )
        .properties(title=ptitle, **fig_size)
    )
    chart = customize_altair_chart(chart)
    return chart


def plot_altair_scatter_chart(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    xtitle: str,
    ytitle: str,
    color_by_col: str,
    legend_title: str,
    ptitle: alt.TitleParams,
    scale_params: Dict[str, List] = dict(
        domain=[True, False], range=["red", "lightgrey"]
    ),
    fig_size: Dict[str, int] = dict(width=700, height=350),
) -> alt.Chart:
    """."""
    chart = (
        alt.Chart(df)
        .mark_point(opacity=0.50, size=65)
        .encode(
            alt.X(xvar, title=xtitle),
            alt.Y(yvar, title=ytitle),
            alt.Color(
                color_by_col,
                title=legend_title,
                scale=alt.Scale(**scale_params),
            ),
        )
        .properties(title=ptitle, **fig_size)
    )
    chart = customize_altair_chart(chart)
    return chart


def plot_grouped_overlapping_altair_bar_chart(
    df: pd.DataFrame,
    xvar: str,
    xtitle: str,
    ytitle: str,
    color_by_col: str,
    legend_title: str,
    ptitle: alt.TitleParams,
    scale_params: Dict[str, List] = dict(
        domain=[False, True], range=["lightgrey", "darkred"]
    ),
    fig_size: Dict[str, int] = dict(width=700, height=350),
) -> alt.Chart:
    """."""
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X(xvar, title=xtitle),
            alt.Y("count()", title=ytitle, scale=alt.Scale(type="symlog")),
            alt.Color(
                color_by_col,
                title=legend_title,
                scale=alt.Scale(**scale_params),
            ),
        )
        .properties(title=ptitle, **fig_size)
    )
    chart = customize_altair_chart(chart, 0, 17, 18)
    return chart


def plot_altair_heatmap(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    textvar: str,
    xtitle: str,
    ytitle: str,
    xsort: List[str],
    ysort: List[str],
    color_by_col: str,
    legend_title: str,
    border_attrs: str,
    cmap: str,
    text_fontsize: int,
    text_alt_condition: alt.expr,
    tooltip: List[alt.Tooltip],
    ptitle: alt.TitleParams,
    fig_size: Dict[str, int] = dict(width=450, height=350),
) -> alt.Chart:
    """."""
    base = alt.Chart(df).encode(
        x=alt.X(xvar, title=xtitle, sort=xsort),
        y=alt.Y(yvar, title=ytitle, sort=ysort),
    )
    chart = base.mark_rect(**border_attrs).encode(
        color=alt.Color(
            color_by_col,
            scale=alt.Scale(scheme=cmap),
            legend=alt.Legend(title=legend_title),
        ),
        tooltip=tooltip,
    )
    text = base.mark_text(baseline="middle", fontSize=text_fontsize).encode(
        text=alt.Text(textvar, format=",.2f"),
        color=alt.condition(
            text_alt_condition, alt.value("white"), alt.value("black")
        ),
    )
    chart = alt.layer(chart, text).properties(title=ptitle, **fig_size)
    chart = customize_altair_chart(chart, 0, 16, 18)
    return chart
