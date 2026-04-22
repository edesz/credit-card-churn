#!/usr/bin/env python3


"""Define helper functions for creating visualizations."""

from typing import Dict, List, Union

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
    label_limit: int = 150,
    legend_label_limit: int = 150,
) -> alt.Chart:
    """Customize altair chart."""
    chart = (
        chart.configure_axis(
            ticks=False,
            labelFontSize=labelFontSize,
            titleFontSize=titleFontSize,
            labelAngle=labelAngle,
            labelLimit=label_limit,
            grid=False,
            domain=False,
        )
        .configure_legend(
            titleLimit=title_width,
            labelFontSize=labelFontSize,
            labelLimit=legend_label_limit,
            titleFontSize=titleFontSize,
            symbolOpacity=1,
        )
        .configure_view(stroke=None)
    )
    return chart


def plot_grouped_overlapping_altair_histogram(
    df: pd.DataFrame,
    num_bins: int,
    xvar: str,
    xtitle: str,
    ytitle: Union[str, None],
    color_by_col: str,
    legend_title: Union[str, None],
    ptitle: alt.TitleParams,
    scale_params: Dict[str, List] = dict(
        domain=[True, False], range=["red", "lightgrey"]
    ),
    y_scale: str = "linear",
    tooltip: List[Union[str, alt.Tooltip, None]] = None,
    fig_size: Dict[str, int] = dict(width=700, height=350),
) -> alt.Chart:
    """."""
    chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.6)
        .encode(
            alt.X(xvar, bin=alt.Bin(maxbins=num_bins), title=xtitle),
            alt.Y(
                "count()",
                stack=None,
                scale=alt.Scale(type=y_scale),
                title=ytitle,
            ),
            alt.Color(
                color_by_col,
                title=legend_title,
                scale=alt.Scale(**scale_params),
            ),
            tooltip=tooltip,
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
    ptitle: alt.TitleParams,
    legend_title: Union[str, None] = None,
    xscale: str = "linear",
    yscale: str = "linear",
    scale_params: Dict[str, List] = dict(
        domain=[True, False], range=["red", "lightgrey"]
    ),
    fig_size: Dict[str, int] = dict(width=700, height=350),
) -> alt.Chart:
    """."""
    chart = (
        alt.Chart(df)
        .mark_circle(size=65, opacity=0.4)
        .encode(
            alt.X(xvar, title=xtitle, scale=alt.Scale(type=xscale)),
            alt.Y(yvar, title=ytitle, scale=alt.Scale(type=yscale)),
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
    y_scale: str = "symlog",
    fig_size: Dict[str, int] = dict(width=700, height=350),
) -> alt.Chart:
    """."""
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X(xvar, title=xtitle),
            alt.Y("count()", title=ytitle, scale=alt.Scale(type=y_scale)),
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
    xsort: List[str],
    ysort: List[str],
    color_by_col: str,
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
        x=alt.X(xvar, title=None, sort=xsort, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(yvar, title=None, sort=ysort, axis=alt.Axis(labelAngle=0)),
    )
    chart = base.mark_rect(**border_attrs).encode(
        color=alt.Color(
            color_by_col,
            scale=alt.Scale(scheme=cmap, domain=[1, -1]),
            legend=None,
        ),
        tooltip=tooltip,
    )
    text = base.mark_text(baseline="middle", fontSize=text_fontsize).encode(
        text=alt.Text(textvar, format=",.3f"),
        color=alt.condition(
            text_alt_condition, alt.value("white"), alt.value("black")
        ),
    )
    chart = alt.layer(chart, text).properties(title=ptitle, **fig_size)
    chart = customize_altair_chart(chart, 0, 16, 18, label_limit=400)
    return chart


def plot_altair_bar_chart(
    df: pd.DataFrame,
    xvar: str,
    xvar2: str,
    yvar: str,
    xtitle: str,
    xtitle2: str,
    ytitle: str,
    y_sort: Union[List[str], str],
    tooltip: List[Union[str, alt.Tooltip]],
    tooltip2: List[Union[str, alt.Tooltip]],
    ptitle: alt.TitleParams,
    x_scale: str = "linear",
    fig_size: dict = dict(width=600, height=200),
):
    """Plot horizontally concatenated bar charts."""
    base = alt.Chart(df).mark_bar().properties(**fig_size)
    frac = base.encode(
        x=alt.X(xvar, title=xtitle),
        y=alt.Y(yvar, sort=y_sort, scale=alt.Scale(type=x_scale), title=ytitle),
        tooltip=tooltip,
    )
    counts = base.encode(
        x=alt.X(xvar2, title=xtitle2),
        y=alt.Y(yvar, sort=y_sort, scale=alt.Scale(type=x_scale), title=ytitle),
        tooltip=tooltip2,
    )
    chart = alt.hconcat(frac, counts).resolve_scale(y="independent")
    chart = customize_altair_chart(chart).properties(title=ptitle)
    return chart


def plot_altair_simple_bar_chart(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    color_by_col: str,
    xtitle: str,
    ytitle: str,
    xsort: str,
    ysort: str,
    ptitle: alt.TitleParams,
    text_color: alt.condition,
    x_scale: str = "linear",
    y_scale: str = "linear",
    text_fmt: str = ".1f",
    text_params: Dict[str, Union[float, str, int]] = dict(
        align="right", baseline="middle", dx=-5, color="white"
    ),
    scale_params: Dict[str, List] = dict(
        domain=[False, True], range=["lightgrey", "darkred"]
    ),
    fig_size: Dict[str, int] = dict(width=700, height=350),
) -> alt.Chart:
    """."""
    base = (
        alt.Chart(df)
        .encode(
            alt.X(
                xvar, title=xtitle, sort=xsort, scale=alt.Scale(type=x_scale)
            ),
            alt.Y(
                yvar, title=ytitle, sort=ysort, scale=alt.Scale(type=y_scale)
            ),
        )
        .properties(title=ptitle, **fig_size)
    )
    bars = base.mark_bar().encode(
        color=alt.Color(
            color_by_col,
            title=None,
            scale=alt.Scale(**scale_params),
            legend=None,
        ),
    )
    text = base.mark_text(**text_params, fontSize=20).encode(
        text=alt.Text(xvar, format=text_fmt), color=text_color
    )
    chart = alt.layer(bars, text)
    chart = customize_altair_chart(
        chart, 0, 17, 18, label_limit=400, legend_label_limit=400
    )
    return chart
