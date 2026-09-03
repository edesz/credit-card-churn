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
    """Apply consistent styling to an Altair chart.

    This function standardizes axis, legend, and view configurations
    such as font sizes, label angles, and grid visibility.

    Args:
        chart: Input Altair chart.
        labelAngle: Angle of axis labels.
        labelFontSize: Font size for axis labels.
        titleFontSize: Font size for titles.
        title_width: Max width for legend titles.
        label_limit: Max length for axis labels.
        legend_label_limit: Max length for legend labels.

    Returns:
        alt.Chart: Styled Altair chart.

    Examples:
        >>> chart = customize_altair_chart(chart)
    """
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


def export_altair_chart(
    alt_chart: alt.Chart,
    fpath: str,
    mode: str = "vega-lite",
    engine: str = "vl-convert",
    override_data_transformer: bool = False,
    embed_options: Dict[str, str] = {"renderer": "svg"},
) -> None:
    """Exports an Altair chart to file.

    Args:
        alt_chart: Altair chart to export.
        fpath: Output file path.
        mode: Serialization mode (e.g., "vega-lite").
        engine: Rendering engine used for export.
        override_data_transformer: Whether to override transformer.
        embed_options: Rendering options for output.

    Returns:
        None

    Examples:
        >>> export_altair_chart(chart, "plot.html")
    """
    alt_chart.save(
        fpath,
        mode=mode,
        engine=engine,
        override_data_transformer=override_data_transformer,
        embed_options=embed_options,
    )


def plot_altair_heatmap(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    textvar: str,
    xsort: List[str],
    ysort: List[str],
    color_by_col: str,
    border_attrs: Dict[str, Union[str, float]],
    scale_params: Dict,
    text_fontsize: int,
    text_alt_condition: alt.expr,
    tooltip: List[alt.Tooltip],
    ptitle: alt.TitleParams,
    x_label_angle: int = 0,
    legend_title: Union[str, None] = None,
    xtitle: Union[str, None] = None,
    ytitle: Union[str, None] = None,
    fig_size: Dict[str, int] = dict(width=450, height=350),
    save_params: Dict[str, Union[str, bool, Dict]] = {},
) -> alt.Chart:
    """Plots a heatmap with annotated values using Altair.

    Args:
        df: Input DataFrame.
        xvar: Column for x-axis.
        yvar: Column for y-axis.
        textvar: Column for cell annotations.
        xsort: Sorting order for x-axis.
        ysort: Sorting order for y-axis.
        color_by_col: Column for color encoding.
        border_attrs: Rectangle styling attributes.
        scale_params: Color scale parameters.
        text_fontsize: Font size for annotations.
        text_alt_condition: Condition for text color.
        tooltip: Tooltip configuration.
        ptitle: Chart title.
        x_label_angle: Angle of x-axis labels.
        legend_title: Legend title.
        xtitle: X-axis title.
        ytitle: Y-axis title.
        fig_size: Chart dimensions.
        save_params: Parameters for saving chart.

    Returns:
        alt.Chart: Generated heatmap chart.

    Examples:
        >>> chart = plot_altair_heatmap(df, ...)
    """
    base = alt.Chart(df).encode(
        x=alt.X(
            xvar,
            title=xtitle,
            sort=xsort,
            axis=alt.Axis(labelAngle=x_label_angle),
        ),
        y=alt.Y(yvar, title=ytitle, sort=ysort, axis=alt.Axis(labelAngle=0)),
    )
    chart = base.mark_rect(**border_attrs).encode(
        color=alt.Color(
            color_by_col,
            scale=alt.Scale(**scale_params),
            legend=None if not legend_title else alt.Legend(title=legend_title),
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
    if save_params:
        export_altair_chart(alt_chart=chart, **save_params)
    return chart
