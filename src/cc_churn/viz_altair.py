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
    label_angle_box: Union[int, float] = -70,
    plot_spacing: Union[int, float] = 25,
    tooltip: List[Union[str, alt.Tooltip, None]] = None,
    save_params: Dict[str, Union[str, bool, Dict]] = {},
    fig_size_hist: Dict[str, int] = dict(width=450, height=350),
    fig_size_bar: Dict[str, int] = dict(width=250, height=350),
) -> alt.Chart:
    """Plots grouped histogram and boxplot charts using Altair.

    This function creates a horizontally concatenated Altair visualization
    consisting of:

    1. An overlapping grouped histogram showing the distribution of a
    numeric variable across categories.
    2. A grouped boxplot summarizing the same variable distribution for
    each category.

    Both charts are colored by the same grouping variable and displayed
    side-by-side to support complementary distribution analysis.

    Args:
        df: The input pandas DataFrame containing plotting data.
        num_bins: The maximum number of histogram bins.
        xvar: The numeric column plotted on the histogram x-axis and
            boxplot y-axis.
        xtitle: The histogram x-axis title.
        ytitle: The histogram y-axis title.
        color_by_col: The categorical column used for grouped coloring.
        legend_title: The legend title for grouped categories.
        ptitle: The Altair chart title configuration.
        scale_params: The dictionary containing Altair color scale
            parameters.
        y_scale: The scale type for the histogram y-axis.
        label_angle_box: The rotation angle for boxplot category labels.
        plot_spacing: The spacing between the histogram and boxplot.
        tooltip: Optional tooltip configuration for the histogram.
        save_params: Optional parameters passed to
            `export_altair_chart()`.
        fig_size_hist: The histogram figure dimensions.
        fig_size_bar: The boxplot figure dimensions.

    Returns:
        alt.Chart: The horizontally concatenated Altair chart containing
            the grouped histogram and grouped boxplot visualizations.

    Examples:
        >>> chart = plot_grouped_overlapping_altair_histogram(
        ...     df=df,
        ...     num_bins=20,
        ...     xvar="credit_limit",
        ...     xtitle="Credit Limit",
        ...     ytitle="Count",
        ...     color_by_col="is_churned",
        ...     legend_title="Churn Status",
        ...     ptitle=alt.TitleParams("Credit Limit Distribution"),
        ... )
    """
    hist_encoding = dict(
        x=alt.X(xvar, bin=alt.Bin(maxbins=num_bins), title=xtitle),
        y=alt.Y(
            "count()",
            stack=None,
            scale=alt.Scale(type=y_scale),
            title=ytitle,
        ),
        color=alt.Color(
            color_by_col,
            title=legend_title,
            scale=alt.Scale(**scale_params),
        ),
    )
    if tooltip is not None:
        hist_encoding["tooltip"] = tooltip

    box_outliers_cfg = {"stroke": "#7f7f7f", "size": 50, "strokeWidth": 1.0}
    box_median_cfg = dict(strokeWidth=1.75, opacity=1.0, stroke="black")
    box_rule_cfg = {"strokeWidth": 3, "color": "#9498a0"}

    histogram = (
        alt.Chart(df)
        .mark_bar(opacity=0.6)
        .encode(**hist_encoding)
        .properties(title=ptitle, **fig_size_hist)
    )
    box = (
        alt.Chart(df)
        .mark_boxplot(
            extent=1.5,
            rule=box_rule_cfg,
            median=alt.MarkConfig(**box_median_cfg),
            outliers=box_outliers_cfg,
            size=20,
        )
        .encode(
            x=alt.X(
                color_by_col,
                axis=alt.Axis(labelAngle=label_angle_box),
                title=None,
            ),
            y=alt.Y(xvar, title=None),
            color=alt.Color(color_by_col, title=None),
        )
        .properties(**fig_size_bar)
    )
    chart = alt.hconcat(histogram, box).configure_concat(spacing=plot_spacing)
    chart = customize_altair_chart(chart)
    if save_params:
        export_altair_chart(alt_chart=chart, **save_params)
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
    save_params: Dict[str, Union[str, bool, Dict]] = {},
) -> alt.Chart:
    """Plots a scatter chart with color grouping using Altair.

    Args:
        df: Input DataFrame.
        xvar: Column for x-axis.
        yvar: Column for y-axis.
        xtitle: X-axis title.
        ytitle: Y-axis title.
        color_by_col: Column for color encoding.
        ptitle: Chart title.
        legend_title: Legend title.
        xscale: Scale type for x-axis.
        yscale: Scale type for y-axis.
        scale_params: Color scale parameters.
        fig_size: Chart dimensions.
        save_params: Parameters for saving chart.

    Returns:
        alt.Chart: Generated scatter chart.

    Examples:
        >>> chart = plot_altair_scatter_chart(df, ...)
    """
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
    if save_params:
        export_altair_chart(alt_chart=chart, **save_params)
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
    save_params: Dict[str, Union[str, bool, Dict]] = {},
) -> alt.Chart:
    """Plots grouped overlapping bar chart using Altair.

    Args:
        df: Input DataFrame.
        xvar: Column for x-axis.
        xtitle: X-axis title.
        ytitle: Y-axis title.
        color_by_col: Column for grouping colors.
        legend_title: Legend title.
        ptitle: Chart title.
        scale_params: Color scale parameters.
        y_scale: Scale type for y-axis.
        fig_size: Chart dimensions.
        save_params: Parameters for saving chart.

    Returns:
        alt.Chart: Generated bar chart.

    Examples:
        >>> chart = plot_grouped_overlapping_altair_bar_chart(df, ...)
    """
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
    if save_params:
        export_altair_chart(alt_chart=chart, **save_params)
    return chart


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
    save_params: Dict[str, Union[str, bool, Dict]] = {},
) -> alt.Chart:
    """Plots horizontally concatenated bar charts.

    Args:
        df: Input DataFrame.
        xvar: Column for first x-axis.
        xvar2: Column for second x-axis.
        yvar: Column for y-axis.
        xtitle: First x-axis title.
        xtitle2: Second x-axis title.
        ytitle: Y-axis title.
        y_sort: Sorting for y-axis.
        tooltip: Tooltip for first chart.
        tooltip2: Tooltip for second chart.
        ptitle: Chart title.
        x_scale: Scale type for x-axis.
        fig_size: Chart dimensions.
        save_params: Parameters for saving chart.

    Returns:
        alt.Chart: Combined bar chart.

    Examples:
        >>> chart = plot_altair_bar_chart(df, ...)
    """
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
    if save_params:
        export_altair_chart(alt_chart=chart, **save_params)
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
    save_params: Dict[str, Union[str, bool, Dict]] = {},
) -> alt.Chart:
    """Plots a bar chart with embedded text labels using Altair.

    Args:
        df: Input DataFrame.
        xvar: Column for x-axis.
        yvar: Column for y-axis.
        color_by_col: Column for color encoding.
        xtitle: X-axis title.
        ytitle: Y-axis title.
        xsort: Sorting for x-axis.
        ysort: Sorting for y-axis.
        ptitle: Chart title.
        text_color: Conditional text color.
        x_scale: Scale type for x-axis.
        y_scale: Scale type for y-axis.
        text_fmt: Format string for text labels.
        text_params: Text styling parameters.
        scale_params: Color scale parameters.
        fig_size: Chart dimensions.
        save_params: Parameters for saving chart.

    Returns:
        alt.Chart: Generated bar chart.

    Examples:
        >>> chart = plot_altair_simple_bar_chart(df, ...)
    """
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
    if save_params:
        export_altair_chart(alt_chart=chart, **save_params)
    return chart


def plot_uplift_curve(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    xtitle: str,
    ytitle: str,
    ptitle: alt.TitleParams,
    tooltip: List[alt.Tooltip],
    plateau: float = 0.05,
    save_params: Dict[str, Union[str, bool, Dict]] = {},
    fig_size: dict = dict(width=400, height=350),
) -> alt.Chart:
    """Generates a cumulative uplift curve.

    Args:
        df_uplift: DataFrame containing the percentile and uplift values.
        xvar: Name of the column to plot on the x-axis.
        yvar: Name of the column to plot on the y-axis.
        xtitle: Label for the x-axis.
        ytitle: Label for the y-axis.
        ptitle: Title configuration for the chart.
        tooltip: List of Altair tooltips to display on hover.
        plateau: The percentile of the elbow in uplift, to draw a vertical line.
        save_params: Parameters for saving chart.
        fig_size: Dictionary specifying width and height of the chart.

    Returns:
        A styled Altair Chart object showing the cumulative uplift curve.

    Example:
        >>> data = {'y': [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        ...         'p': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]}
        >>> df = pd.DataFrame(data)
        >>> df_uplift = calculate_cumulative_uplift(df, 'y', 'p')
        >>> chart = plot_uplift_curve(
        ...     df_uplift=df_uplift,
        ...     xvar='percentile',
        ...     yvar='uplift',
        ...     xtitle='Population Percentile',
        ...     ytitle='Cumulative Uplift',
        ...     ptitle=alt.TitleParams(text='Uplift Curve'),
        ...     tooltip=[alt.Tooltip('percentile'), alt.Tooltip('uplift')]
        ... )
        >>> chart
    """
    line = (
        alt.Chart(df)
        .mark_line(
            point={
                "filled": False,
                "fill": "white",
                "color": "darkred",
                "size": 80,
            },
            color="darkred",
        )
        .encode(
            x=alt.X(xvar, title=xtitle),
            y=alt.Y(yvar, title=ytitle),
            tooltip=tooltip,
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [plateau]}))
        .mark_rule(color="darkgreen", strokeDash=[5, 5])
        .encode(x="x:Q")
    )
    chart = customize_altair_chart(
        alt.layer(line, rule).properties(title=ptitle, **fig_size)
    )
    if save_params:
        export_altair_chart(alt_chart=chart, **save_params)
    return chart


def plot_gain_curve(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    xtitle: str,
    ytitle: str,
    ptitle: alt.TitleParams,
    tooltip: List[alt.Tooltip],
    plateau: float = 0.05,
    save_params: Dict[str, Union[str, bool, Dict]] = {},
    fig_size: dict = dict(width=400, height=350),
) -> alt.Chart:
    """Generates a gain curve showing cumulative positives found.

    Args:
        df: DataFrame containing the percentile and gain values.
        xvar: Name of the column to plot on the x-axis.
        yvar: Name of the column to plot on the y-axis.
        xtitle: Label for the x-axis.
        ytitle: Label for the y-axis.
        ptitle: Title configuration for the chart.
        tooltip: List of Altair tooltips to display on hover.
        plateau: The percentile of the elbow in gain, to draw a vertical line.
        save_params: Parameters for saving chart.
        fig_size: Dictionary specifying width and height of the chart.

    Returns:
        An Altair Chart object showing the cumulative gain curve.

    Example:
        >>> data = {'y': [1, 1, 0, 0, 0], 'p': [0.9, 0.8, 0.4, 0.2, 0.1]}
        >>> df = pd.DataFrame(data)
        >>> df_gain = calculate_cumulative_gain(df, 'y', 'p')
        >>> chart = plot_gain_curve(
        ...     df=df_gain,
        ...     xvar='percentile',
        ...     yvar='gain',
        ...     xtitle='Population Percentile',
        ...     ytitle='Cumulative Gain',
        ...     ptitle=alt.TitleParams(text='Gain Curve'),
        ...     tooltip=[alt.Tooltip('percentile'), alt.Tooltip('gain')]
        ... )
        >>> chart
    """
    # baseline (diagonal line representing random model)
    baseline = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(color="gray", strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q")
    )
    line = (
        alt.Chart(df)
        .mark_line(
            point={
                "filled": False,
                "fill": "white",
                "color": "darkred",
                "size": 80,
            },
            color="darkred",
        )
        .encode(
            x=alt.X(xvar, title=xtitle),
            y=alt.Y(yvar, title=ytitle),
            tooltip=tooltip,
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [plateau]}))
        .mark_rule(color="darkgreen", strokeDash=[5, 5])
        .encode(x="x:Q")
    )
    chart = customize_altair_chart(
        alt.layer(line, baseline, rule).properties(title=ptitle, **fig_size)
    )
    if save_params:
        export_altair_chart(alt_chart=chart, **save_params)
    return chart
