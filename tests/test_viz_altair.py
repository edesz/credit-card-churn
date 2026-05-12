#!/usr/bin/env python3


"""Test altair visualization functions."""

import altair as alt
import pytest

import src.cc_churn.viz_altair as vzu


def check_save(
    expected_calls,
    mock_export,
    chart_type=None,
    expected_fpath="test.html",
) -> None:
    """Validates whether export_altair_chart was called correctly."""

    was_called = mock_export.get("called", False)

    if expected_calls == 1:
        assert was_called is True
        assert "kwargs" in mock_export

        kwargs = mock_export["kwargs"]

        if chart_type:
            assert isinstance(kwargs["alt_chart"], chart_type)
        else:
            import altair as alt

            assert isinstance(
                kwargs["alt_chart"],
                (
                    alt.Chart,
                    alt.LayerChart,
                    alt.HConcatChart,
                ),
            )

        assert kwargs["fpath"] == expected_fpath

    else:
        assert was_called is False or "kwargs" not in mock_export


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_scatter_chart_basic(
    sample_df_plot, mock_export, save_params, expected_calls
):
    """Verifies scatter chart returns a valid Altair Chart object.

    Args:
        None
    """
    chart = vzu.plot_altair_scatter_chart(
        df=sample_df_plot,
        xvar="x",
        yvar="y",
        xtitle="X Axis",
        ytitle="Y Axis",
        color_by_col="category",
        legend_title="Category",
        ptitle=alt.TitleParams(text="Scatter Plot"),
        save_params=save_params,
    )

    assert isinstance(chart, alt.Chart)
    spec = chart.to_dict()

    assert spec["mark"]["type"] == "circle"
    assert spec["encoding"]["x"]["field"] == "x"
    assert spec["encoding"]["y"]["field"] == "y"
    assert spec["encoding"]["color"]["field"] == "category"
    check_save(expected_calls, mock_export)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_scatter_chart_properties(
    sample_df_plot, mock_export, save_params, expected_calls
):
    """Verifies scatter chart includes title and size properties.

    Args:
        None
    """
    chart = vzu.plot_altair_scatter_chart(
        df=sample_df_plot,
        xvar="x",
        yvar="y",
        xtitle="X",
        ytitle="Y",
        color_by_col="category",
        legend_title="Legend",
        ptitle=alt.TitleParams(text="My Title"),
        save_params=save_params,
    )

    spec = chart.to_dict()

    assert "title" in spec
    assert spec["width"] == 700
    assert spec["height"] == 350
    check_save(expected_calls, mock_export)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_scatter_chart_scale(
    sample_df_plot, mock_export, save_params, expected_calls
):
    chart = vzu.plot_altair_scatter_chart(
        df=sample_df_plot,
        xvar="x",
        yvar="y",
        xtitle="X",
        ytitle="Y",
        color_by_col="category",
        ptitle=alt.TitleParams(text="Title"),
        xscale="log",
        yscale="sqrt",
        save_params=save_params,
    )

    spec = chart.to_dict()

    assert spec["encoding"]["x"]["scale"]["type"] == "log"
    assert spec["encoding"]["y"]["scale"]["type"] == "sqrt"
    check_save(expected_calls, mock_export)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_bar_chart_basic(
    sample_df_plot, mock_export, save_params, expected_calls
):
    chart = vzu.plot_grouped_overlapping_altair_bar_chart(
        df=sample_df_plot,
        xvar="group",
        xtitle="Group",
        ytitle="Count",
        color_by_col="category",
        legend_title="Legend",
        ptitle=alt.TitleParams(text="Bar Chart"),
        save_params=save_params,
    )

    spec = chart.to_dict()

    assert spec["mark"]["type"] == "bar"
    assert spec["encoding"]["x"]["field"] == "group"
    assert spec["encoding"]["y"]["aggregate"] == "count"
    check_save(expected_calls, mock_export)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
@pytest.mark.parametrize("scale", ["log", "symlog"])
def test_bar_chart_custom_scale(
    sample_df_plot, mock_export, save_params, expected_calls, scale
):
    chart = vzu.plot_grouped_overlapping_altair_bar_chart(
        df=sample_df_plot,
        xvar="group",
        xtitle="Group",
        ytitle="Count",
        color_by_col="category",
        legend_title="Legend",
        ptitle=alt.TitleParams(text="Bar Chart"),
        y_scale=scale,
        save_params=save_params,
    )

    spec = chart.to_dict()
    assert spec["encoding"]["y"]["scale"]["type"] == scale
    check_save(expected_calls, mock_export)


def test_histogram_encoding(sample_df_plot):
    """Verifies histogram layer encoding configuration."""
    chart = vzu.plot_grouped_overlapping_altair_histogram(
        df=sample_df_plot,
        num_bins=5,
        xvar="value",
        xtitle="Value",
        ytitle="Count",
        color_by_col="flag",
        legend_title="Flag",
        ptitle=alt.TitleParams(text="Histogram"),
    )

    spec = chart.to_dict()

    assert "hconcat" in spec
    assert len(spec["hconcat"]) == 2

    hist_spec = spec["hconcat"][0]

    assert hist_spec["mark"]["type"] == "bar"
    assert hist_spec["encoding"]["x"]["bin"]["maxbins"] == 5
    assert hist_spec["encoding"]["y"]["aggregate"] == "count"
    assert hist_spec["encoding"]["y"]["stack"] is None


@pytest.mark.parametrize("scale", ["log", "symlog", "linear"])
def test_histogram_y_scale(sample_df_plot, scale):
    """Verifies y-axis scaling is correctly applied."""
    chart = vzu.plot_grouped_overlapping_altair_histogram(
        df=sample_df_plot,
        num_bins=5,
        xvar="value",
        xtitle="Value",
        ytitle="Count",
        color_by_col="flag",
        legend_title="Flag",
        ptitle=alt.TitleParams(text="Histogram"),
        y_scale=scale,
    )

    spec = chart.to_dict()

    hist_spec = spec["hconcat"][0]

    assert hist_spec["encoding"]["y"]["scale"]["type"] == scale


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_histogram_returns_chart(
    sample_df_plot,
    mock_export,
    save_params,
    expected_calls,
):
    """Verifies function returns horizontally concatenated chart."""
    chart = vzu.plot_grouped_overlapping_altair_histogram(
        df=sample_df_plot,
        num_bins=5,
        xvar="value",
        xtitle="Value",
        ytitle="Count",
        color_by_col="flag",
        legend_title="Flag",
        ptitle=alt.TitleParams(text="Histogram"),
        save_params=save_params,
    )

    assert isinstance(chart, alt.HConcatChart)

    check_save(expected_calls, mock_export)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_histogram_mark_and_encoding(
    sample_df_plot,
    mock_export,
    save_params,
    expected_calls,
):
    """Verifies histogram uses correct mark and encoding."""
    chart = vzu.plot_grouped_overlapping_altair_histogram(
        df=sample_df_plot,
        num_bins=5,
        xvar="value",
        xtitle="Value",
        ytitle="Count",
        color_by_col="flag",
        legend_title="Flag",
        ptitle=alt.TitleParams(text="Histogram"),
        save_params=save_params,
    )

    spec = chart.to_dict()

    hist_spec = spec["hconcat"][0]

    assert hist_spec["mark"]["type"] == "bar"
    assert hist_spec["encoding"]["x"]["field"] == "value"
    assert hist_spec["encoding"]["y"]["aggregate"] == "count"
    assert hist_spec["encoding"]["color"]["field"] == "flag"

    check_save(expected_calls, mock_export)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_histogram_encoding_with_boxplot(
    sample_df_plot,
    mock_export,
    save_params,
    expected_calls,
):
    """Verifies histogram and boxplot are horizontally concatenated."""
    chart = vzu.plot_grouped_overlapping_altair_histogram(
        df=sample_df_plot,
        num_bins=5,
        xvar="value",
        xtitle="Value",
        ytitle="Count",
        color_by_col="flag",
        legend_title="Flag",
        ptitle=alt.TitleParams(text="Histogram"),
        save_params=save_params,
    )

    spec = chart.to_dict()

    assert "hconcat" in spec
    assert len(spec["hconcat"]) == 2

    hist_spec = spec["hconcat"][0]
    box_spec = spec["hconcat"][1]

    assert hist_spec["mark"]["type"] == "bar"
    assert box_spec["mark"]["type"] == "boxplot"

    assert hist_spec["encoding"]["x"]["bin"]["maxbins"] == 5
    assert hist_spec["encoding"]["y"]["aggregate"] == "count"
    assert hist_spec["encoding"]["y"]["stack"] is None

    check_save(expected_calls, mock_export)


@pytest.mark.parametrize(
    "tooltip, expect_tooltip",
    [
        (None, False),
        ([alt.Tooltip("value:Q")], True),
    ],
)
@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_histogram_tooltip_and_y_scale(
    sample_df_plot,
    mock_export,
    tooltip,
    expect_tooltip,
    save_params,
    expected_calls,
):
    """Verifies tooltip and y-scale configuration."""
    chart = vzu.plot_grouped_overlapping_altair_histogram(
        df=sample_df_plot,
        num_bins=5,
        xvar="value",
        xtitle="Value",
        ytitle="Count",
        color_by_col="flag",
        legend_title="Flag",
        ptitle=alt.TitleParams(text="Histogram"),
        y_scale="log",
        tooltip=tooltip,
        save_params=save_params,
    )

    spec = chart.to_dict()

    hist_spec = spec["hconcat"][0]

    assert hist_spec["encoding"]["y"]["scale"]["type"] == "log"

    if expect_tooltip:
        tooltip_enc = hist_spec["encoding"]["tooltip"]

        assert isinstance(tooltip_enc, list)
        assert tooltip_enc[0]["field"] == "value"
        assert tooltip_enc[0]["type"] == "quantitative"
    else:
        assert "tooltip" not in hist_spec["encoding"]

    check_save(expected_calls, mock_export)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_histogram_binning_applied(
    sample_df_plot,
    mock_export,
    save_params,
    expected_calls,
):
    """Verifies histogram binning configuration."""
    num_bins = 10

    chart = vzu.plot_grouped_overlapping_altair_histogram(
        df=sample_df_plot,
        num_bins=num_bins,
        xvar="value",
        xtitle="Value",
        ytitle="Count",
        color_by_col="flag",
        legend_title="Flag",
        ptitle=alt.TitleParams(text="Histogram"),
        save_params=save_params,
    )

    spec = chart.to_dict()

    hist_spec = spec["hconcat"][0]

    assert hist_spec["encoding"]["x"]["bin"]["maxbins"] == num_bins

    check_save(expected_calls, mock_export)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_histogram_properties(
    sample_df_plot,
    mock_export,
    save_params,
    expected_calls,
):
    """Verifies chart properties and concatenation spacing."""
    chart = vzu.plot_grouped_overlapping_altair_histogram(
        df=sample_df_plot,
        num_bins=5,
        xvar="value",
        xtitle="Value",
        ytitle="Count",
        color_by_col="flag",
        legend_title="Flag",
        ptitle=alt.TitleParams(text="Histogram Title"),
        save_params=save_params,
    )

    spec = chart.to_dict()

    assert "hconcat" in spec
    assert spec["config"]["concat"]["spacing"] == 25

    hist_spec = spec["hconcat"][0]
    box_spec = spec["hconcat"][1]

    assert hist_spec["width"] == 450
    assert hist_spec["height"] == 350

    assert box_spec["width"] == 250
    assert box_spec["height"] == 350

    check_save(expected_calls, mock_export)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_histogram_customization_applied(
    sample_df_plot,
    mock_export,
    save_params,
    expected_calls,
):
    """Verifies customize_altair_chart configuration is applied."""
    chart = vzu.plot_grouped_overlapping_altair_histogram(
        df=sample_df_plot,
        num_bins=5,
        xvar="value",
        xtitle="Value",
        ytitle="Count",
        color_by_col="flag",
        legend_title="Flag",
        ptitle=alt.TitleParams(text="Histogram"),
        save_params=save_params,
    )

    spec = chart.to_dict()

    axis_config = spec["config"]["axis"]

    assert axis_config["ticks"] is False
    assert axis_config["grid"] is False
    assert axis_config["domain"] is False

    assert "legend" in spec["config"]

    check_save(expected_calls, mock_export)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_histogram_color_scale(
    sample_df_plot,
    mock_export,
    save_params,
    expected_calls,
):
    """Verifies custom color scale configuration."""
    chart = vzu.plot_grouped_overlapping_altair_histogram(
        df=sample_df_plot,
        num_bins=5,
        xvar="value",
        xtitle="Value",
        ytitle="Count",
        color_by_col="flag",
        legend_title="Flag",
        ptitle=alt.TitleParams(text="Histogram"),
        scale_params=dict(
            domain=[True, False],
            range=["blue", "grey"],
        ),
        save_params=save_params,
    )

    spec = chart.to_dict()

    hist_spec = spec["hconcat"][0]

    scale = hist_spec["encoding"]["color"]["scale"]

    assert scale["domain"] == [True, False]
    assert scale["range"] == ["blue", "grey"]

    check_save(expected_calls, mock_export)


def test_boxplot_configuration(sample_df_plot):
    """Verifies boxplot configuration and encodings."""
    chart = vzu.plot_grouped_overlapping_altair_histogram(
        df=sample_df_plot,
        num_bins=5,
        xvar="value",
        xtitle="Value",
        ytitle="Count",
        color_by_col="flag",
        legend_title="Flag",
        ptitle=alt.TitleParams(text="Histogram"),
    )

    spec = chart.to_dict()

    box_spec = spec["hconcat"][1]

    assert box_spec["mark"]["type"] == "boxplot"

    assert box_spec["encoding"]["x"]["field"] == "flag"
    assert box_spec["encoding"]["y"]["field"] == "value"
    assert box_spec["encoding"]["color"]["field"] == "flag"

    assert box_spec["mark"]["extent"] == 1.5


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_heatmap_basic(
    sample_df_plot, mock_export, save_params, expected_calls
):
    chart = vzu.plot_altair_heatmap(
        df=sample_df_plot,
        xvar="group",
        yvar="category",
        textvar="value",
        xsort=["A", "B"],
        ysort=[True, False],
        color_by_col="value",
        border_attrs={"stroke": "black"},
        scale_params={"scheme": "viridis"},
        text_fontsize=12,
        text_alt_condition=alt.datum.value > 0.5,
        tooltip=[alt.Tooltip("value:Q")],
        ptitle=alt.TitleParams(text="Heatmap"),
        legend_title="Legend",
        save_params=save_params,
    )

    assert isinstance(chart, alt.LayerChart)

    spec = chart.to_dict()
    assert "layer" in spec
    assert len(spec["layer"]) == 2
    check_save(expected_calls, mock_export, alt.LayerChart)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_heatmap_no_legend(
    sample_df_plot, mock_export, save_params, expected_calls
):
    chart = vzu.plot_altair_heatmap(
        df=sample_df_plot,
        xvar="group",
        yvar="category",
        textvar="value",
        xsort=["A", "B"],
        ysort=[True, False],
        color_by_col="value",
        border_attrs={},
        scale_params={"scheme": "viridis"},
        text_fontsize=12,
        text_alt_condition=alt.datum.value > 0.5,
        tooltip=[],
        ptitle=alt.TitleParams(text="Heatmap"),
        legend_title=None,
        save_params=save_params,
    )

    spec = chart.to_dict()
    rect_layer = spec["layer"][0]

    assert rect_layer["encoding"]["color"]["legend"] is None

    check_save(expected_calls, mock_export, alt.LayerChart)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_heatmap_encodings(
    sample_df_plot, mock_export, save_params, expected_calls
):
    """Verifies heatmap encodings for axes and color.

    Args:
        None
    """
    chart = vzu.plot_altair_heatmap(
        df=sample_df_plot,
        xvar="group",
        yvar="category",
        textvar="value",
        xsort=["A", "B"],
        ysort=[True, False],
        color_by_col="value",
        border_attrs={"stroke": "black"},
        scale_params={"scheme": "viridis"},
        text_fontsize=12,
        text_alt_condition=alt.datum.value > 0.5,
        tooltip=[alt.Tooltip("value:Q")],
        ptitle=alt.TitleParams(text="Heatmap"),
        legend_title="Value",
        save_params=save_params,
    )

    spec = chart.to_dict()

    rect_layer = spec["layer"][0]

    assert rect_layer["encoding"]["x"]["field"] == "group"
    assert rect_layer["encoding"]["y"]["field"] == "category"
    assert rect_layer["encoding"]["color"]["field"] == "value"

    check_save(expected_calls, mock_export, alt.LayerChart)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_heatmap_text_layer(
    sample_df_plot, mock_export, save_params, expected_calls
):
    """Verifies heatmap text layer includes formatted labels.

    Args:
        None
    """
    chart = vzu.plot_altair_heatmap(
        df=sample_df_plot,
        xvar="group",
        yvar="category",
        textvar="value",
        xsort=["A", "B"],
        ysort=[True, False],
        color_by_col="value",
        border_attrs={"stroke": "black"},
        scale_params={"scheme": "viridis"},
        text_fontsize=12,
        text_alt_condition=alt.datum.value > 0.5,
        tooltip=[alt.Tooltip("value:Q")],
        ptitle=alt.TitleParams(text="Heatmap"),
        legend_title=None,
        save_params=save_params,
    )

    spec = chart.to_dict()

    text_layer = spec["layer"][1]

    assert text_layer["mark"]["type"] == "text"
    assert text_layer["encoding"]["text"]["field"] == "value"

    check_save(expected_calls, mock_export, alt.LayerChart)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_simple_bar_chart_layered(
    sample_df_plot, mock_export, save_params, expected_calls
):
    chart = vzu.plot_altair_simple_bar_chart(
        df=sample_df_plot,
        xvar="value",
        yvar="group",
        color_by_col="flag",
        xtitle="Value",
        ytitle="Group",
        xsort=None,
        ysort=None,
        ptitle=alt.TitleParams(text="Simple Bar"),
        text_color=alt.value("white"),
        save_params=save_params,
    )

    assert isinstance(chart, alt.LayerChart)

    spec = chart.to_dict()
    assert "layer" in spec
    assert len(spec["layer"]) == 2

    check_save(expected_calls, mock_export, alt.LayerChart)


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "test.html"}, 1), ({}, 0)],
)
def test_concat_bar_chart(
    sample_df_plot, mock_export, save_params, expected_calls
):
    chart = vzu.plot_altair_bar_chart(
        df=sample_df_plot,
        xvar="value",
        xvar2="value",
        yvar="group",
        xtitle="Frac",
        xtitle2="Count",
        ytitle="Group",
        y_sort=None,
        tooltip=["value"],
        tooltip2=["value"],
        ptitle=alt.TitleParams(text="Bar"),
        save_params=save_params,
    )

    assert isinstance(chart, alt.HConcatChart)

    spec = chart.to_dict()
    assert "hconcat" in spec

    check_save(expected_calls, mock_export, alt.HConcatChart)


def test_save_called(monkeypatch, sample_df_plot):
    called = {}

    def mock_save(*args, **kwargs):
        called["yes"] = True

    monkeypatch.setattr(vzu, "export_altair_chart", mock_save)

    vzu.plot_altair_scatter_chart(
        df=sample_df_plot,
        xvar="x",
        yvar="y",
        xtitle="X",
        ytitle="Y",
        color_by_col="category",
        ptitle=alt.TitleParams(text="Test"),
        save_params={"fpath": "test.html"},
    )

    assert called.get("yes", False)


def test_customize_altair_chart_returns_chart(sample_altair_chart):
    """Verifies function returns an Altair Chart object.

    Args:
        None
    """
    customized = vzu.customize_altair_chart(sample_altair_chart)
    assert isinstance(customized, alt.Chart)


def test_customize_axis_properties(sample_altair_chart):
    """Verifies axis configuration is correctly applied.

    Args:
        None
    """
    customized = vzu.customize_altair_chart(
        sample_altair_chart,
        labelAngle=45,
        labelFontSize=12,
        titleFontSize=14,
    )

    spec = customized.to_dict()

    axis_config = spec["config"]["axis"]

    assert axis_config["labelAngle"] == 45
    assert axis_config["labelFontSize"] == 12
    assert axis_config["titleFontSize"] == 14
    assert axis_config["ticks"] is False
    assert axis_config["grid"] is False
    assert axis_config["domain"] is False


def test_customize_legend_properties(sample_altair_chart):
    """Verifies legend configuration is correctly applied.

    Args:
        None
    """
    customized = vzu.customize_altair_chart(
        sample_altair_chart,
        labelFontSize=11,
        titleFontSize=13,
        title_width=250,
    )

    spec = customized.to_dict()

    legend_config = spec["config"]["legend"]

    assert legend_config["labelFontSize"] == 11
    assert legend_config["titleFontSize"] == 13
    assert legend_config["titleLimit"] == 250


def test_customize_view_properties(sample_altair_chart):
    """Verifies view configuration removes border stroke.

    Args:
        None
    """
    customized = vzu.customize_altair_chart(sample_altair_chart)

    spec = customized.to_dict()

    view_config = spec["config"]["view"]

    assert view_config["stroke"] is None


def test_customize_preserves_chart_encoding(sample_altair_chart):
    """Verifies original chart encodings are preserved after customization.

    Args:
        None
    """
    customized = vzu.customize_altair_chart(sample_altair_chart)

    spec = customized.to_dict()

    assert spec["encoding"]["x"]["field"] == "x"
    assert spec["encoding"]["y"]["field"] == "y"


def test_customize_new_params(sample_altair_chart):
    chart = vzu.customize_altair_chart(
        sample_altair_chart,
        label_limit=123,
        legend_label_limit=456,
    )

    spec = chart.to_dict()

    assert spec["config"]["axis"]["labelLimit"] == 123
    assert spec["config"]["legend"]["labelLimit"] == 456
    assert spec["config"]["legend"]["symbolOpacity"] == 1


def test_save_called(monkeypatch, sample_df_plot):
    called = {}

    def mock_save(*args, **kwargs):
        called["yes"] = True

    monkeypatch.setattr(vzu, "export_altair_chart", mock_save)

    vzu.plot_altair_scatter_chart(
        df=sample_df_plot,
        xvar="x",
        yvar="y",
        xtitle="X",
        ytitle="Y",
        color_by_col="category",
        ptitle=alt.TitleParams(text="Test"),
        save_params={"fpath": "test.html"},
    )

    assert called.get("yes", False)


def test_export_altair_chart_calls_save(monkeypatch):
    called = {}

    def mock_save(self, fpath, **kwargs):
        called["fpath"] = fpath
        called["kwargs"] = kwargs

    # patch altair's alt.Chart.save globally
    monkeypatch.setattr(alt.Chart, "save", mock_save)

    fake_chart = alt.Chart({"data": []})

    vzu.export_altair_chart(
        alt_chart=fake_chart,
        fpath="test.html",
        mode="vega-lite",
        engine="vl-convert",
        override_data_transformer=True,
        embed_options={"renderer": "svg"},
    )

    assert called["fpath"] == "test.html"

    assert called["kwargs"] == {
        "mode": "vega-lite",
        "engine": "vl-convert",
        "override_data_transformer": True,
        "embed_options": {"renderer": "svg"},
    }


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "uplift.html"}, 1), ({}, 0)],
)
def test_uplift_curve_basic(
    df_uplift, mock_export, save_params, expected_calls
):
    """Verifies uplift curve returns layered Altair chart."""
    chart = vzu.plot_uplift_curve(
        df=df_uplift,
        xvar="percentile",
        yvar="uplift",
        xtitle="Percentile",
        ytitle="Uplift",
        ptitle=alt.TitleParams(text="Uplift Curve"),
        tooltip=[alt.Tooltip("percentile"), alt.Tooltip("uplift")],
        save_params=save_params,
    )

    assert isinstance(chart, alt.LayerChart)

    spec = chart.to_dict()

    assert "layer" in spec
    assert len(spec["layer"]) == 2

    check_save(
        expected_calls,
        mock_export,
        expected_fpath=save_params.get("fpath", "test.html"),
    )


def test_uplift_curve_encodings(df_uplift):
    """Verifies uplift curve encoding fields."""
    chart = vzu.plot_uplift_curve(
        df=df_uplift,
        xvar="percentile",
        yvar="uplift",
        xtitle="Percentile",
        ytitle="Uplift",
        ptitle=alt.TitleParams(text="Uplift Curve"),
        tooltip=[alt.Tooltip("percentile"), alt.Tooltip("uplift")],
    )

    spec = chart.to_dict()

    line_layer = spec["layer"][0]

    assert line_layer["mark"]["type"] == "line"
    assert line_layer["encoding"]["x"]["field"] == "percentile"
    assert line_layer["encoding"]["y"]["field"] == "uplift"


def test_uplift_curve_rule_line(df_uplift):
    """Verifies vertical rule is added at plateau."""
    plateau = 0.4

    chart = vzu.plot_uplift_curve(
        df=df_uplift,
        xvar="percentile",
        yvar="uplift",
        xtitle="Percentile",
        ytitle="Uplift",
        ptitle=alt.TitleParams(text="Uplift Curve"),
        tooltip=[alt.Tooltip("percentile")],
        plateau=plateau,
    )

    spec = chart.to_dict()

    rule_layer = spec["layer"][1]

    assert rule_layer["mark"]["type"] == "rule"
    assert rule_layer["encoding"]["x"]["field"] == "x"


def test_uplift_curve_tooltip(df_uplift):
    """Verifies tooltip is correctly applied."""
    tooltip = [alt.Tooltip("percentile"), alt.Tooltip("uplift")]

    chart = vzu.plot_uplift_curve(
        df=df_uplift,
        xvar="percentile",
        yvar="uplift",
        xtitle="Percentile",
        ytitle="Uplift",
        ptitle=alt.TitleParams(text="Uplift Curve"),
        tooltip=tooltip,
    )

    spec = chart.to_dict()

    encoding_tooltip = spec["layer"][0]["encoding"]["tooltip"]

    assert isinstance(encoding_tooltip, list)
    assert encoding_tooltip[0]["field"] == "percentile"


@pytest.mark.parametrize(
    "save_params, expected_calls",
    [({"fpath": "gain.html"}, 1), ({}, 0)],
)
def test_gain_curve_basic(df_gain, mock_export, save_params, expected_calls):
    """Verifies gain curve returns layered chart."""
    chart = vzu.plot_gain_curve(
        df=df_gain,
        xvar="percentile",
        yvar="gain",
        xtitle="Percentile",
        ytitle="Gain",
        ptitle=alt.TitleParams(text="Gain Curve"),
        tooltip=[alt.Tooltip("percentile"), alt.Tooltip("gain")],
        save_params=save_params,
    )

    assert isinstance(chart, alt.LayerChart)

    spec = chart.to_dict()

    assert "layer" in spec
    assert len(spec["layer"]) == 3

    assert spec["layer"][0]["mark"]["type"] == "line"
    assert spec["layer"][1]["mark"]["type"] == "line"
    assert spec["layer"][2]["mark"]["type"] == "rule"

    check_save(
        expected_calls,
        mock_export,
        expected_fpath=save_params.get("fpath", "test.html"),
    )


def test_gain_curve_encodings(df_gain):
    """Verifies gain curve encoding fields."""
    chart = vzu.plot_gain_curve(
        df=df_gain,
        xvar="percentile",
        yvar="gain",
        xtitle="Percentile",
        ytitle="Gain",
        ptitle=alt.TitleParams(text="Gain Curve"),
        tooltip=[alt.Tooltip("percentile"), alt.Tooltip("gain")],
    )

    spec = chart.to_dict()

    line_layer = spec["layer"][0]

    assert line_layer["mark"]["type"] == "line"
    assert line_layer["encoding"]["x"]["field"] == "percentile"
    assert line_layer["encoding"]["y"]["field"] == "gain"


def test_gain_curve_baseline(df_gain):
    """Verifies baseline diagonal line exists."""
    chart = vzu.plot_gain_curve(
        df=df_gain,
        xvar="percentile",
        yvar="gain",
        xtitle="Percentile",
        ytitle="Gain",
        ptitle=alt.TitleParams(text="Gain Curve"),
        tooltip=[alt.Tooltip("percentile")],
    )

    spec = chart.to_dict()

    baseline_layer = spec["layer"][1]

    assert baseline_layer["mark"]["type"] == "line"
    assert baseline_layer["encoding"]["x"]["field"] == "x"
    assert baseline_layer["encoding"]["y"]["field"] == "y"


def test_gain_curve_tooltip(df_gain):
    """Verifies tooltip is correctly applied."""
    tooltip = [alt.Tooltip("percentile"), alt.Tooltip("gain")]

    chart = vzu.plot_gain_curve(
        df=df_gain,
        xvar="percentile",
        yvar="gain",
        xtitle="Percentile",
        ytitle="Gain",
        ptitle=alt.TitleParams(text="Gain Curve"),
        tooltip=tooltip,
    )

    spec = chart.to_dict()

    encoding_tooltip = spec["layer"][0]["encoding"]["tooltip"]

    assert isinstance(encoding_tooltip, list)
    assert encoding_tooltip[0]["field"] == "percentile"
