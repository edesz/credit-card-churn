# #!/usr/bin/env python3


# """Test visualization functions."""

# import altair as alt

# import src.cc_churn.viz_altair as vzu


# def test_scatter_chart_basic(sample_df_plot):
#     """Verifies scatter chart returns a valid Altair Chart object.

#     Args:
#         None
#     """
#     chart = vzu.plot_altair_scatter_chart(
#         df=sample_df_plot,
#         xvar="x",
#         yvar="y",
#         xtitle="X Axis",
#         ytitle="Y Axis",
#         color_by_col="category",
#         legend_title="Category",
#         ptitle=alt.TitleParams(text="Scatter Plot"),
#     )

#     assert isinstance(chart, alt.Chart)
#     spec = chart.to_dict()

#     assert spec["mark"]["type"] == "point"
#     assert spec["encoding"]["x"]["field"] == "x"
#     assert spec["encoding"]["y"]["field"] == "y"
#     assert spec["encoding"]["color"]["field"] == "category"


# def test_scatter_chart_properties(sample_df_plot):
#     """Verifies scatter chart includes title and size properties.

#     Args:
#         None
#     """
#     chart = vzu.plot_altair_scatter_chart(
#         df=sample_df_plot,
#         xvar="x",
#         yvar="y",
#         xtitle="X",
#         ytitle="Y",
#         color_by_col="category",
#         legend_title="Legend",
#         ptitle=alt.TitleParams(text="My Title"),
#     )

#     spec = chart.to_dict()

#     assert "title" in spec
#     assert spec["width"] == 700
#     assert spec["height"] == 350


# def test_bar_chart_basic(sample_df_plot):
#     """Verifies grouped bar chart structure and encoding.

#     Args:
#         None
#     """
#     chart = vzu.plot_grouped_overlapping_altair_bar_chart(
#         df=sample_df_plot,
#         xvar="group",
#         xtitle="Group",
#         ytitle="Count",
#         color_by_col="category",
#         legend_title="Legend",
#         ptitle=alt.TitleParams(text="Bar Chart"),
#     )

#     assert isinstance(chart, alt.Chart)

#     spec = chart.to_dict()

#     assert spec["mark"]["type"] == "bar"
#     assert spec["encoding"]["x"]["field"] == "group"
#     assert spec["encoding"]["y"]["aggregate"] == "count"
#     assert spec["encoding"]["color"]["field"] == "category"


# def test_bar_chart_scale_type(sample_df_plot):
#     """Verifies y-axis uses symlog scale.

#     Args:
#         None
#     """
#     chart = vzu.plot_grouped_overlapping_altair_bar_chart(
#         df=sample_df_plot,
#         xvar="group",
#         xtitle="Group",
#         ytitle="Count",
#         color_by_col="category",
#         legend_title="Legend",
#         ptitle=alt.TitleParams(text="Bar Chart"),
#     )

#     spec = chart.to_dict()

#     assert spec["encoding"]["y"]["scale"]["type"] == "symlog"


# def test_heatmap_basic(sample_df_plot):
#     """Verifies heatmap returns layered Altair chart.

#     Args:
#         None
#     """
#     chart = vzu.plot_altair_heatmap(
#         df=sample_df_plot,
#         xvar="group",
#         yvar="category",
#         textvar="value",
#         xtitle="Group",
#         ytitle="Category",
#         xsort=["A", "B"],
#         ysort=[True, False],
#         color_by_col="value",
#         legend_title="Value",
#         border_attrs={"stroke": "black"},
#         cmap="viridis",
#         text_fontsize=12,
#         text_alt_condition=alt.datum.value > 0.5,
#         tooltip=[alt.Tooltip("value:Q")],
#         ptitle=alt.TitleParams(text="Heatmap"),
#     )

#     assert isinstance(chart, alt.LayerChart)

#     spec = chart.to_dict()

#     # Ensure layering exists (rect + text)
#     assert "layer" in spec
#     assert len(spec["layer"]) == 2


# def test_heatmap_encodings(sample_df_plot):
#     """Verifies heatmap encodings for axes and color.

#     Args:
#         None
#     """
#     chart = vzu.plot_altair_heatmap(
#         df=sample_df_plot,
#         xvar="group",
#         yvar="category",
#         textvar="value",
#         xtitle="Group",
#         ytitle="Category",
#         xsort=["A", "B"],
#         ysort=[True, False],
#         color_by_col="value",
#         legend_title="Value",
#         border_attrs={"stroke": "black"},
#         cmap="viridis",
#         text_fontsize=12,
#         text_alt_condition=alt.datum.value > 0.5,
#         tooltip=[alt.Tooltip("value:Q")],
#         ptitle=alt.TitleParams(text="Heatmap"),
#     )

#     spec = chart.to_dict()

#     rect_layer = spec["layer"][0]

#     assert rect_layer["encoding"]["x"]["field"] == "group"
#     assert rect_layer["encoding"]["y"]["field"] == "category"
#     assert rect_layer["encoding"]["color"]["field"] == "value"


# def test_heatmap_text_layer(sample_df_plot):
#     """Verifies heatmap text layer includes formatted labels.

#     Args:
#         None
#     """
#     chart = vzu.plot_altair_heatmap(
#         df=sample_df_plot,
#         xvar="group",
#         yvar="category",
#         textvar="value",
#         xtitle="Group",
#         ytitle="Category",
#         xsort=["A", "B"],
#         ysort=[True, False],
#         color_by_col="value",
#         legend_title="Value",
#         border_attrs={"stroke": "black"},
#         cmap="viridis",
#         text_fontsize=12,
#         text_alt_condition=alt.datum.value > 0.5,
#         tooltip=[alt.Tooltip("value:Q")],
#         ptitle=alt.TitleParams(text="Heatmap"),
#     )

#     spec = chart.to_dict()

#     text_layer = spec["layer"][1]

#     assert text_layer["mark"]["type"] == "text"
#     assert text_layer["encoding"]["text"]["field"] == "value"


# def test_histogram_returns_chart(sample_df_plot):
#     """Verifies function returns an Altair Chart object.

#     Args:
#         None
#     """
#     chart = vzu.plot_grouped_overlapping_altair_histogram(
#         df=sample_df_plot,
#         num_bins=5,
#         xvar="value",
#         xtitle="Value",
#         ytitle="Count",
#         color_by_col="flag",
#         legend_title="Flag",
#         ptitle=alt.TitleParams(text="Histogram"),
#     )

#     assert isinstance(chart, alt.Chart)


# def test_histogram_mark_and_encoding(sample_df_plot):
#     """Verifies histogram uses bar mark and correct encodings.

#     Args:
#         None
#     """
#     chart = vzu.plot_grouped_overlapping_altair_histogram(
#         df=sample_df_plot,
#         num_bins=5,
#         xvar="value",
#         xtitle="Value",
#         ytitle="Count",
#         color_by_col="flag",
#         legend_title="Flag",
#         ptitle=alt.TitleParams(text="Histogram"),
#     )

#     spec = chart.to_dict()

#     assert spec["mark"]["type"] == "bar"
#     assert spec["encoding"]["x"]["field"] == "value"
#     assert spec["encoding"]["y"]["aggregate"] == "count"
#     assert spec["encoding"]["color"]["field"] == "flag"


# def test_histogram_binning_applied(sample_df_plot):
#     """Verifies binning is applied to x-axis with correct number of bins.

#     Args:
#         None
#     """
#     num_bins = 10

#     chart = vzu.plot_grouped_overlapping_altair_histogram(
#         df=sample_df_plot,
#         num_bins=num_bins,
#         xvar="value",
#         xtitle="Value",
#         ytitle="Count",
#         color_by_col="flag",
#         legend_title="Flag",
#         ptitle=alt.TitleParams(text="Histogram"),
#     )

#     spec = chart.to_dict()

#     assert spec["encoding"]["x"]["bin"]["maxbins"] == num_bins


# def test_histogram_properties(sample_df_plot):
#     """Verifies chart title and figure size properties.

#     Args:
#         None
#     """
#     chart = vzu.plot_grouped_overlapping_altair_histogram(
#         df=sample_df_plot,
#         num_bins=5,
#         xvar="value",
#         xtitle="Value",
#         ytitle="Count",
#         color_by_col="flag",
#         legend_title="Flag",
#         ptitle=alt.TitleParams(text="Histogram Title"),
#     )

#     spec = chart.to_dict()

#     assert "title" in spec
#     assert spec["width"] == 700
#     assert spec["height"] == 350


# def test_histogram_customization_applied(sample_df_plot):
#     """Verifies customize_altair_chart() config is applied.

#     Args:
#         None
#     """
#     chart = vzu.plot_grouped_overlapping_altair_histogram(
#         df=sample_df_plot,
#         num_bins=5,
#         xvar="value",
#         xtitle="Value",
#         ytitle="Count",
#         color_by_col="flag",
#         legend_title="Flag",
#         ptitle=alt.TitleParams(text="Histogram"),
#     )

#     spec = chart.to_dict()

#     # Check axis config from customize_altair_chart
#     axis_config = spec["config"]["axis"]
#     assert axis_config["ticks"] is False
#     assert axis_config["grid"] is False
#     assert axis_config["domain"] is False

#     # Check legend config exists
#     assert "legend" in spec["config"]


# def test_histogram_color_scale(sample_df_plot):
#     """Verifies custom color scale parameters are applied.

#     Args:
#         None
#     """
#     chart = vzu.plot_grouped_overlapping_altair_histogram(
#         df=sample_df_plot,
#         num_bins=5,
#         xvar="value",
#         xtitle="Value",
#         ytitle="Count",
#         color_by_col="flag",
#         legend_title="Flag",
#         ptitle=alt.TitleParams(text="Histogram"),
#         scale_params=dict(domain=[True, False], range=["blue", "grey"]),
#     )

#     spec = chart.to_dict()

#     scale = spec["encoding"]["color"]["scale"]

#     assert scale["domain"] == [True, False]
#     assert scale["range"] == ["blue", "grey"]


# def test_customize_altair_chart_returns_chart(sample_altair_chart):
#     """Verifies function returns an Altair Chart object.

#     Args:
#         None
#     """
#     customized = vzu.customize_altair_chart(sample_altair_chart)
#     assert isinstance(customized, alt.Chart)


# def test_customize_axis_properties(sample_altair_chart):
#     """Verifies axis configuration is correctly applied.

#     Args:
#         None
#     """
#     customized = vzu.customize_altair_chart(
#         sample_altair_chart,
#         labelAngle=45,
#         labelFontSize=12,
#         titleFontSize=14,
#     )

#     spec = customized.to_dict()

#     axis_config = spec["config"]["axis"]

#     assert axis_config["labelAngle"] == 45
#     assert axis_config["labelFontSize"] == 12
#     assert axis_config["titleFontSize"] == 14
#     assert axis_config["ticks"] is False
#     assert axis_config["grid"] is False
#     assert axis_config["domain"] is False


# def test_customize_legend_properties(sample_altair_chart):
#     """Verifies legend configuration is correctly applied.

#     Args:
#         None
#     """
#     customized = vzu.customize_altair_chart(
#         sample_altair_chart,
#         labelFontSize=11,
#         titleFontSize=13,
#         title_width=250,
#     )

#     spec = customized.to_dict()

#     legend_config = spec["config"]["legend"]

#     assert legend_config["labelFontSize"] == 11
#     assert legend_config["titleFontSize"] == 13
#     assert legend_config["titleLimit"] == 250


# def test_customize_view_properties(sample_altair_chart):
#     """Verifies view configuration removes border stroke.

#     Args:
#         None
#     """
#     customized = vzu.customize_altair_chart(sample_altair_chart)

#     spec = customized.to_dict()

#     view_config = spec["config"]["view"]

#     assert view_config["stroke"] is None


# def test_customize_preserves_chart_encoding(sample_altair_chart):
#     """Verifies original chart encodings are preserved after customization.

#     Args:
#         None
#     """
#     customized = vzu.customize_altair_chart(sample_altair_chart)

#     spec = customized.to_dict()

#     assert spec["encoding"]["x"]["field"] == "x"
#     assert spec["encoding"]["y"]["field"] == "y"
