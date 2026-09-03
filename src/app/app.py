from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import altair as alt
import pandas as pd
import panel as pn
from bokeh.resources import INLINE
from dotenv import load_dotenv
from great_tables import GT, loc, md, style

import app.costs_savings as costs_sv
import app.viz_altair as vzu
from app.data import export_to_csv, load_r2_data

_ = alt.renderers.set_embed_options(actions=False)
_ = pn.extension("vega", template="fast")
_ = pn.extension("tabulator")
_ = alt.renderers.enable("png")


PROJ_ROOT = Path.cwd().parent

assert load_dotenv(dotenv_path=PROJ_ROOT.parent / ".env")

# columns to load
columns = [
    "clientnum",
    "customer_age",
    "gender",
    "dependent_count",
    "education_level",
    "marital_status",
    "income_category",
    "card_category",
    "months_on_book",
    "num_products",
    "months_inactive_12_mon",
    "contacts_count_12_mon",
    "credit_limit",
    "total_revolv_bal",
    "avg_open_to_buy",
    "total_amt_chng_q4_q1",
    "total_trans_amt",
    "total_trans_ct",
    # "model_name",
    "total_ct_chng_q4_q1",
    "avg_utilization_ratio",
    "y_pred",
    "y_pred_proba",
    "is_churned",
]
dtypes_ordinals = {
    "income_category": "string[pyarrow]",
    "education_level": "string[pyarrow]",
}
dtypes_categoricals = {
    "gender": "string[pyarrow]",
    "marital_status": "string[pyarrow]",
    "card_category": "string[pyarrow]",
    "dependent_count": "string[pyarrow]",
}

# predictions prefix
# # folder containing predictions
prefix = "cloud-run"
# # prefix of filename with predictions
r2_key_pred = "all_predictions__"

# costs
# # dashboard inputs - part 1/2
interchange_rate = 0.02
apr = 0.18
success_rate = 0.40
intervention_cost = 50
# # not dashboard inputs
card_fees = {"Blue": 0, "Silver": 50, "Gold": 100, "Platinum": 200}
tenure_years = 3
discount = 0.9

# budget
# # dashboard inputs - part 2/2
budget = 20_000

# dashboard
_indicator_kwargs = dict(name="", font_size="24pt", default_color="black")
_card_kwargs = dict(
    collapsible=False,
    collapsed=False,
    # (Top, Right, Bottom, Left) - adds <Left>px left margin
    margin=(5, 10, 5, 10),
    sizing_mode="fixed",
    width=150,
    header_color="white",
)

reports_dir = PROJ_ROOT / "reports"
figures_dir = reports_dir / "figures"

fpath_dashboard = figures_dir / "fig_39_dashboard.html"

columns = tuple(columns)
dtypes_ordinals = frozenset(dtypes_ordinals.items())
dtypes_categoricals = frozenset(dtypes_categoricals.items())

multiplier = (1 - discount**tenure_years) / (1 - discount)


@lru_cache(maxsize=1)
def load_cached_data(
    prefix: str,
    r2_key_pred: str,
    columns: Tuple[str] | None,
    dtypes_categoricals: frozenset,
    dtypes_ordinals: frozenset,
    apr: float,
    multiplier: float,
    success_rate: float,
    intervention_cost: float,
) -> List[pd.DataFrame]:
    """Load and preprocess data once from R2."""
    df_business_metrics, df_summ_per_clv_tier_risk = load_r2_data(
        prefix,
        r2_key_pred,
        columns,
        {k: v for k, v in dtypes_categoricals},
        {k: v for k, v in dtypes_ordinals},
        interchange_rate,
        apr,
        card_fees,
        multiplier,
        success_rate,
        intervention_cost,
    )
    return [df_business_metrics, df_summ_per_clv_tier_risk]


def get_selected_customers(budget, intervention_cost):
    """Dynamically compute selected cohort from cached business metrics."""
    num_customers_max = int(budget / intervention_cost)
    df_business_metrics, df_summ_per_clv_tier_risk = load_cached_data(
        prefix,
        r2_key_pred,
        columns,
        dtypes_categoricals,
        dtypes_ordinals,
        apr,
        multiplier,
        success_rate,
        intervention_cost,
    )

    df_ranked, df_campaign_mix, _, _ = costs_sv.get_cohort_within_budget(
        df_business_metrics,
        df_summ_per_clv_tier_risk.query("expected_savings > 0"),
        intervention_cost=intervention_cost,
        n=num_customers_max,
    )
    return df_ranked, df_campaign_mix


def make_indicators(data):
    df_ranked, df_campaign_mix = data

    cohort_size = df_campaign_mix["num_customers"].sum()
    expected_savings = df_campaign_mix["expected_savings"].sum()
    risk = df_campaign_mix["y_pred_proba"].mean()
    num_segments = len(df_campaign_mix)

    customer_indicator = pn.Card(
        pn.indicators.Number(
            value=cohort_size,
            format="{value:,}",
            **_indicator_kwargs,
        ),
        header_background="teal",
        title="Selected Customers",
        styles={"border": "0.75px lightgrey", "border-radius": "5px"},
        **_card_kwargs,
    )
    savings_indicator = pn.Card(
        pn.indicators.Number(
            value=expected_savings,
            format="${value:,.0f}",
            **_indicator_kwargs,
        ),
        header_background="teal",
        title="Estimated Savings",
        styles={"border": "0.75px lightgrey", "border-radius": "5px"},
        **_card_kwargs,
    )
    risk_indicator = pn.Card(
        pn.indicators.Number(
            value=risk,
            format="{value:.2f}",
            **_indicator_kwargs,
        ),
        header_background="teal",
        title="Average Risk",
        styles={"border": "0.75px lightgrey", "border-radius": "5px"},
        **_card_kwargs,
    )
    segments_indicator = pn.Card(
        pn.indicators.Number(
            value=num_segments,
            format="{value:,}",
            **_indicator_kwargs,
        ),
        header_background="teal",
        title="Selected Segments",
        styles={"border": "0.75px lightgrey", "border-radius": "5px"},
        **_card_kwargs,
    )
    return pn.Row(
        customer_indicator,
        savings_indicator,
        risk_indicator,
        segments_indicator,
    )


def make_heatmap(data):
    _, df_campaign_mix = data

    text_alt_condition = alt.datum.expected_savings_per_customer > 400
    tooltip = [
        alt.Tooltip("num_customers:Q", title="Number of At-Risk Customers"),
        alt.Tooltip(
            "total_intervention_cost:Q", format=",", title="Total Cost"
        ),
        alt.Tooltip(
            "y_pred_proba:Q", format=",.2f", title="Avg. Predicted Probability"
        ),
        alt.Tooltip("clv:Q", format=",.2f", title="Avg. CLV"),
        alt.Tooltip(
            "expected_savings:Q", format=",.2f", title="Expected Savings"
        ),
        alt.Tooltip(
            "savings_error_pct:Q", format=",.2f", title="Savings Error (%)"
        ),
        alt.Tooltip("save_offer:N", title="Recommendation"),
    ]
    ptitle = alt.TitleParams(
        text="",
        fontSize=18,
        font="Arial",
        anchor="start",
        orient="top",
        dx=80,
        offset=10,
    )
    chart_budget = vzu.plot_altair_heatmap(
        df_campaign_mix,
        xvar="value_tier:N",
        yvar="risk_level:N",
        textvar="expected_savings_per_customer:Q",
        xsort=["Bronze", "Silver", "Gold", "Platinum"],
        ysort=["High", "Medium", "Low"],
        color_by_col="expected_savings_per_customer:Q",
        border_attrs=dict(stroke="white", strokeWidth=1.0),
        scale_params=dict(scheme="reds"),
        text_fontsize=18,
        text_alt_condition=text_alt_condition,
        tooltip=tooltip,
        ptitle=ptitle,
        legend_title="",
        xtitle="Revenue Tier (CLV)",
        ytitle="Churn Risk Level",
        save_params={},
        fig_size=dict(width=325, height=350),
    )

    pane = pn.pane.Vega(
        chart_budget,
        sizing_mode="stretch_width",
        # width=700,
        # height=400,
    )
    return pane


def make_summary_table(data):
    _, df_campaign_mix = data

    num_customers = df_campaign_mix["num_customers"].sum()
    gt_table = (
        GT(
            df_campaign_mix[
                [
                    "risk_level",
                    "value_tier",
                    "num_customers",
                    "total_intervention_cost",
                    "expected_savings",
                    "expected_savings_per_customer",
                ]
            ]
        )
        .tab_header(
            md(
                f"**Selected Cohort Covers {num_customers:,} Customers Across "
                f"{len(df_campaign_mix)} Segments**"
            )
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_header(),
        )
        .fmt_number(
            columns=[
                "expected_savings",
                "expected_savings_per_customer",
            ],
            decimals=2,
            sep_mark=",",
            dec_mark=".",
        )
        .fmt_number(
            columns=["total_intervention_cost"],
            decimals=0,
            sep_mark=",",
        )
    )

    pane = pn.pane.HTML(
        gt_table.as_raw_html(make_page=True),
        sizing_mode="stretch_width",
        # width=700,
    )
    return pane


def make_customer_table(data):
    df_ranked, _ = data

    gt_table_2 = (
        GT(
            df_ranked[
                [
                    "clientnum",
                    "customer_age",
                    "gender",
                    "marital_status",
                    "y_pred_proba",
                    "y_pred",
                    "interchange_rev",
                    "annual_rev",
                    "clv",
                    "expected_savings",
                    "risk_level",
                    "value_tier",
                    "save_offer",
                ]
            ].sample(20, random_state=88)
        )
        .tab_header(
            md(
                f"**Sample of 20 of the "
                f"{len(df_ranked):,} Selected Customers**"
            )
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_header(),
        )
        .tab_options(container_height="450px", container_overflow_y="auto")
        .fmt_number(
            columns=[
                "clv",
                "y_pred_proba",
                "interchange_rev",
                "annual_rev",
                "expected_savings",
            ],
            decimals=2,
        )
    )

    pane = pn.pane.HTML(
        gt_table_2.as_raw_html(make_page=True),
        sizing_mode="stretch_width",
    )
    return pane


inputs = dict(
    budget=pn.widgets.DiscreteSlider(
        name="Select the budget available for targeting",
        options=[20_000, 25_000, 30_000, 35_000, 40_000, 45_000],
        value=budget,
    ),
    interchange_rate=pn.widgets.DiscreteSlider(
        name="Select the interchange rate (transaction fees earned)",
        options=[0.02],
        value=interchange_rate,
        orientation="horizontal",
        disabled=True,
    ),
    apr=pn.widgets.DiscreteSlider(
        name="Select the APR (annual fees earned on balance)",
        options=[0.18],
        value=apr,
        orientation="horizontal",
        disabled=True,
    ),
    success_rate=pn.widgets.DiscreteSlider(
        name="Select the success rate",
        options=[0.40],
        value=success_rate,
        orientation="horizontal",
        disabled=True,
    ),
    intervention_cost=pn.widgets.DiscreteSlider(
        name="Select the cost of intervention per customer",
        options=[45, 50, 65, 75, 80, 90, 100, 110, 120],
        value=intervention_cost,
        orientation="horizontal",
    ),
)


selected_customers = pn.bind(
    get_selected_customers,
    budget=inputs["budget"],
    intervention_cost=inputs["intervention_cost"],
)


indicators_panel = pn.panel(
    pn.bind(make_indicators, selected_customers),
    loading_indicator=False,
)


heatmap_panel = pn.panel(
    pn.bind(make_heatmap, selected_customers),
    loading_indicator=False,
)


summary_table_panel = pn.panel(
    pn.bind(make_summary_table, selected_customers),
    loading_indicator=False,
)


card_heatmap_recommendations = pn.Card(
    heatmap_panel,
    title="Cohort Lookup Table as Heatmap",
    name="",
    header_background="teal",
    header_color="white",
    collapsible=True,
    collapsed=False,
    margin=(5, 10, 5, 10),
    sizing_mode="stretch_width",
    # width=700,
    styles={"flex": "1.5", "min-width": "200px"},
)
card_table_recommendations_summary = pn.Card(
    summary_table_panel,
    title="Cohort Summary",
    name="",
    header_background="papayawhip",
    header_color="black",
    collapsible=True,
    collapsed=False,
    margin=(5, 10, 5, 10),
    sizing_mode="stretch_width",
    # width=700,
    styles={"flex": "1.5", "min-width": "200px"},
)
layout_recommendations = pn.FlexBox(
    card_heatmap_recommendations,
    card_table_recommendations_summary,
    flex_direction="row",
    sizing_mode="stretch_width",
)


customer_table_panel = pn.panel(
    pn.bind(make_customer_table, selected_customers),
    loading_indicator=False,
)


card_customer_table = pn.Card(
    customer_table_panel,
    title="Selected Customers",
    header_background="darkgreen",
    header_color="white",
    collapsible=True,
    collapsed=False,
    sizing_mode="stretch_width",
)

column_label = pn.pane.Markdown("### Export Selected Customers", align="center")
download_button = pn.widgets.FileDownload(
    callback=pn.bind(export_to_csv, selected_customers()[0]),
    icon="download",
    filename="selected_cohort.csv",
    label="Export to CSV",
    button_type="primary",
)


dashboard_header = pn.Row(
    pn.pane.Markdown(
        "# Cohort Selection and Targeting Strategies Dashboard",
        styles={
            "color": "white",
            "margin-top": "0px",
            "margin-bottom": "0px",
        },
    ),
    pn.Spacer(sizing_mode="stretch_width"),
    pn.pane.Markdown("Churn Modeling", styles={"color": "lightgrey"}),
    styles={
        "background": "teal",
        "padding": "-5px 10px -5px 10px",
        "align-items": "center",
    },
)


dash = pn.Column(
    dashboard_header,
    pn.Row(pn.pane.Markdown("# Filter the data")),
    pn.Row(
        pn.Column(
            pn.Row(pn.pane.Markdown("## Step 1. Available Budget")),
            pn.Row(inputs["budget"]),
        ),
        pn.Column(
            pn.Row(pn.pane.Markdown("## Step 2. Intervention Cost")),
            pn.Row(inputs["intervention_cost"]),
        ),
    ),
    pn.Row(pn.pane.Markdown("## Fixed Values")),
    pn.Row(
        pn.Column(
            pn.Row(pn.pane.Markdown("## APR")),
            pn.Row(inputs["apr"]),
        ),
        pn.Column(
            pn.Row(pn.pane.Markdown("## Success Rate")),
            pn.Row(inputs["success_rate"]),
        ),
        pn.Column(
            pn.Row(pn.pane.Markdown("## Interchange Rate")),
            pn.Row(inputs["interchange_rate"]),
        ),
    ),
    pn.Row(
        pn.pane.HTML(
            "Performance Summary",
            styles={"font-size": "24px", "font-weight": "500"},
        )
    ),
    indicators_panel,
    pn.Spacer(height=20),
    pn.Row(
        pn.pane.HTML(
            "Recommended Segments and Targeting Strategies",
            styles={"font-size": "24px", "font-weight": "500"},
        )
    ),
    pn.Row(layout_recommendations),
    pn.Spacer(height=20),
    pn.Row(
        pn.pane.HTML(
            "Customers in Cohort",
            styles={"font-size": "24px", "font-weight": "500"},
        )
    ),
    pn.Column(
        pn.Row(card_customer_table),
        pn.Spacer(height=20),
        pn.Row(column_label, download_button),
    ),
)

dash.servable()
