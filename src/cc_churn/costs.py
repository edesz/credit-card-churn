#!/usr/bin/env python3


"""Define helper functions to assign cost to model predictions."""


import numpy as np


def calc_predicted_savings(
    df,
    interchange_rate,
    apr,
    card_fees,
    multiplier,
    success_rate,
    intervention_cost,
):
    """Calculated predicted savings from CLV and success rate."""
    return df.assign(
        interchange_rev=lambda df: df["total_trans_amt"] * interchange_rate,
        interest_rev=lambda df: df["total_revolv_bal"] * apr,
        fee_rev=lambda df: df["card_category"].map(card_fees),
        annual_rev=lambda df: (
            df["interchange_rev"] + df["interest_rev"] + df["fee_rev"]
        ),
        clv=lambda df: df["annual_rev"] * multiplier,
        success_rate=success_rate,
        expected_savings=lambda df: (
            (df["y_pred_proba"] * df["success_rate"] * df["clv"])
            - intervention_cost
        ),
    )


def calc_true_savings(y_pred, is_churned, success_rate, clv, intervention_cost):
    """Calculate net savings relative to true outcome."""
    # intervene, actual churn
    if y_pred == 1 and is_churned == 1:
        return success_rate * clv - intervention_cost
    # intervene, but they’d stay anyway
    elif y_pred == 1 and is_churned == 0:
        return -intervention_cost
    # no intervention
    else:
        return 0


def get_cost(
    df,
    pred_proba_cutoff,
    interchange_rate,
    apr,
    card_fees,
    multiplier,
    success_rate,
    intervention_cost,
):
    """Calculate error in estimated savings."""
    df_costs = (
        calc_predicted_savings(
            df.query(f"y_pred_proba >= {pred_proba_cutoff}"),
            interchange_rate=interchange_rate,
            apr=apr,
            card_fees=card_fees,
            multiplier=multiplier,
            success_rate=success_rate,
            intervention_cost=intervention_cost,
        )
        # get true savings using true outcome (is_churned)
        .assign(
            true_savings=lambda df: np.vectorize(calc_true_savings)(
                df["y_pred"],
                df["is_churned"],
                success_rate,
                df["clv"],
                intervention_cost,
            )
        )
        # ORDER BY to get top recommendations first
        .sort_values("y_pred_proba", ascending=False, ignore_index=True)
        # Cumulative sums
        .assign(
            cum_pred_savings=lambda df: df["expected_savings"].cumsum(),
            cum_true_savings=lambda df: df["true_savings"].cumsum(),
            n=lambda df: np.arange(1, len(df) + 1),
            random_savings=lambda df: df["n"] * df["true_savings"].mean(),
        )
        # incremental lift
        .assign(
            lift=lambda df: df["cum_true_savings"] - df["random_savings"],
            lift_ml=lambda df: df["cum_pred_savings"] - df["random_savings"],
        )
        # ROI
        .assign(
            # 1. compute total intervention cost per N
            # (we assume fixed per-customer cost and we intervene on top N,
            # so number_intervened at top-N is just N)
            total_intervention_cost=lambda df: df["n"] * intervention_cost,
            # 2. ROI at top-N: ratio net_benefit / total_intervention_cost
            ROI=lambda df: (
                df["cum_true_savings"] / df["total_intervention_cost"]
            ),
            ROI_pred=lambda df: (
                df["cum_pred_savings"] / df["total_intervention_cost"]
            ),
            ROI_error=lambda df: 100 * (df["ROI"] - df["ROI_pred"]) / df["ROI"],
            ROI_percent=lambda df: df["ROI"] * 100,
            ROI_percent_pred=lambda df: df["ROI_pred"] * 100,
        )
    )
    pred_total = df_costs["expected_savings"].sum()
    true_total = df_costs["true_savings"].sum()
    # calculate business cost of model (error in predicted savings)
    cost_of_model = 100 * (true_total - pred_total) / true_total
    return [df_costs, pred_total, cost_of_model]
