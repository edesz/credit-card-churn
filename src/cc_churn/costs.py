#!/usr/bin/env python3


"""Define helper functions to assign cost to model predictions using ROI."""


from typing import Dict, Tuple

import numpy as np
import pandas as pd


def calc_predicted_savings(
    df: pd.DataFrame,
    interchange_rate: float,
    apr: float,
    card_fees: Dict[str, float],
    multiplier: float,
    success_rate: float,
    intervention_cost: float,
) -> pd.DataFrame:
    """Calculates expected customer-level savings based on predicted churn risk.

    This function derives multiple revenue components (interchange, interest,
    and fees) to estimate annual revenue and customer lifetime value (CLV).
    It then computes expected savings by combining predicted churn probability,
    intervention success rate, and CLV, adjusted for intervention cost.

    Args:
        df: Input DataFrame containing customer-level features, including
            `total_trans_amt`, `total_revolv_bal`, `card_category`,
            and `y_pred_proba`.
        interchange_rate: Proportion of transaction volume earned as revenue.
        apr: Annual percentage rate applied to revolving balances.
        card_fees: Mapping of card category to annual fee.
        multiplier: Factor used to convert annual revenue into CLV.
        success_rate: Probability that intervention successfully prevents churn.
        intervention_cost: Cost incurred per customer intervention.

    Returns:
        pd.DataFrame: A DataFrame with additional columns:
            - `interchange_rev`, `interest_rev`, `fee_rev`
            - `annual_rev`, `clv`
            - `success_rate`
            - `expected_savings`

    Examples:
        >>> df_out = calc_predicted_savings(
        ...     df,
        ...     interchange_rate=0.02,
        ...     apr=0.15,
        ...     card_fees={"Gold": 100, "Silver": 50},
        ...     multiplier=3.0,
        ...     success_rate=0.25,
        ...     intervention_cost=20.0,
        ... )
    """
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


def calc_true_savings(
    pred: int,
    true: int,
    success_rate: float,
    clv: float,
    intervention_cost: float,
) -> float:
    """Computes realized savings for a single prediction outcome.

    The function evaluates the net benefit of an intervention based on
    predicted action and actual churn outcome.

    Args:
        pred: Binary prediction (1 = intervene, 0 = no intervention).
        true: Actual churn outcome (1 = churned, 0 = retained).
        success_rate: Probability that intervention prevents churn.
        clv: Customer lifetime value.
        intervention_cost: Cost of applying the intervention.

    Returns:
        float: Net realized savings for the given case.

    Examples:
        >>> calc_true_savings(1, 1, 0.3, clv=1000, intervention_cost=50)
        250.0
    """
    # intervene, actual churn
    if pred == 1 and true == 1:
        return success_rate * clv - intervention_cost
    # intervene, but they would stay anyway
    elif pred == 1 and true == 0:
        return -intervention_cost
    # no intervention
    else:
        return 0.0


def get_cost(
    df: pd.DataFrame,
    pred_proba_cutoff: float,
    interchange_rate: float,
    apr: float,
    card_fees: Dict[str, float],
    multiplier: float,
    success_rate: float,
    intervention_cost: float,
    get_atrisk_only: bool = True,
) -> Tuple[pd.DataFrame, float, float]:
    """Evaluates model performance in terms of financial impact and ROI.

    This function filters customers above a prediction probability threshold,
    computes expected and true savings, and derives cumulative savings and ROI
    metrics. It also quantifies the cost of model error as the deviation between
    predicted and actual savings.

    Args:
        df: Input DataFrame containing predictions and true labels, including
            `y_pred_proba`, `y_pred`, and `is_churned`.
        pred_proba_cutoff: Threshold above which customers are selected for
            intervention.
        interchange_rate: Proportion of transaction volume earned as revenue.
        apr: Annual percentage rate applied to revolving balances.
        card_fees: Mapping of card category to annual fee.
        multiplier: Factor used to convert annual revenue into CLV.
        success_rate: Probability that intervention successfully prevents churn.
        intervention_cost: Cost incurred per customer intervention.
        get_atrisk_only: Whether to get at risk (True) or all (False) customers.

    Returns:
        Tuple[pd.DataFrame, float, float]:
            - DataFrame with detailed cost, savings, and ROI metrics
            - Total predicted savings (float)
            - Cost of model error as a percentage (float)

    Raises:
        KeyError: If required columns are missing from the input DataFrame.
        ZeroDivisionError: If true total savings is zero when computing model cost.

    Examples:
        >>> df_costs, pred_total, model_cost = get_cost(
        ...     df,
        ...     pred_proba_cutoff=0.6,
        ...     interchange_rate=0.02,
        ...     apr=0.15,
        ...     card_fees={"Gold": 100, "Silver": 50},
        ...     multiplier=3.0,
        ...     success_rate=0.25,
        ...     intervention_cost=20.0,
        ...     get_atrisk_only=True,
        ... )
    """
    df = (
        df.query(f"y_pred_proba >= {pred_proba_cutoff}")
        if get_atrisk_only
        else df
    )
    df_costs = (
        calc_predicted_savings(
            df,
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
    if true_total == 0:
        raise ZeroDivisionError(
            "True total savings is zero; cannot compute model cost."
        )
    else:
        cost_of_model = 100 * (true_total - pred_total) / true_total
    return [df_costs, pred_total, cost_of_model]
