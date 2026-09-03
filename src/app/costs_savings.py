#!/usr/bin/env python3


"""Define helper functions to assign cost to model predictions using savings."""

from typing import List, Optional, Union

import pandas as pd


def get_save_offer(row: pd.Series) -> str:
    """Assigns a retention offer based on customer risk level and value tier.

    Args:
        row: A pandas Series containing "risk_level" and "value_tier" keys.

    Returns:
        str: A string describing specific marketing or service intervention.
    """
    risk, tier = [row["risk_level"], row["value_tier"]]
    if tier == "Platinum":
        if risk == "High":
            return (
                "Personalized Executive Call + 1-Year Fee Waiver + 25k Points"
            )
        return (
            "Priority Relationship Manager Check-in + Luxury Lounge Access "
            "Upgrade"
        )
    elif tier == "Gold":
        if risk == "High":
            return "Retention Team Call + 50% Annual Fee Rebate + 10k Points"
        return "Personalized Email + 0% Balance Transfer Offer for 6 Months"
    elif tier == "Silver":
        if risk == "High":
            return (
                "Automated 'We Miss You' Email + 5k Bonus Points on Next Spend"
            )
        return "Targeted In-App Notification + New Category Cashback Boost"
    else:
        if risk == "High":
            return "Automated Email + Satisfaction Survey"
        return "Standard Monthly Newsletter + Feature Education"


def add_binned_column(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    bins: Union[int, List[float]],
    labels: Optional[List[str]] = None,
    method: str = "cut",
    **kwargs,
) -> pd.DataFrame:
    """Appends a new binned column to a DataFrame using pandas cut or qcut.

    Args:
        df: The input DataFrame.
        source_col: The name of the existing numeric column to bin.
        target_col: The name of the new column to be created.
        bins: Number of bins or specific bin edges.
        labels: Names for the resulting bins.
        method: 'cut' for fixed edges, 'qcut' for quantiles. Defaults to 'cut'.
        **kwargs: Additional arguments (e.g., include_lowest) passed to pandas.

    Returns:
        pd.DataFrame: The DataFrame with the new binned column appended.
    """
    if method == "qcut":
        df[target_col] = pd.qcut(
            df[source_col], q=bins, labels=labels, **kwargs
        )
    else:
        df[target_col] = pd.cut(
            df[source_col], bins=bins, labels=labels, **kwargs
        )
    return df


def get_buckets(
    df: pd.DataFrame,
    bins_risk_values: List[float] = [0.72, 0.81, 0.9, 1.0],
    bins_risk_labels: List[str] = ["Low", "Medium", "High"],
    bins_clv_tier_values: List[float] = [0, 0.4, 0.7, 0.9, 1.0],
    bins_clv_tier_labels: List[str] = ["Bronze", "Silver", "Gold", "Platinum"],
) -> pd.DataFrame:
    """
    Segments customers by churn risk and CLV value tiers to assign save offers.

    This function applies categorical binning to 'y_pred_proba' (Risk) and 'clv' (Value).
    It then maps these combinations to specific 'save_offer' strategies using
    the get_save_offer logic.

    Args:
        df: Input DataFrame containing 'y_pred_proba' and 'clv' columns.
        bins_risk_values: Numeric boundaries for churn probability levels.
        bins_risk_labels: Descriptive labels for risk levels (e.g., Low, High).
        bins_clv_tier_values: Quantile boundaries (0.0 to 1.0) for CLV segmentation.
        bins_clv_tier_labels: Descriptive labels for value tiers (e.g., Bronze, Gold).

    Returns:
        pd.DataFrame: The original DataFrame enriched with 'risk_level',
            'value_tier', and 'save_offer' columns.
    """
    df = (
        df.pipe(
            add_binned_column,
            source_col="y_pred_proba",
            target_col="risk_level",
            bins=bins_risk_values,
            labels=bins_risk_labels,
            include_lowest=True,
        )
        .pipe(
            add_binned_column,
            source_col="clv",
            target_col="value_tier",
            bins=bins_clv_tier_values,
            labels=bins_clv_tier_labels,
            method="qcut",
        )
        .assign(
            save_offer=lambda x: x.apply(get_save_offer, axis=1).astype(
                "category"
            )
        )
    )
    return df


def filter_matrix_by_limit(
    df: pd.DataFrame,
    col_str: str = "num_customers_cumulative",
    n_max: int = 100,
) -> pd.DataFrame:
    """Filters a DataFrame to include rows up to the first instance of a threshold.

    This function returns all rows where the 'num_customers_cumsum' count is
    below the specified limit, plus the first row that meets or exceeds that
    limit. Any rows occurring after the threshold has been reached are excluded.

    Args:
        df: A pandas DataFrame containing a 'num_customers' column.
        col_str: The name of the column with the cumulative number of customers
        n_max: The numerical threshold for the customer count.

    Returns:
        pd.DataFrame: A subset of the original DataFrame containing rows within
            and including the first instance of the limit.
    """
    # Keep rows until the previous row was >= n_max
    # # this creates a mask for rows where PREVIOUS row was under the limit
    # # This keeps all rows < n_max, plus the first row that exceeds it
    # # (it identifies rows where the sequence hasn't broken the num_customers
    # # threshold yet)
    # # shift(1) looks at the value of the row above. Since the first row has no
    # # above, fill it with 0 (which is less than num_customers), ensuring it is
    # # always checked.
    mask = df[col_str].shift(1, fill_value=0) < n_max
    return df[mask]


def summarize_campaign_mix(
    df: pd.DataFrame,
    col_metric: str,
    intervention_cost: int,
    sort: bool = True,
    add_cumsum: bool = True,
) -> pd.DataFrame:
    """Aggregates campaign performance metrics by risk, tier, and offer.

    Args:
        df: Input DataFrame containing churn predictions and savings data.
        col_metric: The column name to use for sorting results.
        intervention_cost: The fixed dollar cost per customer intervention.
        sort: Whether to sort the resulting DataFrame by col_metric.
        add_cumsum: Whether to add a cumulative sum of the customer count.

    Returns:
        pd.DataFrame: Aggregated metrics including total costs, error
            percentages, and per-customer savings estimates.
    """
    df_agg = (
        df.groupby(["risk_level", "value_tier", "save_offer"], as_index=False)
        .agg(
            {
                "true_savings": "sum",
                "expected_savings": "sum",
                "clientnum": "count",
                "y_pred_proba": "mean",
                "clv": "mean",
            }
        )
        .rename(columns={"clientnum": "num_customers"})
        .assign(
            total_intervention_cost=lambda df: df["num_customers"].mul(
                intervention_cost
            ),
            expected_savings_per_customer=lambda df: df["expected_savings"].div(
                df["num_customers"]
            ),
            savings_error_pct=lambda df: df["expected_savings"]
            .sub(df["true_savings"])
            .div(df["true_savings"])
            .mul(100),
        )
    )
    if sort:
        df_agg = df_agg.sort_values(by=[col_metric], ascending=False)
    if add_cumsum:
        df_agg = df_agg.assign(
            num_customers_cumsum=lambda df: df["num_customers"].cumsum()
        )
    return df_agg


def get_cohort_within_budget(
    df: pd.DataFrame,
    df_matrix_agg: pd.DataFrame,
    intervention_cost: int = 50,
    n: int = 100,
) -> List[Union[pd.DataFrame, float]]:
    """Identifies the top N customers for intervention based on budget.

    Args:
        df: Raw customer DataFrame containing expected savings.
        df_matrix_agg: Aggregated performance matrix by risk and tier.
        intervention_cost: The cost in dollars per targeted customer.
        n: The maximum number of customers to include in the cohort.

    Returns:
        List[Union[pd.DataFrame, float]]: A list containing the ranked
            customer DataFrame, the campaign mix summary, the total
            realizable impact, and the average benefit per customer.
    """
    # step 1. get combinations of risk_level and value_tier that contain the
    # required maximum number of customers (n)
    df_realizable_clv_tier_risk_level = (
        df_matrix_agg.nlargest(n, ["expected_savings_per_customer"])
        .assign(num_customers_cumsum=lambda df: df["num_customers"].cumsum())
        .pipe(filter_matrix_by_limit, "num_customers_cumsum", n)
    )

    # step 1. (contd.) get required combinations of risk_level and value_tier
    combo_filter = " | ".join(
        "((risk_level == '"
        + df_realizable_clv_tier_risk_level["risk_level"].astype(str)
        + "') & (value_tier == '"
        + df_realizable_clv_tier_risk_level["value_tier"].astype(str)
        + "'))"
    )

    # step 2. get top customers from raw customer data
    df_ranked = (
        # get customers who meet required combinations of risk_level & value_tier
        df.query(combo_filter)
        # get top N customers who meet these criteria by expected savings
        .nlargest(n, ["expected_savings"])
    )

    # step 3. show breakdown of combinations of risk levels and value tiers from
    # where these top N customers come
    campaign_mix = summarize_campaign_mix(
        df_ranked, "expected_savings_per_customer", intervention_cost
    )

    # step 4. calculate the realizable benefit (average expected savings per
    # customer, across all tiers)
    total_impact = df_ranked["expected_savings"].sum()
    average_efficiency = df_ranked["expected_savings"].mean()
    print(f"Total Realizable Impact: ${total_impact:,.2f}")
    print(f"Average Benefit per Intervention: ${average_efficiency:,.2f}")
    return [df_ranked, campaign_mix, total_impact, average_efficiency]
