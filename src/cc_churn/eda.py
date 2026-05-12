#!/usr/bin/env python3


"""Define utilities that are used to transform data for use in EDA."""

import pandas as pd


def get_grouped_churn_rate(df: pd.DataFrame, col_to_group: str) -> pd.DataFrame:
    """Computes churn rate and distribution by a grouping column.

    This function aggregates customer counts by a specified feature and
    churn outcome, then derives churn rates and group proportions. The
    result is a summary table showing the number of churned and
    non-churned customers, along with percentage metrics.

    Args:
        df: The input DataFrame containing customer data. Must include
            `clientnum` and `outcome` columns.
        col_to_group: The column name used to group customers (e.g.,
            categorical feature).

    Returns:
        pd.DataFrame: The aggregated DataFrame with the following columns:
            - grouping column (`col_to_group`)
            - Churned: count of churned customers
            - Did not Churn: count of retained customers
            - total: total customers per group
            - frac_churned: churn rate (%) per group
            - frac_total: percentage contribution of group to dataset

    Raises:
        KeyError: If required columns are missing from the DataFrame.

    Examples:
        >>> df_grouped = get_grouped_churn_rate(df, "card_category")
    """
    df_agg = (
        df.groupby([col_to_group, "outcome"], as_index=False)
        .agg({"clientnum": "count"})
        .pivot(index=col_to_group, columns="outcome", values="clientnum")
        .reindex(columns=["Churned", "Did not Churn"], fill_value=0)
        .assign(
            total=lambda df: df.sum(axis=1),
            frac_churned=lambda df: df["Churned"].div(df["total"]).mul(100),
            frac_total=lambda df: df["total"].div(df["total"].sum()).mul(100),
        )
        .reset_index()
    )
    return df_agg
