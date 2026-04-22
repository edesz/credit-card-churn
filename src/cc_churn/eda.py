#!/usr/bin/env python3


"""Define utilities that are used to transform data for use in EDA."""


import pandas as pd


def get_grouped_churn_rate(df: pd.DataFrame, col_to_group: str) -> pd.DataFrame:
    """Get churn rate by group."""
    df_agg = (
        df.groupby([col_to_group, "outcome"], as_index=False)
        .agg({"clientnum": "count"})
        .pivot(index=col_to_group, columns="outcome")
        .set_axis(["Churned", "Did not Churn"], axis=1)
        .assign(
            total=lambda df: df.sum(axis=1),
            frac_churned=lambda df: df["Churned"].div(df["total"]).mul(100),
            frac_total=lambda df: df["total"].div(df["total"].sum()).mul(100),
        )
        .reset_index()
    )
    return df_agg
