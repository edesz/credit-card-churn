#!/usr/bin/env python3


"""Test EDA data aggregation utility functions."""

import pandas as pd
import pytest

from src.cc_churn.eda import get_grouped_churn_rate


@pytest.mark.parametrize(
    "col_to_group",
    [
        "gender",
        "marital_status",
        "card_category",
        "dependent_count",
        "education_level",
        "income_category",
    ],
)
def test_get_grouped_churn_rate_basic(base_df, col_to_group):
    """Verifies grouped churn rate computation for valid categorical columns.

    Args:
        base_df: Pytest fixture providing the full churn dataset.
        col_to_group: Column name used for grouping.
    """
    df_out = get_grouped_churn_rate(base_df, col_to_group)

    assert isinstance(df_out, pd.DataFrame)
    assert col_to_group in df_out.columns
    assert "Churned" in df_out.columns
    assert "Did not Churn" in df_out.columns
    assert "total" in df_out.columns
    assert "frac_churned" in df_out.columns
    assert "frac_total" in df_out.columns


def test_totals_match_original_counts(base_df):
    """Ensures aggregated totals equal original dataset size.

    Args:
        base_df: Pytest fixture providing the full churn dataset.
    """
    df_out = get_grouped_churn_rate(base_df, "gender")

    assert df_out["total"].sum() == len(base_df)


def test_churn_rate_bounds(base_df):
    """Checks churn rates are valid percentages between 0 and 100.

    Args:
        base_df: Pytest fixture providing the full churn dataset.
    """
    df_out = get_grouped_churn_rate(base_df, "card_category")

    assert (
        (df_out["frac_churned"] >= 0) & (df_out["frac_churned"] <= 100)
    ).all()


def test_frac_total_sums_to_100(base_df):
    """Validates that group proportions sum approximately to 100%.

    Args:
        base_df: Pytest fixture providing the full churn dataset.
    """
    df_out = get_grouped_churn_rate(base_df, "marital_status")

    total_pct = df_out["frac_total"].sum()
    assert pytest.approx(total_pct, rel=1e-3) == 100.0


def test_zero_churn_group_handled():
    """Ensures groups with zero churn are handled without errors.

    Args:
        None
    """
    df = pd.DataFrame(
        {
            "clientnum": [1, 2, 3],
            "outcome": ["Did not Churn", "Did not Churn", "Did not Churn"],
            "gender": ["M", "M", "F"],
        }
    )

    df_out = get_grouped_churn_rate(df, "gender")

    assert (df_out["Churned"] == 0).all()
    assert (df_out["frac_churned"] == 0).all()


def test_missing_required_columns_raises():
    """Ensures KeyError is raised when required columns are missing.

    Args:
        None
    """
    df = pd.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(KeyError):
        get_grouped_churn_rate(df, "A")


def test_group_column_preserved(base_df):
    """Ensures grouping column values are preserved in output.

    Args:
        base_df: Pytest fixture providing the full churn dataset.
    """
    col = "education_level"
    df_out = get_grouped_churn_rate(base_df, col)

    assert set(df_out[col]) == set(base_df[col].dropna().unique())
