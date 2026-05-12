#!/usr/bin/env python3


"""Test methods to estimate business savings from customer attributes."""

import numpy as np
import pandas as pd

import src.cc_churn.costs_savings as cssv


def test_get_save_offer_high_platinum():
    """Verifies correct offer is returned for high-risk Platinum customers.

    Args:
        None
    """
    row = pd.Series({"risk_level": "High", "value_tier": "Platinum"})
    result = cssv.get_save_offer(row)
    assert "Executive Call" in result


def test_get_save_offer_low_gold():
    """Verifies correct offer is returned for low-risk Gold customers.

    Args:
        None
    """
    row = pd.Series({"risk_level": "Low", "value_tier": "Gold"})
    result = cssv.get_save_offer(row)
    assert "Personalized Email" in result


def test_get_save_offer_default_branch():
    """Verifies fallback logic for unknown tiers.

    Args:
        None
    """
    row = pd.Series({"risk_level": "Low", "value_tier": "Unknown"})
    result = cssv.get_save_offer(row)
    assert "Newsletter" in result


def test_add_binned_column_cut():
    """Verifies binning using pandas cut creates expected categories.

    Args:
        None
    """
    df = pd.DataFrame({"x": [0.1, 0.5, 0.9]})
    result = cssv.add_binned_column(
        df.copy(),
        source_col="x",
        target_col="bin",
        bins=[0, 0.5, 1],
        labels=["Low", "High"],
        include_lowest=True,
    )
    assert "bin" in result.columns
    assert result["bin"].notna().all()


def test_add_binned_column_qcut():
    """Verifies binning using pandas qcut creates quantile-based bins.

    Args:
        None
    """
    df = pd.DataFrame({"x": np.linspace(0, 1, 10)})
    result = cssv.add_binned_column(
        df.copy(),
        source_col="x",
        target_col="bin",
        bins=2,
        method="qcut",
    )
    assert result["bin"].nunique() == 2


def test_get_buckets_adds_columns(df_business_metrics):
    """Verifies risk_level, value_tier, and save_offer columns are added.

    Args:
        df_business_metrics: Fixture with prediction + CLV data.
    """
    result = cssv.get_buckets(df_business_metrics.copy())

    for col in ["risk_level", "value_tier", "save_offer"]:
        assert col in result.columns

    assert result["save_offer"].dtype.name == "category"


def test_filter_matrix_by_limit_basic():
    """Verifies rows are retained up to and including threshold crossing.

    Args:
        None
    """
    df = pd.DataFrame({"num_customers_cumulative": [10, 30, 60, 120, 150]})

    result = cssv.filter_matrix_by_limit(
        df, "num_customers_cumulative", n_max=100
    )

    # this should include a first row with a value >= 100 (i.e. 120)
    assert result.iloc[-1]["num_customers_cumulative"] == 120
    assert len(result) == 4


def test_summarize_campaign_mix_basic(df_business_metrics):
    """Verifies aggregation produces expected columns and calculations.

    Args:
        df_business_metrics: Fixture with required columns.
    """
    bins_risk_values = [0.5, 0.667, 0.833, 1.0]
    df = cssv.get_buckets(
        df_business_metrics.copy(),
        bins_risk_values=bins_risk_values,
        bins_risk_labels=["Low", "Medium", "High"],
    )
    assert (
        round(df.query("y_pred == 1")["y_pred_proba"].min(), 2)
        == bins_risk_values[0]
    )

    result = cssv.summarize_campaign_mix(
        df,
        col_metric="expected_savings",
        intervention_cost=50,
        sort=False,
        add_cumsum=False,
    )

    expected_cols = [
        "risk_level",
        "value_tier",
        "save_offer",
        "num_customers",
        "total_intervention_cost",
        "expected_savings_per_customer",
    ]

    for col in expected_cols:
        assert col in result.columns

    assert result["num_customers"].sum() == len(df)


def test_summarize_campaign_mix_cumsum_flag(df_business_metrics):
    """Verifies cumulative sum column is conditionally added.

    Args:
        df_business_metrics: Fixture with required columns.
    """
    df = cssv.get_buckets(df_business_metrics.copy())

    result = cssv.summarize_campaign_mix(
        df,
        col_metric="expected_savings",
        intervention_cost=50,
        add_cumsum=True,
    )

    assert "num_customers_cumsum" in result.columns


def test_get_cohort_within_budget_outputs(df_business_metrics):
    """Verifies cohort selection returns expected outputs and shapes.

    Args:
        df_business_metrics: Fixture with savings and predictions.
    """
    df = cssv.get_buckets(df_business_metrics.copy())

    df_agg = cssv.summarize_campaign_mix(
        df,
        col_metric="expected_savings_per_customer",
        intervention_cost=50,
        sort=False,
        add_cumsum=False,
    )

    df_ranked, campaign_mix, total, avg = cssv.get_cohort_within_budget(
        df,
        df_agg,
        intervention_cost=50,
        n=50,
    )

    assert isinstance(df_ranked, pd.DataFrame)
    assert isinstance(campaign_mix, pd.DataFrame)
    assert isinstance(total, float)
    assert isinstance(avg, float)

    assert len(df_ranked) <= 50


def test_get_cohort_within_budget_sorted(df_business_metrics):
    """Verifies selected cohort is sorted by expected savings.

    Args:
        df_business_metrics: Fixture with savings and predictions.
    """
    df = cssv.get_buckets(df_business_metrics.copy())

    df_agg = cssv.summarize_campaign_mix(
        df,
        col_metric="expected_savings_per_customer",
        intervention_cost=50,
        sort=False,
        add_cumsum=False,
    )

    df_ranked, _, _, _ = cssv.get_cohort_within_budget(
        df,
        df_agg,
        intervention_cost=50,
        n=50,
    )

    assert df_ranked["expected_savings"].is_monotonic_decreasing
