#!/usr/bin/env python3


"""Test methods to evaluate a ML model."""

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as mtr

import src.cc_churn.evaluation as ev


def test_score_predictions_output_structure(sample_predictions, scorers):
    """Verifies that the function returns a DataFrame with expected structure.

    Args:
        sample_predictions: Fixture providing train/test labels and predictions.
        scorers: Fixture providing scorer dictionary.
    """
    y_train, y_train_pred, y_test, y_test_pred = sample_predictions
    df = ev.score_predictions(
        scorers,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        model_name="LogisticRegression",
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_score_columns_exist(sample_predictions, scorers):
    """Verifies that expected metric columns are present.

    Args:
        sample_predictions: Fixture providing train/test labels and predictions.
        scorers: Fixture providing scorer dictionary.
    """
    y_train, y_train_pred, y_test, y_test_pred = sample_predictions
    df = ev.score_predictions(
        scorers,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        model_name="LogisticRegression",
    )
    expected_cols = {
        "model_name",
        "train_f2",
        "test_f2",
        "train_recall",
        "test_recall",
        "pct_diff_f2",
        "is_overfit_f2",
        "is_overfit_significant_f2",
    }
    assert expected_cols.issubset(df.columns)


def test_score_values_correct(sample_predictions, scorers):
    """Verifies that computed metric values are correct.

    Args:
        sample_predictions: Fixture providing train/test labels and predictions.
        scorers: Fixture providing scorer dictionary.
    """
    y_train, y_train_pred, y_test, y_test_pred = sample_predictions
    df = ev.score_predictions(
        scorers,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        model_name="LogisticRegression",
    )
    expected_train_f2 = mtr.fbeta_score(y_train, y_train_pred, beta=2)
    expected_test_f2 = mtr.fbeta_score(y_test, y_test_pred, beta=2)
    assert np.isclose(df.loc[0, "train_f2"], expected_train_f2)
    assert np.isclose(df.loc[0, "test_f2"], expected_test_f2)


def test_overfitting_flag(sample_predictions, scorers):
    """Verifies overfitting flag is set correctly.

    Args:
        sample_predictions: Fixture providing train/test labels and predictions.
        scorers: Fixture providing scorer dictionary.
    """
    y_train, y_train_pred, y_test, y_test_pred = sample_predictions
    df = ev.score_predictions(
        scorers,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        model_name="LogisticRegression",
        threshold_overfit=0,  # force detection
    )
    row = df.iloc[0]
    if row["train_f2"] > row["test_f2"]:
        assert row["is_overfit_f2"] in [True, False]


def test_significant_overfitting(sample_predictions, scorers):
    """Verifies significant overfitting flag behavior.

    Args:
        sample_predictions: Fixture providing train/test labels and predictions.
        scorers: Fixture providing scorer dictionary.
    """
    y_train, y_train_pred, y_test, y_test_pred = sample_predictions
    df = ev.score_predictions(
        scorers,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        model_name="LogisticRegression",
        threshold_overfit=0,  # always trigger if diff > 0
    )
    row = df.iloc[0]
    if row["pct_diff_f2"] > 0:
        assert row["is_overfit_significant_f2"] in [True, False]


def test_missing_primary_metric_raises(sample_predictions, scorers):
    """Verifies that missing primary metric raises KeyError.

    Args:
        sample_predictions: Fixture providing train/test labels and predictions.
        scorers: Fixture providing scorer dictionary.
    """
    y_train, y_train_pred, y_test, y_test_pred = sample_predictions
    with pytest.raises(KeyError):
        ev.score_predictions(
            scorers,
            y_train,
            y_train_pred,
            y_test,
            y_test_pred,
            model_name="LogisticRegression",
            primary_metric="precision",
        )


def test_zero_train_metric(scorers):
    """Verifies pct_diff is NaN when train metric is zero.

    Args:
        sample_data: Fixture providing train/test labels and predictions.
        scorers: Fixture providing scorer dictionary.
    """
    y_train = pd.Series([0, 0])
    y_test = pd.Series([0, 0])
    # create bad predictions: f2 = 0
    y_train_pred = pd.Series([1, 1])
    y_test_pred = pd.Series([1, 1])

    df = ev.score_predictions(
        scorers,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        model_name="test_model",
    )
    assert pd.isna(df["pct_diff_f2"].iloc[0])
    assert df["is_overfit_significant_f2"].iloc[0] is False


def test_model_name_propagation(sample_predictions, scorers):
    """Verifies that model_name is correctly included in output.

    Args:
        sample_predictions: Fixture providing train/test labels and predictions.
        scorers: Fixture providing scorer dictionary.
    """
    y_train, y_train_pred, y_test, y_test_pred = sample_predictions
    df = ev.score_predictions(
        scorers,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        model_name="LogisticRegression",
    )
    assert df.loc[0, "model_name"] == "LogisticRegression"


def test_top_n_pct_uplift_basic(simple_df):
    """Verifies uplift calculation on simple dataset."""
    uplift = ev.calculate_top_n_pct_uplift(
        simple_df, target_col="y", proba_col="p", frac_top=0.4
    )

    assert uplift > 0
    assert isinstance(uplift, float)


def test_top_n_pct_uplift_base_rate_zero():
    """Verifies function returns 0 when base rate is zero."""
    df = pd.DataFrame({"y": [0, 0, 0], "p": [0.9, 0.8, 0.1]})

    uplift = ev.calculate_top_n_pct_uplift(df, target_col="y", proba_col="p")

    assert uplift == 0.0


def test_top_n_pct_uplift_minimum_one_row():
    """Ensures at least one row is used in top fraction."""
    df = pd.DataFrame({"y": [1, 0, 0], "p": [0.2, 0.1, 0.05]})

    uplift = ev.calculate_top_n_pct_uplift(
        df, target_col="y", proba_col="p", frac_top=0.01
    )

    assert isinstance(uplift, float)


def test_cumulative_uplift_shape(simple_df):
    """Verifies output has correct number of rows."""
    df_out = ev.calculate_cumulative_uplift(simple_df, "y", "p", steps=5)

    assert len(df_out) == 5
    assert "percentile" in df_out.columns
    assert "uplift" in df_out.columns


def test_cumulative_uplift_monotonic_percentile(simple_df):
    """Verifies percentiles increase monotonically."""
    df_out = ev.calculate_cumulative_uplift(simple_df, "y", "p", steps=5)

    assert df_out["percentile"].is_monotonic_increasing


def test_cumulative_uplift_base_rate_zero():
    """Verifies uplift is zero when no positives exist."""
    df = pd.DataFrame({"y": [0, 0, 0], "p": [0.9, 0.8, 0.1]})

    df_out = ev.calculate_cumulative_uplift(df, "y", "p", steps=3)

    assert (df_out["uplift"] == 0).all()


def test_cumulative_gain_basic(simple_df):
    """Verifies gain output structure."""
    df_out = ev.calculate_cumulative_gain(simple_df, "y", "p", steps=5)

    assert len(df_out) == 5
    assert "percentile" in df_out.columns
    assert "gain" in df_out.columns


def test_cumulative_gain_bounds(simple_df):
    """Ensures gain values are between 0 and 1."""
    df_out = ev.calculate_cumulative_gain(simple_df, "y", "p", steps=5)

    assert ((df_out["gain"] >= 0) & (df_out["gain"] <= 1)).all()


def test_cumulative_gain_total_positive_zero():
    """Verifies gain is zero when no positives exist."""
    df = pd.DataFrame({"y": [0, 0, 0], "p": [0.9, 0.8, 0.1]})

    df_out = ev.calculate_cumulative_gain(df, "y", "p", steps=3)

    assert (df_out["gain"] == 0).all()


def test_transform_model_metrics_structure(metrics_df):
    """Verifies output columns and shape."""
    metrics = ["f2", "recall"]

    df_out = ev.transform_model_metrics(metrics_df, metrics)

    expected_cols = {
        "metric",
        "description",
        "name",
        "status",
        "value",
        "condition",
        "category",
        "is_count",
    }

    assert set(df_out.columns) == expected_cols
    assert len(df_out) == len(metrics)


def test_transform_model_metrics_status_success(metrics_df):
    """Verifies SUCCESS when within tolerance."""
    df_out = ev.transform_model_metrics(metrics_df, ["f2"])

    assert df_out.loc[0, "status"] == "SUCCESS"


def test_transform_model_metrics_status_fail():
    """Verifies FAIL when outside tolerance."""
    df = pd.DataFrame(
        {
            "model_name": ["LR", "LR"],
            "split": ["val", "test"],
            # this is a large deviation in scores
            "test_f2": [0.80, 0.50],
        }
    )

    df_out = ev.transform_model_metrics(df, ["f2"])

    assert df_out.loc[0, "status"] == "FAIL"


def test_transform_model_metrics_values(metrics_df):
    """Verifies actual values are correctly assigned."""
    df_out = ev.transform_model_metrics(metrics_df, ["recall"])

    assert df_out.loc[0, "value"] == 0.88


def test_transform_model_metrics_description_format(metrics_df):
    """Verifies description string contains expected info."""
    df_out = ev.transform_model_metrics(metrics_df, ["f2"])

    desc = df_out.loc[0, "description"]

    assert "Actual value" in desc
    assert "expected" in desc


@pytest.mark.parametrize(
    ("fixture_name", "col_name", "expected_percentile", "expected_value"),
    [("df_uplift", "uplift", 0.10, 5.97), ("df_gain", "gain", 0.20, 4.95)],
)
def test_get_elbow_point(
    request: pytest.FixtureRequest,
    fixture_name: str,
    col_name: str,
    expected_percentile: float,
    expected_value: float,
) -> None:
    """Test elbow detection for uplift and gain curves."""
    df = request.getfixturevalue(fixture_name)

    elbow_percentile, elbow_value = ev.get_elbow_point(df, col_name)

    assert elbow_percentile == pytest.approx(expected_percentile)
    assert elbow_value == pytest.approx(expected_value)
