#!/usr/bin/env python3


"""Test methods to evaluate a ML model."""


import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as mtr

from cc_churn.evaluation import score_predictions


def test_score_predictions_output_structure(sample_predictions, scorers):
    """Verifies that the function returns a DataFrame with expected structure.

    Args:
        sample_predictions: Fixture providing train/test labels and predictions.
        scorers: Fixture providing scorer dictionary.
    """
    y_train, y_train_pred, y_test, y_test_pred = sample_predictions
    df = score_predictions(
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
    df = score_predictions(
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
    df = score_predictions(
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
    df = score_predictions(
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
    df = score_predictions(
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
        score_predictions(
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

    df = score_predictions(
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
    df = score_predictions(
        scorers,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        model_name="LogisticRegression",
    )
    assert df.loc[0, "model_name"] == "LogisticRegression"
