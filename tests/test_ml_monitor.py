#!/usr/bin/env python3


"""Test methods to run ML monitoring tests with evidently."""

import pandas as pd
import pytest

import src.cc_churn.ml_monitor as mlm


def test_run_column_data_tests_returns_dataframe(
    evidently_datasets, sample_metrics
):
    """Verifies function returns a non-empty DataFrame."""
    ds_train, ds_test = evidently_datasets

    df = mlm.run_column_data_tests(
        metrics=sample_metrics,
        dataset_test=ds_test,
        dataset_train=ds_train,
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_expected_columns_present(evidently_datasets, sample_metrics):
    """Ensures expected output columns exist."""
    ds_train, ds_test = evidently_datasets

    df = mlm.run_column_data_tests(sample_metrics, ds_test, ds_train)

    expected_cols = {
        "id",
        "column",
        "metric_name",
        "value",
        "threshold",
        "condition",
        "category",
        "is_count",
        "description",
        "tolerance",
        "status",
    }

    assert expected_cols.issubset(df.columns)


def test_metric_values_not_null(evidently_datasets, sample_metrics):
    """Checks metric values are computed."""
    ds_train, ds_test = evidently_datasets

    df = mlm.run_column_data_tests(sample_metrics, ds_test, ds_train)

    assert df["value"].notna().any()


def test_merge_preserves_rows(evidently_datasets, sample_metrics):
    """Ensures merge does not drop all rows."""
    ds_train, ds_test = evidently_datasets

    df = mlm.run_column_data_tests(sample_metrics, ds_test, ds_train)

    assert len(df) > 0


def test_category_filtering_logic(evidently_datasets, sample_metrics):
    """Validates category/is_count filtering condition."""
    ds_train, ds_test = evidently_datasets

    df = mlm.run_column_data_tests(sample_metrics, ds_test, ds_train)

    # Condition from implementation
    valid_rows = df[
        ((df["category"].apply(len) == 0) & (df["is_count"] == True))
        | ((df["category"].apply(len) >= 1) & (df["is_count"] == False))
    ]

    # All rows should satisfy filter
    assert len(valid_rows) == len(df)


def test_empty_metrics_raises_error(evidently_datasets):
    dataset_train, dataset_test = evidently_datasets

    with pytest.raises(Exception):
        mlm.run_column_data_tests(
            metrics=[],
            dataset_test=dataset_test,
            dataset_train=dataset_train,
        )


def test_threshold_column_exists(evidently_datasets, sample_metrics):
    """Ensures threshold column exists even if NaN."""
    ds_train, ds_test = evidently_datasets

    df = mlm.run_column_data_tests(sample_metrics, ds_test, ds_train)

    assert "threshold" in df.columns


def test_status_values_valid(evidently_datasets, sample_metrics):
    """Ensures status values are valid strings."""
    ds_train, ds_test = evidently_datasets

    df = mlm.run_column_data_tests(sample_metrics, ds_test, ds_train)

    assert df["status"].dropna().apply(lambda x: isinstance(x, str)).all()


def test_metric_name_parsing(evidently_datasets, sample_metrics):
    """Ensures metric_name parsing removes parentheses."""
    ds_train, ds_test = evidently_datasets

    df = mlm.run_column_data_tests(sample_metrics, ds_test, ds_train)

    # No metric names should contain '(' after split
    assert not df["metric_name"].str.contains(r"\(").any()


def test_metric_name_parsing(evidently_datasets, sample_metrics):
    """Ensures metric_name is properly cleaned."""
    ds_train, ds_test = evidently_datasets

    df = mlm.run_column_data_tests(sample_metrics, ds_test, ds_train)

    assert not df["metric_name"].str.contains(r"\(").any()


def test_metric_values_present(evidently_datasets, sample_metrics):
    """Ensures metric values are computed."""
    ds_train, ds_test = evidently_datasets

    df = mlm.run_column_data_tests(sample_metrics, ds_test, ds_train)

    assert df["value"].notna().any()
