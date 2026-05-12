#!/usr/bin/env python3


"""Define helper functions to run ML monitoring tests."""

from typing import List

import numpy as np
import pandas as pd
from evidently import Dataset, Report


def run_column_data_tests(
    metrics: List, dataset_test: Dataset, dataset_train: Dataset
) -> pd.DataFrame:
    """Runs Evidently column-level tests and returns structured results.

    This function executes an Evidently `Report` using the provided
    metrics on a test dataset relative to a reference (train) dataset.
    It extracts both metric values and test results, then merges them
    into a single DataFrame for analysis.

    The output includes metric values, thresholds, and corresponding
    test conditions such as tolerance and pass/fail status. Filtering is
    applied to align count-based and category-based tests appropriately.

    Args:
        metrics: List of Evidently metric objects to evaluate.
        dataset_test: Dataset to evaluate (typically validation/test).
        dataset_train: Reference dataset (typically training data).

    Returns:
        pd.DataFrame: Combined DataFrame with:
            - metric identifiers and names
            - column names
            - metric values and thresholds
            - test descriptions and tolerances
            - test status (e.g., "SUCCESS", "FAIL")

    Raises:
        KeyError: If expected keys are missing in the evaluation output.
        AttributeError: If metric or test objects are malformed.

    Examples:
        >>> df_results = run_column_data_tests(
        ...     metrics=metrics,
        ...     dataset_test=test_ds,
        ...     dataset_train=train_ds,
        ... )
    """
    report = Report(metrics, include_tests=True)
    my_eval = report.run(dataset_test, dataset_train)
    eval_dict = my_eval.dict()

    df_metrics = pd.DataFrame.from_records(
        [
            {
                "id": r["id"],
                "column": r["config"]["column"],
                "metric_name": r["metric_name"].split("(")[0],
                "value": (
                    r["value"]["count"]
                    if isinstance(r["value"], dict)
                    else r["value"]
                ),
                "threshold": (
                    r["config"]["threshold"]
                    if "threshold" in list(r["config"])
                    else np.nan
                ),
            }
            for r in eval_dict["metrics"]
        ]
    )
    df_tests = pd.DataFrame.from_records(
        [
            {
                "condition": test["id"],
                "column": test["metric_config"]["params"]["column"],
                "category": (
                    test["metric_config"]["params"]["categories"]
                    if "categories" in list(test["metric_config"]["params"])
                    else []
                ),
                "id": test["metric_config"]["metric_id"],
                "metric_name": test["metric_config"]["params"]["type"].split(
                    ":"
                )[-1],
                "is_count": (
                    test["bound_test"]["is_count"]
                    if "is_count" in list(test["bound_test"])
                    else True
                ),
                "description": test["description"],
                "tolerance": (
                    test["test_config"]["expected"]["relative"]
                    if "expected" in list(test["test_config"])
                    else np.nan
                ),
                "status": test["status"].value,
            }
            for test in eval_dict["tests"]
        ]
    )
    df_summary = df_metrics.merge(
        df_tests.query(
            "((category.str.len() == 0) & (is_count == True)) | "
            "((category.str.len() >= 1) & (is_count == False))"
        ),
        on=["column", "id", "metric_name"],
        how="left",
    )
    return df_summary
