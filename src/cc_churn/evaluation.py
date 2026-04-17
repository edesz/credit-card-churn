#!/usr/bin/env python3


"""Define helper functions for model evaluation."""

from typing import Dict

import numpy as np
import pandas as pd


def score_predictions(
    scorers: Dict[str, object],
    y_train: pd.Series,
    y_train_pred: pd.Series,
    y_test: pd.Series,
    y_test_pred: pd.Series,
    model_name: str,
    primary_metric: str = "f2",
    threshold_overfit: float = 5,
) -> pd.DataFrame:
    """Computes train/test metrics and evaluates overfitting.

    This function applies a dictionary of scorer objects to predicted
    labels for both training and test datasets. It returns a DataFrame
    with metric values and overfitting diagnostics based on a selected
    primary metric.

    Overfitting is assessed by comparing train and test scores:
        - Percentage difference between train and test scores.
        - Boolean flag indicating if train > test.
        - Boolean flag for significant overfitting based on a threshold.

    Args:
        scorers: Dictionary mapping metric names to sklearn scorer
            objects. Each scorer must expose `_score_func` and `_kwargs`.
        y_train: True labels for the training dataset.
        y_train_pred: Predicted labels for the training dataset.
        y_test: True labels for the test dataset.
        y_test_pred: Predicted labels for the test dataset.
        model_name: Name of the model for labeling results.
        primary_metric: Metric used for overfitting diagnostics.
        threshold_overfit: Percentage threshold to flag significant
            overfitting.

    Returns:
        pd.DataFrame: Single-row DataFrame containing:
            - train/test scores for each metric
            - percentage difference for the primary metric
            - overfitting indicators
            - model name identifier

    Raises:
        KeyError: If `primary_metric` is not present in `scorers`.
        AttributeError: If scorer objects lack required attributes.

    Examples:
        >>> df_scores = score_predictions(
        ...     scorers=scorers,
        ...     y_train=y_train,
        ...     y_train_pred=y_train_pred,
        ...     y_test=y_test,
        ...     y_test_pred=y_test_pred,
        ...     model_name="logreg",
        ... )
    """
    scores_eval = dict(model_name=model_name)
    scores_eval.update(
        {
            f"train_{k}": scorers[k]._score_func(
                y_train, y_train_pred, **scorers[k]._kwargs
            )
            for k, _ in scorers.items()
        }
    )
    scores_eval.update(
        {
            f"test_{k}": scorers[k]._score_func(
                y_test, y_test_pred, **scorers[k]._kwargs
            )
            for k, _ in scorers.items()
        }
    )

    # estimate overfitting
    df_scores = (
        pd.DataFrame.from_records([scores_eval])
        .convert_dtypes(dtype_backend="pyarrow")
        .assign(
            pct_diff=lambda df: (
                (
                    df[f"train_{primary_metric}"] - df[f"test_{primary_metric}"]
                ).abs()
                / df[f"train_{primary_metric}"].replace(0, np.nan)
                * 100
            ),
            is_overfit=lambda df: (
                (
                    df[f"train_{primary_metric}"] > df[f"test_{primary_metric}"]
                ).astype("bool[pyarrow]")
            ),
            is_overfit_significant=lambda df: (
                (
                    df["is_overfit"].fillna(False)
                    & (df["pct_diff"].fillna(0) > threshold_overfit)
                ).astype("bool[pyarrow]")
            ),
        )
        .rename(
            columns={
                "pct_diff": f"pct_diff_{primary_metric}",
                "is_overfit": f"is_overfit_{primary_metric}",
                "is_overfit_significant": (
                    f"is_overfit_significant_{primary_metric}"
                ),
            }
        )
    )
    return df_scores
