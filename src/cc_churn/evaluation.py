#!/usr/bin/env python3


"""Define helper functions for model evaluation."""

from typing import Dict, List

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


def calculate_top_n_pct_uplift(
    df: pd.DataFrame, target_col: str, proba_col: str, frac_top: float = 0.05
) -> float:
    """Calculates the uplift for the top 5% of predicted probabilities.

    Args:
        df: The input DataFrame containing actual labels and probabilities.
        target_col: Name of the column with the true binary labels (0 or 1).
        proba_col: Name of column with predicted probabilities for class 1.
        frac_top: Fraction of top customers by predicted probability.

    Returns:
        The uplift value as the ratio of top 5% density to the base rate.

    Example:
        >>> data = {'y': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ...         'p': [0.9, 0.1, 0.2, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1]}
        >>> df = pd.DataFrame(data)
        >>> calculate_top_5pct_uplift(df, 'y', 'p')
        5.0
    """
    # Sort by probability in descending order
    df_sorted = df.sort_values(by=proba_col, ascending=False)

    # Calculate the number of rows that make up the top N%
    n_top = max(1, int(len(df_sorted) * frac_top))

    # Calculate the positive rate in the top N%
    top_n_rate = df_sorted[target_col].head(n_top).mean()

    # Calculate the positive rate across the entire dataset
    base_rate = df_sorted[target_col].mean()

    # Avoid division by zero if the base rate is 0
    if base_rate == 0:
        return 0.0

    return top_n_rate / base_rate


def calculate_cumulative_uplift(
    df: pd.DataFrame, target_col: str, proba_col: str, steps: int = 20
) -> pd.DataFrame:
    """Calculates cumulative uplift across defined population percentiles.

    Args:
        df: The input DataFrame containing actual labels and probabilities.
        target_col: Name of the column with the true binary labels (0 or 1).
        proba_col: Name of column with predicted probabilities for class 1.
        steps: Number of quantiles to split the data into (default 20).

    Returns:
        df_cumul_uplift: DataFrame containing percentiles and their
                             corresponding uplift.

    Example:
        >>> data = {'y': [1, 0, 1, 0, 0], 'p': [0.9, 0.8, 0.7, 0.4, 0.1]}
        >>> df = pd.DataFrame(data)
        >>> calculate_cumulative_uplift(df, 'y', 'p', steps=5)
           percentile  uplift
        0         0.2     2.5
        1         0.4     2.5
        2         0.6     1.666667
        3         0.8     1.25
        4         1.0     1.0
    """
    df_sorted = df.sort_values(by=proba_col, ascending=False).reset_index()
    base_rate = df_sorted[target_col].mean()
    results = []

    for i in range(1, steps + 1):
        percentile = i / steps
        n_rows = max(1, int(len(df_sorted) * percentile))
        cum_rate = df_sorted[target_col].head(n_rows).mean()
        uplift = cum_rate / base_rate if base_rate > 0 else 0.0
        results.append({"percentile": percentile, "uplift": uplift})
    df_cumul_uplift = pd.DataFrame(results)
    return df_cumul_uplift


def calculate_cumulative_gain(
    df: pd.DataFrame, target_col: str, proba_col: str, steps: int = 20
) -> pd.DataFrame:
    """Calculates cumulative gain across defined population percentiles.

    Args:
        df: The input DataFrame containing actual labels and probabilities.
        target_col: Name of the column with the true binary labels (0 or 1).
        proba_col: Name of column with predicted probabilities for class 1.
        steps: Number of quantiles to split the data into (default 20).

    Returns:
        df_cumul_gain: DataFrame containing percentiles and their corresponding
                           cumulative gain.

    Example:
        >>> data = {'y': [1, 1, 0, 0, 0], 'p': [0.9, 0.8, 0.4, 0.2, 0.1]}
        >>> df = pd.DataFrame(data)
        >>> calculate_cumulative_gain(df, 'y', 'p', steps=5)
           percentile  gain
        0         0.2   0.5
        1         0.4   1.0
        2         0.6   1.0
        3         0.8   1.0
        4         1.0   1.0
    """
    df_sorted = df.sort_values(by=proba_col, ascending=False).reset_index()
    total_positives = df_sorted[target_col].sum()

    results = []
    for i in range(1, steps + 1):
        percentile = i / steps
        n_rows = max(1, int(len(df_sorted) * percentile))
        positives_found = df_sorted[target_col].head(n_rows).sum()
        gain = positives_found / total_positives if total_positives > 0 else 0.0
        results.append({"percentile": percentile, "gain": gain})
    df_cumul_gain = pd.DataFrame.from_records(results)
    return df_cumul_gain


def get_elbow_point(df: pd.DataFrame, col_name: str) -> List[float]:
    """Detect the elbow point in a cumulative uplift or gain curve.

    The elbow is identified as the point preceding the largest increase
    in slope magnitude. This corresponds to the point at which the curve
    begins to decline much more rapidly, indicating diminishing returns.

    The input DataFrame must contain the following columns:

    - ``percentile``: cumulative population percentile
    - ``uplift``: cumulative uplift or gain value

    The method uses a discrete approximation of the first and second
    derivatives:

    1. First derivative:
       Computes the slope between adjacent points.

    2. Second derivative:
       Computes the change in slope between adjacent segments.

    The elbow is defined as the point immediately before the largest
    increase in slope.

    Args:
        df: The DataFrame containing ``percentile`` and ``uplift`` columns.
        col_name: The name of the curve column (``uplift`` or ``gain``).

    Returns:
        List containing:

        - elbow percentile
        - uplift or gain value at the elbow

    Example:
        >>> elbow_pctile, elbow_value = get_elbow_point(df_uplift, 'uplift')
        >>> print(elbow_pctile)
        0.10
        >>> print(elbow_value)
        5.967018827035081
    """
    # extract percentile and uplift/gain arrays
    x = df["percentile"].to_numpy()
    y = df[col_name].to_numpy()

    # compute the first derivative (slope)
    # the uplift curve decreases as percentile increases, so the raw
    # slope is negative. For this reason, multiply by -1 to make larger
    # values represent steeper declines.
    slopes = -np.diff(y) / np.diff(x)

    # compute the second derivative as the change in slope between
    # adjacent segments
    slope_change = np.diff(slopes)

    # locate the elbow index
    # np.argmax() returns the index of the largest increase in slope.
    # So, add +1 to map back to the corresponding percentile point.
    elbow_idx = np.argmax(slope_change) + 1

    # extract elbow coordinates
    elbow_percentile = x[elbow_idx]
    elbow_value = y[elbow_idx]
    return [elbow_percentile, elbow_value]


def transform_model_metrics(
    df: pd.DataFrame, metrics: List[str]
) -> pd.DataFrame:
    """
    Transforms model metrics into a Evidently drift-analysis summary format.

    Calculates expected values from the 'val' split and compares them
    against the 'test' split to determine success based on a 10% error margin.

    Args:
        df: DataFrame containing columns 'model_name', 'split', 'test_<m1>',
            'test_<m2>', 'test_<m3>', etc.
        metrics: List containing the names of ML scoring metrics.

    Returns:
        A formatted DataFrame with columns: metric, description, name,
        status, value, condition, category, and is_count.

    Example:
        >>> data = {
        ...     'model_name': ['LR', 'LR'],
        ...     'split': ['val', 'test'],
        ...     'test_f2': [0.80, 0.75],
        ...     'test_recall': [0.90, 0.88],
        ...     'test_prauc': [0.65, 0.60],
        ... }
        >>> df_input = pd.DataFrame(data, ['f2', 'recall', "prauc"])
        >>> df_result = transform_model_metrics(df_input)
    """
    # get baseline values from 'val' split
    val_subset = df.query("split == 'val'").iloc[0]

    results = []
    test_df = df.query("split == 'test'")

    for m in metrics:
        actual = test_df[f"test_{m}"].iloc[0]
        expected = val_subset[f"test_{m}"]
        error = expected * 0.1

        # check if actual value is within 10% of expected
        is_success = (expected - error) <= actual <= (expected + error)
        status = "SUCCESS" if is_success else "FAIL"

        m_name = m.upper()
        results.append(
            {
                "metric": m,
                "description": (
                    f"{m_name} metric: Actual value {actual:.4f}  "
                    f"expected {expected:.4f} ± {error:.4f}"
                ),
                "name": f"{m_name} metric: Equal {expected:.4f} ± {error:.3f}",
                "status": status,
                "value": actual,
                "condition": "drift",
                "category": "[]",
                "is_count": False,
            }
        )

    transformed_df = pd.DataFrame(results)
    return transformed_df
