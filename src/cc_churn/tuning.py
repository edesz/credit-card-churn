#!/usr/bin/env python3


"""Define utilities to compare tuned ML model scores."""

from typing import Dict

import pandas as pd


def combine_cv_scores_thresholds(
    cv_results_tuned_models: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combines CV results and extracts thresholds for each fold.

    This function concatenates cross-validation results from multiple
    models and extracts the best decision threshold from each fitted
    estimator. It also assigns fold indices for outer CV.

    Args:
        cv_results_tuned_models: Dictionary mapping model names to their
            cross-validation result DataFrames.

    Returns:
        pd.DataFrame: Combined DataFrame with:
            - model name
            - outer CV fold index
            - best threshold per fold
            - associated evaluation metrics

    Raises:
        AttributeError: If an estimator does not expose `best_threshold_`.

    Examples:
        >>> df_combined = combine_cv_scores_thresholds(cv_results)
    """
    df_cv_scores_outer = pd.concat(
        [
            v.assign(
                cv_fold_outer=lambda df: df.index + 1,
                threshold=lambda df: df["estimator"].apply(
                    lambda est: float(est.best_threshold_)
                ),
            ).drop(columns=["estimator"])
            for _, v in cv_results_tuned_models.items()
        ],
        ignore_index=True,
    )
    columns_to_move = ["model_name", "cv_fold_outer", "threshold"]
    reordered_columns = [
        col for col in df_cv_scores_outer.columns if col not in columns_to_move
    ]
    return df_cv_scores_outer[columns_to_move + reordered_columns]


def agg_cv_scores_thresholds(
    df_cv_scores_outer: pd.DataFrame,
    primary_metric: str,
    threshold_overfit: float,
) -> pd.DataFrame:
    """Aggregates CV scores and evaluates overfitting across models.

    This function computes mean cross-validation scores per model and
    derives overfitting diagnostics by comparing train and test scores
    for a primary metric. It also flags significant overfitting based
    on a percentage difference threshold.

    Args:
        df_cv_scores_outer: DataFrame containing fold-level CV results.
        primary_metric: Metric used to assess performance and overfitting.
        threshold_overfit: Percentage threshold for significant overfit.

    Returns:
        pd.DataFrame: Aggregated DataFrame with:
            - mean train/test metrics per model
            - percentage difference for primary metric
            - boolean overfitting indicators
            - sorted by descending test performance

    Raises:
        KeyError: If required metric columns are missing.

    Examples:
        >>> df_agg = agg_cv_scores_thresholds(
        ...     df_cv_scores_outer=df_combined,
        ...     primary_metric="f2",
        ...     threshold_overfit=5.0,
        ... )
    """
    df_cv_scores_agg = (
        df_cv_scores_outer.drop(columns=["cv_fold_outer"])
        .groupby(["model_name", "feat_group", "run_id"], as_index=False)
        .mean()
        .convert_dtypes(dtype_backend="pyarrow")
        .assign(
            pct_diff=lambda df: (
                (
                    df[f"train_{primary_metric}"] - df[f"test_{primary_metric}"]
                ).abs()
                / df[f"train_{primary_metric}"]
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
        .sort_values(
            by=[f"test_{primary_metric}"],
            ascending=[False],
            ignore_index=True,
        )
    )
    return df_cv_scores_agg
