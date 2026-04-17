#!/usr/bin/env python3


"""Define utilities to extract Metaflow attributes."""

from typing import List, Union

from metaflow import Run
from sklearn.pipeline import Pipeline


def get_metaflow_run_artifacts(
    run_id: str,
) -> List[Union[Pipeline, object, List[str], float, Run]]:
    """Extracts model artifacts and metadata from a Metaflow run.

    This function retrieves key artifacts from a completed Metaflow run,
    including preprocessing steps, trained model, pipeline, feature
    names, and the average decision threshold across CV folds.

    It assumes the run contains attributes such as `transformers_
    preprocessors`, `pipe`, and `df_cv` with trained estimators.

    Args:
        run_id: Identifier of the Metaflow run (e.g., "12345").

    Returns:
        List[Union[Pipeline, object, List[str], float, Run]]:
            - Preprocessing pipeline constructed from transformers.
            - Trained model (classifier).
            - Full pipeline object from CV results.
            - List of preprocessed feature names.
            - Average decision threshold across CV estimators.
            - Metaflow `Run` object.

    Raises:
        AttributeError: If expected artifacts are missing in the run.
        KeyError: If required keys are not present in stored data.

    Examples:
        >>> artifacts = get_metaflow_run_artifacts("12345")
        >>> preproc, model, pipe, features, threshold, run = artifacts
    """
    # get Metaflow run object
    run = Run(f"ValidationFlow/{run_id}")

    # get preprocessor
    transformer_preprocessor = Pipeline(run.data.transformers_preprocessors)

    # get classifier (model) from the clf step of Pipeline
    model = run.data.pipe.named_steps["clf"]

    # get Pipeline
    pipe = run.data.df_cv.head(1)["estimator"].squeeze()

    # get pre-processed feature names
    features = (
        pipe.estimator_.named_steps["pre"].get_feature_names_out().tolist()
    )

    # get average classifier (model) decision threshold
    avg_decision_threshold = (
        run.data.df_cv["estimator"]
        .apply(lambda row: row.best_threshold_)
        .mean()
    )
    return [
        transformer_preprocessor,
        model,
        pipe,
        features,
        avg_decision_threshold,
        run,
    ]
