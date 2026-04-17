#!/usr/bin/env python3


"""Test Metaflow convenience utilities."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.cc_churn.mflow_utils import get_metaflow_run_artifacts


class MockEstimator:
    def __init__(self, threshold):
        self.best_threshold_ = threshold


class MockFittedPipeline:
    def __init__(self, threshold):
        self.best_threshold_ = threshold
        self.estimator_ = Pipeline(
            [
                ("pre", MagicMock()),
            ]
        )
        self.estimator_.named_steps["pre"].get_feature_names_out = MagicMock(
            return_value=np.array(["x1", "x2"])
        )


@patch("src.cc_churn.mflow_utils.Run")
def test_get_metaflow_run_artifacts_success(mock_run_class):
    """Verifies artifacts are correctly extracted from a Metaflow run.

    Args:
        mock_run_class: Mocked Metaflow Run class.
    """

    class MockFullEstimator:
        def __init__(self, threshold):
            self.best_threshold_ = threshold
            self.estimator_ = Pipeline(
                [
                    ("pre", MagicMock()),
                ]
            )
            self.estimator_.named_steps["pre"].get_feature_names_out = (
                MagicMock(return_value=np.array(["x1", "x2"]))
            )

    # Create mock run
    mock_run = MagicMock()
    mock_run_class.return_value = mock_run

    # Mock transformers
    mock_run.data.transformers_preprocessors = [
        ("step1", MagicMock()),
    ]

    # Mock model
    mock_model = LogisticRegression()
    mock_run.data.pipe = Pipeline([("clf", mock_model)])

    # Mock df_cv
    df_cv = pd.DataFrame(
        {
            "estimator": [
                MockFullEstimator(0.6),
                MockFullEstimator(0.8),
            ]
        }
    )

    mock_run.data.df_cv = df_cv

    # Call function
    artifacts = get_metaflow_run_artifacts("123")

    preproc, model, pipe, features, threshold, run = artifacts

    # Assertions
    assert isinstance(preproc, Pipeline)
    assert model == mock_model

    # First row estimator should be selected as pipe
    assert pipe == df_cv.iloc[0]["estimator"]

    assert features == ["x1", "x2"]
    assert threshold == pytest.approx((0.6 + 0.8) / 2)

    assert run == mock_run


@patch("src.cc_churn.mflow_utils.Run")
def test_threshold_mean_computation(mock_run_class):
    """Verifies average threshold is computed correctly.

    Args:
        mock_run_class: Mocked Metaflow Run class.
    """
    mock_run = MagicMock()
    mock_run_class.return_value = mock_run

    mock_run.data.transformers_preprocessors = []
    mock_run.data.pipe = Pipeline([("clf", LogisticRegression())])

    df_cv = pd.DataFrame(
        {
            "estimator": [
                MockEstimator(0.5),
                MockEstimator(0.7),
                MockEstimator(0.9),
            ]
        }
    )

    # Inject a proper pipeline for feature extraction
    df_cv.iloc[0, 0] = MockFittedPipeline(0.5)

    mock_run.data.df_cv = df_cv

    _, _, _, _, threshold, _ = get_metaflow_run_artifacts("123")

    assert threshold == pytest.approx((0.5 + 0.7 + 0.9) / 3)


@patch("src.cc_churn.mflow_utils.Run")
def test_feature_extraction(mock_run_class):
    """Verifies feature names are extracted from pipeline.

    Args:
        mock_run_class: Mocked Metaflow Run class.
    """
    mock_run = MagicMock()
    mock_run_class.return_value = mock_run

    mock_run.data.transformers_preprocessors = []
    mock_run.data.pipe = Pipeline([("clf", LogisticRegression())])

    mock_pipe = MockFittedPipeline(0.7)

    df_cv = pd.DataFrame({"estimator": [mock_pipe]})
    mock_run.data.df_cv = df_cv

    _, _, _, features, _, _ = get_metaflow_run_artifacts("123")

    assert features == ["x1", "x2"]


@patch("src.cc_churn.mflow_utils.Run")
def test_missing_attributes_raises(mock_run_class):
    """Verifies missing artifacts raise AttributeError.

    Args:
        mock_run_class: Mocked Metaflow Run class.
    """
    mock_run = MagicMock()
    mock_run_class.return_value = mock_run

    # Use a strict object without the required attributes
    # transformers_preprocessors and df_cv
    mock_run.data = SimpleNamespace(
        pipe=Pipeline([("clf", LogisticRegression())])
    )

    with pytest.raises(AttributeError):
        get_metaflow_run_artifacts("123")


@patch("src.cc_churn.mflow_utils.Run")
def test_empty_df_cv_raises(mock_run_class):
    """Verifies empty df_cv raises an error.

    Args:
        mock_run_class: Mocked Metaflow Run class.
    """
    mock_run = MagicMock()
    mock_run_class.return_value = mock_run

    mock_run.data.transformers_preprocessors = []
    mock_run.data.pipe = Pipeline([("clf", LogisticRegression())])
    mock_run.data.df_cv = pd.DataFrame()

    with pytest.raises(Exception):
        get_metaflow_run_artifacts("123")
