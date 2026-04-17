#!/usr/bin/env python3


"""Test methods to explain a trained model."""


import numpy as np
import pytest
from sklearn.base import clone

import src.cc_churn.explanation as expl


def test_get_shap_values_output_types(trained_model):
    """Test Basic output structure."""
    clf, X_pre = trained_model

    shap_values, shap_exp = expl.get_shap_values(clf, X_pre)

    assert isinstance(shap_values, np.ndarray)
    assert hasattr(shap_exp, "values")


def test_get_shap_values_shape(trained_model):
    """Test output shape for both outputs."""
    clf, X_pre = trained_model

    shap_values, shap_exp = expl.get_shap_values(clf, X_pre)

    assert shap_values.shape == X_pre.shape
    assert shap_exp.values.shape == X_pre.shape


def test_shap_outputs_not_identical(trained_model):
    """Test that outputs are not identical."""
    clf, X_pre = trained_model

    shap_values, shap_exp = expl.get_shap_values(clf, X_pre)

    assert not np.testing.assert_array_equal(shap_values, shap_exp.values)


def test_feature_order_preserved(trained_model):
    """Test that expected number of features are returned."""
    clf, X_pre = trained_model

    _, shap_exp = expl.get_shap_values(clf, X_pre)

    assert shap_exp.values.shape[1] == X_pre.shape[1]


def test_subset_input(trained_model):
    """Test that SHAP works on a subset of the features."""
    clf, X_pre = trained_model

    X_subset = X_pre.iloc[:10]

    shap_values, shap_exp = expl.get_shap_values(clf, X_subset)

    assert shap_values.shape[0] == 10
    assert shap_exp.values.shape[0] == 10


def test_array_input_type(trained_model):
    """Test that a numpy array can be provided as input."""
    clf, _ = trained_model
    X_array = np.random.rand(10, 5)

    shap_values, shap_exp = expl.get_shap_values(clf, X_array)
    assert shap_values.shape[0] == X_array.shape[0]
    assert shap_exp.values.shape[1] == X_array.shape[1]


def test_unfitted_model_raises(trained_model):
    """Test behaviour on untrained model."""
    clf, X_pre = trained_model

    clf_untrained = clone(clf)

    with pytest.raises(TypeError):
        expl.get_shap_values(clf_untrained, X_pre)
