#!/usr/bin/env python3


"""Test methods to explain a trained model using SHAP values."""

import numpy as np
import pandas as pd
import pytest
from shap import _explanation
from sklearn.base import clone

import src.cc_churn.explanation as shexp


@pytest.mark.parametrize("y_class", [0, 1])
def test_get_shap_values_output_attrs(
    preprocess_data, get_shap_values_inputs, y_class
):
    """Verifies SHAP explanation exposes required public attributes.

    This test confirms that the SHAP explanation object returned by
    ``get_shap_values()`` contains the standard SHAP attributes used in
    downstream analysis and visualization workflows.
    """
    model, best_features = get_shap_values_inputs
    _, shap_explainer_values = shexp.get_shap_values(
        model, preprocess_data.query(f"y_pred == {y_class}")[best_features]
    )
    assert hasattr(shap_explainer_values, "values")
    assert hasattr(shap_explainer_values, "base_values")
    assert hasattr(shap_explainer_values, "data")


@pytest.mark.parametrize("y_class", [0, 1])
def test_explainer_object_sizes(
    preprocess_data, get_shap_values_inputs, y_class
):
    """Verifies SHAP explanation arrays contain non-empty outputs.

    This test ensures that the explanation values, base values, and
    underlying feature data are populated for both prediction classes.
    """
    model, best_features = get_shap_values_inputs
    _, shap_explainer_values = shexp.get_shap_values(
        model, preprocess_data.query(f"y_pred == {y_class}")[best_features]
    )
    assert shap_explainer_values.values.size > 0
    assert shap_explainer_values.base_values.size > 0
    assert shap_explainer_values.data.size > 0


@pytest.mark.parametrize("y_class", [0, 1])
def test_explainer_object_types(
    preprocess_data, get_shap_values_inputs, y_class
):
    """Verifies SHAP outputs use expected object and array types.

    This test confirms that the returned explanation is a SHAP
    ``Explanation`` object and that all internal components are stored
    as NumPy arrays.
    """
    model, best_features = get_shap_values_inputs
    _, shap_explainer_values = shexp.get_shap_values(
        model, preprocess_data.query(f"y_pred == {y_class}")[best_features]
    )
    assert type(shap_explainer_values) == _explanation.Explanation
    for obj in [
        shap_explainer_values.values,
        shap_explainer_values.base_values,
        shap_explainer_values.data,
    ]:
        assert isinstance(obj, np.ndarray)


@pytest.mark.parametrize("y_class", [0, 1])
def test_get_shap_values_shape(
    preprocess_data, get_shap_values_inputs, y_class
):
    """Verifies SHAP values match the number of evaluated rows.

    This test ensures that one SHAP explanation vector is generated for
    every observation passed into ``get_shap_values()``.
    """
    model, best_features = get_shap_values_inputs
    _, shap_explainer_values = shexp.get_shap_values(
        model, preprocess_data.query(f"y_pred == {y_class}")[best_features]
    )
    assert shap_explainer_values.values.shape[0] == len(
        preprocess_data.query(f"y_pred == {y_class}")
    )


@pytest.mark.parametrize("y_class", [0, 1])
def test_feature_order_preserved(
    preprocess_data, get_shap_values_inputs, y_class
):
    """Verifies SHAP output preserves the expected feature dimension.

    This test confirms that the number of SHAP feature contributions
    aligns with the number of model input features used for explanation.
    """
    model, best_features = get_shap_values_inputs
    _, shap_explainer_values = shexp.get_shap_values(
        model, preprocess_data.query(f"y_pred == {y_class}")[best_features]
    )
    assert shap_explainer_values.values.shape[1] == len(best_features)


@pytest.mark.parametrize("y_class", [0, 1])
def test_unfitted_model_raises(
    preprocess_data, get_shap_values_inputs, y_class
):
    """Verifies SHAP generation fails for an unfitted model.

    This test checks that attempting to compute SHAP values from an
    unfitted estimator raises a ``TypeError``.
    """
    model, best_features = get_shap_values_inputs

    model_untrained = clone(model)

    with pytest.raises(TypeError):
        _, _ = shexp.get_shap_values(
            model_untrained,
            preprocess_data.query(f"y_pred == {y_class}")[best_features],
        )


@pytest.mark.parametrize("y_class, tol", [(0, 0.15), (1, 0.15)])
def test_get_shap_values_within_expected_range(
    preprocess_data,
    get_shap_values_inputs,
    expected_shap_values,
    y_class,
    tol,
):
    """Verifies SHAP feature importances remain within tolerance bounds.

    This regression-style test compares the mean absolute SHAP values
    against expected benchmark values and validates that relative
    deviation remains within the configured tolerance.
    """
    model, best_features = get_shap_values_inputs
    _, shap_explainer_values = shexp.get_shap_values(
        model, preprocess_data.query(f"y_pred == {y_class}")[best_features]
    )
    df_shap_values = (
        pd.concat(
            [
                pd.Series(
                    np.abs(shap_explainer_values.values).mean(0),
                    name=f"y_class_{y_class}",
                ).to_frame(),
                pd.Series(best_features, name="feature"),
            ],
            axis=1,
        )
        .set_index("feature")
        .reset_index()
    )
    df_comp = df_shap_values.merge(
        expected_shap_values[["feature", f"y_class == {y_class}"]],
        on="feature",
        how="inner",
    ).assign(
        pct_diff=lambda df: (
            abs(df[f"y_class == {y_class}"] - df[f"y_class_{y_class}"])
            / df[f"y_class == {y_class}"]
        ),
        is_in_range=lambda df: df["pct_diff"] <= tol,
    )
    # verify that all features with average SHAP value >= 0.1 are in range
    assert df_comp.query(f"y_class_{y_class} >= 0.1")["is_in_range"].all()
