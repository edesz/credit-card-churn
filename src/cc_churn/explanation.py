#!/usr/bin/env python3


"""Define helper functions for model interpretation."""

from typing import List, Union

import numpy as np
import pandas as pd
import shap


def get_shap_values(
    model, X: Union[np.ndarray, pd.DataFrame]
) -> List[Union[np.ndarray, shap._explanation.Explanation]]:
    """Computes SHAP values using both legacy and new SHAP APIs.

    This function initializes a SHAP explainer for the given model and
    computes feature attributions using two approaches:

    1. `explainer.shap_values(X)` which returns a NumPy array.
    2. `explainer(X)` which returns a SHAP `Explanation` object.

    The function validates that both outputs have the same shape as the
    input feature matrix. It also asserts that the two outputs are not
    identical, reflecting differences between legacy and newer SHAP APIs.

    Args:
        model: A trained model compatible with SHAP explainers.
        X: Input feature matrix for which SHAP values are computed.

    Returns:
        List[Union[np.ndarray, shap._explanation.Explanation]]:
            - First element: SHAP values as a NumPy array with shape
              `(n_samples, n_features)`.
            - Second element: SHAP `Explanation` object containing
              `.values` with the same shape as `X` and additional
              metadata (e.g., base values, feature names).

    Raises:
        AssertionError: If SHAP outputs do not match expected types or
            shapes, or if both SHAP computation methods return identical
            results.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.ensemble import RandomForestClassifier
        >>>
        >>> model = RandomForestClassifier().fit(X_train, y_train)
        >>> shap_values, shap_exp = get_shap_values(model, X_test)
        >>> shap_values.shape
        (len(X_test), X_test.shape[1])
    """
    explainer = shap.Explainer(model)

    shap_values = explainer.shap_values(X)
    assert type(shap_values) == np.ndarray
    assert shap_values.shape == X.shape

    shap_explainer_values = explainer(X)
    assert not np.testing.assert_array_equal(
        shap_explainer_values.values, shap_values
    )
    assert type(shap_explainer_values.values) == np.ndarray
    assert shap_explainer_values.values.shape == X.shape
    return [shap_values, shap_explainer_values]
