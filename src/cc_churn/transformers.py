#!/usr/bin/env python3


"""Define custom scikit-learn transformers."""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CategoryCombiner2(BaseEstimator, TransformerMixin):
    """Combines predefined categorical and ordinal levels into groups.

    This transformer applies fixed mappings to selected columns to reduce
    cardinality and group semantically similar levels. It is intended for
    feature engineering where domain-specific category consolidation is
    required prior to modeling.

    The transformation is deterministic and column-specific:
        - `income_category`: high-income buckets merged into "$80K+".
        - `education_level`: advanced degrees merged into one group.
        - `marital_status`: rare/unknown values grouped as "Other".
        - `dependent_count`: higher counts grouped as "4+".

    Attributes:
        feature_names_in_: List of input feature names observed during fit.

    Notes:
        - Assumes the presence of required columns in `X`.
        - Does not infer mappings dynamically; rules are hard-coded.
        - Preserves all other columns unchanged.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "income_category": ["$60K", "$80K - $120K"],
        ...     "education_level": ["Graduate", "Doctorate"],
        ...     "marital_status": ["Married", "Unknown"],
        ...     "dependent_count": ["3", "5"],
        ... })
        >>> combiner = CategoryCombiner2()
        >>> _ = combiner.fit(df)
        >>> combiner.transform(df)
    """

    def fit(self, X, y=None) -> Self:
        """Stores feature names from the input DataFrame.

        Args:
            X: Input DataFrame containing categorical features.
            y: Optional target values (ignored).

        Returns:
            Self: Fitted transformer instance.

        Examples:
            >>> combiner.fit(df)
        """
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies predefined category grouping rules to the DataFrame.

        Args:
            X: Input DataFrame containing required categorical columns.

        Returns:
            pd.DataFrame: Transformed DataFrame with grouped categories.

        Raises:
            ValueError: If input is not a valid DataFrame.
            KeyError: If required columns are missing.
            NotFittedError: If called before `fit`.

        Examples:
            >>> combiner.transform(df)
        """
        check_is_fitted(self, "feature_names_in_")
        return X.assign(
            income_category=lambda df: df["income_category"].replace(
                ["$80K - $120K", "$120K +"], "$80K+"
            ),
            education_level=lambda df: df["education_level"].replace(
                ["Post-Graduate", "Doctorate"], "Post-Graduate"
            ),
            marital_status=lambda df: df["marital_status"].replace(
                ["Unknown", "Divorced"], "Other"
            ),
            dependent_count=lambda df: df["dependent_count"].replace(
                ["4", "5"], "4+"
            ),
        )

    def get_feature_names_out(self, input_features=None):
        """Returns output feature names.

        Args:
            input_features: Optional list of input feature names.

        Returns:
            np.ndarray: Array of output feature names.

        Examples:
            >>> combiner.get_feature_names_out()
        """
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return np.array(self.feature_names_in_, dtype=object)
        return np.array(
            [f"x{i}" for i in range(self.n_features_in_)], dtype=object
        )
