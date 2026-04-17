# #!/usr/bin/env python3


# """Test custom scikit-learn transformer."""

# import numpy as np
# import pandas as pd
# import pytest
# from sklearn.base import clone
# from sklearn.exceptions import NotFittedError


# def test_fit_sets_feature_names(base_df, combiner):
#     """Verifies that fit stores input feature names.

#     Args:
#         base_df: Pytest fixture providing a sample DataFrame.
#     """
#     combiner.fit(base_df)
#     assert hasattr(combiner, "feature_names_in_")
#     assert combiner.feature_names_in_ == list(base_df.columns)


# def test_transform_replaces_categories(base_df, combiner):
#     """Verifies that transform correctly groups specified categories.

#     Args:
#         base_df: Pytest fixture providing a sample DataFrame.
#     """
#     combiner.fit(base_df)
#     df_out = combiner.transform(base_df)
#     assert "$80K+" in df_out["income_category"].unique().tolist()
#     assert "Post-Graduate" in df_out["education_level"].unique()
#     assert "Other" in df_out["marital_status"].unique()
#     print(df_out["dependent_count"].unique().tolist())


# def test_transform_does_not_mutate_input(base_df, combiner):
#     """Verifies that transform does not modify the original DataFrame.

#     Args:
#         base_df: Pytest fixture providing a sample DataFrame.
#     """
#     combiner.fit(base_df)
#     df_copy = base_df.copy()
#     _ = combiner.transform(base_df)
#     assert base_df.equals(df_copy)


# def test_transform_requires_fit(base_df, combiner):
#     """Verifies that transform raises an error if called before fit.

#     Args:
#         base_df: Pytest fixture providing a sample DataFrame.
#     """
#     combiner = clone(combiner)
#     with pytest.raises(NotFittedError):
#         combiner.transform(base_df)


# def test_get_feature_names_out_after_fit(base_df, combiner):
#     """Verifies get_feature_names_out returns correct names after fit.

#     Args:
#         base_df: Pytest fixture providing a sample DataFrame.
#     """
#     combiner.fit(base_df)
#     names = combiner.get_feature_names_out()
#     assert isinstance(names, np.ndarray)
#     assert list(names) == list(base_df.columns)


# def test_get_feature_names_out_with_input_features(combiner):
#     """Verifies get_feature_names_out respects provided input features."""
#     input_features = ["col1", "col2"]
#     combiner = clone(combiner)
#     names = combiner.get_feature_names_out(input_features=input_features)
#     assert list(names) == input_features


# def test_get_feature_names_out_without_fit(combiner):
#     """Verifies fallback feature naming when transformer is not fitted."""
#     combiner = clone(combiner)
#     combiner.n_features_in_ = 2
#     names = combiner.get_feature_names_out()
#     assert list(names) == ["x0", "x1"]


# def test_pipeline_integration_numericals(data_split, pipe):
#     """Verifies that CategoryCombiner2 integrates correctly within a full pipeline.

#     The transformer is applied prior to preprocessing and model training.
#     This test ensures that the pipeline fits and produces predictions
#     without errors.

#     Args:
#         base_df: Pytest fixture providing a sample DataFrame including
#             both features and target column `is_churned`.
#     """
#     X_train, y_train, _, _ = data_split
#     pipe.fit(X_train, y_train)
#     preds = pipe.predict(X_train)
#     assert len(preds) == len(X_train)


# def test_pipeline_feature_names(data_split, pipe):
#     """Verifies that feature names are preserved through the full pipeline.

#     This test ensures that after applying CategoryCombiner2 and the
#     preprocessing pipeline, feature names remain consistent and include
#     expected transformed columns.

#     Args:
#         base_df: Pytest fixture providing a sample DataFrame including
#             both features and target column `is_churned`.
#     """
#     X_train, y_train, _, _ = data_split
#     pipe.fit(X_train, y_train)
#     feature_names = pipe.named_steps["pre"].get_feature_names_out()
#     # in conftest.py, categoricals are excluded in pipe so only check ordinals
#     expected_cols = ["income_category", "education_level"]
#     assert set(expected_cols).issubset(set(feature_names))


# def test_transform_only_affects_target_columns(base_df, combiner):
#     """Verifies that only intended columns are modified.

#     Args:
#         base_df: Pytest fixture providing a sample DataFrame.
#     """
#     combiner.fit(base_df)
#     df_out = combiner.transform(base_df)
#     unchanged_cols = set(base_df.columns) - set(
#         [
#             "income_category",
#             "education_level",
#             "marital_status",
#             "dependent_count",
#         ]
#     )
#     for col in unchanged_cols:
#         assert base_df[col].equals(df_out[col])


# def test_multiple_columns_specific_logic(combiner):
#     """Verifies transformer applies only defined hardcoded transformations."""
#     df = pd.DataFrame(
#         {
#             "income_category": ["$120K +", "$80K - $120K"],
#             "education_level": ["Doctorate", "Graduate"],
#             "marital_status": ["Divorced", "Married"],
#             "dependent_count": ["4", "2"],
#         }
#     )
#     combiner.fit(df)
#     result = combiner.transform(df)
#     assert result["income_category"].tolist() == ["$80K+", "$80K+"]
#     assert result["education_level"].tolist() == ["Post-Graduate", "Graduate"]
#     assert result["marital_status"].tolist() == ["Other", "Married"]
#     assert result["dependent_count"].tolist() == ["4+", "2"]
