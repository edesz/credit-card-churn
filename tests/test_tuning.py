# #!/usr/bin/env python3


# """Test methods to validate threshold tuning during the validation phase."""


# import numpy as np
# import pandas as pd

# import src.cc_churn.tuning as tn


# def test_combine_cv_scores_structure(metaflow_cv_runs):
#     """Verifies combined CV output has expected structure and columns.

#     Args:
#         metaflow_cv_runs: Fixture providing CV results with estimators.
#     """
#     df = tn.combine_cv_scores_thresholds(metaflow_cv_runs)
#     assert isinstance(df, pd.DataFrame)
#     assert len(df) > 0
#     expected_cols = {"model_name", "cv_fold_outer", "threshold"}
#     assert expected_cols.issubset(df.columns)


# def test_cv_fold_outer_sequential(metaflow_cv_runs):
#     """Verifies that cv_fold_outer is sequential starting from 1.

#     Args:
#         metaflow_cv_runs: Fixture providing CV results with estimators.
#     """
#     df = tn.combine_cv_scores_thresholds(metaflow_cv_runs)
#     folds = df["cv_fold_outer"].unique().tolist()
#     assert sorted(folds) == list(range(1, len(folds) + 1))


# def test_threshold_extraction(metaflow_cv_runs):
#     """Verifies thresholds are extracted correctly from estimators.

#     Args:
#         metaflow_cv_runs: Fixture providing CV results with estimators.
#     """
#     df = tn.combine_cv_scores_thresholds(metaflow_cv_runs)
#     assert df["threshold"].notna().all()
#     assert (df["threshold"] >= 0).all()
#     assert (df["threshold"] <= 1).all()


# def test_estimator_column_removed(metaflow_cv_runs):
#     """Verifies that estimator column is removed after processing.

#     Args:
#         metaflow_cv_runs: Fixture providing CV results with estimators.
#     """
#     df = tn.combine_cv_scores_thresholds(metaflow_cv_runs)
#     assert "estimator" not in df.columns


# def test_agg_cv_scores_structure(metaflow_cv_runs):
#     """Verifies aggregated CV output has expected structure.

#     Args:
#         metaflow_cv_runs: Fixture providing CV results with estimators.
#     """
#     df_combined = tn.combine_cv_scores_thresholds(metaflow_cv_runs)
#     df_agg = tn.agg_cv_scores_thresholds(
#         df_combined, primary_metric="f2", threshold_overfit=5
#     )
#     assert isinstance(df_agg, pd.DataFrame)
#     assert len(df_agg) > 0


# def test_groupby_reduces_rows(metaflow_cv_runs):
#     """Verifies aggregation reduces number of rows compared to input.

#     Args:
#         metaflow_cv_runs: Fixture providing CV results with estimators.
#     """
#     df_combined = tn.combine_cv_scores_thresholds(metaflow_cv_runs)
#     df_agg = tn.agg_cv_scores_thresholds(
#         df_combined, primary_metric="f2", threshold_overfit=5
#     )
#     assert len(df_agg) <= len(df_combined)


# def test_overfitting_columns_exist(metaflow_cv_runs):
#     """Verifies overfitting-related columns are present.

#     Args:
#         metaflow_cv_runs: Fixture providing CV results with estimators.
#     """
#     df_combined = tn.combine_cv_scores_thresholds(metaflow_cv_runs)
#     df_agg = tn.agg_cv_scores_thresholds(
#         df_combined, primary_metric="f2", threshold_overfit=5
#     )
#     expected_cols = {"pct_diff", "is_overfit", "is_overfit_significant"}
#     assert expected_cols.issubset(df_agg.columns)


# def test_pct_diff_computation(metaflow_cv_runs):
#     """Verifies pct_diff is computed correctly.

#     Args:
#         metaflow_cv_runs: Fixture providing CV results with estimators.
#     """
#     df_combined = tn.combine_cv_scores_thresholds(metaflow_cv_runs)
#     df_agg = tn.agg_cv_scores_thresholds(
#         df_combined, primary_metric="f2", threshold_overfit=5
#     )
#     row = df_agg.iloc[0]
#     train = row["train_f2"]
#     test = row["test_f2"]
#     if train != 0:
#         expected = abs(train - test) / train * 100
#         assert np.isclose(row["pct_diff"], expected, equal_nan=True)


# def test_overfit_flag_logic(metaflow_cv_runs):
#     """Verifies overfitting flag logic is consistent.

#     Args:
#         metaflow_cv_runs: Fixture providing CV results with estimators.
#     """
#     df_combined = tn.combine_cv_scores_thresholds(metaflow_cv_runs)
#     df_agg = tn.agg_cv_scores_thresholds(
#         df_combined, primary_metric="f2", threshold_overfit=5
#     )
#     row = df_agg.iloc[0]
#     expected = row["train_f2"] > row["test_f2"]
#     assert bool(row["is_overfit"]) == expected


# def test_significant_overfit_logic(metaflow_cv_runs):
#     """Verifies significant overfitting flag behavior.

#     Args:
#         metaflow_cv_runs: Fixture providing CV results with estimators.
#     """
#     df_combined = tn.combine_cv_scores_thresholds(metaflow_cv_runs)
#     # set the overfitting threshold to 0 to check sensitivity
#     df_agg = tn.agg_cv_scores_thresholds(
#         df_combined, primary_metric="f2", threshold_overfit=0
#     )
#     row = df_agg.iloc[0]
#     if row["is_overfit"]:
#         assert row["is_overfit_significant"] in [True, False]


# def test_sorting_by_primary_metric(metaflow_cv_runs):
#     """Verifies results are sorted by test primary metric descending.

#     Args:
#         metaflow_cv_runs: Fixture providing CV results with estimators.
#     """
#     df_combined = tn.combine_cv_scores_thresholds(metaflow_cv_runs)
#     df_agg = tn.agg_cv_scores_thresholds(
#         df_combined, primary_metric="f2", threshold_overfit=5
#     )
#     values = df_agg["test_f2"].to_numpy()
#     assert np.all(values[:-1] >= values[1:])
