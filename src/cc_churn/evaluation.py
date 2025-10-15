#!/usr/bin/env python3


"""Define helper functions for model evaluation."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def evaluate_model(
    preprocessor,
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    scorers,
    class_weights=None,
    primary_metric="f2",
    threshold_overfit=5,
    threshold_scoring=0.5,
) -> list[pd.DataFrame]:
    """."""
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    # train on full train split
    _ = pipe.fit(X_train, y_train.to_numpy().ravel())

    # predict full test split
    if "Dummy" not in model_name:
        assert model.class_weight == class_weights
        y_train_pred_proba = pd.Series(
            pipe.predict_proba(X_train)[:, 1],
            index=X_train.index,
            dtype="float64[pyarrow]",
        )
        y_train_pred = pd.Series(
            (y_train_pred_proba >= threshold_scoring).astype(int),
            index=X_train.index,
            dtype="int16[pyarrow]",
        )
        y_test_pred_proba = pd.Series(
            pipe.predict_proba(X_test)[:, 1],
            index=X_test.index,
            dtype="float64[pyarrow]",
        )
        y_test_pred = pd.Series(
            (y_test_pred_proba >= threshold_scoring).astype(int),
            index=X_test.index,
            dtype="int16[pyarrow]",
        )
    else:
        y_train_pred = pd.Series(
            pipe.predict(X_train),
            index=X_train.index,
            dtype="int16[pyarrow]",
        )
        y_train_pred_proba = np.nan
        y_test_pred = pd.Series(
            pipe.predict(X_test),
            index=X_test.index,
            dtype="int16[pyarrow]",
        )
        y_test_pred_proba = np.nan
    df_test_pred = pd.concat(
        [
            X_test.assign(y_pred=y_test_pred, y_pred_proba=y_test_pred_proba),
            y_test,
        ],
        axis=1,
    )
    assert len(df_test_pred) == len(X_test)

    # get scores on train and test splits
    scores_eval = dict(model=model_name)
    scores_eval.update(
        {
            f"train_{k}": scorers[k]._score_func(y_train, y_train_pred)
            for k, v in scorers.items()
        }
    )
    scores_eval.update(
        {
            f"test_{k}": scorers[k]._score_func(y_test, y_test_pred)
            for k, v in scorers.items()
        }
    )

    # estimate overfitting
    df_scores_test = (
        pd.DataFrame.from_records([scores_eval])
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
                    (df["is_overfit"] == True)
                    & (df["pct_diff"] > threshold_overfit)
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
    return [df_scores_test, df_test_pred.assign(model_name=model_name)]
