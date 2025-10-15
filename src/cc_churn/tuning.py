#!/usr/bin/env python3


"""Define utilities to tune ML models."""

from datetime import datetime

import pandas as pd
import sklearn.model_selection as mds
from sklearn.pipeline import Pipeline


def tune_threshold_cv(
    preprocessor,
    model,
    model_name_param,
    n_cv_folds,
    scorers,
    primary_metric,
    features,
    X_train,
    y_train,
):
    """."""
    pipe = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", model)]
    )
    tuned_model = mds.TunedThresholdClassifierCV(
        estimator=pipe,
        scoring=scorers[primary_metric],
        cv=mds.StratifiedKFold(
            n_splits=n_cv_folds, shuffle=True, random_state=88
        ),
        n_jobs=-1,
    )
    # params = {"sample_weight": sample_weights}
    params = None
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"({model_name_param}) Start: {start_time_str}...", end="")
    df = pd.DataFrame(
        mds.cross_validate(
            tuned_model,
            X_train[features],
            y_train.to_numpy().ravel(),
            scoring=scorers,
            cv=mds.StratifiedKFold(
                n_splits=n_cv_folds, shuffle=True, random_state=88
            ),
            params=params,
            return_train_score=True,
            return_estimator=True,
            n_jobs=-1,
        )
    ).assign(model_name=model_name_param)
    end_time = datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    time_difference = (end_time - start_time).total_seconds()
    print(f"done at {end_time_str} ({time_difference:,.2f} s).")
    return df
