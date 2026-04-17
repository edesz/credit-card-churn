#!/usr/bin/env python3


"""Define pytest fixtures."""


from pathlib import Path

import altair as alt
import boto3
import numpy as np
import pandas as pd
import pytest
import sklearn.model_selection as mds
import sklearn.preprocessing as pp
from moto import mock_aws
from sklearn import preprocessing as pp
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.cc_churn.costs import calc_predicted_savings, calc_true_savings
from src.cc_churn.scoring import get_scorers
from src.cc_churn.transformers import CategoryCombiner2


@pytest.fixture(scope="session", params=[1.0])
def base_df(request):
    url = (
        "https://raw.githubusercontent.com/azar-s91/dataset/refs/heads/master/"
        "BankChurners.csv"
    )
    dtypes_ordinals = {
        "income_category": "string[pyarrow]",
        "education_level": "string[pyarrow]",
    }
    dtypes_categoricals = {
        "gender": "string[pyarrow]",
        "marital_status": "string[pyarrow]",
        "card_category": "string[pyarrow]",
        "dependent_count": "string[pyarrow]",
    }

    df = (
        pd.read_csv(url)
        .rename(columns=str.lower)
        .rename(
            columns={
                # "Income_Category": "income_category",
                "Card_Category": "card_category",
                "Total_Trans_Amt": "total_trans_amt",
                "total_revolving_bal": "total_revolv_bal",
                "attrition_flag": "is_churned",
            }
        )
        .astype(dtypes_categoricals)
        .astype(dtypes_ordinals)
    )

    label_mapper = {"Existing Customer": 0, "Attrited Customer": 1}
    df["is_churned"] = df["is_churned"].map(label_mapper)

    # Add mock predictions
    np.random.seed(42)
    df["y_pred_proba"] = np.random.uniform(0, 1, len(df))
    df["y_pred"] = (df["y_pred_proba"] >= 0.5).astype(int)

    return df.sample(frac=request.param, random_state=88)


@pytest.fixture(scope="session")
def data_split(base_df):
    label = "is_churned"

    df_train, df_test = mds.train_test_split(
        base_df,
        test_size=0.2,
        random_state=88,
        stratify=base_df[label],
    )

    X_train = df_train.drop(columns=[label])
    y_train = df_train[label]

    X_test = df_test.drop(columns=[label])
    y_test = df_test[label]

    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="session")
def params():
    return dict(
        pred_proba_cutoff=0.72,
        interchange_rate=0.02,
        apr=0.18,
        card_fees={"Blue": 0, "Silver": 50, "Gold": 100, "Platinum": 200},
        multiplier=3.0,
        success_rate=0.40,
        intervention_cost=50,
    )


# @pytest.fixture(scope="session")
# def class_weights(data_split):
#     _, y_train, _, _ = data_split

#     weights = skut.compute_class_weight(
#         class_weight="balanced",
#         classes=np.array([0, 1]),
#         y=y_train.to_numpy(),
#     )

#     return {0: float(weights[0]), 1: float(weights[1])}


# @pytest.fixture(scope="session")
# def preprocessor():
#     numeric_features = [
#         "months_on_book",
#         "total_trans_amt",
#         "total_trans_ct",
#         "total_revolv_bal",
#     ]

#     return ColumnTransformer(
#         transformers=[
#             ("num", Pipeline([("scaler", pp.MinMaxScaler())]), numeric_features)
#         ],
#         remainder="drop",
#     )


# @pytest.fixture(scope="session")
# def model():
#     return skens.HistGradientBoostingClassifier(
#         max_depth=3,
#         l2_regularization=0.25,
#         class_weight="balanced",
#         random_state=42,
#     )


@pytest.fixture(scope="session")
def sample_predictions():
    y_train = pd.Series([0, 0, 1, 1])
    y_train_pred = pd.Series([0, 1, 1, 1])

    y_test = pd.Series([0, 0, 1, 1])
    y_test_pred = pd.Series([0, 0, 0, 1])

    return y_train, y_train_pred, y_test, y_test_pred


@pytest.fixture(scope="session")
def scorers():
    scorers = get_scorers(["f2", "recall"])
    return scorers


# @pytest.fixture(scope="session")
# def trained_model(data_split, model, preprocessor):
#     X_train, y_train, _, _ = data_split

#     _ = preprocessor.fit(X_train)
#     columns_transformed = (
#         preprocessor.named_transformers_["num"].get_feature_names_out().tolist()
#     )
#     X_train_pre = pd.DataFrame(
#         preprocessor.transform(X_train),
#         columns=columns_transformed,
#         index=X_train.index,
#     )
#     model.fit(X_train_pre, y_train)
#     return model, X_train_pre


@pytest.fixture(scope="session")
def binary_data():
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = pd.Series([0, 1, 0, 1])
    y_pred_proba = pd.Series([0.278, 0.916, 0.764, 0.543])
    return y_true, y_pred, y_pred_proba


@pytest.fixture(scope="session")
def combiner():
    return CategoryCombiner2()


def get_features(X):
    numericals = X.select_dtypes(include=np.number).columns.tolist()
    categoricals = ["gender", "marital_status", "dependent_count"]
    ordinal_features = ["income_category", "education_level"]
    return numericals, categoricals, ordinal_features


def get_preprocessor(
    ordinal_transformer, numeric_transformer, numericals, ordinals
):
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numericals),
            ("cat", ordinal_transformer, ordinals),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return preprocessor


def get_pipeline(preprocessor, combiner):
    cat_ord_grouper = combiner.set_output(transform="pandas")
    return Pipeline(
        [
            ("combiner", cat_ord_grouper),
            ("pre", preprocessor),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=100)),
        ]
    ).set_output(transform="pandas")


@pytest.fixture(scope="session")
def pipe(data_split, combiner):
    X, _, _, _ = data_split
    # LogisticRegression requires OHE, but OHE is excluded for speed so
    # categorical features (which require OHE) are not extracted
    numericals, _, ordinals = get_features(X)
    numeric_transformer = Pipeline(steps=[("sc", pp.MinMaxScaler())])
    ordinal_transformer = Pipeline(
        [
            (
                "oe",
                pp.OrdinalEncoder(
                    categories=[
                        [
                            "Unknown",
                            "Less than $40K",
                            "$40K - $60K",
                            "$60K - $80K",
                            "$80K+",
                        ],
                        [
                            "Unknown",
                            "Uneducated",
                            "High School",
                            "College",
                            "Graduate",
                            "Post-Graduate",
                        ],
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
            )
        ]
    )
    return get_pipeline(
        get_preprocessor(
            ordinal_transformer, numeric_transformer, numericals, ordinals
        ),
        combiner,
    )


@pytest.fixture(scope="session")
def df_cv(pipe, data_split):
    primary_metric_val = "prauc"
    scorers = get_scorers(["prauc", "f2", "recall", "rocauc"])
    scoring = scorers[primary_metric_val]
    cv = mds.StratifiedKFold(5, shuffle=True, random_state=88)

    X_train, y_train, _, _ = data_split
    model_fpath = "models/03/LogisticRegression.joblib"
    tuned_model = mds.TunedThresholdClassifierCV(
        estimator=pipe, scoring=scoring, cv=cv
    )
    df_cv = pd.DataFrame(
        mds.cross_validate(
            tuned_model,
            X_train,
            y_train.to_numpy().ravel(),
            scoring=scorers,
            cv=cv,
            params=None,
            return_train_score=True,
            return_estimator=True,
            n_jobs=-1,
        )
    ).assign(
        feat_group="[numericals_1,ordinals]",
        model_name=Path(model_fpath).name.replace(".joblib", ""),
    )
    return df_cv


@pytest.fixture(scope="session")
def metaflow_cv_runs(df_cv):
    cv_results_tuned_models = {
        f"{df_cv['model_name'].head(1).squeeze()}_1747328437": (
            df_cv.assign(
                run_id="1747328437",
                experiment_num=lambda df: (
                    df["feat_group"].map(
                        {
                            "[numericals_1]": 1,
                            "[numericals_2]": 2,
                            "[numericals_1,ordinals]": 3,
                            "[numericals_1,ordinals,categoricals_ohe_encoding]": 4,
                            "[numericals_1,ordinals,categoricals_no_encoding]": 5,
                        }
                    )
                ),
            )
        )
    }
    return cv_results_tuned_models


@pytest.fixture(scope="session")
def sample_df():
    return pd.DataFrame(
        {
            "A": [1, 2, np.nan, 2],
            "B": ["x", "y", "y", "z"],
            "C": [0.1, -0.6, 0.3, 0.8],
        }
    )


@pytest.fixture(scope="session")
def df_business_metrics(base_df, params):
    """Calculate business metrics for model predictions."""
    df = (
        base_df.query("y_pred == 1")
        .pipe(
            lambda df: calc_predicted_savings(
                df,
                params["interchange_rate"],
                params["apr"],
                params["card_fees"],
                params["multiplier"],
                params["success_rate"],
                params["intervention_cost"],
            )
        )
        .assign(
            true_savings=lambda df: np.vectorize(calc_true_savings)(
                pred=df["y_pred"],
                true=df["is_churned"],
                success_rate=params["success_rate"],
                clv=df["clv"],
                intervention_cost=params["intervention_cost"],
            )
        )
    )
    return df


@pytest.fixture(scope="session")
def sample_r2_df():
    """Create a simple DataFrame for testing with boto3."""
    dict_values = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
    return pd.DataFrame(dict_values).convert_dtypes(dtype_backend="pyarrow")


@pytest.fixture(scope="session")
def s3_setup():
    """Creates a mocked S3 client and bucket.

    Returns:
        Tuple[boto3.client, str]: S3 client and bucket name.
    """
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        bucket = "test-bucket"
        client.create_bucket(Bucket=bucket)
        yield client, bucket


@pytest.fixture(scope="session")
def sample_df_plot():
    """Creates a small sample DataFrame for plotting tests.

    Args:
        None

    Returns:
        pd.DataFrame: Simple dataset for visualization tests.
    """
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40],
            "category": [True, False, True, False],
            "group": ["A", "A", "B", "B"],
            "value": [0.1, 0.5, 0.8, 0.3],
            "flag": [True, False, False, True],
        }
    )


@pytest.fixture(scope="session")
def sample_altair_chart():
    """Creates a simple Altair chart for testing customization.

    Args:
        None

    Returns:
        alt.Chart: Basic scatter chart.
    """
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    chart = alt.Chart(df).mark_point().encode(x="x", y="y")
    return chart
