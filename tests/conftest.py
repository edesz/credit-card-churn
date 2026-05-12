#!/usr/bin/env python3


"""Define pytest fixtures."""

import random
from pathlib import Path
from typing import List, Union

import altair as alt
import boto3
import evidently.metrics as em
import numpy as np
import pandas as pd
import pytest
import sklearn.ensemble as skens
import sklearn.model_selection as mds
import sklearn.preprocessing as pp
from evidently import BinaryClassification, DataDefinition, Dataset
from moto import mock_aws
from sklearn import preprocessing as pp
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import src.cc_churn.viz_altair as vzu
import src.r2.io_utils as r2io
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
                "Card_Category": "card_category",
                "Total_Trans_Amt": "total_trans_amt",
                "total_revolving_bal": "total_revolv_bal",
                "attrition_flag": "is_churned",
                "total_relationship_count": "num_products",
            }
        )
        .astype(dtypes_categoricals)
        .astype(dtypes_ordinals)
    )

    label_mapper = {"Existing Customer": 0, "Attrited Customer": 1}
    df = df.assign(
        is_churned=lambda df: df["is_churned"].map(label_mapper),
        outcome=lambda df: df["is_churned"].map(
            {1: "Churned", 0: "Did not Churn"}
        ),
    )

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

    df_train = df_train.sample(frac=1.0, random_state=88)

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


@pytest.fixture
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


@pytest.fixture
def mock_latest_and_read(monkeypatch):
    """Mocks key lookup and parquet reader."""
    calls = {}

    def mock_get_latest(*args, **kwargs):
        calls["key_called"] = True
        return "some/key.parquet.gzip"

    def mock_read_parquet(*args, **kwargs):
        calls["read_called"] = True
        calls["kwargs"] = kwargs
        return pd.DataFrame({"a": [1, 2]})

    monkeypatch.setattr(r2io, "get_latest_s3_file_optimized", mock_get_latest)
    monkeypatch.setattr(r2io, "pandas_read_parquet_r2", mock_read_parquet)
    return calls


@pytest.fixture
def sample_joblib_object():
    """Provides a simple serializable Python object."""
    return {"model": "xgboost", "score": 0.91}


@pytest.fixture
def mock_latest_and_joblib(monkeypatch):
    """Mocks latest-key lookup and joblib loading."""
    calls = {}

    def mock_get_latest(*args, **kwargs):
        calls["key_called"] = True
        return "models/latest.joblib"

    def mock_load(*args, **kwargs):
        calls["load_called"] = True
        calls["kwargs"] = kwargs
        return {"model": "rf"}

    monkeypatch.setattr(
        r2io,
        "get_latest_s3_file_optimized",
        mock_get_latest,
    )

    monkeypatch.setattr(
        r2io,
        "joblib_load_key_from_r2",
        mock_load,
    )

    return calls


@pytest.fixture
def mock_export(monkeypatch):
    called = {}

    def _mock(*args, **kwargs):
        called["called"] = True
        called["kwargs"] = kwargs

    monkeypatch.setattr(vzu, "export_altair_chart", _mock)
    return called


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


@pytest.fixture
def sample_code():
    """Provides sample Python code as file content.

    Args:
        None

    Returns:
        str: Multi-line Python code string.
    """
    return "print('hello')\nprint('world')\nprint('!')\n"


@pytest.fixture
def simple_df():
    """Creates a simple dataset for uplift/gain tests."""
    return pd.DataFrame(
        {
            "y": [1, 0, 1, 0, 0],
            "p": [0.9, 0.8, 0.7, 0.4, 0.1],
        }
    )


@pytest.fixture
def metrics_df():
    """Creates sample metrics DataFrame."""
    return pd.DataFrame(
        {
            "model_name": ["LR", "LR"],
            "split": ["val", "test"],
            "test_f2": [0.80, 0.75],
            "test_recall": [0.90, 0.88],
        }
    )


# @pytest.fixture
# def sample_uplift_df():
#     """Creates sample uplift curve data."""
#     return pd.DataFrame(
#         {
#             "percentile": [0.2, 0.4, 0.6, 0.8, 1.0],
#             "uplift": [2.0, 1.8, 1.5, 1.2, 1.0],
#         }
#     )


# @pytest.fixture
# def sample_gain_df():
#     """Creates sample gain curve data."""
#     return pd.DataFrame(
#         {
#             "percentile": [0.2, 0.4, 0.6, 0.8, 1.0],
#             "gain": [0.4, 0.7, 0.85, 0.95, 1.0],
#         }
#     )


@pytest.fixture
def df_uplift() -> pd.DataFrame:
    """Fixture providing a cumulative uplift curve."""
    return pd.DataFrame(
        {
            "percentile": [i / 100 for i in range(5, 55, 5)],
            "uplift": [
                6.05,
                5.97,
                5.40,
                4.69,
                3.89,
                3.28,
                2.84,
                2.49,
                2.22,
                2.00,
            ],
        }
    )


@pytest.fixture
def df_gain() -> pd.DataFrame:
    """Fixture providing a cumulative gain curve."""
    return pd.DataFrame(
        {
            "percentile": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            "gain": [5.00, 4.95, 3.90, 3.10, 2.60, 2.30],
        }
    )


@pytest.fixture(scope="session")
def evidently_datasets(data_split):
    X_train, y_train, X_test, y_test = data_split

    df_train = X_train.copy()
    df_train["is_churned"] = y_train

    df_test = X_test.copy()
    df_test["is_churned"] = y_test

    # Required columns for Evidently
    df_train = df_train.assign(
        y_true=df_train["is_churned"].astype(str),
        y_pred=df_train["y_pred"].astype(str),
    )
    df_test = df_test.assign(
        y_true=df_test["is_churned"].astype(str),
        y_pred=df_test["y_pred"].astype(str),
    )

    ordinal_features = ["income_category", "education_level"]
    categorical_features = [
        "gender",
        "marital_status",
        "card_category",
        "dependent_count",
    ]
    numeric_features = list(
        set(X_train.select_dtypes(include=["number"]).columns.tolist())
        - set(["clientnum", "y_pred", "y_pred_proba", "is_churned"])
    )

    data_definition = DataDefinition(
        numerical_columns=numeric_features + ["y_pred_proba"],
        categorical_columns=(
            ordinal_features + categorical_features + ["y_pred", "y_true"]
        ),
        classification=[
            BinaryClassification(
                target="y_true",
                prediction_labels="y_pred",
                prediction_probas="y_pred_proba",
                pos_label="1",
            )
        ],
    )

    dataset_train = Dataset.from_pandas(
        df_train, data_definition=data_definition
    )
    dataset_test = Dataset.from_pandas(df_test, data_definition=data_definition)

    return dataset_train, dataset_test


@pytest.fixture(scope="session")
def sample_metrics(base_df):
    """Provides a realistic set of drift metrics used in production."""

    numeric_features = list(
        set(base_df.select_dtypes(include=["number"]).columns.tolist())
        - set(["clientnum", "y_pred", "y_pred_proba", "is_churned"])
    )

    ordinal_features = ["income_category", "education_level"]
    categorical_features = [
        "gender",
        "marital_status",
        "card_category",
        "dependent_count",
    ]

    metrics = [
        em.ValueDrift(column=c, method="wasserstein", threshold=0.05)
        for c in numeric_features + ["y_pred_proba"]
    ] + [
        em.ValueDrift(column=c, method="jensenshannon", threshold=0.05)
        for c in ordinal_features + categorical_features + ["y_pred", "y_true"]
    ]

    return metrics


@pytest.fixture(scope="session", params=[9])
def numeric_features(request):
    numericals = [
        "total_trans_amt",
        "total_revolv_bal",
        "total_ct_chng_q4_q1",
        "num_products",
        "total_amt_chng_q4_q1",
        "months_inactive_12_mon",
        "contacts_count_12_mon",
        "credit_limit",
        "months_on_book",
    ]
    numericals_shuffled = random.sample(numericals, k=request.param)
    return numericals_shuffled


@pytest.fixture(scope="session")
def get_Xy(data_split) -> List[Union[pd.DataFrame, pd.Series]]:
    X_train, y_train, X_test, y_test = data_split
    X, y = [pd.concat([X_train, X_test]), pd.concat([y_train, y_test])]
    return [X, y]


@pytest.fixture(scope="session")
def pipeline(numeric_features) -> Pipeline:
    numeric_transformer = Pipeline([("scaler", pp.MinMaxScaler())])
    transformers = [("num", numeric_transformer, numeric_features)]

    preprocessor = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    ).set_output(transform="pandas")

    transformers_preprocessors = [("pre", preprocessor)]

    clf = skens.HistGradientBoostingClassifier(
        max_depth=3,
        l2_regularization=0.25,
        class_weight="balanced",
        random_state=42,
    )
    pipe = Pipeline(transformers_preprocessors + [("clf", clf)]).set_output(
        transform="pandas"
    )
    return pipe


@pytest.fixture(scope="session")
def trained_pipeline(pipeline, get_Xy) -> List[Union[Pipeline, np.ndarray]]:
    X, y = get_Xy
    _ = pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    return [pipeline, y_pred]


@pytest.fixture(scope="session")
def preprocess_data(trained_pipeline, get_Xy) -> pd.DataFrame:
    pipe, y_pred = trained_pipeline
    X, y = get_Xy

    pipe_transf_pre, _ = [pipe.named_steps["pre"], pipe.named_steps["clf"]]
    best_features = (
        pipe.named_steps["pre"]
        .named_transformers_["num"]
        .named_steps["scaler"]
        .get_feature_names_out()
        .tolist()
    )

    features_non_preprocessed = list(set(list(X)) - set(best_features))

    _ = pipe_transf_pre.fit(X)
    columns_transformed = list(
        pipe_transf_pre.get_feature_names_out(input_features=list(X)).tolist()
    )

    X_transformed_preprocessed = pd.DataFrame(
        pipe_transf_pre.transform(X),
        columns=columns_transformed,
        index=X.index,
    )

    df_transf_pre = pd.concat(
        [
            X[features_non_preprocessed],
            X_transformed_preprocessed,
            y.rename("y"),
        ],
        axis=1,
    )[list(X) + ["y"]]

    df_transf_pre = df_transf_pre.assign(y_pred=y_pred)
    return df_transf_pre


@pytest.fixture(scope="session")
def get_shap_values_inputs(trained_pipeline) -> List:
    pipe, _ = trained_pipeline
    model = pipe.named_steps["clf"]
    best_features = (
        pipe.named_steps["pre"]
        .named_transformers_["num"]
        .named_steps["scaler"]
        .get_feature_names_out()
        .tolist()
    )
    return [model, best_features]


@pytest.fixture(scope="session")
def expected_shap_values() -> pd.DataFrame:
    expected = [
        {
            "feature": "total_trans_amt",
            "y_class == 1": 2.0885696067733392,
            "y_class == 0": 1.1094070900869124,
        },
        {
            "feature": "total_revolv_bal",
            "y_class == 1": 0.9105218915463017,
            "y_class == 0": 0.5541132815538157,
        },
        {
            "feature": "total_ct_chng_q4_q1",
            "y_class == 1": 0.7297208040328211,
            "y_class == 0": 0.3456090024349042,
        },
        {
            "feature": "num_products",
            "y_class == 1": 0.47973882653631456,
            "y_class == 0": 0.26269088194467005,
        },
        {
            "feature": "total_amt_chng_q4_q1",
            "y_class == 1": 0.32504312949926467,
            "y_class == 0": 0.2770954480182687,
        },
        {
            "feature": "months_inactive_12_mon",
            "y_class == 1": 0.27239414601820405,
            "y_class == 0": 0.3196610225527143,
        },
        {
            "feature": "contacts_count_12_mon",
            "y_class == 1": 0.2290909334104806,
            "y_class == 0": 0.23490676106894415,
        },
        {
            "feature": "credit_limit",
            "y_class == 1": 0.056199224258841184,
            "y_class == 0": 0.06308792379982318,
        },
        {
            "feature": "months_on_book",
            "y_class == 1": 0.09400804735828071,
            "y_class == 0": 0.08297436432449141,
        },
    ]
    return pd.DataFrame.from_records(expected)
