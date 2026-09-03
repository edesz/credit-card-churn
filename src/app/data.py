#!/usr/bin/env python3


"""Define helper functions to assign cost to model inference predictions."""

import os
from io import StringIO
from typing import Dict, List, Tuple, Union

import boto3
import numpy as np
import pandas as pd

from app import costs
from app import costs_savings as costs_sv
from app import io_utils as r2io


def get_metrics(
    df: pd.DataFrame,
    interchange_rate: float,
    apr: float,
    card_fees: Dict[str, int],
    multiplier: float,
    success_rate: float,
    intervention_cost: Union[float, int],
) -> List[pd.DataFrame]:
    """."""
    df_business_metrics = (
        df.pipe(
            lambda df: costs.calc_predicted_savings(
                df,
                interchange_rate=interchange_rate,
                apr=apr,
                card_fees=card_fees,
                multiplier=multiplier,
                success_rate=success_rate,
                intervention_cost=intervention_cost,
            )
        )
        .assign(
            true_savings=lambda df: np.vectorize(costs.calc_true_savings)(
                pred=df["y_pred"],
                true=df["is_churned"],
                success_rate=success_rate,
                clv=df["clv"],
                intervention_cost=intervention_cost,
            )
        )
        .pipe(
            costs_sv.get_buckets,
            bins_risk_values=[0.72, 0.81, 0.9, 1.0],
            bins_risk_labels=["Low", "Medium", "High"],
            bins_clv_tier_values=[0, 0.4, 0.7, 0.9, 1.0],
            bins_clv_tier_labels=["Bronze", "Silver", "Gold", "Platinum"],
        )
    )
    df_summary_per_clv_tier_risk_level = costs_sv.summarize_campaign_mix(
        df_business_metrics,
        "expected_savings_per_customer",
        intervention_cost,
        False,
        False,
    )
    return [df_business_metrics, df_summary_per_clv_tier_risk_level]


def get_data_v2(
    columns: Tuple[str],
    dtypes_categoricals: Dict[str, str],
    dtypes_ordinals: Dict[str, str],
    prefix: str,
    r2_key_pred: str,
) -> pd.DataFrame:
    """."""
    account_id = os.getenv("ACCOUNT_ID")

    s3_client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        region_name="auto",
    )
    df_all_pred = (
        r2io.pandas_read_latest_parquet_r2(
            s3_client,
            os.getenv("BUCKET_NAME"),
            f"{prefix}/",
            r2_key_pred,
            ".parquet.gzip",
            columns,
        )
        .astype({"card_category": "category"})
        .query("y_pred == 1")
        .astype(dtypes_categoricals)
        .astype(dtypes_ordinals)
    )
    return df_all_pred


def load_r2_data(
    prefix: str,
    r2_key_pred: str,
    columns: List[str],
    dtypes_categoricals: Dict[str, str],
    dtypes_ordinals: Dict[str, str],
    interchange_rate: float,
    apr: float,
    card_fees: Dict[str, int],
    multiplier: float,
    success_rate: float,
    intervention_cost: Union[float, int],
) -> List[pd.DataFrame]:
    """."""
    outputs = get_metrics(
        get_data_v2(
            columns,
            dtypes_categoricals,
            dtypes_ordinals,
            prefix,
            r2_key_pred,
        ),
        interchange_rate,
        apr,
        card_fees,
        multiplier,
        success_rate,
        intervention_cost,
    )
    return outputs


def export_to_csv(df: pd.DataFrame) -> StringIO:
    """."""
    sio = StringIO()
    df.to_csv(sio, index=False)
    sio.seek(0)
    return sio
