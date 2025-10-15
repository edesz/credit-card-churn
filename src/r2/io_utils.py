#!/usr/bin/env python3


"""Define helper functions to import to and export from R2 bucket."""

import tempfile
from io import BytesIO

import botocore.exceptions
import joblib
import pandas as pd


def pandas_read_parquet_r2(s3_client, bucket_name, r2_key, columns=None):
    """Read parquet file from private R2 bucket."""
    s3_object = s3_client.get_object(Bucket=bucket_name, Key=r2_key)
    df = pd.read_parquet(
        BytesIO(s3_object["Body"].read()),
        columns=columns,
        dtype_backend="pyarrow",
    )
    return df


def pandas_read_filtered_parquets_r2(
    s3_client, bucket_name, key_prefix, cols_to_load
):
    """Read parquet files using partial filename from private R2 bucket."""
    s3_objects = s3_client.list_objects_v2(
        Bucket=bucket_name, Prefix=key_prefix, MaxKeys=1
    )
    assert s3_objects["ResponseMetadata"]["HTTPStatusCode"] == 200
    df = pd.concat(
        [
            pandas_read_parquet_r2(
                s3_client, bucket_name, obj["Key"], columns=cols_to_load
            )
            for obj in s3_objects["Contents"]
        ],
        ignore_index=True,
    )
    return df


def export_df_to_r2(s3_client, df, bucket_name, r2_key):
    """Export DataFrame to file in private R2 bucket, if not present."""
    try:
        s3_client.head_object(Bucket=bucket_name, Key=r2_key)
        print(f"Key {r2_key} already exists in bucket {bucket_name}")
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"Key {r2_key} does not exist in bucket {bucket_name}")
            buffer = BytesIO()
            df.to_parquet(
                buffer, index=False, engine="pyarrow", compression="gzip"
            )
            response = s3_client.put_object(
                Bucket=bucket_name, Key=r2_key, Body=buffer.getvalue()
            )
            assert response["ResponseMetadata"]["HTTPStatusCode"] == 200
            print(f"Exported {len(df):,} rows to key: {r2_key}")
        elif e.response["Error"]["Code"] == "403":
            print(f"Access denied to bucket {bucket_name} or key {r2_key}")
        else:
            print(f"An unexpected error occurred: {e}")


def joblib_dump_to_r2(s3_client, pipe, bucket_name, r2_key):
    """Export trained pipeline to file in private R2 bucket, if not present."""
    try:
        s3_client.head_object(Bucket=bucket_name, Key=r2_key)
        print(f"Key {r2_key} already exists in bucket {bucket_name}")
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"Key {r2_key} does not exist in bucket {bucket_name}")
            with tempfile.TemporaryFile() as fp:
                # Dump the object to the in-memory file
                joblib.dump(pipe, fp)

                # Seek to the beginning of the file before uploading
                fp.seek(0)

                # Upload the in-memory file to our team's R2 bucket
                s3_client.put_object(
                    Body=fp.read(), Bucket=bucket_name, Key=r2_key
                )
            # verify key is present in bucket
            key_prefix = "__".join(r2_key.split("__", 3)[:-1])
            s3_objects = s3_client.list_objects_v2(
                Bucket=bucket_name, Prefix=key_prefix, MaxKeys=1
            )
            assert s3_objects["ResponseMetadata"]["HTTPStatusCode"] == 200
            print(f"Saved pipeline to key: {r2_key}")
        elif e.response["Error"]["Code"] == "403":
            print(f"Access denied to bucket {bucket_name} or key {r2_key}")
        else:
            print(f"An unexpected error occurred: {e}")
