#!/usr/bin/env python3


"""Define helper functions to import to and export from R2 bucket."""

import tempfile
from io import BytesIO
from typing import Any, Dict, List, Optional

import botocore.exceptions
import joblib
import pandas as pd
from boto3 import resource as boto3_resource
from botocore.config import Config
from mypy_boto3_s3 import S3Client


def pandas_read_parquet_r2(
    s3_client: S3Client,
    bucket_name: str,
    r2_key: str,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Reads a parquet file from a private R2 bucket into a DataFrame.

    This function retrieves a parquet object from an R2 (S3-compatible)
    bucket and loads it into a pandas DataFrame using an in-memory buffer.

    Args:
        s3_client: The authenticated boto3-compatible S3 client.
        bucket_name: The name of the R2 bucket.
        r2_key: The object key (path) to the parquet file.
        columns: The optional list of columns to load.

    Returns:
        pd.DataFrame: A loaded DataFrame with optional column filtering.

    Raises:
        botocore.exceptions.ClientError: If object retrieval fails.

    Examples:
        >>> df = pandas_read_parquet_r2(client, "bucket", "file.parquet")
    """
    s3_object = s3_client.get_object(Bucket=bucket_name, Key=r2_key)
    df = pd.read_parquet(
        BytesIO(s3_object["Body"].read()),
        columns=columns,
        dtype_backend="pyarrow",
    )
    return df


def pandas_read_filtered_parquets_r2(
    s3_client: S3Client,
    bucket_name: str,
    key_prefix: str,
    cols_to_load: List[str],
) -> pd.DataFrame:
    """Reads parquet files from R2 bucket using a key prefix filter.

    This function lists objects matching a prefix and loads them into a
    single concatenated DataFrame.

    Args:
        s3_client: The authenticated boto3-compatible S3 client.
        bucket_name: The name of the R2 bucket.
        key_prefix: The prefix used to filter parquet files.
        cols_to_load: The list of columns to load from each file.

    Returns:
        pd.DataFrame: The concatenated DataFrame from matching parquet files.

    Raises:
        AssertionError: If the list_objects API call fails.

    Examples:
        >>> df = pandas_read_filtered_parquets_r2(
        ...     client, "bucket", "prefix/", ["col1", "col2"]
        ... )
    """
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


def pandas_read_xlsx_r2(
    s3_client: S3Client, bucket_name: str, r2_key: str, dtypes: Dict[str, str]
) -> pd.DataFrame:
    """Reads an Excel file from a private R2 bucket into a DataFrame.

    Args:
        s3_client: The authenticated boto3-compatible S3 client.
        bucket_name: The name of the R2 bucket.
        r2_key: The object key (path) to the Excel file.
        dtypes: The dictionary mapping column names to data types.

    Returns:
        pd.DataFrame: The loaded DataFrame with specified dtypes.

    Raises:
        botocore.exceptions.ClientError: If object retrieval fails.

    Examples:
        >>> df = pandas_read_xlsx_r2(
        ...     client, "bucket", "file.xlsx", {"col": "string"}
        ... )
    """
    s3_object = s3_client.get_object(Bucket=bucket_name, Key=r2_key)
    df = pd.read_excel(
        BytesIO(s3_object["Body"].read()),
        dtype=dtypes,
        dtype_backend="pyarrow",
    )
    return df


def export_df_to_r2(
    s3_client: S3Client,
    df: pd.DataFrame,
    bucket_name: str,
    r2_key: str,
    verbose: bool = True,
) -> None:
    """Exports a DataFrame to an R2 bucket if the object does not exist.

    The DataFrame is serialized as a compressed parquet file and uploaded
    only if the key is not already present.

    Args:
        s3_client: The authenticated boto3-compatible S3 client.
        df: The DataFrame to export.
        bucket_name: The name of the R2 bucket.
        r2_key: The object key (path) for the parquet file.
        verbose: Whether to print status messages.

    Returns:
        None

    Raises:
        botocore.exceptions.ClientError: For unexpected S3 errors.

    Examples:
        >>> export_df_to_r2(client, df, "bucket", "data/file.parquet")
    """
    try:
        s3_client.head_object(Bucket=bucket_name, Key=r2_key)
        print(f"Key {r2_key} already exists in bucket {bucket_name}")
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            if verbose:
                print(f"Key {r2_key} does not exist in bucket {bucket_name}")
            buffer = BytesIO()
            df.to_parquet(
                buffer, index=False, engine="pyarrow", compression="gzip"
            )
            response = s3_client.put_object(
                Bucket=bucket_name, Key=r2_key, Body=buffer.getvalue()
            )
            assert response["ResponseMetadata"]["HTTPStatusCode"] == 200
            if verbose:
                print(f"Exported {len(df):,} rows to key: {r2_key}")
        elif e.response["Error"]["Code"] == "403":
            if verbose:
                print(f"Access denied to bucket {bucket_name} or key {r2_key}")
        else:
            if verbose:
                print(f"An unexpected error occurred: {e}")


def joblib_dump_to_r2(
    s3_client: S3Client,
    pipe: Any,
    bucket_name: str,
    r2_key: str,
    verbose: bool = True,
) -> None:
    """Serializes and uploads a Python object to an R2 bucket.

    The object (e.g., trained pipeline) is saved using joblib and uploaded
    only if the target key does not already exist.

    Args:
        s3_client: The authenticated boto3-compatible S3 client.
        pipe: Python object to serialize (e.g., model pipeline).
        bucket_name: The name of the R2 bucket.
        r2_key: The object key (path) for the saved object.
        verbose: Whether to print status messages.

    Returns:
        None

    Raises:
        botocore.exceptions.ClientError: For unexpected S3 errors.

    Examples:
        >>> joblib_dump_to_r2(client, pipe, "bucket", "model.joblib")
    """
    try:
        s3_client.head_object(Bucket=bucket_name, Key=r2_key)
        print(f"Key {r2_key} already exists in bucket {bucket_name}")
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            if verbose:
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
            if verbose:
                print(f"Saved pipeline to key: {r2_key}")
        elif e.response["Error"]["Code"] == "403":
            if verbose:
                print(f"Access denied to bucket {bucket_name} or key {r2_key}")
        else:
            if verbose:
                print(f"An unexpected error occurred: {e}")


def joblib_load_from_r2(
    s3_client: S3Client, bucket_name: str, key_prefix: str
) -> Any:
    """Loads a joblib-serialized object from an R2 bucket.

    This function retrieves the latest object matching a prefix and
    deserializes it using joblib.

    Args:
        s3_client: The authenticated boto3-compatible S3 client.
        bucket_name: The name of the R2 bucket.
        key_prefix: The prefix used to locate the object.

    Returns:
        Any: A deserialized Python object.

    Raises:
        AssertionError: If object listing fails.

    Examples:
        >>> model = joblib_load_from_r2(client, "bucket", "model_prefix")
    """
    # get list of objects based on key prefix
    s3_objects = s3_client.list_objects_v2(
        Bucket=bucket_name, Prefix=key_prefix, MaxKeys=1
    )
    assert s3_objects["ResponseMetadata"]["HTTPStatusCode"] == 200
    # # get last object from list
    r2_key = list(map(lambda x: x["Key"], s3_objects["Contents"]))[-1]

    # get s3_resource from s3_client
    credentials = s3_client._request_signer._credentials
    s3_resource = boto3_resource(
        "s3",
        endpoint_url=s3_client.meta.endpoint_url,
        aws_access_key_id=credentials.access_key,
        aws_secret_access_key=credentials.secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    # get bucket object from s3_resource
    s3_bucket = s3_resource.Bucket(bucket_name)

    # load object from bucket
    with BytesIO() as data:
        s3_bucket.download_fileobj(r2_key, data)
        data.seek(0)
        object = joblib.load(data)
    return object


def get_latest_s3_file_optimized(
    s3_client: S3Client,
    bucket_name: str,
    base_prefix: str,
    file_pattern: str,
    file_extension: str = ".parquet.gzip",
):
    """Finds the newest object key by checking folders in reverse order.

    Optimizes S3 listing by using a delimiter to identify date-based
    subfolders and sorting them in reverse chronological order to locate
    the latest file efficiently.

    Args:
        s3_client: The authenticated boto3-compatible S3 client.
        bucket_name: The name of the R2 bucket.
        base_prefix: The root prefix to begin listing from (e.g. 'cloud-run').
        file_pattern: The substring required in the object key.
        file_extension: The extension of the files in the R2 bucket.

    Returns:
        str: The S3 Key of the newest file, or None if not found.

    Example:
        >>> client = boto3.client('s3')
        >>> # Search for latest file in multiple sub-folders
        >>> key = get_latest_s3_file_optimized(
        ...     s3_client=client,
        ...     bucket_name='my-bucket',
        ...     base_prefix='cloud-run/',
        ...     file_pattern='all_predictions__',
        ...     file_extension='.parquet.gzip'
        ... )
        >>> # Search for latest file in root of bucket
        >>> key = get_latest_s3_file_optimized(
        ...     s3_client=client,
        ...     bucket_name='my-bucket',
        ...     base_prefix='',
        ...     file_pattern='all_predictions__',
        ...     file_extension='.parquet.gzip'
        ... )
    """
    paginator = s3_client.get_paginator("list_objects_v2")

    # get subfolders (CommonPrefixes) using a delimiter
    folders = []
    page_iterator = paginator.paginate(
        Bucket=bucket_name, Prefix=base_prefix, Delimiter="/"
    )
    for page in page_iterator:
        for prefix in page.get("CommonPrefixes", []):
            folders.append(prefix["Prefix"])

    # sort folders in reverse (latest date folders first)
    folders.sort(reverse=True)

    # find the newest file in the most recent folder
    for folder in folders:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder)
        matches = [
            obj
            for obj in response.get("Contents", [])
            if file_pattern in obj["Key"]
            and obj["Key"].endswith(file_extension)
        ]

        if matches:
            # if matches exist in this newest folder, get the file with the
            # latest LastModified in this specific folder
            latest_key = max(matches, key=lambda x: x["LastModified"])["Key"]
            return latest_key
    return None


def pandas_read_latest_parquet_r2(
    s3_client: S3Client,
    bucket_name: str,
    base_prefix: str,
    file_pattern: str,
    file_extension: str = ".parquet.gzip",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Finds the latest S3 key and reads it into a Pandas DataFrame.

    This function coordinates the discovery of the newest file and handles
    the data ingestion using the specified pyarrow backend.

    Args:
        s3_client: The authenticated boto3-compatible S3 client.
        bucket_name: The name of the R2 bucket.
        base_prefix: The root prefix to search within.
        file_pattern: The substring to match in the filename.
        columns: The specific columns to load.

    Returns:
        pd.DataFrame: The loaded DataFrame, or an empty DataFrame if no
            file is found.

    Example:
        >>> client = boto3.client('s3')
        >>> # Load from latest file in multiple sub-folders
        >>> df = pandas_read_latest_parquet_r2(
        ...     s3_client=client,
        ...     bucket_name='my-bucket',
        ...     base_prefix='cloud-run/',
        ...     file_pattern='all_predictions__',
        ...     file_extension='.parquet.gzip'
        ... )
        >>> # Load from latest file in root of bucket
        >>> df = pandas_read_latest_parquet_r2(
        ...     s3_client=client,
        ...     bucket_name='my-bucket',
        ...     base_prefix='',
        ...     file_pattern='all_predictions__',
        ...     file_extension='.parquet.gzip'
        ... )
    """
    latest_key = get_latest_s3_file_optimized(
        s3_client, bucket_name, base_prefix, file_pattern, file_extension
    )

    if not latest_key:
        return pd.DataFrame()

    df = pandas_read_parquet_r2(
        s3_client=s3_client,
        bucket_name=bucket_name,
        r2_key=latest_key,
        columns=columns,
    )
    return df


def joblib_load_key_from_r2(
    s3_client: Any, bucket_name: str, r2_key: str
) -> Any:
    """Loads a joblib-serialized object from an R2 bucket.

    This function retrieves an object from a key and deserializes
    it using joblib.

    Args:
        s3_client: The authenticated boto3-compatible S3 client.
        bucket_name: The name of the R2 bucket.
        r2_key: The key of the object in the R2 bucket.

    Returns:
        Any: A deserialized Python object.

    Raises:
        AssertionError: If object listing fails.

    Examples:
        >>> model = joblib_load_from_r2(
        ...     s3_client, "my-bucket", "cloud-folder/model.joblib"
        ... )
    """
    credentials = s3_client._request_signer._credentials
    s3_resource = boto3_resource(
        "s3",
        endpoint_url=s3_client.meta.endpoint_url,
        aws_access_key_id=credentials.access_key,
        aws_secret_access_key=credentials.secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    # get bucket object from s3_resource
    s3_bucket = s3_resource.Bucket(bucket_name)

    # load object from bucket
    with BytesIO() as data:
        s3_bucket.download_fileobj(r2_key, data)
        data.seek(0)
        object = joblib.load(data)
    return object


def joblib_read_latest_r2(
    s3_client: S3Client,
    bucket_name: str,
    base_prefix: str,
    file_pattern: str,
    file_extension: str = ".joblib",
) -> Any:
    """Finds and loads the latest joblib object from an R2 bucket.

    This function identifies the newest matching object key using
    `get_latest_s3_file_optimized()` and deserializes the object using
    `joblib_load_from_r2()`.

    Args:
        s3_client: The authenticated boto3-compatible S3 client.
        bucket_name: The name of the R2 bucket.
        base_prefix: The root prefix to search within.
        file_pattern: The substring to match in the filename.
        file_extension: The extension of the serialized joblib files.

    Returns:
        Any: The deserialized Python object, or None if no matching
            file is found.

    Example:
        >>> client = boto3.client("s3")
        >>> # Load latest model from multiple sub-folders
        >>> pipe_best = joblib_read_latest_r2(
        ...     s3_client=client,
        ...     bucket_name="my-bucket",
        ...     base_prefix="cloud-run/",
        ...     file_pattern="best_model__",
        ...     file_extension=".joblib",
        ... )
        >>>
        >>> # Load latest model from root of bucket
        >>> pipe_best = joblib_read_latest_r2(
        ...     s3_client=client,
        ...     bucket_name="my-bucket",
        ...     base_prefix="",
        ...     file_pattern="best_model__",
        ...     file_extension=".joblib",
        ... )
    """
    latest_key = get_latest_s3_file_optimized(
        s3_client, bucket_name, base_prefix, file_pattern, file_extension
    )

    if not latest_key:
        return None

    obj = joblib_load_key_from_r2(
        s3_client=s3_client, bucket_name=bucket_name, r2_key=latest_key
    )
    return obj
