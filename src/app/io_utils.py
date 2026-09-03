#!/usr/bin/env python3


"""Define helper functions to import to and export from R2 bucket."""

from io import BytesIO
from typing import List, Optional

import pandas as pd
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
