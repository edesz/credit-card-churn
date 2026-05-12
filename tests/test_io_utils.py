#!/usr/bin/env python3


"""Test methods read from and write to private R2 bucket using boto3."""

from io import BytesIO

import botocore.exceptions
import joblib
import pandas as pd
import pytest

import src.r2.io_utils as r2io


def test_pandas_read_parquet_r2(s3_setup, sample_r2_df):
    """Verifies parquet file can be written and read from R2.

    Args:
        s3_setup: Mocked S3 client and bucket.
        sample_r2_df: Sample DataFrame.
    """
    client, bucket = s3_setup

    buffer = BytesIO()
    sample_r2_df.to_parquet(buffer, index=False)
    buffer.seek(0)

    client.put_object(Bucket=bucket, Key="test.parquet", Body=buffer.read())
    df = r2io.pandas_read_parquet_r2(client, bucket, "test.parquet")
    pd.testing.assert_frame_equal(df, sample_r2_df)


def test_pandas_read_filtered_parquets_r2(s3_setup, sample_r2_df):
    """Verifies multiple parquet files are concatenated.

    Args:
        s3_setup: Mocked S3 client and bucket.
        sample_r2_df: Sample DataFrame.
    """
    client, bucket = s3_setup

    buffer = BytesIO()
    sample_r2_df.to_parquet(buffer, index=False)
    buffer.seek(0)

    client.put_object(
        Bucket=bucket,
        Key=f"prefix/file_1.parquet",
        Body=buffer.read(),
    )
    df = r2io.pandas_read_filtered_parquets_r2(
        client, bucket, "prefix/", ["a", "b"]
    )
    assert len(df) == 3


def test_pandas_read_xlsx_r2(s3_setup, sample_r2_df):
    """Verifies Excel file can be read from R2.

    Args:
        s3_setup: Mocked S3 client and bucket.
        sample_r2_df: Sample DataFrame.
    """
    client, bucket = s3_setup

    buffer = BytesIO()
    sample_r2_df.to_excel(buffer, index=False)
    buffer.seek(0)

    client.put_object(Bucket=bucket, Key="test.xlsx", Body=buffer.read())

    df = r2io.pandas_read_xlsx_r2(
        client,
        bucket,
        "test.xlsx",
        dtypes={"a": "int64[pyarrow]", "b": "string[pyarrow]"},
    )

    assert list(df.columns) == ["a", "b"]
    assert len(df) == 3


def test_export_df_to_r2_new_key(s3_setup, sample_r2_df):
    """Verifies DataFrame is uploaded when key does not exist.

    Args:
        s3_setup: Mocked S3 client and bucket.
        sample_r2_df: Sample DataFrame.
    """
    client, bucket = s3_setup
    r2io.export_df_to_r2(client, sample_r2_df, bucket, "data.parquet")
    response = client.get_object(Bucket=bucket, Key="data.parquet")
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200


def test_export_df_to_r2_existing_key(s3_setup, sample_r2_df):
    """Verifies existing key prevents overwrite.

    Args:
        s3_setup: Mocked S3 client and bucket.
        sample_r2_df: Sample DataFrame.
    """
    client, bucket = s3_setup
    client.put_object(Bucket=bucket, Key="data.parquet", Body=b"dummy")
    r2io.export_df_to_r2(client, sample_r2_df, bucket, "data.parquet")
    response = client.get_object(Bucket=bucket, Key="data.parquet")
    assert response["Body"].read() == b"dummy"


def test_joblib_dump_to_r2(s3_setup):
    """Verifies object is serialized and uploaded.

    Args:
        s3_setup: Mocked S3 client and bucket.
    """
    client, bucket = s3_setup
    obj = {"a": 1}
    r2io.joblib_dump_to_r2(client, obj, bucket, "model__v1.joblib")
    objects = client.list_objects_v2(Bucket=bucket)
    assert "Contents" in objects


def test_joblib_load_from_r2(s3_setup):
    """Verifies joblib object is correctly loaded from R2.

    Args:
        s3_setup: Mocked S3 client and bucket.
    """
    client, bucket = s3_setup
    obj = {"test": 123}
    buffer = BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)

    key = "model__v1.joblib"
    client.put_object(Bucket=bucket, Key=key, Body=buffer.read())
    loaded = r2io.joblib_load_from_r2(client, bucket, "model")
    assert loaded == obj


def test_pandas_read_parquet_r2_missing_key(s3_setup):
    """Verifies missing object raises ClientError.

    Args:
        s3_setup: Mocked S3 client and bucket.
    """
    client, bucket = s3_setup
    with pytest.raises(botocore.exceptions.ClientError):
        r2io.pandas_read_parquet_r2(client, bucket, "missing.parquet")


def test_filtered_parquet_bad_status(monkeypatch, s3_setup):
    """Verifies assertion triggers on bad list_objects response.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        s3_setup: Mocked S3 client and bucket.
    """
    client, bucket = s3_setup

    def fake_list(*args, **kwargs):
        return {"ResponseMetadata": {"HTTPStatusCode": 500}}

    monkeypatch.setattr(client, "list_objects_v2", fake_list)
    with pytest.raises(AssertionError):
        r2io.pandas_read_filtered_parquets_r2(client, bucket, "prefix", ["a"])


def _upload_object(client, bucket, key, body=b"data"):
    """Helper to upload object to mocked S3."""
    client.put_object(Bucket=bucket, Key=key, Body=body)


def test_get_latest_s3_file_returns_latest_key(s3_setup):
    """Verifies latest file is selected from newest folder.

    Args:
        None

    Returns:
        None
    """
    client, bucket = s3_setup

    # Older folder
    _upload_object(
        client,
        bucket,
        "cloud-run/2024-01-01/file__a.parquet.gzip",
    )

    # Newer folder
    _upload_object(
        client,
        bucket,
        "cloud-run/2024-02-01/file__b.parquet.gzip",
    )

    key = r2io.get_latest_s3_file_optimized(
        client,
        bucket,
        base_prefix="cloud-run/",
        file_pattern="file__",
    )

    assert key.endswith("file__b.parquet.gzip")


def test_get_latest_s3_file_filters_pattern(s3_setup):
    """Verifies only matching pattern files are considered.

    Args:
        None

    Returns:
        None
    """
    client, bucket = s3_setup

    _upload_object(
        client,
        bucket,
        "cloud-run/2024-02-01/other.parquet.gzip",
    )
    _upload_object(
        client,
        bucket,
        "cloud-run/2024-02-01/match__file.parquet.gzip",
    )

    key = r2io.get_latest_s3_file_optimized(
        client,
        bucket,
        base_prefix="cloud-run/",
        file_pattern="match__",
    )

    assert "match__file" in key


def test_get_latest_s3_file_returns_none_if_no_match(s3_setup):
    """Verifies None is returned when no matching file exists.

    Args:
        None

    Returns:
        None
    """
    client, bucket = s3_setup

    _upload_object(
        client,
        bucket,
        "cloud-run/2024-02-01/file.parquet.gzip",
    )

    key = r2io.get_latest_s3_file_optimized(
        client,
        bucket,
        base_prefix="cloud-run/",
        file_pattern="does_not_exist",
    )

    assert key is None


def test_pandas_read_latest_parquet_success(s3_setup, mock_latest_and_read):
    """Verifies latest file is loaded into DataFrame.

    Args:
        None

    Returns:
        None
    """
    client, bucket = s3_setup

    df = r2io.pandas_read_latest_parquet_r2(
        s3_client=client,
        bucket_name=bucket,
        base_prefix="cloud-run/",
        file_pattern="file__",
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    assert mock_latest_and_read.get("key_called") is True
    assert mock_latest_and_read.get("read_called") is True


def test_pandas_read_latest_parquet_passes_columns(
    s3_setup, mock_latest_and_read
):
    """Verifies columns argument is passed to reader.

    Args:
        None

    Returns:
        None
    """
    client, bucket = s3_setup

    cols = ["a"]

    r2io.pandas_read_latest_parquet_r2(
        s3_client=client,
        bucket_name=bucket,
        base_prefix="cloud-run/",
        file_pattern="file__",
        columns=cols,
    )

    kwargs = mock_latest_and_read["kwargs"]
    assert kwargs["columns"] == cols


def test_pandas_read_latest_parquet_returns_empty_if_no_key(
    s3_setup, monkeypatch
):
    """Verifies empty DataFrame is returned when no file found.

    Args:
        None

    Returns:
        None
    """
    client, bucket = s3_setup

    monkeypatch.setattr(
        r2io,
        "get_latest_s3_file_optimized",
        lambda *args, **kwargs: None,
    )

    df = r2io.pandas_read_latest_parquet_r2(
        s3_client=client,
        bucket_name=bucket,
        base_prefix="cloud-run/",
        file_pattern="file__",
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_latest_s3_file_optimized_parquet(s3_setup):
    """Verifies latest parquet key is returned from newest folder."""
    client, bucket = s3_setup

    client.put_object(
        Bucket=bucket,
        Key=(
            "cloud-run/2025-01-01/" "all_predictions__2025-01-01.parquet.gzip"
        ),
        Body=b"old",
    )

    client.put_object(
        Bucket=bucket,
        Key=(
            "cloud-run/2025-02-01/" "all_predictions__2025-02-01.parquet.gzip"
        ),
        Body=b"new",
    )

    latest_key = r2io.get_latest_s3_file_optimized(
        s3_client=client,
        bucket_name=bucket,
        base_prefix="cloud-run/",
        file_pattern="all_predictions__",
        file_extension=".parquet.gzip",
    )

    assert latest_key == (
        "cloud-run/2025-02-01/" "all_predictions__2025-02-01.parquet.gzip"
    )


def test_get_latest_s3_file_optimized_joblib(s3_setup):
    """Verifies latest joblib key is returned."""
    client, bucket = s3_setup

    client.put_object(
        Bucket=bucket,
        Key="models/2025-01-01/best_model__v1.joblib",
        Body=b"old-model",
    )

    client.put_object(
        Bucket=bucket,
        Key="models/2025-02-01/best_model__v2.joblib",
        Body=b"new-model",
    )

    latest_key = r2io.get_latest_s3_file_optimized(
        s3_client=client,
        bucket_name=bucket,
        base_prefix="models/",
        file_pattern="best_model__",
        file_extension=".joblib",
    )

    assert latest_key == "models/2025-02-01/best_model__v2.joblib"


def test_get_latest_s3_file_optimized_returns_none(s3_setup):
    """Verifies None is returned when no matching file exists."""
    client, bucket = s3_setup

    client.put_object(
        Bucket=bucket,
        Key="cloud-run/2025-01-01/random.txt",
        Body=b"abc",
    )

    latest_key = r2io.get_latest_s3_file_optimized(
        s3_client=client,
        bucket_name=bucket,
        base_prefix="cloud-run/",
        file_pattern="all_predictions__",
        file_extension=".parquet.gzip",
    )

    assert latest_key is None


def test_pandas_read_latest_parquet_r2(
    mock_latest_and_read,
):
    """Verifies latest parquet file is discovered and loaded."""
    df = r2io.pandas_read_latest_parquet_r2(
        s3_client="mock-client",
        bucket_name="bucket",
        base_prefix="cloud-run/",
        file_pattern="all_predictions__",
        file_extension=".parquet.gzip",
        columns=["a"],
    )

    assert isinstance(df, pd.DataFrame)
    assert mock_latest_and_read["key_called"] is True
    assert mock_latest_and_read["read_called"] is True

    kwargs = mock_latest_and_read["kwargs"]

    assert kwargs["bucket_name"] == "bucket"
    assert kwargs["r2_key"] == "some/key.parquet.gzip"
    assert kwargs["columns"] == ["a"]


def test_pandas_read_latest_parquet_r2_empty(monkeypatch):
    """Verifies empty DataFrame is returned when no key exists."""

    def mock_get_latest(*args, **kwargs):
        return None

    monkeypatch.setattr(
        r2io,
        "get_latest_s3_file_optimized",
        mock_get_latest,
    )

    df = r2io.pandas_read_latest_parquet_r2(
        s3_client="mock-client",
        bucket_name="bucket",
        base_prefix="cloud-run/",
        file_pattern="all_predictions__",
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_joblib_load_key_from_r2(
    s3_setup,
    sample_joblib_object,
):
    """Verifies joblib object is loaded correctly from R2."""
    client, bucket = s3_setup

    buffer = BytesIO()
    joblib.dump(sample_joblib_object, buffer)
    buffer.seek(0)

    client.put_object(
        Bucket=bucket,
        Key="models/model.joblib",
        Body=buffer.read(),
    )

    loaded_obj = r2io.joblib_load_key_from_r2(
        s3_client=client,
        bucket_name=bucket,
        r2_key="models/model.joblib",
    )

    assert loaded_obj == sample_joblib_object


@pytest.mark.parametrize("verbose", [True, False])
def test_export_df_to_r2_verbose_404(
    s3_setup,
    sample_r2_df,
    capsys,
    verbose,
):
    """Verifies verbose logging for missing-key upload flow."""
    client, bucket = s3_setup

    r2_key = "exports/test_file.parquet.gzip"

    r2io.export_df_to_r2(
        s3_client=client,
        df=sample_r2_df,
        bucket_name=bucket,
        r2_key=r2_key,
        verbose=verbose,
    )

    captured = capsys.readouterr()

    if verbose:
        assert f"Key {r2_key} does not exist" in captured.out
        assert f"Exported {len(sample_r2_df):,} rows" in captured.out
    else:
        assert captured.out == ""


@pytest.mark.parametrize("verbose", [True, False])
def test_export_df_to_r2_verbose_existing_key(
    s3_setup,
    sample_r2_df,
    capsys,
    verbose,
):
    """Verifies existing-key message handling."""
    client, bucket = s3_setup

    r2_key = "exports/existing.parquet.gzip"

    client.put_object(
        Bucket=bucket,
        Key=r2_key,
        Body=b"already-exists",
    )

    r2io.export_df_to_r2(
        s3_client=client,
        df=sample_r2_df,
        bucket_name=bucket,
        r2_key=r2_key,
        verbose=verbose,
    )

    captured = capsys.readouterr()

    # This print occurs regardless of verbose flag
    assert f"Key {r2_key} already exists" in captured.out


@pytest.mark.parametrize("verbose", [True, False])
def test_export_df_to_r2_verbose_403(
    monkeypatch,
    sample_r2_df,
    capsys,
    verbose,
):
    """Verifies verbose logging for access denied errors."""

    class MockS3Client:
        def head_object(self, **kwargs):
            raise botocore.exceptions.ClientError(
                error_response={"Error": {"Code": "403"}},
                operation_name="HeadObject",
            )

    r2io.export_df_to_r2(
        s3_client=MockS3Client(),
        df=sample_r2_df,
        bucket_name="bucket",
        r2_key="test.parquet.gzip",
        verbose=verbose,
    )

    captured = capsys.readouterr()

    if verbose:
        assert "Access denied to bucket" in captured.out
    else:
        assert captured.out == ""


@pytest.mark.parametrize("verbose", [True, False])
def test_export_df_to_r2_verbose_unexpected_error(
    monkeypatch,
    sample_r2_df,
    capsys,
    verbose,
):
    """Verifies verbose logging for unexpected errors."""

    class MockS3Client:
        def head_object(self, **kwargs):
            raise botocore.exceptions.ClientError(
                error_response={"Error": {"Code": "500"}},
                operation_name="HeadObject",
            )

    r2io.export_df_to_r2(
        s3_client=MockS3Client(),
        df=sample_r2_df,
        bucket_name="bucket",
        r2_key="test.parquet.gzip",
        verbose=verbose,
    )

    captured = capsys.readouterr()

    if verbose:
        assert "An unexpected error occurred" in captured.out
    else:
        assert captured.out == ""


@pytest.mark.parametrize("verbose", [True, False])
def test_joblib_dump_to_r2_verbose_404(
    s3_setup,
    capsys,
    verbose,
):
    """Verifies verbose logging for successful joblib upload."""
    client, bucket = s3_setup

    obj = {"model": "xgb"}

    r2_key = "models/best_model__2025.joblib"

    r2io.joblib_dump_to_r2(
        s3_client=client,
        pipe=obj,
        bucket_name=bucket,
        r2_key=r2_key,
        verbose=verbose,
    )

    captured = capsys.readouterr()

    if verbose:
        assert f"Key {r2_key} does not exist" in captured.out
        assert f"Saved pipeline to key: {r2_key}" in captured.out
    else:
        assert captured.out == ""


@pytest.mark.parametrize("verbose", [True, False])
def test_joblib_dump_to_r2_verbose_existing_key(
    s3_setup,
    capsys,
    verbose,
):
    """Verifies existing-key message handling for joblib uploads."""
    client, bucket = s3_setup

    r2_key = "models/existing.joblib"

    client.put_object(
        Bucket=bucket,
        Key=r2_key,
        Body=b"existing-model",
    )

    r2io.joblib_dump_to_r2(
        s3_client=client,
        pipe={"model": "rf"},
        bucket_name=bucket,
        r2_key=r2_key,
        verbose=verbose,
    )

    captured = capsys.readouterr()

    # This print occurs regardless of verbose flag
    assert f"Key {r2_key} already exists" in captured.out


@pytest.mark.parametrize("verbose", [True, False])
def test_joblib_dump_to_r2_verbose_403(
    capsys,
    verbose,
):
    """Verifies verbose logging for access denied joblib uploads."""

    class MockS3Client:
        def head_object(self, **kwargs):
            raise botocore.exceptions.ClientError(
                error_response={"Error": {"Code": "403"}},
                operation_name="HeadObject",
            )

    r2io.joblib_dump_to_r2(
        s3_client=MockS3Client(),
        pipe={"model": "rf"},
        bucket_name="bucket",
        r2_key="model.joblib",
        verbose=verbose,
    )

    captured = capsys.readouterr()

    if verbose:
        assert "Access denied to bucket" in captured.out
    else:
        assert captured.out == ""


@pytest.mark.parametrize("verbose", [True, False])
def test_joblib_dump_to_r2_verbose_unexpected_error(
    capsys,
    verbose,
):
    """Verifies verbose logging for unexpected joblib upload errors."""

    class MockS3Client:
        def head_object(self, **kwargs):
            raise botocore.exceptions.ClientError(
                error_response={"Error": {"Code": "500"}},
                operation_name="HeadObject",
            )

    r2io.joblib_dump_to_r2(
        s3_client=MockS3Client(),
        pipe={"model": "rf"},
        bucket_name="bucket",
        r2_key="model.joblib",
        verbose=verbose,
    )

    captured = capsys.readouterr()

    if verbose:
        assert "An unexpected error occurred" in captured.out
    else:
        assert captured.out == ""


def test_joblib_read_latest_r2(mock_latest_and_joblib):
    """Verifies latest joblib object is discovered and loaded."""
    obj = r2io.joblib_read_latest_r2(
        s3_client="mock-client",
        bucket_name="bucket",
        base_prefix="models/",
        file_pattern="best_model__",
        file_extension=".joblib",
    )

    assert obj == {"model": "rf"}

    assert mock_latest_and_joblib["key_called"] is True
    assert mock_latest_and_joblib["load_called"] is True

    kwargs = mock_latest_and_joblib["kwargs"]

    assert kwargs["bucket_name"] == "bucket"
    assert kwargs["r2_key"] == "models/latest.joblib"


def test_joblib_read_latest_r2_returns_none(monkeypatch):
    """Verifies None is returned when no joblib key exists."""

    def mock_get_latest(*args, **kwargs):
        return None

    monkeypatch.setattr(
        r2io,
        "get_latest_s3_file_optimized",
        mock_get_latest,
    )

    obj = r2io.joblib_read_latest_r2(
        s3_client="mock-client",
        bucket_name="bucket",
        base_prefix="models/",
        file_pattern="best_model__",
    )

    assert obj is None
