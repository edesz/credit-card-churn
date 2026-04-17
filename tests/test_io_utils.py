# #!/usr/bin/env python3


# """Test methods read from and write to private R2 bucket using boto3."""

# from io import BytesIO

# import botocore.exceptions
# import joblib
# import pandas as pd
# import pytest

# import src.r2.io_utils as r2io


# def test_pandas_read_parquet_r2(s3_setup, sample_r2_df):
#     """Verifies parquet file can be written and read from R2.

#     Args:
#         s3_setup: Mocked S3 client and bucket.
#         sample_r2_df: Sample DataFrame.
#     """
#     client, bucket = s3_setup

#     buffer = BytesIO()
#     sample_r2_df.to_parquet(buffer, index=False)
#     buffer.seek(0)

#     client.put_object(Bucket=bucket, Key="test.parquet", Body=buffer.read())
#     df = r2io.pandas_read_parquet_r2(client, bucket, "test.parquet")
#     pd.testing.assert_frame_equal(df, sample_r2_df)


# def test_pandas_read_filtered_parquets_r2(s3_setup, sample_r2_df):
#     """Verifies multiple parquet files are concatenated.

#     Args:
#         s3_setup: Mocked S3 client and bucket.
#         sample_r2_df: Sample DataFrame.
#     """
#     client, bucket = s3_setup

#     buffer = BytesIO()
#     sample_r2_df.to_parquet(buffer, index=False)
#     buffer.seek(0)

#     client.put_object(
#         Bucket=bucket,
#         Key=f"prefix/file_1.parquet",
#         Body=buffer.read(),
#     )
#     df = r2io.pandas_read_filtered_parquets_r2(
#         client, bucket, "prefix/", ["a", "b"]
#     )
#     assert len(df) == 3


# def test_pandas_read_xlsx_r2(s3_setup, sample_r2_df):
#     """Verifies Excel file can be read from R2.

#     Args:
#         s3_setup: Mocked S3 client and bucket.
#         sample_r2_df: Sample DataFrame.
#     """
#     client, bucket = s3_setup

#     buffer = BytesIO()
#     sample_r2_df.to_excel(buffer, index=False)
#     buffer.seek(0)

#     client.put_object(Bucket=bucket, Key="test.xlsx", Body=buffer.read())

#     df = r2io.pandas_read_xlsx_r2(
#         client,
#         bucket,
#         "test.xlsx",
#         dtypes={"a": "int64[pyarrow]", "b": "string[pyarrow]"},
#     )

#     assert list(df.columns) == ["a", "b"]
#     assert len(df) == 3


# def test_export_df_to_r2_new_key(s3_setup, sample_r2_df):
#     """Verifies DataFrame is uploaded when key does not exist.

#     Args:
#         s3_setup: Mocked S3 client and bucket.
#         sample_r2_df: Sample DataFrame.
#     """
#     client, bucket = s3_setup
#     r2io.export_df_to_r2(client, sample_r2_df, bucket, "data.parquet")
#     response = client.get_object(Bucket=bucket, Key="data.parquet")
#     assert response["ResponseMetadata"]["HTTPStatusCode"] == 200


# def test_export_df_to_r2_existing_key(s3_setup, sample_r2_df):
#     """Verifies existing key prevents overwrite.

#     Args:
#         s3_setup: Mocked S3 client and bucket.
#         sample_r2_df: Sample DataFrame.
#     """
#     client, bucket = s3_setup
#     client.put_object(Bucket=bucket, Key="data.parquet", Body=b"dummy")
#     r2io.export_df_to_r2(client, sample_r2_df, bucket, "data.parquet")
#     response = client.get_object(Bucket=bucket, Key="data.parquet")
#     assert response["Body"].read() == b"dummy"


# def test_joblib_dump_to_r2(s3_setup):
#     """Verifies object is serialized and uploaded.

#     Args:
#         s3_setup: Mocked S3 client and bucket.
#     """
#     client, bucket = s3_setup
#     obj = {"a": 1}
#     r2io.joblib_dump_to_r2(client, obj, bucket, "model__v1.joblib")
#     objects = client.list_objects_v2(Bucket=bucket)
#     assert "Contents" in objects


# def test_joblib_load_from_r2(s3_setup):
#     """Verifies joblib object is correctly loaded from R2.

#     Args:
#         s3_setup: Mocked S3 client and bucket.
#     """
#     client, bucket = s3_setup
#     obj = {"test": 123}
#     buffer = BytesIO()
#     joblib.dump(obj, buffer)
#     buffer.seek(0)

#     key = "model__v1.joblib"
#     client.put_object(Bucket=bucket, Key=key, Body=buffer.read())
#     loaded = r2io.joblib_load_from_r2(client, bucket, "model")
#     assert loaded == obj


# def test_pandas_read_parquet_r2_missing_key(s3_setup):
#     """Verifies missing object raises ClientError.

#     Args:
#         s3_setup: Mocked S3 client and bucket.
#     """
#     client, bucket = s3_setup
#     with pytest.raises(botocore.exceptions.ClientError):
#         r2io.pandas_read_parquet_r2(client, bucket, "missing.parquet")


# def test_filtered_parquet_bad_status(monkeypatch, s3_setup):
#     """Verifies assertion triggers on bad list_objects response.

#     Args:
#         monkeypatch: Pytest monkeypatch fixture.
#         s3_setup: Mocked S3 client and bucket.
#     """
#     client, bucket = s3_setup

#     def fake_list(*args, **kwargs):
#         return {"ResponseMetadata": {"HTTPStatusCode": 500}}

#     monkeypatch.setattr(client, "list_objects_v2", fake_list)
#     with pytest.raises(AssertionError):
#         r2io.pandas_read_filtered_parquets_r2(client, bucket, "prefix", ["a"])
