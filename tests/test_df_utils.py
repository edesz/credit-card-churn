#!/usr/bin/env python3


"""Test DataFrame utilities."""

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import src.utils.df_utils as du


def test_show_df_output_types(sample_df):
    summary, mem = du.show_df(sample_df)

    assert isinstance(summary, pd.DataFrame)
    assert isinstance(mem, float)


def test_show_df_columns(sample_df):
    summary, _ = du.show_df(sample_df)

    expected_cols = {"num_missing", "num_unique", "dtype"}
    assert expected_cols.issubset(summary.columns)


def test_show_df_missing_counts(sample_df):
    summary, _ = du.show_df(sample_df)

    assert summary.loc["A", "num_missing"] == 1
    assert summary.loc["B", "num_missing"] == 0


def test_show_df_unique_counts(sample_df):
    summary, _ = du.show_df(sample_df)

    # exclude NaNs in output values
    assert summary.loc["A", "num_unique"] == 2
    assert summary.loc["B", "num_unique"] == 3


def test_show_df_dtype(sample_df):
    summary, _ = du.show_df(sample_df)

    assert str(summary.loc["A", "dtype"]).startswith("float")
    assert str(summary.loc["B", "dtype"]) == "str"


def test_show_df_memory_positive(sample_df):
    _, mem = du.show_df(sample_df)

    assert mem > 0


def test_show_df_shape_consistency(sample_df):
    summary, _ = du.show_df(sample_df)

    assert len(summary) == sample_df.shape[1]


def test_highlight_abs_greater_basic(sample_df):
    df_numeric = sample_df[["C"]]

    styled = du.highlight_abs_greater(df_numeric, threshold=0.5)

    expected = [
        # 0.1
        "",
        # -0.6
        "background-color: yellow",
        # 0.3
        "",
        # 0.8
        "background-color: yellow",
    ]

    assert styled["C"].tolist() == expected


def test_highlight_abs_greater_all_below_threshold():
    df = pd.DataFrame({"A": [0.1, 0.2, -0.3]})

    styled = du.highlight_abs_greater(df, threshold=1.0)

    assert (styled == "").all().all()


def test_highlight_abs_greater_all_above_threshold():
    df = pd.DataFrame({"A": [1.1, -2.0, 3.5]})

    styled = du.highlight_abs_greater(df, threshold=0.5)

    assert (styled == "background-color: yellow").all().all()


def test_highlight_abs_greater_shape():
    df = pd.DataFrame(np.random.randn(5, 3), columns=list("ABC"))

    styled = du.highlight_abs_greater(df)

    assert styled.shape == df.shape


def test_highlight_abs_greater_preserves_index_columns(sample_df):
    df_numeric = sample_df[["C"]]

    styled = du.highlight_abs_greater(df_numeric)

    assert (styled.index == df_numeric.index).all()
    assert (styled.columns == df_numeric.columns).all()


def test_highlight_abs_greater_with_zero_threshold():
    df = pd.DataFrame({"A": [0, 1, -1]})

    styled = du.highlight_abs_greater(df, threshold=0)

    expected = ["", "background-color: yellow", "background-color: yellow"]

    assert styled["A"].tolist() == expected


def test_highlight_abs_greater_empty_df():
    """Test edge case of highlighting empty DataFrame."""
    df = pd.DataFrame()

    styled = du.highlight_abs_greater(df)

    assert styled.empty


def test_show_df_empty_df():
    """Test edge case of showing empty DataFrame."""
    df = pd.DataFrame()

    summary, mem = du.show_df(df)

    assert summary.empty
    assert mem >= 0


def test_show_df_verbose_true_displays_dataframe(sample_df, monkeypatch):
    """Verifies summary DataFrame is displayed when verbose=True.

    Args:
        None

    Returns:
        None
    """
    calls = {}

    def mock_display(obj):
        calls["called"] = True
        calls["obj"] = obj

    monkeypatch.setattr(du, "display", mock_display)

    df_summary, _ = du.show_df(sample_df, verbose=True)

    assert calls.get("called") is True

    displayed_df = calls["obj"]

    assert isinstance(displayed_df, type(df_summary))

    assert_frame_equal(displayed_df, df_summary)


def test_show_df_verbose_false_does_not_display(sample_df, monkeypatch):
    """Verifies display is not called when verbose=False.

    Args:
        None

    Returns:
        None
    """
    calls = {}

    def mock_display(obj):
        calls["called"] = True

    monkeypatch.setattr(du, "display", mock_display)

    du.show_df(sample_df, verbose=False)

    assert calls.get("called") is not True
