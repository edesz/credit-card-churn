#!/usr/bin/env python3


"""Define helper functions to inspect a pandas DataFrame."""

from typing import List, Union

import numpy as np
import pandas as pd
from IPython.display import display


def show_df(df: pd.DataFrame) -> List[Union[pd.DataFrame, float]]:
    """Displays DataFrame summary and memory usage.

    This function computes and displays a summary including missing
    values, unique counts, and data types for each column. It also
    prints memory usage.

    Args:
        df: Input DataFrame.

    Returns:
        List[Union[pd.DataFrame, float]]:
            - Summary DataFrame with column diagnostics.
            - Total memory usage in KB.

    Examples:
        >>> summary, mem = show_df(df)
    """
    df_summary = (
        df.isna()
        .sum()
        .rename("num_missing")
        .to_frame()
        .merge(
            pd.DataFrame.from_dict(
                {c: df[c].nunique() for c in list(df)},
                columns=["num_unique"],
                orient="index",
            ),
            left_index=True,
            right_index=True,
        )
        .merge(
            df.dtypes.rename("dtype").to_frame(),
            left_index=True,
            right_index=True,
        )
    )
    total_memory_KB = df.memory_usage(index=True, deep=True).sum() / 1000
    display(df_summary)
    print(
        f"Shape: {len(df):,} rows X {df.shape[1]:,} columns, Memory Usage: "
        f"{total_memory_KB:.3f} KB"
    )
    return [df_summary, total_memory_KB]


def highlight_abs_greater(
    df: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    """Highlights values exceeding a threshold in absolute terms.

    This function creates a style mask that highlights cells where the
    absolute value exceeds a specified threshold.

    Args:
        df: Input DataFrame of numeric values.
        threshold: Threshold for highlighting.

    Returns:
        pd.DataFrame: DataFrame of CSS styles for highlighting.

    Examples:
        >>> df.style.apply(highlight_abs_greater, threshold=0.5)
    """
    # create mask where absolute value is greater than a threshold
    mask = df.abs() > threshold
    # get CSS for true and empty string for false
    return pd.DataFrame(
        np.where(mask, "background-color: yellow", ""),
        index=df.index,
        columns=df.columns,
    )
