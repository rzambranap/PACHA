"""
Temporal alignment utilities module.

This module provides functions for aligning time series data from different
sources with potentially different timestamps.
"""

import numpy as np
import pandas as pd


def align_dataframes_by_time_fixed(df1, df2, time_diff_minutes=6):
    """
    Align two DataFrames based on their datetime indices within a time tolerance.

    Finds matching pairs of timestamps between two DataFrames where the time
    difference is within the specified tolerance, and aligns both DataFrames
    to the average (midpoint) timestamps.

    Parameters
    ----------
    df1 : pandas.DataFrame
        First DataFrame with datetime index.
    df2 : pandas.DataFrame
        Second DataFrame with datetime index.
    time_diff_minutes : float, optional
        Maximum time difference in minutes for matching. Default is 6.

    Returns
    -------
    sel_df1 : pandas.DataFrame
        Aligned subset of df1 with averaged timestamps.
    sel_df2 : pandas.DataFrame
        Aligned subset of df2 with averaged timestamps.

    Examples
    --------
    >>> dates1 = pd.date_range('2024-01-01 00:00:00', periods=5, freq='T')
    >>> dates2 = pd.date_range('2024-01-01 00:02:30', periods=5, freq='T')
    >>> df1 = pd.DataFrame({"A": range(5)}, index=dates1)
    >>> df2 = pd.DataFrame({"B": range(5, 10)}, index=dates2)
    >>> aligned_df1, aligned_df2 = align_dataframes_by_time_fixed(df1, df2)
    """
    # Find insertion points for df2's indices into df1's indices array
    idx = np.searchsorted(df1.index.to_numpy(), df2.index.to_numpy())
    # Ensure idx does not go out of bounds for df1
    idx = np.minimum(idx, len(df1.index) - 1)

    # Creating DataFrame with corresponding indices from both DataFrames
    _dict = {"t1": df1.index.to_numpy()[idx], "t2": df2.index.to_numpy()}
    df = pd.DataFrame(_dict)

    # Calculate time differences in minutes between paired indices
    diff = [(item.total_seconds() / 60) for item in (df.t1 - df.t2).tolist()]
    df.loc[:, 'diff'] = diff

    # Filter pairs of indices by the specified time difference
    mask = df.loc[:, 'diff'].abs() <= time_diff_minutes
    selection = df.loc[mask]

    # Select rows from the original DataFrames based on the filtered indices
    sel_df1 = df1.loc[selection.t1.tolist()]
    sel_df2 = df2.loc[selection.t2.tolist()]

    # Align indices of the selected rows to their average (midpoint)
    idx = sel_df1.index + (sel_df1.index - sel_df2.index) / 2
    sel_df1.index, sel_df2.index = idx, idx

    return sel_df1, sel_df2


def match_closest_time_indexes(df1, df2):
    """
    Match rows from df1 to their closest timestamps in df2.

    For each timestamp in df1, finds the closest timestamp in df2 and
    creates a combined DataFrame.

    Parameters
    ----------
    df1 : pandas.DataFrame
        First DataFrame with datetime index.
    df2 : pandas.DataFrame
        Second DataFrame with datetime index.

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame with matched rows, indexed by df1's timestamps.

    Notes
    -----
    This function uses pandas.DataFrame.append which is deprecated.
    Consider using align_dataframes_by_time_fixed for better performance.

    Examples
    --------
    >>> matched = match_closest_time_indexes(radar_df, satellite_df)
    """
    matched_df = pd.DataFrame(columns=df1.columns)

    for idx1, row1 in df1.iterrows():
        closest_idx2 = (df2.index - idx1).argmin()
        matched_row = pd.concat([row1, df2.iloc[closest_idx2]])
        matched_df = matched_df.append(matched_row, ignore_index=True)

    matched_df.index = df1.index
    return matched_df
