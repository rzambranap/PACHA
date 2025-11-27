"""
FUNCEME gauge data readers.

This module provides functions for reading and processing rain gauge data
from FUNCEME (Fundação Cearense de Meteorologia e Recursos Hídricos) CSV files.
"""

import os
import io
import numpy as np
import pandas as pd


def read_csv_with_comments(file_path, comment_char, kwargs_pd):
    """
    Read CSV file extracting only comment lines.

    Reads lines that start with the comment character and parses them
    as CSV data.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    comment_char : str
        Character that indicates a comment line (e.g., '#').
    kwargs_pd : dict
        Additional keyword arguments passed to pandas.read_csv.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the parsed comment lines.

    Examples
    --------
    >>> df = read_csv_with_comments("data.csv", '#', {'sep': ':'})
    """
    # Open the file and filter lines starting with the comment character
    with open(file_path, 'r') as file:
        filtered_lines = [line[1:] for line in file if line.startswith(comment_char)]

    # Create a StringIO object to simulate a file-like object from the filtered lines
    string_io_obj = io.StringIO(''.join(filtered_lines))

    # Read the filtered lines using read_csv()
    df = pd.read_csv(string_io_obj, **kwargs_pd)

    return df


def parse_numeric_columns(df):
    """
    Convert columns to numeric type where possible.

    Attempts to convert each column in the DataFrame to numeric type.
    Columns that cannot be converted are left unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with potentially mixed types.

    Returns
    -------
    pandas.DataFrame
        DataFrame with numeric columns converted.

    Examples
    --------
    >>> df = pd.DataFrame({'a': ['1', '2'], 'b': ['x', 'y']})
    >>> df = parse_numeric_columns(df)
    >>> print(df.dtypes)
    """
    for column in df.columns:
        if pd.to_numeric(df[column], errors='coerce').notnull().all():
            df[column] = pd.to_numeric(df[column])
    return df


def read_data_metadata_from_csv_funceme(path):
    """
    Read data and metadata from a FUNCEME gauge CSV file.

    Parses the FUNCEME CSV format which contains metadata in comment lines
    and time series data in the main body.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    df_data : pandas.DataFrame
        Data DataFrame with time-indexed precipitation data.
    df_metadata : pandas.DataFrame
        Metadata DataFrame with station information.

    Notes
    -----
    The function handles timezone conversion from 'America/Sao_Paulo' to UTC
    for sub-daily data.

    Examples
    --------
    >>> data, metadata = read_data_metadata_from_csv_funceme("gauge_data.csv")
    >>> print(metadata.columns)
    """
    # Read data from CSV file, skip rows starting with '#', and parse dates
    df_data = pd.read_csv(path, comment='#', parse_dates=[0])
    metadata_rename_dict = {
        'Código': 'id',
        'Nome': 'name',
        'Latitude': 'lat',
        'Longitude': 'lon',
        'Altitude': 'alt'
    }

    # Rename the 'data' column to 'time' and set it as the index
    df_data = df_data.rename({'data': 'time'}, axis=1)
    df_data = df_data.set_index('time')

    # Calculate the timestep in nanoseconds and minutes
    timestep_ns = np.median(np.diff(df_data.index.to_numpy()))
    timestep_mins = (timestep_ns.astype(float) / (1e9)) / 60

    # Read metadata from CSV file with comments
    df_metadata = read_csv_with_comments(path, '#', kwargs_pd={'sep': ':', 'header': None})

    # Set the index of the metadata DataFrame and rename columns
    df_metadata = df_metadata.set_index(0).rename(metadata_rename_dict).T

    # Parse numeric columns in the metadata DataFrame
    df_metadata = parse_numeric_columns(df_metadata)

    # Add calculated timestep information to the metadata DataFrame
    df_metadata.loc[:, 'timestep_mins'] = timestep_mins
    df_metadata.loc[:, 'timestep_ns'] = timestep_ns

    # Calculate the duration
    df_metadata.loc[:, 'duration'] = pd.to_timedelta(timestep_ns * df_data.shape[0])

    # Set the 'id' column as the index
    df_metadata = df_metadata.set_index('id')
    df_metadata.loc[:, 'id'] = df_metadata.index

    # Replace ' nan' values with NaN
    df_metadata = df_metadata.replace(' nan', np.nan)

    if df_metadata.duration.tolist()[0] < pd.to_timedelta('1 day'):
        # Add Brazil's timezone and convert to UTC
        df_data.index = df_data.index.tz_localize('America/Sao_Paulo').tz_convert('UTC')
        # Remove the timezone information from the data index
        df_data.index = df_data.index.tz_localize(None)

    df_data.columns = df_metadata.id.tolist()

    return df_data, df_metadata


def get_timedelta_string(delta):
    """
    Convert a timedelta to a human-readable string.

    Parameters
    ----------
    delta : pandas.Timedelta
        Time delta to convert.

    Returns
    -------
    str
        String representation (e.g., '1DAY', '6HOUR', '15MIN').

    Examples
    --------
    >>> get_timedelta_string(pd.Timedelta(hours=6))
    '6HOUR'
    """
    if delta >= pd.Timedelta(days=1):
        days = delta.components.days
        return f'{days}DAY'
    elif delta >= pd.Timedelta(hours=1):
        hours = delta.components.hours
        return f'{hours}HOUR'
    elif delta >= pd.Timedelta(minutes=1):
        minutes = delta.components.minutes
        return f'{minutes}MIN'
    else:
        return 'MIN'


def get_file_paths(directory):
    """
    Recursively get all CSV file paths from a directory.

    Parameters
    ----------
    directory : str
        Root directory to search.

    Returns
    -------
    list
        Sorted list of CSV file paths.

    Examples
    --------
    >>> paths = get_file_paths("/data/gauges")
    >>> print(len(paths))
    """
    fpaths = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if name.endswith('.csv'):
                fpaths.append(os.path.join(path, name))
    fpaths.sort()
    return fpaths


def get_unique_files(fpaths):
    """
    Filter file paths to keep only unique files based on name and size.

    Parameters
    ----------
    fpaths : list
        List of file paths.

    Returns
    -------
    pandas.DataFrame
        DataFrame with unique file paths, names, and sizes.

    Examples
    --------
    >>> unique = get_unique_files(["/data/file1.csv", "/data/file2.csv"])
    """
    sizes = [os.path.getsize(fpath) for fpath in fpaths]
    fnames = [os.path.split(fpath)[-1] for fpath in fpaths]

    df_paths = pd.DataFrame({
        'paths': fpaths,
        'names': fnames,
        'file_size_bytes': sizes
    })
    df_paths = df_paths.sort_values('names').drop_duplicates(['names', 'file_size_bytes'])
    return df_paths


def process_data_file(path):
    """
    Process a single FUNCEME data file.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    nid : str or None
        Station ID, or None if processing failed.
    df_data : pandas.DataFrame or None
        Processed data, or None if processing failed.
    df_metadata : pandas.DataFrame or None
        Processed metadata, or None if processing failed.

    Examples
    --------
    >>> nid, data, meta = process_data_file("station_001.csv")
    """
    df_data, df_metadata = read_data_metadata_from_csv_funceme(path)
    nid = df_metadata.id.values[0]
    if df_metadata.duration.isnull().any():
        return None, None, None
    else:
        return nid, df_data, df_metadata


def process_files(df_paths):
    """
    Process multiple FUNCEME data files.

    Parameters
    ----------
    df_paths : pandas.DataFrame
        DataFrame with 'paths' column containing file paths.

    Returns
    -------
    dfs_data : dict
        Dictionary mapping station IDs to data DataFrames.
    df_metadata_full : pandas.DataFrame
        Combined metadata for all stations.

    Examples
    --------
    >>> data_dict, metadata = process_files(unique_files_df)
    """
    dfs_data = {}
    dfs_metadata = []

    for path in df_paths['paths']:
        nid, df_data, df_metadata = process_data_file(path)
        if nid is not None:
            dfs_data[nid] = df_data
            dfs_metadata.append(df_metadata)

    df_metadata_full = pd.concat(dfs_metadata)
    return dfs_data, df_metadata_full


def organize_data_by_step_lengths(df_metadata_full, dfs_data):
    """
    Organize data by timestep lengths.

    Groups data by their temporal resolution (e.g., 6MIN, 1HOUR, 1DAY).

    Parameters
    ----------
    df_metadata_full : pandas.DataFrame
        Combined metadata for all stations.
    dfs_data : dict
        Dictionary mapping station IDs to data DataFrames.

    Returns
    -------
    dict
        Dictionary with timestep strings as keys and dictionaries
        containing 'timedelta', 'df_metadata', and 'df_data' as values.

    Examples
    --------
    >>> organized = organize_data_by_step_lengths(metadata, data_dict)
    >>> print(organized.keys())  # ['6MIN', '1HOUR']
    """
    step_lengths = list(set(df_metadata_full.timestep_ns.tolist()))
    organized_by_step_lengths = {}

    for t_step_length in step_lengths:
        key = get_timedelta_string(t_step_length)
        relevant_meta_df = df_metadata_full.loc[df_metadata_full.timestep_ns == t_step_length]
        relevant_ids = relevant_meta_df.id.tolist()
        relevant_data_dfs = [dfs_data[nid] for nid in relevant_ids]
        relevant_data_dfs = [df[~df.index.duplicated(keep='first')] for df in relevant_data_dfs]
        relevant_data_df = pd.concat(relevant_data_dfs, axis=1, join='outer')

        organized_by_step_lengths[key] = {
            'timedelta': t_step_length,
            'df_metadata': relevant_meta_df,
            'df_data': relevant_data_df
        }

    return organized_by_step_lengths


def process_funceme_csv_files(dir_path):
    """
    Process all FUNCEME CSV files in a directory.

    Main entry point for processing FUNCEME gauge data. Reads all CSV files,
    extracts metadata and data, and organizes by timestep length.

    Parameters
    ----------
    dir_path : str
        Path to directory containing FUNCEME CSV files.

    Returns
    -------
    dict
        Dictionary organized by timestep lengths, each containing
        'timedelta', 'df_metadata', and 'df_data'.

    Examples
    --------
    >>> result = process_funceme_csv_files("/data/funceme_gauges")
    >>> hourly_data = result['1HOUR']['df_data']
    """
    # Get the file paths
    fpaths = get_file_paths(dir_path)

    # Get unique files based on name and size
    df_paths = get_unique_files(fpaths)

    # Process the data files
    dfs_data, df_metadata_full = process_files(df_paths)

    # Organize data by step lengths
    organized_by_step_lengths = organize_data_by_step_lengths(df_metadata_full, dfs_data)

    return organized_by_step_lengths


def save_funceme_csvs_to_organized_csvs(organized_by_step_lengths, out_dir, prefix, suffix):
    """
    Save organized FUNCEME data to CSV files.

    Saves data and metadata to separate CSV files for each timestep length.

    Parameters
    ----------
    organized_by_step_lengths : dict
        Dictionary organized by timestep lengths (from process_funceme_csv_files).
    out_dir : str
        Output directory path.
    prefix : str
        Prefix for output filenames.
    suffix : str
        Suffix for output filenames (including extension).

    Examples
    --------
    >>> save_funceme_csvs_to_organized_csvs(organized, "/output", "funceme", ".csv")
    """
    step_lengths = organized_by_step_lengths.keys()
    for length in step_lengths:
        df_data = organized_by_step_lengths[length]['df_data']
        df_metadata = organized_by_step_lengths[length]['df_metadata']
        fname_metadata = f'{prefix}_{length}_metadata{suffix}'
        fpath_metadata = f'{out_dir}{fname_metadata}'
        fname_data = f'{prefix}_{length}{suffix}'
        fpath_data = f'{out_dir}{fname_data}'
        df_data.to_csv(fpath_data)
        df_metadata.to_csv(fpath_metadata)
    return
