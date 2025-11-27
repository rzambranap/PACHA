"""
File utilities module.

This module provides utility functions for handling file paths and parsing
datetime information from filenames.
"""

import os
from datetime import datetime as dt
import pandas as pd


def get_df_dates_filepaths(directory_path, date_char_st=None,
                           date_char_nd=None, date_format=None):
    """
    Create a DataFrame mapping file paths to their corresponding datetimes.

    Walks through a directory and extracts datetime information from filenames,
    creating an indexed DataFrame for easy time-based file selection.

    Parameters
    ----------
    directory_path : str
        Directory to map out. Best used with absolute paths.
    date_char_st : int, optional
        Starting character position of date in filename.
    date_char_nd : int, optional
        Ending character position of date in filename.
    date_format : str, optional
        Date format string following datetime conventions.

    Returns
    -------
    pandas.DataFrame
        DataFrame with 'paths' column, indexed by datetime.

    Notes
    -----
    If date_char_st, date_char_nd, and date_format are not provided,
    the function attempts to auto-detect the datetime format based on
    filename patterns.

    Examples
    --------
    >>> files = get_df_dates_filepaths("/data/radar")
    >>> daily_files = files.loc["2023-01-01"]
    """
    paths_x = []
    dates_x = []
    for root, dirs, files in os.walk(directory_path):
        for name in files:
            fpath = os.path.join(root, name)
            if fpath[-4:] == 'tore':
                break
            if (date_char_st is None) or (date_char_nd is None) or (date_format is None):
                parser = get_dt_parser(name)
                date = parser(name)
            else:
                date_str = name[date_char_st:date_char_nd]
                date = dt.strptime(date_str, date_format)
            paths_x.append(fpath)
            dates_x.append(date)

    files_x = pd.DataFrame.from_dict({
        'paths': paths_x,
        'time': dates_x
    })
    files_x = files_x.sort_values('time')
    files_x = files_x.set_index('time')
    return files_x


def get_dt_parser(fname):
    """
    Get appropriate datetime parser based on filename pattern.

    Parameters
    ----------
    fname : str
        Filename to analyze.

    Returns
    -------
    callable
        Parser function for the detected file type.

    Raises
    ------
    ValueError
        If filename doesn't match any known patterns.

    Examples
    --------
    >>> parser = get_dt_parser("IMERG_20230101_0000.nc")
    >>> dt = parser("IMERG_20230101_0000.nc")
    """
    for key in dtparsers.keys():
        str2match = strstomatch[key]
        if str2match in fname:
            return dtparsers[key]
    raise ValueError(f"fname {fname} didn't match any known files")


def get_imerg_datetime(fname):
    """
    Extract datetime from IMERG filename.

    Parameters
    ----------
    fname : str
        IMERG filename.

    Returns
    -------
    pandas.Timestamp
        Parsed datetime.

    Examples
    --------
    >>> dt = get_imerg_datetime("IMERG.20230101.0000.nc")
    """
    date_portion = fname.split('.')[4].split('-')[0:2]
    datestr = (date_portion[0] + date_portion[1]).replace('S', '')
    return pd.to_datetime(datestr)


def get_gsmap_datetime(fname):
    """
    Extract datetime from GSMaP filename.

    Parameters
    ----------
    fname : str
        GSMaP filename.

    Returns
    -------
    pandas.Timestamp
        Parsed datetime.

    Examples
    --------
    >>> dt = get_gsmap_datetime("gsmap.20230101.0000.dat.gz")
    """
    date_portion = fname.split('.')[1:3]
    datestr = (date_portion[0] + date_portion[1])
    return pd.to_datetime(datestr)


def get_romuald_datetime(fname):
    """
    Extract datetime from Romuald radar filename.

    Parameters
    ----------
    fname : str
        Romuald radar filename.

    Returns
    -------
    pandas.Timestamp
        Parsed datetime.

    Examples
    --------
    >>> dt = get_romuald_datetime("Romuald_20230101120000_scan.nc")
    """
    datestr = fname.split('_')[-2]
    return pd.to_datetime(datestr)


def get_funrad_datetime(fname):
    """
    Extract datetime from Funceme radar filename.

    Parameters
    ----------
    fname : str
        Funceme radar filename.

    Returns
    -------
    pandas.Timestamp
        Parsed datetime.

    Examples
    --------
    >>> dt = get_funrad_datetime("230101120000.RAW")
    """
    datestr = ''.join(filter(str.isdigit, fname.split('.')[0]))
    return pd.to_datetime(datestr, format='%y%m%d%H%M%S')


def get_recrad_datetime(fname):
    """
    Extract datetime from Recife radar filename.

    Parameters
    ----------
    fname : str
        Recife radar filename.

    Returns
    -------
    pandas.Timestamp
        Parsed datetime.

    Examples
    --------
    >>> dt = get_recrad_datetime("202301011200.vol")
    """
    datestr = ''.join(filter(str.isdigit, fname.split('.')[0]))[:-2]
    return pd.to_datetime(datestr)


def get_generalized_datetime(fname):
    """
    Extract datetime from generalized L1/L2 filename format.

    Parameters
    ----------
    fname : str
        L1 or L2 data filename.

    Returns
    -------
    pandas.Timestamp
        Parsed datetime.

    Examples
    --------
    >>> dt = get_generalized_datetime("L1_radar_S20230101T1200.nc")
    """
    datestr = fname.split('_')[-2]
    return pd.to_datetime(datestr, format='S%Y%m%dT%H%M')


def get_fpaths_from_dir(dirpath):
    """
    Get all file paths from a directory recursively.

    Parameters
    ----------
    dirpath : str
        Directory path to search.

    Returns
    -------
    list
        List of file paths.

    Examples
    --------
    >>> paths = get_fpaths_from_dir("/data/radar")
    """
    fpaths = []
    for root, dirs, files in os.walk(dirpath):
        for name in files:
            fpath = os.path.join(root, name)
            fpaths.append(fpath)
    return fpaths


# Mapping of file types to their parser functions
dtparsers = {
    'imerg': get_imerg_datetime,
    'gsmap': get_gsmap_datetime,
    'romuald': get_romuald_datetime,
    'funrad': get_funrad_datetime,
    'recrad': get_recrad_datetime,
    'level1': get_generalized_datetime,
    'level2': get_generalized_datetime
}

# Mapping of file types to their identifying substrings
strstomatch = {
    'imerg': 'IMERG',
    'gsmap': 'gsmap',
    'romuald': 'Romuald',
    'funrad': '.RAW',
    'recrad': '.vol',
    'level1': 'L1_',
    'level2': 'L2_'
}
