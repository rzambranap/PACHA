"""
Instrument classes and utilities.

This module provides classes for representing different precipitation
measurement instruments including gauges, radars, and satellite products.
"""

import numpy as np
import pandas as pd
import xarray as xr
import haversine as hvs
from .file_utils import get_df_dates_filepaths


class Gauges:
    """
    Class representing a rain gauge network.

    Parameters
    ----------
    data_path : str
        Path to CSV file containing gauge data.
    meta_path : str
        Path to CSV file containing gauge metadata.

    Attributes
    ----------
    data : pandas.DataFrame
        Time-indexed DataFrame with gauge measurements.
    metadata : pandas.DataFrame
        DataFrame with station metadata (lat, lon, etc.).

    Examples
    --------
    >>> gauges = Gauges("/data/gauges.csv", "/data/gauge_metadata.csv")
    >>> gauges.calc_distance_to_point((-5.0, -39.0))
    """

    # mapping of column names to standard names
    colname_mappings = {
        'name': ['NOM_USUEL', 'name', 'station_name', 'STATION_NAME'],
        'lat': ['lat', 'latitude', 'LAT'],
        'lon': ['lon', 'longitude', 'LON'],
        'alt': ['alt', 'altitude', 'ALT'],
    }

    def __init__(self, data_path, meta_path):
        """Initialize gauge network from data and metadata files."""
        meta = pd.read_csv(meta_path, index_col=0)
        data = pd.read_csv(data_path, index_col=0, parse_dates=[0])
        if isinstance(meta.index[0], (int, np.int64)):
            data.columns = data.columns.astype(int)
        self.data = data
        self.metadata = meta
        # standardize metadata column names
        for std_name, possible_names in self.colname_mappings.items():
            for col in possible_names:
                if col in self.metadata.columns:
                    self.metadata = self.metadata.rename(columns={col: std_name})
                    break
        return

    def calc_distance_to_point(self, point, return_df=False):
        """
        Calculate distances from all gauges to a reference point.

        Parameters
        ----------
        point : tuple
            (lat, lon) coordinates of the reference point.
        return_df : bool, optional
            If True, return the updated metadata DataFrame. Default is False.

        Returns
        -------
        pandas.DataFrame or None
            Updated metadata if return_df is True, otherwise None.

        Raises
        ------
        ValueError
            If point is not a tuple.

        Examples
        --------
        >>> gauges.calc_distance_to_point((-5.0, -39.0))
        >>> print(gauges.metadata['distance_to_point'])
        """
        if not isinstance(point, tuple):
            raise ValueError(f'point must be a tuple (lat, lon), instead is {type(point)}')
        gau_coords = self.metadata.loc[:, ['lat', 'lon']].to_numpy()
        rad_coords = np.array(point)
        distances = hvs.haversine_vector(rad_coords, gau_coords, comb=True)
        self.metadata.loc[:, 'distance_to_point'] = distances
        if return_df:
            return self.metadata
        return

    def get_coords_station(self, station):
        """
        Get coordinates for a specific station.

        Parameters
        ----------
        station : str or int
            Station identifier.

        Returns
        -------
        tuple
            (longitude, latitude) of the station.

        Examples
        --------
        >>> lon, lat = gauges.get_coords_station('ST001')
        """
        st_df = self.metadata.loc[station, :]
        lat = st_df.lat
        lon = st_df.lon
        return lon, lat


class Radar:
    """
    Class representing a weather radar.

    Parameters
    ----------
    region : str
        Region identifier for the radar.
    name : str
        Name of the radar.

    Attributes
    ----------
    region : str
        Region identifier.
    name : str
        Radar name.
    l0_catalog, l1_catalog, l2_catalog : pandas.DataFrame
        Data catalogs for different processing levels (set via build_catalog).

    Examples
    --------
    >>> radar = Radar("ceara", "fortaleza")
    >>> radar.build_catalog(2, "/data/radar/l2")
    >>> sample = radar.load_sample(level=2)
    """

    def __init__(self, region, name):
        """Initialize radar with region and name."""
        self.region = region
        self.name = name
        return

    def build_catalog(self, level, dirpath, **kwargs):
        """
        Build a file catalog for a specific processing level.

        Parameters
        ----------
        level : int
            Processing level (0, 1, or 2).
        dirpath : str
            Directory path containing the data files.
        **kwargs
            Additional arguments passed to get_df_dates_filepaths.

        Examples
        --------
        >>> radar.build_catalog(2, "/data/radar/l2")
        """
        catalog = self.__catalog_builders[level](dirpath, **kwargs)
        setattr(self, f'l{level}_catalog', catalog)
        return

    def load_sample(self, level=None, date=None):
        """
        Load a sample dataset from the catalog.

        Parameters
        ----------
        level : int, optional
            Processing level. Default is 2.
        date : str or datetime, optional
            Date to load. If None, loads the first file.

        Returns
        -------
        xarray.Dataset
            Loaded dataset.

        Raises
        ------
        ValueError
            If no catalog has been built.

        Examples
        --------
        >>> ds = radar.load_sample(level=2, date="2023-01-01")
        """
        if level is None:
            retrieve_catalog = 'l2_catalog'
        else:
            retrieve_catalog = f'l{level}_catalog'
        if hasattr(self, retrieve_catalog):
            cat = getattr(self, retrieve_catalog)
            if date is None:
                return xr.open_dataset(cat.iloc[0, 0])
            else:
                fpath = cat.loc[date, 'paths']
                if not isinstance(fpath, str):
                    fpath = fpath.iloc[0]
                return xr.open_dataset(fpath)
        else:
            raise ValueError(f'no catalog yet built for {self.name}')

    __catalog_builders = {
        0: get_df_dates_filepaths,
        1: get_df_dates_filepaths,
        2: get_df_dates_filepaths
    }


class SPP:
    """
    Class representing a Satellite Precipitation Product.

    Parameters
    ----------
    product : str
        Product name (e.g., 'IMERG', 'GSMaP').
    name : str
        Identifier name for this product instance.

    Attributes
    ----------
    product : str
        Product name.
    name : str
        Instance name.
    l0_catalog through l3_catalog : pandas.DataFrame
        Data catalogs for different processing levels.

    Examples
    --------
    >>> imerg = SPP("IMERG", "imerg_v06")
    >>> imerg.build_catalog(0, "/data/imerg")
    >>> sample = imerg.load_sample(level=0)
    """

    def __init__(self, product, name):
        """Initialize satellite product with product type and name."""
        self.product = product
        self.name = name
        return

    def build_catalog(self, level, dirpath, **kwargs):
        """
        Build a file catalog for a specific processing level.

        Parameters
        ----------
        level : int
            Processing level (0, 1, 2, or 3).
        dirpath : str
            Directory path containing the data files.
        **kwargs
            Additional arguments passed to get_df_dates_filepaths.

        Examples
        --------
        >>> spp.build_catalog(0, "/data/satellite")
        """
        catalog = self.__catalog_builders[level](dirpath, **kwargs)
        setattr(self, f'l{level}_catalog', catalog)
        return

    def load_sample(self, level=None, date=None):
        """
        Load a sample dataset from the catalog.

        Parameters
        ----------
        level : int, optional
            Processing level. Default is 2.
        date : str or datetime, optional
            Date to load. If None, loads the first file.

        Returns
        -------
        xarray.Dataset
            Loaded dataset.

        Raises
        ------
        ValueError
            If no catalog has been built.

        Examples
        --------
        >>> ds = spp.load_sample(level=0, date="2023-01-01")
        """
        if level is None:
            retrieve_catalog = 'l2_catalog'
        else:
            retrieve_catalog = f'l{level}_catalog'
        if hasattr(self, retrieve_catalog):
            cat = getattr(self, retrieve_catalog)
            if date is None:
                return xr.open_dataset(cat.iloc[0, 0])
            else:
                fpath = cat.loc[date, 'paths']
                if not isinstance(fpath, str):
                    fpath = fpath.iloc[0]
                return xr.open_dataset(fpath)
        else:
            raise ValueError(f'no catalog yet built for {self.name}')

    __catalog_builders = {
        0: get_df_dates_filepaths,
        1: get_df_dates_filepaths,
        2: get_df_dates_filepaths,
        3: get_df_dates_filepaths
    }


def determine_analysis_scope_for_gauge_network(
    full_gauge_dataset, df_files_radar,
    daily_threshold=20,
    min_gauge_points=6, min_hours_radar=8,
    min_radar_files=32
):
    """
    Determine the analysis scope for gauge-radar comparison.

    Identifies stations and dates suitable for analysis based on
    data availability and precipitation thresholds.

    Parameters
    ----------
    full_gauge_dataset : pandas.DataFrame
        Complete gauge dataset with time index.
    df_files_radar : pandas.DataFrame
        DataFrame with radar file timestamps.
    daily_threshold : float, optional
        Minimum daily precipitation to consider (mm). Default is 20.
    min_gauge_points : int, optional
        Minimum number of gauge data points required. Default is 6.
    min_hours_radar : int, optional
        Minimum hours of radar data required. Default is 8.
    min_radar_files : int, optional
        Minimum number of radar files required. Default is 32.

    Returns
    -------
    pandas.DataFrame
        Summary of eligible stations with columns:
        'npoints', 'dates', 'time_step', 'ndays', 'gauge_data_for_station'.

    Notes
    -----
    This function helps identify co-located gauge-radar data for
    calibration and validation studies.
    """
    unique_dates_radar = list(set([wea.date() for wea in df_files_radar.index.tolist()]))
    dates_to_test = [str(wea) for wea in unique_dates_radar]
    gauge_data = pd.concat([full_gauge_dataset.loc[day] for day in dates_to_test])
    stations = gauge_data.columns.tolist()

    summary_of_eligible_stations = {}
    for station in stations:
        ds = gauge_data.loc[:, station]
        ds_acc_daily = ds.groupby(pd.Grouper(freq='D')).sum()
        ds_over_threshold = ds_acc_daily.loc[ds_acc_daily > daily_threshold]
        dates_for_station = [str(item.date()) for item in ds_over_threshold.index.tolist()]

        if len(dates_for_station) < 1:
            continue

        gauge_data_for_station = pd.concat(
            [gauge_data.loc[day, station] for day in dates_for_station]
        )
        gauge_data_for_station = gauge_data_for_station.dropna()
        npoints = gauge_data_for_station.shape[0]

        if npoints > min_gauge_points:
            time_step = gauge_data_for_station.index[1] - gauge_data_for_station.index[0]
            mins = int(((time_step.to_numpy().astype(float)) / 1e9) / 60)
            summary_of_eligible_stations[station] = {
                'npoints': npoints,
                'dates': dates_for_station,
                'time_step': mins,
                'ndays': len(dates_for_station),
                'gauge_data_for_station': gauge_data_for_station
            }

    return pd.DataFrame.from_dict(summary_of_eligible_stations).T
