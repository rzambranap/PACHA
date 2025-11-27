"""
Geospatial utilities module.

This module provides classes and functions for handling geospatial operations
including bounding boxes, coordinate extraction, and land/sea masking.
"""

import numpy as np
import xarray as xr
from global_land_mask import globe


class BBox:
    """
    Bounding box class for geospatial extent definition.

    Parameters
    ----------
    min_lat : float
        Minimum latitude (southern boundary).
    min_lon : float
        Minimum longitude (western boundary).
    max_lat : float
        Maximum latitude (northern boundary).
    max_lon : float
        Maximum longitude (eastern boundary).

    Attributes
    ----------
    min_lat : float
        Minimum latitude.
    min_lon : float
        Minimum longitude.
    max_lat : float
        Maximum latitude.
    max_lon : float
        Maximum longitude.

    Examples
    --------
    >>> bbox = BBox(min_lat=-6.0, min_lon=-40.0, max_lat=-4.0, max_lon=-38.0)
    >>> print(bbox.to_shape_compliant())
    """

    def __init__(self, min_lat, min_lon, max_lat, max_lon):
        """Initialize bounding box with coordinates."""
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon

    def to_shape_compliant(self):
        """
        Convert to shapefile-compliant format [min_lon, min_lat, max_lon, max_lat].

        Returns
        -------
        list
            Coordinates in [min_lon, min_lat, max_lon, max_lat] order.
        """
        shp = [self.min_lon, self.min_lat, self.max_lon, self.max_lat]
        return shp


def get_data_in_bbox(data, bbox):
    """
    Extract data within a bounding box from xarray Dataset or DataArray.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Dataset or DataArray from which to extract data.
    bbox : BBox
        Bounding box defining the region of interest.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Subset of data within the bounding box.

    Examples
    --------
    >>> bbox = BBox(-6, -40, -4, -38)
    >>> subset = get_data_in_bbox(ds, bbox)
    """
    data = data.sel({'lon': slice(bbox.min_lon, bbox.max_lon)})
    if data['lat'][0] < data['lat'][-1]:
        # Latitude Ordered Negative to Positive
        return data.sel({'lat': slice(bbox.min_lat, bbox.max_lat)})
    else:
        # Latitude Ordered Positive to Negative
        return data.sel({'lat': slice(bbox.max_lat, bbox.min_lat)})


def get_bbox_from_xarray(array, lon_name='lon', lat_name='lat'):
    """
    Create a BBox from xarray Dataset or DataArray coordinates.

    Parameters
    ----------
    array : xarray.Dataset or xarray.DataArray
        Dataset or DataArray from which to extract bounding box.
    lon_name : str, optional
        Name of longitude dimension. Default is 'lon'.
    lat_name : str, optional
        Name of latitude dimension. Default is 'lat'.

    Returns
    -------
    BBox
        Bounding box encompassing the data extent.

    Examples
    --------
    >>> bbox = get_bbox_from_xarray(ds)
    >>> print(f"Extent: {bbox.min_lon} to {bbox.max_lon}")
    """
    lons = array[lon_name]
    lats = array[lat_name]

    min_lat = lats.min().data
    max_lat = lats.max().data
    min_lon = lons.min().data
    max_lon = lons.max().data

    box = BBox(
        min_lat=min_lat, min_lon=min_lon,
        max_lat=max_lat, max_lon=max_lon
    )
    return box


def add_land_mask(data):
    """
    Add a land mask to an xarray dataset or data array.

    Uses the global_land_mask library to create a boolean mask where
    True indicates land and False indicates ocean.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Data to add land mask to. Must have 'lat' and 'lon' coordinates.

    Returns
    -------
    xarray.Dataset
        Dataset with 'mask' variable added.

    Examples
    --------
    >>> ds_with_mask = add_land_mask(ds)
    >>> land_only = ds_with_mask.where(ds_with_mask.mask)
    """
    if isinstance(data, xr.DataArray):
        data = data.to_dataset()
    lat = data.lat.to_numpy()
    lon = data.lon.to_numpy()
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    globe_land_mask = globe.is_land(lat_grid, lon_grid)
    data = data.assign({'mask': (('lat', 'lon'), globe_land_mask)})
    return data


def divide_dataset_land_sea(dataset):
    """
    Split a dataset into land and sea components.

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        Input dataset with 'lat' and 'lon' dimensions.

    Returns
    -------
    ds_land : xarray.Dataset or xarray.DataArray
        Data over land only (sea values are NaN).
    ds_sea : xarray.Dataset or xarray.DataArray
        Data over sea only (land values are NaN).

    Examples
    --------
    >>> land_data, sea_data = divide_dataset_land_sea(precip_ds)
    """
    ds_mask = add_land_mask(dataset)
    ds_land = dataset.where(ds_mask.mask)
    ds_sea = dataset.where(~ds_mask.mask)
    return ds_land, ds_sea


def subdivide(ds, nrows, ncols):
    """
    Subdivide a dataset into a grid of smaller regions.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset to subdivide.
    nrows : int
        Number of rows in the output grid.
    ncols : int
        Number of columns in the output grid.

    Returns
    -------
    list
        List of xarray Datasets representing the subdivisions.

    Examples
    --------
    >>> divisions = subdivide(ds, nrows=2, ncols=3)
    >>> print(len(divisions))  # 6
    """
    lons = ds.lon.to_numpy()
    lats = ds.lat.to_numpy()
    nlons = lons.shape[0]
    nlats = lats.shape[0]
    col_sze = nlons // ncols
    row_sze = nlats // nrows
    col_slices = [slice(i * col_sze, (i + 1) * col_sze) for i in range(ncols)]
    row_slices = [slice(i * row_sze, (i + 1) * row_sze) for i in range(nrows)]
    divisions = []
    for j in range(ncols):
        for i in range(nrows):
            slice_col = col_slices[j]
            slice_row = row_slices[i]
            div = ds[:, slice_row, slice_col]
            divisions.append(div)
    return divisions


def get_val_coord(lon, lat, scan):
    """
    Extract value from a gridded dataset at specific coordinates.

    Parameters
    ----------
    lon : float
        Longitude coordinate.
    lat : float
        Latitude coordinate.
    scan : xarray.Dataset or xarray.DataArray
        Gridded data to extract from.

    Returns
    -------
    xarray.DataArray
        Value(s) at the specified location.

    Examples
    --------
    >>> val = get_val_coord(-39.5, -5.0, radar_ds)
    """
    if len(scan.lat.shape) == 1:
        return scan.sel({'lat': lat, 'lon': lon}, method='nearest')
    if len(scan.lat.shape) > 2:
        abslat = np.abs(scan.lat.mean(dim='time') - lat)
        abslon = np.abs(scan.lon.mean(dim='time') - lon)
    else:
        abslat = np.abs(scan.lat - lat)
        abslon = np.abs(scan.lon - lon)
    c = np.maximum(abslon.values, abslat.values)
    xloc, yloc = np.where(c == np.min(c))
    point_value = scan[:, xloc[0], yloc[0]]
    return point_value


def ds_to_df_at_point(ds, point):
    """
    Extract time series at a point as a DataFrame.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Gridded dataset.
    point : tuple
        (longitude, latitude) coordinates.

    Returns
    -------
    pandas.DataFrame
        Time series at the specified point.

    Examples
    --------
    >>> df = ds_to_df_at_point(precip_ds, (-39.5, -5.0))
    """
    lon, lat = point
    if isinstance(ds, xr.DataArray):
        vars = [ds.name]
    if isinstance(ds, xr.Dataset):
        vars = list(ds.data_vars)
    ds_at_point = get_val_coord(lon, lat, ds)
    df_at_point = ds_at_point.to_dataframe()
    df_at_point = df_at_point.loc[:, vars]
    return df_at_point


def calc_ds_center(sample):
    """
    Calculate the geographic center of a dataset.

    Parameters
    ----------
    sample : xarray.Dataset or xarray.DataArray
        Dataset with 'lon' and 'lat' coordinates.

    Returns
    -------
    tuple
        (lon0, lat0) center coordinates.

    Examples
    --------
    >>> lon0, lat0 = calc_ds_center(radar_ds)
    """
    lon_max, lon_min = sample.lon.values.max(), sample.lon.values.min()
    lat_max, lat_min = sample.lat.values.max(), sample.lat.values.min()

    lon0 = lon_min + (lon_max - lon_min) / 2
    lat0 = lat_min + (lat_max - lat_min) / 2

    return lon0, lat0


def get_bbox_from_gdf(gdf):
    """
    Create a BBox from a GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with geometry.

    Returns
    -------
    BBox
        Bounding box of the GeoDataFrame.

    Examples
    --------
    >>> bbox = get_bbox_from_gdf(catchment_gdf)
    """
    minlon, minlat = gdf.boundary.get_coordinates().min()
    maxlon, maxlat = gdf.boundary.get_coordinates().max()
    return BBox(minlat, minlon, maxlat, maxlon)


def get_surrounding_bbox_from_bboxs(bboxs):
    """
    Create a bounding box that encompasses multiple bounding boxes.

    Parameters
    ----------
    bboxs : list
        List of BBox objects.

    Returns
    -------
    BBox
        Bounding box encompassing all input boxes.

    Examples
    --------
    >>> combined_bbox = get_surrounding_bbox_from_bboxs([bbox1, bbox2])
    """
    minlat = np.min([i.min_lat for i in bboxs])
    minlon = np.min([i.min_lon for i in bboxs])
    maxlat = np.max([i.max_lat for i in bboxs])
    maxlon = np.max([i.max_lon for i in bboxs])
    return BBox(minlat, minlon, maxlat, maxlon)


def get_bbox_from_df(df):
    """
    Create a BBox from a DataFrame with lat/lon columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'lat' and 'lon' columns.

    Returns
    -------
    BBox
        Bounding box of the point locations.

    Examples
    --------
    >>> bbox = get_bbox_from_df(station_metadata)
    """
    min_lat, max_lat = df.lat.min(), df.lat.max()
    min_lon, max_lon = df.lon.min(), df.lon.max()
    return BBox(min_lat, min_lon, max_lat, max_lon)


def get_extent_from_bbox(bbox, pad=0):
    """
    Convert a BBox to matplotlib/cartopy extent format.

    Parameters
    ----------
    bbox : BBox
        Bounding box.
    pad : float, optional
        Padding to add to all sides. Default is 0.

    Returns
    -------
    tuple
        (min_lon, max_lon, min_lat, max_lat) extent tuple.

    Examples
    --------
    >>> extent = get_extent_from_bbox(bbox, pad=0.5)
    >>> ax.set_extent(extent)
    """
    return (
        bbox.min_lon - pad,
        bbox.max_lon + pad,
        bbox.min_lat - pad,
        bbox.max_lat + pad
    )
