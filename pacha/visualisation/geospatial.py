"""
Geospatial visualization module.

This module provides functions for creating geospatial maps and visualizations
of precipitation data using matplotlib and cartopy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import xarray as xr
import pyproj
from shapely.geometry import Point
from shapely.ops import transform
import geopandas as gpd
from matplotlib.patches import Rectangle


def save_fig(
    fig,
    fig_name,
    fig_dir,
    fig_fmt,
    fig_size=None,
    save=True,
    dpi=300,
    transparent_png=True,
):
    """
    Save a matplotlib figure to file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure instance to save.
    fig_name : str
        Filename (without extension).
    fig_dir : str
        Directory path for saving.
    fig_fmt : str
        Format ('pdf', 'png', etc.).
    fig_size : tuple, optional
        Size in inches (width, height). Default is None (use current size).
    save : bool, optional
        If False, skip saving. Default is True.
    dpi : int, optional
        Resolution for raster formats. Default is 300.
    transparent_png : bool, optional
        Make PNG background transparent. Default is True.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3])
    >>> save_fig(fig, 'my_plot', '/output', 'png')
    """
    if not save:
        return
    if fig_size is not None:
        fig.set_size_inches(fig_size, forward=False)
    else:
        fig.set_size_inches(fig.get_size_inches(), forward=False)

    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    pth = os.path.join(fig_dir, f'{fig_name}.{fig_fmt.lower()}')

    if fig_fmt == 'pdf':
        metadata = {
            'Creator': '',
            'Producer': '',
            'CreationDate': None
        }
        fig.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(pth, bbox_inches='tight', dpi=dpi)
    else:
        try:
            fig.savefig(pth, bbox_inches='tight')
        except Exception as e:
            print(f"Cannot save figure: {e}")


def create_circle(coord, radius_km):
    """
    Create a circular polygon at given coordinates.

    Parameters
    ----------
    coord : tuple
        (longitude, latitude) of the circle center.
    radius_km : float
        Radius of the circle in kilometers.

    Returns
    -------
    shapely.geometry.Polygon
        Circle polygon in WGS84 coordinates.

    Examples
    --------
    >>> circle = create_circle((-39.0, -5.0), 150)
    """
    central_point = Point(coord)

    wgs84 = pyproj.CRS("EPSG:4326")
    aeqd_proj = pyproj.CRS(proj="aeqd", lat_0=coord[1], lon_0=coord[0])

    project = pyproj.Transformer.from_crs(wgs84, aeqd_proj, always_xy=True).transform
    central_point_aeqd = transform(project, central_point)

    buffer_aeqd = central_point_aeqd.buffer(radius_km * 1000)

    project_back = pyproj.Transformer.from_crs(aeqd_proj, wgs84, always_xy=True).transform
    buffer_wgs84 = transform(project_back, buffer_aeqd)

    return buffer_wgs84


def create_radar_circles_and_centers(radar_csv_path, radius_km=150):
    """
    Create radar coverage circles and center points from a CSV file.

    Parameters
    ----------
    radar_csv_path : str
        Path to CSV file with radar coordinates (columns: 'lat', 'lon', 'name').
    radius_km : float, optional
        Radius of coverage circles in km. Default is 150.

    Returns
    -------
    circle_gdf : geopandas.GeoDataFrame
        GeoDataFrame with radar circle geometries.
    central_points : geopandas.GeoSeries
        GeoSeries with circle center points.

    Examples
    --------
    >>> circles, centers = create_radar_circles_and_centers("radars.csv")
    """
    rad_df = pd.read_csv(radar_csv_path)
    # Filter to region of interest
    rad_df = rad_df.loc[rad_df.loc[:, 'lat'] < 9]
    rad_df = rad_df.loc[rad_df.loc[:, 'lon'] > -80]
    rad_df = rad_df.loc[rad_df.loc[:, 'lon'] < -50]
    rad_df = rad_df.loc[rad_df.loc[:, 'lat'] > -6]

    coordinates = rad_df[['lon', 'lat']].values

    circles = [create_circle(coord, radius_km) for coord in coordinates]

    circle_gdf = gpd.GeoDataFrame(
        {'name': rad_df['name'], 'geometry': circles},
        crs="EPSG:4326"
    )
    central_points = gpd.GeoSeries(
        [Point(coord) for coord in coordinates],
        crs="EPSG:4326"
    )

    return circle_gdf, central_points


def plot_stamen(ax=None, fig=None, level=2):
    """
    Add a Stamen terrain basemap to a matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to add map to. If None, creates new axes.
    fig : matplotlib.figure.Figure, optional
        Figure to use. If None, uses current figure.
    level : int, optional
        Zoom level (higher = more detail). Default is 2.

    Examples
    --------
    >>> fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
    >>> plot_stamen(ax=ax, level=3)
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    stamen_terrain = cimgt.Stamen('terrain-background')
    ax.add_image(stamen_terrain, level)
    return


def plot_simple_map(ax=None, fig=None):
    """
    Add simple map features (coastlines, borders, rivers) to axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to add features to.
    fig : matplotlib.figure.Figure, optional
        Figure to use.

    Examples
    --------
    >>> fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
    >>> plot_simple_map(ax=ax)
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND)
    gl = ax.gridlines()
    gl.bottom_labels = True
    gl.left_labels = True
    return


def plot_cml_network(df_metadata,
                     lons1_col='lons1', lons2_col='lons2',
                     lats1_col='lats1', lats2_col='lats2',
                     ax=None, c='b'):
    """
    Plot a commercial microwave link network.

    Parameters
    ----------
    df_metadata : pandas.DataFrame
        Metadata with link endpoint coordinates.
    lons1_col : str, optional
        Column name for first endpoint longitudes. Default is 'lons1'.
    lons2_col : str, optional
        Column name for second endpoint longitudes. Default is 'lons2'.
    lats1_col : str, optional
        Column name for first endpoint latitudes. Default is 'lats1'.
    lats2_col : str, optional
        Column name for second endpoint latitudes. Default is 'lats2'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    c : str, optional
        Line color. Default is 'b'.

    Examples
    --------
    >>> plot_cml_network(links_df, ax=ax)
    """
    if ax is None:
        ax = plt.gca()
    idxs = df_metadata.index.tolist()
    for idx in idxs:
        lons = df_metadata.loc[idx, lons1_col], df_metadata.loc[idx, lons2_col]
        lats = df_metadata.loc[idx, lats1_col], df_metadata.loc[idx, lats2_col]
        ax.plot(lons, lats, c=c)
    return


def plot_gauge_network(df, ax=None, display_names=False, c='b'):
    """
    Plot rain gauge locations.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'lat', 'lon' columns and optionally 'name'/'station'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    display_names : bool, optional
        If True, display station names. Default is False.
    c : str, optional
        Marker color. Default is 'b'.

    Examples
    --------
    >>> plot_gauge_network(gauge_metadata, ax=ax, display_names=True)
    """
    if ax is None:
        ax = plt.gca()
    ax.scatter(df['lon'], df['lat'], c=c, marker='x', s=50)

    name_list = ['name', 'station']
    if display_names:
        name_index = [i for i in name_list if i in df.columns][0]
        for _, row in df.iterrows():
            name = row[name_index]
            x, y = row['lon'], row['lat']

            marker_size = 50
            text_size = marker_size / 10
            text_offset = marker_size * 0.05

            ax.annotate(
                name, (x, y),
                xytext=(text_offset, -text_offset),
                textcoords='offset points',
                fontsize=text_size,
                ha='left', va='bottom'
            )

    return


def plot_radar_coverage(sample, ax=None, fig=None):
    """
    Plot radar coverage area.

    Parameters
    ----------
    sample : xarray.Dataset or xarray.DataArray
        Radar data with lat/lon coordinates.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    fig : matplotlib.figure.Figure, optional
        Figure to use.

    Examples
    --------
    >>> plot_radar_coverage(radar_ds, ax=ax)
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    if isinstance(sample, xr.Dataset):
        fields = list(sample.data_vars)
        da = sample[fields[0]]
    elif isinstance(sample, xr.DataArray):
        da = sample
    else:
        raise TypeError("can only plot if xr dataset or data array")

    if 'time' in da.dims:
        single_tstep = da.isel({'time': 0})
    else:
        single_tstep = da

    single_tstep = single_tstep * 0
    coords = list(single_tstep.coords)

    if ('lat' in coords) and ('lon' in coords):
        single_tstep.plot(x='lon', y='lat', add_colorbar=False, alpha=0.5, ax=ax)

    return


def decide_type_plot(ds):
    """
    Determine appropriate plot type based on dataset dimensions.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset to analyze.

    Returns
    -------
    str
        'time_series_plot' for 1D data, 'spatial_plot' for 2D data.

    Raises
    ------
    ValueError
        If dimensions are not 1 or 2.
    """
    dims = ds.dims
    ndims = len(dims)
    if ndims == 1:
        return 'time_series_plot'
    elif ndims == 2:
        return 'spatial_plot'
    else:
        raise ValueError(f'{ndims} dims in dataset, not yet supported')


def plot_ds(ds, joint=False, plot_map=True, match_scale=True,
            pcp_var=None, discretize=False, max_global=None, **kwargs):
    """
    Plot multiple variables from a dataset side by side.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with multiple variables to plot.
    joint : bool, optional
        Not yet implemented. Default is False.
    plot_map : bool, optional
        If True, add map features. Default is True.
    match_scale : bool, optional
        If True, use same color scale for all plots. Default is True.
    pcp_var : str, optional
        Precipitation variable name (not yet used). Default is None.
    discretize : bool, optional
        If True, use discrete color levels. Default is False.
    max_global : float, optional
        Maximum value for color scale. Default is None.
    **kwargs
        Additional arguments passed to xarray.plot().

    Returns
    -------
    None

    Examples
    --------
    >>> plot_ds(precip_ds, match_scale=True)
    """
    vars = list(ds.data_vars)
    nvars = len(vars)

    global_min = np.min(np.array([ds[var].min().values for var in vars]))
    global_max = np.max(np.array([ds[var].max().values for var in vars]))
    if max_global is not None:
        global_max = max_global

    plot_type = decide_type_plot(ds)

    figsize = ((7 + 2) * nvars, 7)
    fig, axes = plt.subplots(
        1, nvars,
        subplot_kw=dict(projection=ccrs.PlateCarree()),
        figsize=figsize
    )

    for i in range(nvars):
        ax = axes[i]
        var = vars[i]
        if plot_map:
            plot_simple_map(ax=ax)
        vmin, vmax = ds[var].min().values, ds[var].max().values

        if match_scale:
            vmin, vmax = global_min, global_max
        if discretize:
            lvs = calculate_plotting_levels([ds[v] for v in vars], max_global=max_global)
            ds[var].plot(x='lon', y='lat', ax=ax, levels=lvs, **kwargs)
            ax.set_title(var)
        else:
            ds[var].plot(x='lon', y='lat', ax=ax, vmin=vmin, vmax=vmax, **kwargs)
            ax.set_title(var)
    return


def plot_bbox_rectangle(bbox, style=None):
    """
    Plot a bounding box rectangle on the current axes.

    Parameters
    ----------
    bbox : BBox
        Bounding box to plot.
    style : dict, optional
        Style parameters for the rectangle. Default is black outline.

    Examples
    --------
    >>> plot_bbox_rectangle(region_bbox, style={'edgecolor': 'red'})
    """
    if style is None:
        style = {
            'linewidth': 1,
            'edgecolor': 'black',
            'facecolor': 'none',
            'alpha': 0.5
        }
    rect = Rectangle(
        (bbox.min_lon, bbox.min_lat),
        bbox.max_lon - bbox.min_lon,
        bbox.max_lat - bbox.min_lat,
        transform=ccrs.PlateCarree(),
        **style
    )
    ax = plt.gca()
    ax.add_patch(rect)
    return


def calculate_plotting_levels(das_to_plot, nlevs=None, max_global=None):
    """
    Calculate appropriate color levels for precipitation plotting.

    Parameters
    ----------
    das_to_plot : list or xarray.Dataset
        DataArrays or Dataset to analyze.
    nlevs : int, optional
        Number of levels. Default is auto-calculated.
    max_global : float, optional
        Maximum value for levels. Default is data maximum.

    Returns
    -------
    numpy.ndarray
        Array of level values.

    Examples
    --------
    >>> levels = calculate_plotting_levels([radar_da, satellite_da])
    """
    if isinstance(das_to_plot, xr.Dataset):
        das_to_plot = [das_to_plot[i] for i in das_to_plot.data_vars]
    if not isinstance(das_to_plot, list):
        das_to_plot = [das_to_plot]

    maxs = [i.max().values for i in das_to_plot]
    mins = [i.min().values for i in das_to_plot]
    vmax = np.max(maxs)
    if max_global is not None:
        vmax = max_global
    vmin = np.min(mins)

    if vmax // 10 >= 3:
        vmax = int(vmax // 10 * 10 + 10)
        if nlevs is None:
            nlevs = vmax // 10 + 1
        lvs = np.linspace(0, vmax, nlevs)
    else:
        vmax = int(vmax // 1 * 1 + 1)
        if nlevs is None:
            nlevs = vmax // 1 + 1
        lvs = np.linspace(int(vmin // 1), vmax, nlevs)
    return lvs
