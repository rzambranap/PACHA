"""
Statistical visualization module.

This module provides functions for creating statistical plots such as
scatter plots with linear regression fits.
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def plot_fit(x, y, ax=None, print_vals=False):
    """
    Plot a linear fit with R-squared value on scatter plot.

    Performs linear regression on the data and plots the fitted line
    along with a 1:1 reference line.

    Parameters
    ----------
    x : array-like
        X-axis data (must not contain NaN or Inf).
    y : array-like
        Y-axis data (must not contain NaN or Inf).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    print_vals : bool, optional
        If True, print regression statistics. Default is False.

    Returns
    -------
    float
        Slope of the fitted line, or 0 if fitting failed.

    Notes
    -----
    The function plots:
    - Red dashed line: linear regression fit
    - Blue line: 1:1 reference line
    - Legend with equation and R² value

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.scatter(observed, simulated, alpha=0.5)
    >>> slope = plot_fit(observed, simulated, ax=ax)
    >>> plt.show()
    """
    try:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

        xmin = np.min(x)
        xmax = np.max(x)
        xvec = np.linspace(xmin, xmax, 50)
        yvec = slope * xvec + intercept

        if print_vals:
            print('rvalue', r_value)
            print('slope', slope)
            print('intercept', intercept)
            print('pvalue', p_value)
            print('std_err', std_err)

        if ax is None:
            ax = plt.gca()

        label = (
            f'y = {slope:0.2f}x + {intercept:0.2f}\n'
            f'r² = {r_value**2:0.3f}'
        )
        ax.plot(xvec, yvec, c='r', linestyle='--', label=label)
        ax.plot(np.linspace(0, xmax, 10), np.linspace(0, xmax, 10), c='b')

        return slope

    except Exception as E:
        print('Error fitting data')
        print(E)
        return 0


def plot_histogram_line(df, column, bins=30, density=False, ax=None, **kwargs):
    """
    Plots a histogram as a step line (with straight tops) for a given DataFrame column.

    Parameters:
    - df: Pandas DataFrame
    - column: Column name as a string
    - bins: Number of bins (default=30)
    - density: Whether to normalize to a probability density (default=True)
    - color: Line color (default='blue')
    - ax: Matplotlib Axes object (optional)
    - **kwargs: Additional parameters for matplotlib's step function (e.g., linestyle, linewidth)
    """
    if column not in df:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    data = df[column].dropna()  # Drop NaNs to avoid errors

    # Compute histogram
    counts, bin_edges = np.histogram(data, bins=bins, density=density)

    # Create figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Plot step line
    ax.step(bin_edges[:-1], counts, where='post', **kwargs)
    ax.set_xlabel(column)
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(f"Histogram of {column} (Step Plot)")
    ax.grid(True)

    # Show only if no external figure is used
    if ax is None:
        plt.show()