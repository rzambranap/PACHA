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
            print('std_errr', std_err)

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
