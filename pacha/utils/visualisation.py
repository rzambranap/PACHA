"""
Visualization style utilities module.

This module provides classes and functions for creating consistent matplotlib
styles for lines, markers, and patches with legend support.
"""

from copy import copy
from abc import ABC, abstractmethod
from matplotlib.lines import Line2D
from matplotlib import patches as mpatches


class Style(ABC):
    """
    Abstract base class for matplotlib styles.

    Parameters
    ----------
    color : str, optional
        Color for the style element. Default is 'b' (blue).
    alpha : float, optional
        Transparency value (0-1). Default is 1.

    Attributes
    ----------
    color : str
        Color value.
    alpha : float
        Transparency value.
    """

    def __init__(self, color='b', alpha=1):
        """Initialize style with color and alpha."""
        self.color = color
        self.alpha = alpha

    def get_style(self):
        """
        Get style parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary of style parameters.
        """
        return self.__dict__

    def get_legend_element(self, label):
        """
        Generate a legend handle for this style.

        Parameters
        ----------
        label : str
            Label for the legend entry.

        Returns
        -------
        list
            List containing the legend handle.
        """
        return self._create_legend_handle(label)

    @abstractmethod
    def _create_legend_handle(self, label):
        """
        Create the legend handle (implemented by subclasses).

        Parameters
        ----------
        label : str
            Label for the legend entry.

        Returns
        -------
        list
            List containing the legend handle.
        """
        pass


class Line(Style):
    """
    Line style class for matplotlib line plots.

    Parameters
    ----------
    linestyle : str, optional
        Line style (e.g., '-', '--', '-.', ':'). Default is '-'.
    linewidth : float, optional
        Line width. Default is 1.
    **kwargs
        Additional arguments passed to Style base class.

    Examples
    --------
    >>> line_style = Line(color='red', linestyle='--', linewidth=2)
    >>> ax.plot(x, y, **line_style.get_style())
    """

    def __init__(self, linestyle='-', linewidth=1, **kwargs):
        """Initialize line style."""
        super().__init__(**kwargs)
        self.linestyle = linestyle
        self.linewidth = linewidth

    def _create_legend_handle(self, label):
        """Create legend handle for line style."""
        return [Line2D(
            [0], [0],
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            color=self.color,
            alpha=self.alpha,
            label=label
        )]


class Marker(Style):
    """
    Marker style class for matplotlib scatter/point plots.

    Parameters
    ----------
    marker : str, optional
        Marker symbol. Default is 'x'.
    markersize : float, optional
        Marker size. Default is 10.
    **kwargs
        Additional arguments passed to Style base class.

    Examples
    --------
    >>> marker_style = Marker(color='blue', marker='o', markersize=8)
    >>> ax.scatter(x, y, **marker_style.get_style())
    """

    def __init__(self, marker='x', markersize=10, **kwargs):
        """Initialize marker style."""
        super().__init__(**kwargs)
        self.marker = marker
        self.markersize = markersize

    def _create_legend_handle(self, label):
        """Create legend handle for marker style."""
        return [Line2D(
            [0], [0],
            linestyle='none',
            marker=self.marker,
            markersize=self.markersize,
            color=self.color,
            alpha=self.alpha,
            label=label
        )]


class Patch(Style):
    """
    Patch style class for matplotlib filled regions.

    Parameters
    ----------
    hatch : str, optional
        Hatch pattern (e.g., '/', '\\', 'x'). Default is None.
    edgecolor : str, optional
        Edge color. Default is None.
    **kwargs
        Additional arguments passed to Style base class.

    Examples
    --------
    >>> patch_style = Patch(color='green', alpha=0.5, hatch='//')
    """

    def __init__(self, hatch=None, edgecolor=None, **kwargs):
        """Initialize patch style."""
        super().__init__(**kwargs)
        self.hatch = hatch
        self.edgecolor = edgecolor

    def _create_legend_handle(self, label):
        """Create legend handle for patch style."""
        return [mpatches.Patch(
            color=self.color,
            alpha=self.alpha,
            hatch=self.hatch,
            edgecolor=self.edgecolor,
            label=label
        )]


def from_styles_create_legend_elements_v1(styles):
    """
    Create legend elements from a dictionary of styles.

    Parameters
    ----------
    styles : dict
        Dictionary mapping labels to style dictionaries.

    Returns
    -------
    list
        List of Line2D objects for use with ax.legend().

    Examples
    --------
    >>> styles = {'radar': {'color': 'blue'}, 'satellite': {'color': 'red'}}
    >>> legend_elements = from_styles_create_legend_elements_v1(styles)
    >>> ax.legend(handles=legend_elements)
    """
    legend_elements = []
    for key in styles.keys():
        st = copy(styles[key])
        if 'edgecolor' in st:
            st = from_patch_style_create_line_style(st)
        if 'markersize' in st:
            st['markersize'] = 5
        if 'linestyle' not in st:
            st['linestyle'] = 'none'
        line = Line2D([0], [0], label=key, **st)
        legend_elements.append(line)
    return legend_elements


def from_patch_style_create_line_style(patch_style):
    """
    Convert a patch style dictionary to a line style dictionary.

    Parameters
    ----------
    patch_style : dict
        Dictionary with patch style parameters.

    Returns
    -------
    dict
        Dictionary with line style parameters.

    Examples
    --------
    >>> patch = {'linewidth': 2, 'edgecolor': 'black', 'linestyle': '-'}
    >>> line = from_patch_style_create_line_style(patch)
    """
    linestyle = {}
    linestyle['linewidth'] = patch_style['linewidth']
    linestyle['color'] = patch_style['edgecolor']
    linestyle['linestyle'] = patch_style['linestyle']
    linestyle['alpha'] = patch_style['alpha']
    return linestyle
