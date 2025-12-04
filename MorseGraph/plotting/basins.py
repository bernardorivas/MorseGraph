"""
Basins of attraction visualization functions.

This module provides functions for plotting basins of attraction and data coverage.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, Set, FrozenSet, Optional

from ..grids import AbstractGrid
from .utils import setup_figure_and_axes


def _get_basin_color(attractor, basins, morse_graph):
    """Get color for a basin from morse_graph or fallback colormap."""
    import networkx as nx
    
    if morse_graph and isinstance(morse_graph, nx.DiGraph) and attractor in morse_graph.nodes:
        if 'color' in morse_graph.nodes[attractor]:
            return morse_graph.nodes[attractor]['color']
    
    # Fallback to viridis colormap
    colors_cmap = plt.colormaps.get_cmap('viridis')
    return colors_cmap(list(basins.keys()).index(attractor) / max(len(basins) - 1, 1))


def _plot_box_rectangle(ax, box, color, alpha=0.3, **kwargs):
    """Plot a single box as a rectangle."""
    rect = Rectangle((box[0, 0], box[0, 1]),
                     box[1, 0] - box[0, 0],
                     box[1, 1] - box[0, 1],
                     facecolor=color,
                     edgecolor='none',
                     alpha=alpha, **kwargs)
    ax.add_patch(rect)


def plot_basins_of_attraction(grid: AbstractGrid, basins,
                             morse_graph = None, ax: plt.Axes = None,
                             show_outside: bool = False, **kwargs):
    """
    Plots the basins of attraction with colors matching the Morse sets.

    :param grid: The grid object used for the computation.
    :param basins: Dict[FrozenSet[int], Set[int]]: box-level basins (from compute_all_morse_set_basins)
    :param morse_graph: The Morse graph with color attributes. If provided, uses colors from node attributes.
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param show_outside: If True, paint boxes not in any basin black (boxes mapping outside domain).
    :param kwargs: Additional keyword arguments to pass to Rectangle patches.
    """
    _, ax, _ = setup_figure_and_axes(ax)

    all_boxes = grid.get_boxes()

    # Plot each basin
    for attractor, basin in basins.items():
        color = _get_basin_color(attractor, basins, morse_graph)

        # Box-level basins: basin is a set of box indices
        # Plot basin boxes with lower opacity
        for box_index in basin:
            if box_index < len(all_boxes):
                _plot_box_rectangle(ax, all_boxes[box_index], color, alpha=0.3, **kwargs)

        # Plot attractor itself with full opacity
        for box_index in attractor:
            if box_index < len(all_boxes):
                _plot_box_rectangle(ax, all_boxes[box_index], color, alpha=1.0, **kwargs)

    # Optionally show boxes that don't belong to any basin (map outside domain)
    if show_outside:
        all_box_indices = set(range(len(all_boxes)))
        basin_box_indices = set().union(*basins.values())
        outside_boxes = all_box_indices - basin_box_indices

        # Paint outside boxes black
        for box_index in outside_boxes:
            if box_index < len(all_boxes):
                _plot_box_rectangle(ax, all_boxes[box_index], 'black', alpha=0.5)

    ax.set_xlim(grid.bounds[0, 0], grid.bounds[1, 0])
    ax.set_ylim(grid.bounds[0, 1], grid.bounds[1, 1])
    ax.set_aspect('equal', adjustable='box')


def plot_data_coverage(grid: AbstractGrid, box_map_data, 
                       ax: plt.Axes = None, colormap: str = 'viridis'):
    """
    Visualize how many data points are in each box (2D only).

    User interprets the visualization themselves - no statistics.

    :param grid: The grid
    :param box_map_data: BoxMapData instance with box_to_points
    :param ax: Matplotlib axes
    :param colormap: Color scheme for data density
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if grid.dim != 2:
        raise ValueError("Data coverage plot only supports 2D grids")

    all_boxes = grid.get_boxes()

    # Count points per box
    counts = []
    for box_idx in range(len(all_boxes)):
        if box_idx in box_map_data.box_to_points:
            counts.append(len(box_map_data.box_to_points[box_idx]))
        else:
            counts.append(0)
    counts = np.array(counts)

    # Plot boxes colored by count
    cmap = plt.colormaps.get_cmap(colormap)
    max_count = np.max(counts) if np.max(counts) > 0 else 1

    rects = []
    colors = []

    for box_idx, box in enumerate(all_boxes):
        count = counts[box_idx]
        color = cmap(count / max_count)

        rect = Rectangle(
            (box[0, 0], box[0, 1]),
            box[1, 0] - box[0, 0],
            box[1, 1] - box[0, 1],
            facecolor=color,
            edgecolor='gray',
            alpha=0.7
        )
        rects.append(rect)
        colors.append(color)

    for rect in rects:
        ax.add_patch(rect)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=max_count))
    plt.colorbar(sm, ax=ax, label='Number of data points')

    ax.set_xlim(grid.bounds[0, 0], grid.bounds[1, 0])
    ax.set_ylim(grid.bounds[0, 1], grid.bounds[1, 1])
    ax.set_aspect('equal')
    ax.set_title('Data Coverage per Box')


def plot_data_points_overlay(grid: AbstractGrid, X: np.ndarray, Y: np.ndarray,
                            ax: plt.Axes = None, max_arrows: int = 500):
    """
    Overlay data points (X as dots, Y as arrows) on grid (2D only).

    Shows raw data distribution - user decides what it means.

    :param grid: The grid
    :param X: Input data points (N, 2)
    :param Y: Output data points (N, 2)
    :param ax: Matplotlib axes
    :param max_arrows: Maximum number of arrows to plot (subsampled if exceeded)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if X.shape[1] != 2:
        raise ValueError("Overlay plot only supports 2D data")

    # Plot grid boxes lightly
    all_boxes = grid.get_boxes()
    for box in all_boxes:
        rect = Rectangle(
            (box[0, 0], box[0, 1]),
            box[1, 0] - box[0, 0],
            box[1, 1] - box[0, 1],
            facecolor='none',
            edgecolor='lightgray'
        )
        ax.add_patch(rect)

    # Plot data: X points and X→Y arrows
    ax.scatter(X[:, 0], X[:, 1], c='blue', s=10, alpha=0.5, label='X (current)')

    # Subsample for arrows (too many arrows clutters)
    n_arrows = min(max_arrows, len(X))
    if n_arrows < len(X):
        indices = np.random.choice(len(X), n_arrows, replace=False)
    else:
        indices = np.arange(len(X))

    # Compute arrow scale
    domain_size = np.max(grid.bounds[1] - grid.bounds[0])
    arrow_head_width = domain_size * 0.02
    arrow_head_length = domain_size * 0.02

    for i in indices:
        dx = Y[i, 0] - X[i, 0]
        dy = Y[i, 1] - X[i, 1]
        ax.arrow(X[i, 0], X[i, 1], dx, dy,
                head_width=arrow_head_width, head_length=arrow_head_length,
                fc='red', ec='red', alpha=0.3)

    ax.set_xlim(grid.bounds[0, 0], grid.bounds[1, 0])
    ax.set_ylim(grid.bounds[0, 1], grid.bounds[1, 1])
    ax.set_aspect('equal')
    ax.set_title('Data Points and Transitions')
    ax.legend()

