"""
Latent space visualization functions.

This module provides functions for visualizing and working with latent spaces,
including plotting latent space data, classifying points to Morse sets, and
visualizing data boxes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from typing import Dict, Optional, Any

from .utils import setup_figure_and_axes, finalize_plot, normalize_colormap


def plot_latent_space_2d(z_data, latent_bounds, morse_graph=None, output_path=None, title="Latent Space (2D)",
                       equilibria_latent: Dict[str, np.ndarray] = None, barycenters_latent: Dict[int, list] = None, ax=None):
    """
    Plot 2D latent space with data points and optionally Morse sets.

    Args:
        z_data: Encoded data points in latent space (N x 2 array)
        latent_bounds: [[lower_x, lower_y], [upper_x, upper_y]]
        morse_graph: Optional CMGDB MorseGraph to overlay Morse sets
        output_path: Path to save figure (if None, displays instead)
        title: Plot title
        equilibria_latent: Optional dict of equilibrium points in latent space to plot, e.g. {'EQ1': [z1,z2]}
        barycenters_latent: Optional dict mapping Morse set index to list of encoded barycenters (from compute_encoded_barycenters)
        ax: Matplotlib axes to plot on (if None, creates new figure)

    Example:
        >>> plot_latent_space_2d(z_train, latent_bounds, morse_graph, 'latent_space.png',
        ...                      equilibria_latent={'EQ': [0.1, 0.2]},
        ...                      barycenters_latent={0: [[0.1, 0.2]], 1: [[0.3, 0.4]]})
    """
    # Create figure if ax not provided
    fig, ax, should_close = setup_figure_and_axes(ax, figsize=(10, 10))

    # Plot data points (light background)
    ax.scatter(z_data[:, 0], z_data[:, 1], c='lightgray', s=1, alpha=0.2, label='Data', rasterized=True, zorder=0)

    # Plot Morse sets if provided (2D Latent Morse Sets)
    # Use Viridis colormap for 2D sets
    if morse_graph is not None:
        try:
            num_morse_sets = morse_graph.num_vertices()
            cmap_2d = cm.viridis
            colors_2d = normalize_colormap(cmap_2d, num_morse_sets)

            for i in range(num_morse_sets):
                boxes = morse_graph.morse_set_boxes(i)
                if boxes:
                    label = f'2D Morse Set {i}'
                    for j, box in enumerate(boxes):
                        rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                       facecolor=colors_2d[i], edgecolor='black', linewidth=0.5, alpha=0.6,
                                       label=label if j == 0 else None, zorder=5)
                        ax.add_patch(rect)
        except AttributeError:
            # morse_graph might be NetworkX, not CMGDB
            pass

    # Plot encoded 3D barycenters if provided (3D Ground Truth projections)
    # Use Cool colormap for 3D barycenters to distinguish from 2D sets
    if barycenters_latent:
        num_bary_sets = len(barycenters_latent)
        cmap_3d = cm.cool
        # Assuming indices are 0-based and correspond to 3D morse sets count roughly
        # We normalize based on number of keys for now, or could use max index
        max_idx = max(barycenters_latent.keys()) if barycenters_latent else 1
        
        for morse_set_idx, barys in barycenters_latent.items():
            if barys:
                barys_array = np.array(barys)
                color = cmap_3d(morse_set_idx / max(max_idx, 1))
                ax.scatter(barys_array[:, 0], barys_array[:, 1],
                          marker='X', s=80, c=[color], edgecolors='black', linewidths=0.5,
                          label=f'E(3D Barycenter {morse_set_idx})' if morse_set_idx < 5 else None,
                          zorder=10)

    if equilibria_latent:
        for name, point in equilibria_latent.items():
            ax.plot(point[0], point[1],
                   marker='*', markersize=18, color='red', markeredgecolor='black',
                   label=name, zorder=20, linestyle='none')

    ax.set_xlim(latent_bounds[0][0], latent_bounds[1][0])
    ax.set_ylim(latent_bounds[0][1], latent_bounds[1][1])
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_aspect('equal')
    ax.set_title(title)
    
    # Improve legend
    handles, labels_legend = ax.get_legend_handles_labels()
    # Filter duplicate labels if any
    by_label = dict(zip(labels_legend, handles))
    # Sort logic could be added, but default order is usually fine
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small', framealpha=0.9)

    finalize_plot(fig, ax, output_path, should_close)


def classify_points_to_morse_sets(points, morse_graph, bounds, subdiv_max):
    """
    Classify which Morse set each point belongs to.

    For each point, finds the box at max subdivision that contains it,
    then checks which Morse set that box belongs to.

    Args:
        points: Array of points (N x D)
        morse_graph: CMGDB MorseGraph object
        bounds: [[lower_1, ..., lower_D], [upper_1, ..., upper_D]]
        subdiv_max: Maximum subdivision depth used in computation

    Returns:
        Array of Morse set indices for each point (-1 if not in any Morse set)
    """
    points = np.array(points)
    lower = np.array(bounds[0])
    upper = np.array(bounds[1])
    dim = len(lower)
    n_boxes = 2 ** subdiv_max
    box_width = (upper - lower) / n_boxes

    # Build mapping from boxes to Morse sets
    box_to_morse_set = {}
    for i in range(morse_graph.num_vertices()):
        boxes = morse_graph.morse_set_boxes(i)
        for box in boxes:
            box_tuple = tuple(box)
            box_to_morse_set[box_tuple] = i

    # Classify each point
    classifications = np.full(len(points), -1, dtype=int)
    for idx, point in enumerate(points):
        box_idx = np.floor((point - lower) / box_width).astype(int)
        box_idx = np.clip(box_idx, 0, n_boxes - 1)
        box_lower = lower + box_idx * box_width
        box_upper = lower + (box_idx + 1) * box_width
        box_rect = tuple(list(box_lower) + list(box_upper))
        if box_rect in box_to_morse_set:
            classifications[idx] = box_to_morse_set[box_rect]

    return classifications


def plot_data_boxes(ax, z_data, bounds, subdiv_max, box_color='white', box_alpha=0.3):
    """
    Plot boxes containing data points as white rectangles.

    Args:
        ax: Matplotlib axes
        z_data: Array of data points (N x 2)
        bounds: [[lower_x, lower_y], [upper_x, upper_y]]
        subdiv_max: Maximum subdivision depth
        box_color: Color for boxes
        box_alpha: Transparency for boxes
    """
    lower = np.array(bounds[0])
    upper = np.array(bounds[1])
    n_boxes = 2 ** subdiv_max
    box_width = (upper - lower) / n_boxes

    # Find unique boxes containing data
    data_boxes = set()
    for point in z_data:
        box_idx = np.floor((point - lower) / box_width).astype(int)
        box_idx = np.clip(box_idx, 0, n_boxes - 1)
        data_boxes.add(tuple(box_idx))

    # Plot boxes
    for box_idx in data_boxes:
        box_lower = lower + np.array(box_idx) * box_width
        rect = Rectangle(
            box_lower,
            box_width[0],
            box_width[1],
            facecolor=box_color,
            edgecolor='none',
            alpha=box_alpha
        )
        ax.add_patch(rect)

