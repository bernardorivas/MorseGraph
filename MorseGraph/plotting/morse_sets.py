"""
Morse sets visualization functions.

This module provides functions for plotting Morse sets in 2D and 3D, including
scatter plots, cuboid visualizations, and projections.
"""

import os
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, Optional, Any

from ..grids import AbstractGrid
from .utils import (
    get_morse_set_color,
    _box_to_cuboid_faces,
    configure_3d_axes,
    finalize_plot,
    setup_figure_and_axes,
    extract_trajectory_data,
    sample_trajectories,
)


def plot_morse_sets(grid: AbstractGrid, morse_graph: nx.DiGraph, ax: plt.Axes = None,
                   box_map: nx.DiGraph = None, show_outside: bool = False, **kwargs):
    """
    Plots the Morse sets on a grid.

    :param grid: The grid object used for the computation.
    :param morse_graph: The Morse graph (NetworkX DiGraph) containing the Morse sets as nodes.
                       Each node should have a 'color' attribute (assigned by compute_morse_graph).
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param box_map: The BoxMap (directed graph) from compute_box_map(). Required if show_outside=True.
    :param show_outside: If True, paint boxes that map outside the domain in grey. Requires box_map.
    :param kwargs: Additional keyword arguments to pass to the PatchCollection.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Extract morse sets from the NetworkX graph
    morse_sets = morse_graph.nodes()

    # Get all boxes from the grid
    all_boxes = grid.get_boxes()

    rects = []
    colors = []

    for morse_set in morse_sets:
        color = get_morse_set_color(
            morse_set, morse_graph, 
            list(morse_sets).index(morse_set), 
            len(morse_sets)
        )

        for box_index in morse_set:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, 0], box[0, 1]),
                               box[1, 0] - box[0, 0],
                               box[1, 1] - box[0, 1])
                rects.append(rect)
                colors.append(color)

    if rects:
        pc = PatchCollection(rects, facecolors=colors, alpha=0.7, **kwargs)
        ax.add_collection(pc)

    # Optionally show boxes that map outside the domain (have data but no transitions)
    if show_outside and box_map is not None:
        # Collect all boxes in Morse sets
        morse_set_boxes = set()
        for morse_set in morse_sets:
            morse_set_boxes.update(morse_set)

        # Find boxes with out_degree == 0 that are not in any Morse set
        # These boxes have data but map outside the domain
        outside_boxes = set()
        for node in box_map.nodes():
            if box_map.out_degree(node) == 0 and node not in morse_set_boxes:
                outside_boxes.add(node)

        # Paint outside boxes grey
        for box_index in outside_boxes:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, 0], box[0, 1]),
                               box[1, 0] - box[0, 0],
                               box[1, 1] - box[0, 1],
                               facecolor='grey',
                               edgecolor='none',
                               alpha=0.4)
                ax.add_patch(rect)

    ax.set_xlim(grid.bounds[0, 0], grid.bounds[1, 0])
    ax.set_ylim(grid.bounds[0, 1], grid.bounds[1, 1])
    ax.set_aspect('equal', adjustable='box')


def plot_morse_sets_3d(grid: AbstractGrid, morse_graph: nx.DiGraph, ax: plt.Axes = None,
                       box_map: nx.DiGraph = None, show_outside: bool = False, alpha: float = 0.3, **kwargs):
    """
    Plots the Morse sets on a 3D grid.

    :param grid: The grid object used for the computation (must be 3D).
    :param morse_graph: The Morse graph (NetworkX DiGraph) containing the Morse sets as nodes.
                       Each node should have a 'color' attribute (assigned by compute_morse_graph).
    :param ax: The matplotlib 3D axes to plot on. If None, a new figure and 3D axes are created.
    :param box_map: The BoxMap (directed graph) from compute_box_map(). Required if show_outside=True.
    :param show_outside: If True, paint boxes that map outside the domain in grey. Requires box_map.
    :param alpha: Transparency level for the boxes (0.0-1.0). Lower values show more structure.
    :param kwargs: Additional keyword arguments to pass to Poly3DCollection.
    """
    if grid.dim != 3:
        raise ValueError(f"plot_morse_sets_3d requires a 3D grid, got {grid.dim}D")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Extract morse sets from the NetworkX graph
    morse_sets = morse_graph.nodes()

    # Get all boxes from the grid
    all_boxes = grid.get_boxes()

    def box_to_cuboid_faces(box):
        """
        Convert a box (shape (2, 3)) to the 8 vertices and 6 faces of a cuboid.
        Returns a list of 6 faces, each face is a list of 4 vertices (x, y, z).
        """
        x_min, y_min, z_min = box[0]
        x_max, y_max, z_max = box[1]

        # Define 8 vertices
        vertices = [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ]

        # Define 6 faces (each face is a list of 4 vertex indices)
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        ]

        return faces

    # Plot Morse sets
    for morse_set in morse_sets:
        color = get_morse_set_color(
            morse_set, morse_graph, 
            list(morse_sets).index(morse_set), 
            len(morse_sets)
        )

        # Collect all faces for this Morse set
        faces = []
        for box_index in morse_set:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                faces.extend(box_to_cuboid_faces(box))

        if faces:
            # Create 3D polygon collection for this Morse set
            poly3d = Poly3DCollection(faces, facecolors=color, alpha=alpha,
                                     edgecolors='none', **kwargs)
            ax.add_collection3d(poly3d)

    # Optionally show boxes that map outside the domain
    if show_outside and box_map is not None:
        # Collect all boxes in Morse sets
        morse_set_boxes = set()
        for morse_set in morse_sets:
            morse_set_boxes.update(morse_set)

        # Find boxes with out_degree == 0 that are not in any Morse set
        outside_boxes = set()
        for node in box_map.nodes():
            if box_map.out_degree(node) == 0 and node not in morse_set_boxes:
                outside_boxes.add(node)

        # Paint outside boxes grey
        outside_faces = []
        for box_index in outside_boxes:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                outside_faces.extend(box_to_cuboid_faces(box))

        if outside_faces:
            poly3d_outside = Poly3DCollection(outside_faces, facecolors='grey',
                                             alpha=alpha * 0.5, edgecolors='none')
            ax.add_collection3d(poly3d_outside)

    # Set axis limits
    ax.set_xlim(grid.bounds[0, 0], grid.bounds[1, 0])
    ax.set_ylim(grid.bounds[0, 1], grid.bounds[1, 1])
    ax.set_zlim(grid.bounds[0, 2], grid.bounds[1, 2])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def _setup_3d_morse_plot(fig_w: float, fig_h: float, num_morse_sets: int):
    """Create 3D figure and axes for Morse set plotting."""
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_morse_sets-1)
    return fig, ax, cmap_norm


def _render_morse_set_cuboids(ax, morse_graph, morse_node, cmap_norm, alpha: float):
    """Render cuboids for a single Morse set."""
    morse_set_boxes = morse_graph.morse_set_boxes(morse_node)
    clr = matplotlib.colors.to_hex(cm.cool(cmap_norm(morse_node)), keep_alpha=True)
    
    faces = []
    for box in morse_set_boxes:
        faces.extend(_box_to_cuboid_faces(box))
    
    if faces:
        poly3d = Poly3DCollection(faces, facecolors=clr, alpha=alpha,
                                 edgecolors='none', linewidths=0)
        ax.add_collection3d(poly3d)


def _finalize_3d_morse_plot(ax, domain_bounds, labels: Dict[str, str], 
                            title: str, elev: float, azim: float, 
                            fontsize: int, output_path: str, dpi: int, fig):
    """Set axis properties, labels, and save/display the plot."""
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=fontsize)
    
    # Use common utility for axis configuration
    configure_3d_axes(ax, domain_bounds, labels, title, fontsize)
    
    # Use common utility for finalization (but override dpi if provided)
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def _check_timeout(start_time, timeout_seconds, context=""):
    """Check if timeout has been exceeded and return status."""
    import time
    elapsed = time.time() - start_time
    if elapsed > timeout_seconds:
        if context:
            print(f"  WARNING: 3D cuboid rendering exceeded timeout ({timeout_seconds}s) {context}.")
        else:
            print(f"  WARNING: 3D cuboid rendering exceeded timeout ({timeout_seconds}s).")
        print(f"  Skipping 3D cuboid plot. Use scatter plot instead for dense Morse sets.")
        return True
    return False


def _render_all_morse_set_cuboids(ax, morse_graph, num_morse_sets, cmap_norm, alpha, start_time, timeout_seconds):
    """Render all Morse set cuboids with timeout checking."""
    for morse_node in range(num_morse_sets):
        if _check_timeout(start_time, timeout_seconds, 
                         f"after processing {morse_node}/{num_morse_sets} Morse sets"):
            return False
        _render_morse_set_cuboids(ax, morse_graph, morse_node, cmap_norm, alpha)
    return True


def plot_morse_sets_3d(morse_graph, domain_bounds, output_path=None, title="Morse Sets (3D)",
                             alpha: float = 0.3, labels: Dict[str, str] = None, elev=30, azim=45, 
                             fig_w=10, fig_h=10, fontsize=15, dpi=150, timeout_seconds=30, **kwargs):
    """
    Plots the Morse sets from a CMGDB MorseGraph object on a 3D grid.
    Follows CMGDB.PlotMorseSets3D style with Poly3DCollection cuboids.
    
    If rendering takes longer than timeout_seconds, the plot is skipped.

    :param morse_graph: The CMGDB.MorseGraph object.
    :param domain_bounds: [[lower_x, lower_y, lower_z], [upper_x, upper_y, upper_z]]
    :param output_path: Path to save figure (if None, displays instead)
    :param title: Plot title
    :param alpha: Transparency level for the boxes (0.0-1.0).
    :param labels: Optional dict for axis labels, e.g. {'x': 'log(M)', 'y': 'log(A)', 'z': 'log(D)'}
    :param elev: Elevation angle for 3D view (degrees)
    :param azim: Azimuth angle for 3D view (degrees)
    :param fig_w: Figure width in inches
    :param fig_h: Figure height in inches
    :param fontsize: Font size for axis labels
    :param dpi: DPI for saved figure
    :param timeout_seconds: Maximum time to spend rendering (default 30s). If exceeded, plot is skipped.
    :param kwargs: Additional keyword arguments (deprecated, kept for backwards compatibility)
    :return: True if plot succeeded, False if skipped due to timeout
    """
    import time
    
    start_time = time.time()
    num_morse_sets = morse_graph.num_vertices()
    total_boxes = sum(len(morse_graph.morse_set_boxes(i)) for i in range(num_morse_sets))
    
    print(f"  Plotting {total_boxes} boxes across {num_morse_sets} Morse sets...")
    
    fig, ax, cmap_norm = _setup_3d_morse_plot(fig_w, fig_h, num_morse_sets)

    # Render all Morse sets with timeout checking
    if not _render_all_morse_set_cuboids(ax, morse_graph, num_morse_sets, cmap_norm, alpha, 
                                         start_time, timeout_seconds):
        plt.close(fig)
        return False

    # Check timeout before finalization
    if _check_timeout(start_time, timeout_seconds, "during finalization"):
        plt.close(fig)
        return False
    
    _finalize_3d_morse_plot(ax, domain_bounds, labels, title, elev, azim, 
                           fontsize, output_path, dpi, fig)
    
    elapsed = time.time() - start_time
    print(f"  3D cuboid plot completed in {elapsed:.1f}s")
    return True


def _prepare_morse_set_scatter_data(morse_graph, domain_bounds, scale_factor):
    """Prepare scatter plot data for Morse sets."""
    vertices = morse_graph.vertices()
    num_morse_sets = len(vertices)
    
    domain_bounds_arr = np.array(domain_bounds)
    x_min, y_min, z_min = domain_bounds_arr[0]
    x_max, y_max, z_max = domain_bounds_arr[1]
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    scatter_data = {}
    vertex_to_color_idx = {v: i for i, v in enumerate(vertices)}
    
    for v in vertices:
        boxes = morse_graph.morse_set_boxes(v)
        if boxes:
            dim = len(boxes[0]) // 2
            X, Y, Z, S = [], [], [], []
            
            for box in boxes:
                center = np.array([(box[d] + box[d+dim]) / 2.0 for d in range(dim)])
                sizes = np.array([box[d+dim] - box[d] for d in range(dim)])
                
                X.append(center[0])
                Y.append(center[1])
                Z.append(center[2])
                
                normalized_sizes = sizes / np.array([x_range, y_range, z_range])
                marker_size = (scale_factor * 100 * np.mean(normalized_sizes)) ** 2
                S.append(marker_size)
            
            if X:
                scatter_data[v] = {
                    'X': X, 'Y': Y, 'Z': Z, 'S': S,
                    'color_idx': vertex_to_color_idx[v]
                }
    
    return scatter_data, num_morse_sets, domain_bounds_arr


def _plot_morse_set_scatter_points(ax, scatter_data, num_morse_sets, cmap):
    """Plot Morse set boxes as scatter points."""
    for v, data in scatter_data.items():
        color_idx = data['color_idx']
        color = cmap(color_idx / max(num_morse_sets - 1, 1))
        ax.scatter(data['X'], data['Y'], data['Z'], 
                  s=data['S'], c=[color] * len(data['X']), 
                  alpha=0.5, marker='s', label=f'Morse Set {v}', 
                  edgecolors='none')


def _plot_equilibria_and_orbits(ax, equilibria, periodic_orbits):
    """Plot equilibrium points and periodic orbits."""
    if equilibria:
        for name, point in equilibria.items():
            ax.scatter(point[0], point[1], point[2],
                      c='red', marker='*', s=200, label=name, 
                      zorder=100, edgecolors='darkred', linewidth=1)
    
    if periodic_orbits:
        for orbit_name, orbit_points in periodic_orbits.items():
            if orbit_points is not None and len(orbit_points) > 0:
                ax.scatter(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2],
                          c='orange', marker='o', s=80, label=orbit_name, 
                          zorder=90, alpha=0.8, edgecolors='darkorange', linewidth=1)
                orbit_closed = np.vstack([orbit_points, orbit_points[0:1]])
                ax.plot(orbit_closed[:, 0], orbit_closed[:, 1], orbit_closed[:, 2],
                       'orange', alpha=0.6, linewidth=1.5, zorder=85)


def plot_morse_sets_3d_scatter(morse_graph, domain_bounds, output_path=None, title="Morse Sets (3D)",
                               labels: Dict[str, str] = None, equilibria: Dict[str, np.ndarray] = None,
                               periodic_orbits: Dict[str, np.ndarray] = None,
                               scale_factor: float = 1.0,
                               data_overlay: np.ndarray = None):
    """
    Plots the Morse sets from a CMGDB MorseGraph object on a 3D scatter plot of box centers.
    Box marker sizes are proportional to box dimensions in data units.
    
    Adapted from Marcio Gameiro's PlotMorseSets implementation.

    :param morse_graph: The CMGDB.MorseGraph object.
    :param domain_bounds: [[lower_x, lower_y, lower_z], [upper_x, upper_y, upper_z]]
    :param output_path: Path to save figure (if None, displays instead)
    :param title: Plot title
    :param labels: Optional dict for axis labels, e.g. {'x': 'X', 'y': 'Y', 'z': 'Z'}
    :param equilibria: Optional dict of equilibrium points to plot
    :param periodic_orbits: Optional dict of periodic orbits to plot
    :param scale_factor: Scale factor for marker sizes (default 1.0)
    :param data_overlay: Optional data to overlay (N x 3 array)
    """
    fig, ax, _ = setup_figure_and_axes(ax=None, figsize=(12, 10), projection='3d')
    
    scatter_data, num_morse_sets, domain_bounds_arr = _prepare_morse_set_scatter_data(
        morse_graph, domain_bounds, scale_factor
    )
    
    cmap = cm.cool
    _plot_morse_set_scatter_points(ax, scatter_data, num_morse_sets, cmap)
    _plot_equilibria_and_orbits(ax, equilibria, periodic_orbits)
    
    if data_overlay is not None and len(data_overlay) > 0:
        ax.scatter(data_overlay[:, 0], data_overlay[:, 1], data_overlay[:, 2],
                  c='black', s=2, alpha=0.1, label='Data')
    
    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}
    
    configure_3d_axes(ax, domain_bounds, labels, title, fontsize=12)
    
    if num_morse_sets <= 15:
        ax.legend(loc='upper left', fontsize=10)
    
    finalize_plot(fig, ax, output_path, should_close=True)


def plot_morse_sets_3d_projections(
    morse_graph,
    barycenters_3d,
    output_dir,
    system_name,
    domain_bounds,
    cmap=cm.cool,
    prefix=""
):
    """
    Plots 2D projections of 3D Morse sets onto the xy, xz, and yz planes.

    Args:
        morse_graph: CMGDB MorseGraph object.
        barycenters_3d: Dictionary of barycenters for each Morse set (keyed by vertex).
        output_dir: Directory to save the projection plots.
        system_name: Name of the system for plot titles.
        domain_bounds: The bounds of the 3D domain.
        cmap: Colormap to use for Morse sets.
        prefix: Optional filename prefix (e.g., "03") for ordering.
    """
    if morse_graph is None:
        print("Morse graph is None, skipping 3D projections.")
        return

    from .utils import normalize_colormap
    
    vertices = morse_graph.vertices()
    num_morse_sets = len(vertices)
    vertex_to_color_idx = {v: i for i, v in enumerate(vertices)}
    colors = normalize_colormap(cmap, num_morse_sets)

    # Projections: (dim1, dim2, name, label1, label2)
    projections = [(0, 1, 'xy', 'X', 'Y'), (0, 2, 'xz', 'X', 'Z'), (1, 2, 'yz', 'Y', 'Z')]

    file_prefix = f"{prefix}_" if prefix else ""

    for dim1_idx, dim2_idx, proj_name, label1, label2 in projections:
        fig, ax = plt.subplots(figsize=(9, 9), dpi=100)
        ax.set_title(f"{system_name} - 3D Morse Sets Projection ({label1}-{label2})", fontsize=14)
        ax.set_xlabel(label1, fontsize=12)
        ax.set_ylabel(label2, fontsize=12)
        ax.set_xlim(domain_bounds[0][dim1_idx], domain_bounds[1][dim1_idx])
        ax.set_ylim(domain_bounds[0][dim2_idx], domain_bounds[1][dim2_idx])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.4)

        # Plot barycenters for each Morse set
        for v in vertices:
            color_idx = vertex_to_color_idx[v]
            if v in barycenters_3d and barycenters_3d[v]:
                barys = np.array(barycenters_3d[v])
                ax.scatter(barys[:, dim1_idx], barys[:, dim2_idx],
                           c=[colors[color_idx]], s=60, alpha=0.7, marker='o',
                           label=f'Morse Set {v}', edgecolors='none')
        
        # Add legend if reasonable number of sets
        if num_morse_sets <= 15:
            ax.legend(loc='best', fontsize=10)

        plt.tight_layout()
        filename = f"{file_prefix}morse_sets_3d_projection_{proj_name}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved 3D Morse set projections to {output_dir}")


def plot_morse_sets_3d_with_trajectories(
    morse_graph,
    domain_bounds,
    output_path=None,
    title="Morse Sets (3D) with Trajectories",
    labels=None,
    trajectory_data=None,
    n_trajectories=100,
    use_tail_only=True,
    tail_fraction=0.5,
    scale_factor=1.0
):
    """
    Plot 3D Morse sets with trajectory tail overlay.
    
    Box marker sizes are proportional to box dimensions in data units.
    Trajectories shown as low-alpha scatter to visualize attractor behavior.

    :param morse_graph: CMGDB MorseGraph object.
    :param domain_bounds: [[lower_x, lower_y, lower_z], [upper_x, upper_y, upper_z]]
    :param output_path: Path to save figure
    :param title: Plot title
    :param labels: Optional dict for axis labels
    :param trajectory_data: np.ndarray of shape (N, n_points, 3) or dict with 'Y_trajectories'
    :param n_trajectories: Number of trajectories to sample and plot
    :param use_tail_only: If True, plot only the tail of each trajectory
    :param tail_fraction: Fraction of trajectory to use as tail (e.g., 0.5 = last 50%)
    :param scale_factor: Scale factor for Morse set marker sizes
    """
    fig, ax, _ = setup_figure_and_axes(ax=None, figsize=(13, 11), projection='3d')

    vertices = morse_graph.vertices()
    num_morse_sets = len(vertices)
    
    cmap = cm.cool
    
    # Use common scatter data preparation
    scatter_data, _, _ = _prepare_morse_set_scatter_data(morse_graph, domain_bounds, scale_factor)
    _plot_morse_set_scatter_points(ax, scatter_data, num_morse_sets, cmap)

    # Plot trajectory data using common utility
    traj_data = extract_trajectory_data(trajectory_data, use_tail_only, tail_fraction)
    if traj_data is not None and len(traj_data) > 0:
        traj_data = sample_trajectories(traj_data, n_trajectories)
        
        # Flatten if 3D array
        if len(traj_data.shape) == 3:
            traj_data = traj_data.reshape(-1, 3)
        
        # Plot trajectory tails
        ax.scatter(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2],
                  c='black', s=2, alpha=0.15, label=f'Trajectory Tails (n={len(traj_data)})',
                  edgecolors='none')

    # Set axis properties using common utility
    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}
    
    configure_3d_axes(ax, domain_bounds, labels, title, fontsize=12)
    
    if num_morse_sets <= 15:
        ax.legend(loc='upper left', fontsize=10)

    finalize_plot(fig, ax, output_path, should_close=True)


def plot_morse_sets_3d_projections_with_trajectories(
    morse_graph,
    barycenters_3d,
    trajectory_data,
    output_dir,
    system_name,
    domain_bounds,
    cmap=cm.cool,
    prefix="",
    n_trajectories=100,
    use_tail_only=True,
    tail_fraction=0.5
):
    """
    Plot 2D projections of 3D Morse sets with trajectory tail overlay.
    
    Creates XY, XZ, and YZ projections showing Morse set barycenters and
    trajectory tail points to visualize attractor structure.

    :param morse_graph: CMGDB MorseGraph object.
    :param barycenters_3d: Dictionary of barycenters for each Morse set (keyed by vertex).
    :param trajectory_data: Either np.ndarray or dict with trajectory data.
    :param output_dir: Directory to save the projection plots.
    :param system_name: Name of the system for plot titles.
    :param domain_bounds: The bounds of the 3D domain.
    :param cmap: Colormap to use for Morse sets.
    :param prefix: Optional filename prefix for ordering.
    :param n_trajectories: Number of trajectories to sample and plot.
    :param use_tail_only: If True, plot only the tail of each trajectory.
    :param tail_fraction: Fraction of trajectory to use as tail.
    """
    if morse_graph is None:
        return

    vertices = morse_graph.vertices()
    num_morse_sets = len(vertices)
    
    from .utils import normalize_colormap
    
    vertex_to_color_idx = {v: i for i, v in enumerate(vertices)}
    colors = normalize_colormap(cmap, num_morse_sets)

    # Process trajectory data using common utility
    traj_data = extract_trajectory_data(trajectory_data, use_tail_only, tail_fraction)

    projections = [(0, 1, 'xy', 'X', 'Y'), (0, 2, 'xz', 'X', 'Z'), (1, 2, 'yz', 'Y', 'Z')]
    file_prefix = f"{prefix}_" if prefix else ""

    for dim1_idx, dim2_idx, proj_name, label1, label2 in projections:
        fig, ax, _ = setup_figure_and_axes(ax=None, figsize=(10, 9))
        ax.set_title(f"{system_name} - {label1}-{label2} Projection with Trajectories", fontsize=14)
        ax.set_xlabel(label1, fontsize=12)
        ax.set_ylabel(label2, fontsize=12)
        ax.set_xlim(domain_bounds[0][dim1_idx], domain_bounds[1][dim1_idx])
        ax.set_ylim(domain_bounds[0][dim2_idx], domain_bounds[1][dim2_idx])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.4)

        # Plot Morse sets
        for v in vertices:
            color_idx = vertex_to_color_idx[v]
            if v in barycenters_3d and barycenters_3d[v]:
                barys = np.array(barycenters_3d[v])
                ax.scatter(barys[:, dim1_idx], barys[:, dim2_idx],
                           c=[colors[color_idx]], s=80, alpha=0.7, marker='o',
                           label=f'Morse Set {v}', edgecolors='none', zorder=10)

        # Plot trajectory projections
        if traj_data is not None and len(traj_data) > 0:
            traj_subset = sample_trajectories(traj_data, n_trajectories)
            
            # Flatten and project trajectories
            traj_flat = traj_subset.reshape(-1, 3)
            ax.scatter(traj_flat[:, dim1_idx], traj_flat[:, dim2_idx],
                      c='black', s=2, alpha=0.12, label=f'Trajectory Tails',
                      edgecolors='none', zorder=5)

        if num_morse_sets <= 15:
            ax.legend(loc='best', fontsize=10)

        finalize_plot(fig, ax, os.path.join(output_dir, 
            f"{file_prefix}morse_sets_3d_projection_{proj_name}_with_trajectories.png"), 
            should_close=True)

    print(f"  Saved 3D Morse set projections with trajectories to {output_dir}")

