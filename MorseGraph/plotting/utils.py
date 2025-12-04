"""
Plotting utility functions.

This module provides helper functions for plotting Morse graphs, including
color management, graph statistics, figure/axes setup, and common utilities.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, Any, Tuple, Optional


def get_num_morse_sets(morse_graph) -> int:
    """
    Get number of Morse sets from CMGDB or NetworkX graph.
    
    Args:
        morse_graph: CMGDB MorseGraph or NetworkX DiGraph
        
    Returns:
        Number of Morse sets (vertices) in the graph
    """
    if hasattr(morse_graph, 'num_vertices'):
        return morse_graph.num_vertices()
    if hasattr(morse_graph, 'nodes'):
        return len(morse_graph.nodes())
    return 0


def get_morse_set_color(morse_set, morse_graph, index: int, 
                       num_sets: int, default_cmap='tab10'):
    """
    Get color for Morse set with fallback to default colormap.
    
    Args:
        morse_set: Morse set node
        morse_graph: NetworkX DiGraph containing the Morse set
        index: Index of the Morse set
        num_sets: Total number of Morse sets
        default_cmap: Default colormap name if no color attribute exists
        
    Returns:
        Color (RGBA tuple or string)
    """
    if isinstance(morse_graph, dict):
        # Handle case where morse_graph is actually a node dict
        if 'color' in morse_set:
            return morse_set['color']
    elif hasattr(morse_graph, 'nodes'):
        if morse_set in morse_graph.nodes and 'color' in morse_graph.nodes[morse_set]:
            color = morse_graph.nodes[morse_set]['color']
            # Convert numpy floats to python floats for compatibility
            if hasattr(color, '__iter__') and not isinstance(color, str):
                return tuple(float(c) for c in color)
            return color
    
    # Fallback to colormap
    import matplotlib.pyplot as plt
    cmap = plt.colormaps.get_cmap(default_cmap)
    color = cmap(index / max(num_sets, 10))
    # Convert numpy floats to python floats
    if hasattr(color, '__iter__') and not isinstance(color, str):
        return tuple(float(c) for c in color)
    return color


def get_morse_set_colors(num_morse_sets: int, colormap: str = 'tab10'):
    """
    Get consistent color scheme for Morse sets.
    
    Args:
        num_morse_sets: Number of Morse sets
        colormap: Matplotlib colormap name
        
    Returns:
        List of RGB color tuples
    """
    import matplotlib.pyplot as plt
    cmap = plt.colormaps.get_cmap(colormap)
    if num_morse_sets <= 10:
        colors = [cmap(i) for i in range(num_morse_sets)]
    else:
        colors = [cmap(i / num_morse_sets) for i in range(num_morse_sets)]
    return colors


def create_latent_grid(bounds, num_points=50):
    """
    Generate regular grid in latent space for sampling.
    
    Args:
        bounds: [[lower_x, lower_y], [upper_x, upper_y]]
        num_points: Number of points per dimension
        
    Returns:
        Array of grid points (num_points^2 x 2)
    """
    x = np.linspace(bounds[0][0], bounds[1][0], num_points)
    y = np.linspace(bounds[0][1], bounds[1][1], num_points)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return grid


def compute_encoded_barycenters(barycenters_3d, encoder, device):
    """
    Project 3D barycenters to latent space using encoder.
    
    Args:
        barycenters_3d: Dict mapping Morse set index to list of 3D barycenters
        encoder: Trained encoder model (PyTorch)
        device: torch device
        
    Returns:
        Dict mapping Morse set index to list of 2D encoded barycenters
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for compute_encoded_barycenters")
    
    barycenters_latent = {}
    for morse_set_idx, barys in barycenters_3d.items():
        if len(barys) == 0:
            barycenters_latent[morse_set_idx] = []
            continue
        
        barys_array = np.array(barys)
        with torch.no_grad():
            barys_tensor = torch.FloatTensor(barys_array).to(device)
            encoded = encoder(barys_tensor).cpu().numpy()
        barycenters_latent[morse_set_idx] = [encoded[i] for i in range(len(encoded))]
    
    return barycenters_latent


def _box_to_cuboid_faces(box):
    """
    Convert a 3D box to 6 faces for Poly3DCollection rendering.
    
    Args:
        box: Box in format [x_min, y_min, z_min, x_max, y_max, z_max]
        
    Returns:
        List of 6 faces, each face is a list of 4 vertices
    """
    x_min, y_min, z_min = box[0], box[1], box[2]
    x_max, y_max, z_max = box[3], box[4], box[5]

    vertices = [
        [x_min, y_min, z_min],  # 0: bottom-front-left
        [x_max, y_min, z_min],  # 1: bottom-front-right
        [x_max, y_max, z_min],  # 2: bottom-back-right
        [x_min, y_max, z_min],  # 3: bottom-back-left
        [x_min, y_min, z_max],  # 4: top-front-left
        [x_max, y_min, z_max],  # 5: top-front-right
        [x_max, y_max, z_max],  # 6: top-back-right
        [x_min, y_max, z_max],  # 7: top-back-left
    ]

    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom (z=min)
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top (z=max)
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front (y=min)
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back (y=max)
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left (x=min)
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right (x=max)
    ]

    return faces


def _get_graph_statistics(morse_graph) -> Dict[str, Any]:
    """
    Get statistics about a Morse graph.
    
    Args:
        morse_graph: CMGDB MorseGraph or NetworkX DiGraph
        
    Returns:
        Dictionary with graph statistics
    """
    import networkx as nx
    
    if isinstance(morse_graph, nx.DiGraph):
        num_nodes = morse_graph.number_of_nodes()
        num_edges = morse_graph.number_of_edges()
        num_attractors = sum(1 for n in morse_graph.nodes() if morse_graph.out_degree(n) == 0)
        num_repellers = sum(1 for n in morse_graph.nodes() if morse_graph.in_degree(n) == 0)
    elif hasattr(morse_graph, 'num_vertices'):
        num_nodes = morse_graph.num_vertices()
        num_edges = len(morse_graph.edges())
        num_attractors = sum(1 for v in morse_graph.vertices() if len(morse_graph.adjacencies(v)) == 0)
        num_repellers = 0  # Would need to compute reverse graph
    else:
        return {}
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_attractors': num_attractors,
        'num_repellers': num_repellers,
    }


# =============================================================================
# Common Plotting Patterns
# =============================================================================

def setup_figure_and_axes(ax=None, figsize=(10, 10), projection=None):
    """
    Standardized figure/axes setup with cleanup handling.
    
    Args:
        ax: Existing matplotlib axes (if None, creates new)
        figsize: Figure size tuple (width, height)
        projection: Projection type (e.g., '3d' for 3D plots)
        
    Returns:
        Tuple of (fig, ax, should_close) where should_close indicates
        whether the figure should be closed after plotting
    """
    if ax is None:
        if projection:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection=projection)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.get_figure(), ax, False


def finalize_plot(fig, ax, output_path=None, should_close=True, tight_layout=True):
    """
    Standardized plot finalization.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
        output_path: Path to save figure (if None, displays instead)
        should_close: Whether to close the figure after saving
        tight_layout: Whether to apply tight_layout
    """
    if tight_layout:
        plt.tight_layout()
    
    if output_path and should_close:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    elif should_close:
        plt.show()


def configure_3d_axes(ax, domain_bounds, labels=None, title=None, fontsize=11):
    """
    Standardized 3D axis configuration.
    
    Args:
        ax: 3D matplotlib axes
        domain_bounds: [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        labels: Dict with 'x', 'y', 'z' keys for axis labels
        title: Plot title (optional)
        fontsize: Font size for labels and title
    """
    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}
    
    ax.set_xlim(domain_bounds[0][0], domain_bounds[1][0])
    ax.set_ylim(domain_bounds[0][1], domain_bounds[1][1])
    ax.set_zlim(domain_bounds[0][2], domain_bounds[1][2])
    ax.set_xlabel(labels.get('x', 'X'), fontsize=fontsize)
    ax.set_ylabel(labels.get('y', 'Y'), fontsize=fontsize)
    ax.set_zlabel(labels.get('z', 'Z'), fontsize=fontsize)
    
    if title:
        ax.set_title(title, fontsize=fontsize + 3, fontweight='bold')


def normalize_colormap(cmap, num_items):
    """
    Standardized colormap normalization.
    
    Args:
        cmap: Matplotlib colormap
        num_items: Number of items to generate colors for
        
    Returns:
        List of colors normalized across the colormap
    """
    if num_items == 0:
        return []
    if num_items == 1:
        return [cmap(0)]
    return [cmap(i / max(num_items - 1, 1)) for i in range(num_items)]


def extract_trajectory_data(trajectory_data, use_tail_only=False, tail_fraction=0.5):
    """
    Extract and process trajectory data from various input formats.
    
    Handles:
    - Dictionary with 'Y_trajectories' or 'X_trajectories' keys
    - Direct numpy array input
    - Tail extraction for trajectory visualization
    
    Args:
        trajectory_data: Either np.ndarray or dict with trajectory data
        use_tail_only: If True, extract only the tail portion
        tail_fraction: Fraction of trajectory to use as tail (e.g., 0.5 = last 50%)
        
    Returns:
        Processed trajectory data as numpy array, or None if invalid
    """
    if trajectory_data is None:
        return None
    
    # Extract from dictionary if needed
    if isinstance(trajectory_data, dict):
        if 'Y_trajectories' in trajectory_data:
            traj_data = trajectory_data['Y_trajectories']
        elif 'X_trajectories' in trajectory_data:
            traj_data = trajectory_data['X_trajectories']
        else:
            return None
    else:
        traj_data = trajectory_data
    
    if traj_data is None or len(traj_data) == 0:
        return None
    
    # Extract tail if requested
    if use_tail_only and len(traj_data.shape) == 3:
        tail_start = int(traj_data.shape[1] * (1 - tail_fraction))
        traj_data = traj_data[:, tail_start:, :]
    
    return traj_data


def sample_trajectories(traj_data, n_trajectories):
    """
    Sample a subset of trajectories if there are more than requested.
    
    Args:
        traj_data: Trajectory data array (N, n_points, dim) or (N, dim)
        n_trajectories: Maximum number of trajectories to return
        
    Returns:
        Sampled trajectory data
    """
    if traj_data is None or len(traj_data) == 0:
        return traj_data
    
    if len(traj_data) > n_trajectories:
        indices = np.random.choice(len(traj_data), n_trajectories, replace=False)
        return traj_data[indices]
    
    return traj_data

