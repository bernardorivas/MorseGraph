"""
Comparison plotting functions for Morse graphs.

This module provides functions for comparing 3D and 2D Morse graphs,
including preimage classification visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from typing import Dict, Optional, Any
from io import BytesIO

try:
    import CMGDB
    from PIL import Image
    _CMGDB_AVAILABLE = True
except ImportError:
    _CMGDB_AVAILABLE = False

# Import from other plotting modules
from .utils import (
    get_num_morse_sets, 
    get_morse_set_colors,
    configure_3d_axes,
    normalize_colormap,
)
from .graphs import plot_morse_graph_diagram

# Import from latent module
from .latent import plot_latent_space_2d, classify_points_to_morse_sets


def _get_graph_statistics(morse_graph):
    """Extract number of nodes and edges from CMGDB or NetworkX graph."""
    if hasattr(morse_graph, 'num_vertices'):
        num_nodes = morse_graph.num_vertices()
        num_edges = sum(1 for v in range(num_nodes) for _ in morse_graph.adjacencies(v))
    else:
        num_nodes = len(morse_graph.nodes())
        num_edges = len(morse_graph.edges())
    return num_nodes, num_edges


def _plot_comparison_top_row(fig, morse_graph_3d, morse_graph_2d_data, 
                             morse_graph_2d_restricted, stats, title_prefix):
    """Plot top row of Morse graph diagrams."""
    ax1 = fig.add_subplot(2, 3, 1)
    plot_morse_graph_diagram(
        morse_graph_3d,
        output_path=None,
        title=f"{title_prefix}3D Morse Graph\n({stats['num_morse_sets_3d']} sets, {stats['num_edges_3d']} edges)",
        ax=ax1
    )

    ax2 = fig.add_subplot(2, 3, 2)
    plot_morse_graph_diagram(
        morse_graph_2d_data,
        output_path=None,
        title=f"{title_prefix}Learned Latent Dynamics (2D) - Data\n({stats['num_morse_sets_2d_data']} sets, {stats['num_edges_2d_data']} edges)",
        ax=ax2
    )

    ax3 = fig.add_subplot(2, 3, 3)
    plot_morse_graph_diagram(
        morse_graph_2d_restricted,
        output_path=None,
        title=f"{title_prefix}Learned Latent Dynamics (2D) - Restricted\n({stats['num_morse_sets_2d_restricted']} sets, {stats['num_edges_2d_restricted']} edges)",
        ax=ax3
    )


def _plot_comparison_bottom_row(fig, morse_graph_3d, morse_graph_2d_data,
                                morse_graph_2d_restricted, barycenters_3d,
                                z_data, z_restricted, latent_bounds, domain_bounds,
                                equilibria, equilibria_latent, labels, title_prefix):
    """Plot bottom row of spatial visualizations."""
    from mpl_toolkits.mplot3d import Axes3D
    
    num_morse_sets_3d = get_num_morse_sets(morse_graph_3d)
    
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    if hasattr(morse_graph_3d, 'morse_set_boxes'):
        for morse_idx in range(num_morse_sets_3d):
            boxes = morse_graph_3d.morse_set_boxes(morse_idx)
            if boxes and morse_idx in barycenters_3d and barycenters_3d[morse_idx]:
                barys = np.array(barycenters_3d[morse_idx])
                if barys.ndim == 1:
                    barys = barys.reshape(1, -1)
                color = cm.cool(morse_idx / max(num_morse_sets_3d - 1, 1))
                ax4.scatter(barys[:, 0], barys[:, 1], barys[:, 2],
                           c=[color], s=50, alpha=0.6)
    
    if equilibria:
        for name, eq_point in equilibria.items():
            ax4.scatter([eq_point[0]], [eq_point[1]], [eq_point[2]],
                       c='red', marker='*', s=200, label=name, zorder=10)
    
    ax4.set_xlabel(labels['x'])
    ax4.set_ylabel(labels['y'])
    ax4.set_zlabel(labels['z'])
    ax4.set_title(f"{title_prefix}3D Morse Sets")
    ax4.set_xlim(domain_bounds[0][0], domain_bounds[1][0])
    ax4.set_ylim(domain_bounds[0][1], domain_bounds[1][1])
    ax4.set_zlim(domain_bounds[0][2], domain_bounds[1][2])
    if equilibria:
        ax4.legend()

    ax5 = fig.add_subplot(2, 3, 5)
    plot_latent_space_2d(
        z_data, latent_bounds,
        morse_graph=morse_graph_2d_data,
        output_path=None,
        title=f"{title_prefix}Latent Space (Data)",
        equilibria_latent=equilibria_latent,
        ax=ax5
    )

    ax6 = fig.add_subplot(2, 3, 6)
    plot_latent_space_2d(
        z_restricted, latent_bounds,
        morse_graph=morse_graph_2d_restricted,
        output_path=None,
        title=f"{title_prefix}Latent Space (Restricted)",
        equilibria_latent=equilibria_latent,
        ax=ax6
    )


def plot_morse_graph_comparison(
    morse_graph_3d, 
    morse_graph_2d_data, 
    morse_graph_2d_restricted, 
    barycenters_3d, 
    encoder, 
    device, 
    z_data, 
    z_restricted, 
    latent_bounds, 
    domain_bounds, 
    output_path=None, 
    title_prefix="", 
    equilibria=None, 
    equilibria_latent=None, 
    labels=None
):
    """
    Create comprehensive side-by-side comparison of 3D and 2D Morse graphs.

    Creates a 2x3 figure comparing:
    - Top row: Morse graph diagrams (3D, 2D Data, 2D Restricted)
    - Bottom row: Corresponding visualizations (3D scatter, 2D latent spaces)

    This helps assess:
    - Agreement between 3D ground truth and 2D approximations
    - Differences between BoxMapData and Domain-Restricted methods
    - Overall quality of dimension reduction

    Args:
        morse_graph_3d: CMGDB MorseGraph object for 3D computation
        morse_graph_2d_data: NetworkX graph for 2D BoxMapData method
        morse_graph_2d_restricted: NetworkX graph for 2D Domain-Restricted method
        barycenters_3d: Dict mapping Morse set index to 3D barycenter coordinates
        encoder: PyTorch encoder model
        device: PyTorch device
        z_data: Encoded training data (N, latent_dim)
        z_restricted: Encoded large sample for domain restriction (M, latent_dim)
        latent_bounds: [[z0_min, z1_min], [z0_max, z1_max]]
        domain_bounds: [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        output_path: Path to save figure
        title_prefix: Prefix for subplot titles
        equilibria: Dict of equilibrium points in 3D space (optional)
        equilibria_latent: Dict of equilibrium points in latent space (optional)
        labels: Dict with 'x', 'y', 'z' keys for axis labels (optional)

    Returns:
        Dictionary with statistics:
            - 'num_morse_sets_3d': Number of Morse sets in 3D
            - 'num_morse_sets_2d_data': Number in 2D Data method
            - 'num_morse_sets_2d_restricted': Number in 2D Restricted method
            - 'num_edges_3d': Number of connections in 3D
            - 'num_edges_2d_data': Number in 2D Data
            - 'num_edges_2d_restricted': Number in 2D Restricted

    Example:
        >>> stats = plot_morse_graph_comparison(
        ...     mg3d, mg2d_data, mg2d_restricted,
        ...     barycenters, encoder, device,
        ...     z_train, z_large, latent_bounds, domain_bounds,
        ...     output_path='results/morse_comparison.png',
        ...     equilibria={'Eq': equilibrium_3d},
        ...     equilibria_latent={'Eq': equilibrium_2d}
        ... )
    """
    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}

    stats = {
        'num_morse_sets_3d': _get_graph_statistics(morse_graph_3d)[0],
        'num_edges_3d': _get_graph_statistics(morse_graph_3d)[1],
        'num_morse_sets_2d_data': _get_graph_statistics(morse_graph_2d_data)[0],
        'num_edges_2d_data': _get_graph_statistics(morse_graph_2d_data)[1],
        'num_morse_sets_2d_restricted': _get_graph_statistics(morse_graph_2d_restricted)[0],
        'num_edges_2d_restricted': _get_graph_statistics(morse_graph_2d_restricted)[1],
    }

    fig = plt.figure(figsize=(21, 12))

    _plot_comparison_top_row(
        fig, morse_graph_3d, morse_graph_2d_data, morse_graph_2d_restricted,
        stats, title_prefix
    )

    _plot_comparison_bottom_row(
        fig, morse_graph_3d, morse_graph_2d_data, morse_graph_2d_restricted,
        barycenters_3d, z_data, z_restricted, latent_bounds, domain_bounds,
        equilibria, equilibria_latent, labels, title_prefix
    )

    from .utils import finalize_plot
    finalize_plot(fig, ax=None, output_path=output_path, should_close=True)

    return stats


def _plot_morse_sets_rectangles(ax, morse_graph, cmap_colors, alpha=0.2, zorder=0):
    """
    Plot morse sets as rectangles - faithful to actual box bounds.

    This approach directly draws each box as a Rectangle patch in data coordinates,
    ensuring the visual representation exactly matches the computational structure.

    Args:
        ax: Matplotlib axis to plot on
        morse_graph: CMGDB MorseGraph object
        cmap_colors: List of colors for each morse set
        alpha: Transparency of rectangles (default: 0.2 for subtle background)
        zorder: Z-order for layering (default: 0 to place behind everything)
    """
    num_morse_sets = morse_graph.num_vertices()

    for morse_idx in range(num_morse_sets):
        boxes = morse_graph.morse_set_boxes(morse_idx)
        if not boxes:
            continue

        for box in boxes:
            # Box format from CMGDB for 2D: [xmin, ymin, xmax, ymax]
            rect = Rectangle(
                (box[0], box[1]),           # lower-left corner (xmin, ymin)
                box[2] - box[0],            # width (xmax - xmin)
                box[3] - box[1],            # height (ymax - ymin)
                facecolor=cmap_colors[morse_idx],
                alpha=alpha,
                edgecolor='none',
                zorder=zorder
            )
            ax.add_patch(rect)


def _plot_morse_graph_diagram_subplot(ax, morse_graph, cmap, title: str):
    """Plot a Morse graph diagram as a subplot using CMGDB."""
    if not _CMGDB_AVAILABLE:
        ax.text(0.5, 0.5, 'CMGDB not available', ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
        return
    
    gv_source = CMGDB.PlotMorseGraph(morse_graph, cmap=cmap)
    png_bytes = gv_source.pipe(format='png')
    img = Image.open(BytesIO(png_bytes))
    
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')


def _plot_3d_barycenters_with_equilibria(
    ax, morse_graph_3d, domain_bounds_3d, cmap_3d,
    equilibria=None, periodic_orbits=None, labels=None, title_prefix=""
):
    """Plot 3D Morse set barycenters with equilibria and periodic orbits."""
    num_morse_sets_3d = morse_graph_3d.num_vertices()
    node_colors_3d = normalize_colormap(cmap_3d, num_morse_sets_3d)
    
    for morse_idx in range(num_morse_sets_3d):
        boxes = morse_graph_3d.morse_set_boxes(morse_idx)
        if boxes:
            dim = len(boxes[0]) // 2
            centers = np.array([[(b[d] + b[d+dim]) / 2.0 for d in range(dim)] 
                               for b in boxes])
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                      c=[node_colors_3d[morse_idx]], s=20, alpha=0.4, marker='s')
    
    if equilibria:
        for name, eq_point in equilibria.items():
            ax.scatter([eq_point[0]], [eq_point[1]], [eq_point[2]],
                      c='red', marker='*', s=300, label=name, zorder=10)
    
    if periodic_orbits:
        for orbit_name, orbit_points in periodic_orbits.items():
            if orbit_points is not None and len(orbit_points) > 0:
                ax.scatter(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2],
                          c='orange', marker='o', s=100, label=orbit_name, 
                          zorder=90, alpha=0.9)
                orbit_closed = np.vstack([orbit_points, orbit_points[0:1]])
                ax.plot(orbit_closed[:, 0], orbit_closed[:, 1], orbit_closed[:, 2],
                       'orange', linewidth=2.5, alpha=0.7, zorder=85)
    
    configure_3d_axes(ax, domain_bounds_3d, labels, 
                      f'{title_prefix}3D Phase Space', fontsize=11)
    if equilibria or periodic_orbits:
        ax.legend(fontsize=9, loc='upper right')


def _plot_2d_latent_space_with_morse_sets(
    ax, morse_graph_2d, z_data, latent_bounds_2d, cmap_2d,
    equilibria_latent=None, title_prefix=""
):
    """Plot 2D latent space with Morse sets and equilibria."""
    num_morse_sets_2d = morse_graph_2d.num_vertices()
    
    ax.scatter(z_data[:, 0], z_data[:, 1], c='lightgray', s=1, 
              alpha=0.3, rasterized=True, zorder=1)
    
    node_colors_2d = normalize_colormap(cmap_2d, num_morse_sets_2d)
    _plot_morse_sets_rectangles(ax, morse_graph_2d, node_colors_2d)
    
    if equilibria_latent:
        for name, point_latent in equilibria_latent.items():
            ax.scatter(point_latent[0], point_latent[1],
                      c='red', marker='*', s=300, label=name, 
                      zorder=100, alpha=0.95)
        ax.legend(fontsize=9, loc='upper right')
    
    ax.set_xlim(latent_bounds_2d[0][0], latent_bounds_2d[1][0])
    ax.set_ylim(latent_bounds_2d[0][1], latent_bounds_2d[1][1])
    ax.set_xlabel('Latent Dim 0', fontsize=11)
    ax.set_ylabel('Latent Dim 1', fontsize=11)
    ax.set_title(f'{title_prefix}Learned Latent Dynamics (2D)', 
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')


def _count_morse_graph_edges(morse_graph) -> int:
    """Count total edges in a CMGDB Morse graph."""
    return sum(len(list(morse_graph.adjacencies(i))) 
               for i in range(morse_graph.num_vertices()))


def plot_2x2_morse_comparison(
    morse_graph_3d,
    morse_graph_2d, 
    domain_bounds_3d, 
    latent_bounds_2d, 
    encoder, 
    device, 
    z_data, 
    output_path=None, 
    title_prefix="", 
    equilibria=None, 
    periodic_orbits=None, 
    equilibria_latent=None, 
    labels=None
):
    """
    Creates a clean 2x2 comparison figure showing:
    - Top-left: 3D Morse graph diagram (using CMGDB.PlotMorseGraph)
    - Top-right: 2D Morse graph diagram (using CMGDB.PlotMorseGraph)
    - Bottom-left: 3D scatter (barycenters with equilibria/orbits)
    - Bottom-right: 2D latent space scatter (using rectangle patches) 
    
    Args:
        morse_graph_3d: CMGDB MorseGraph object for 3D
        morse_graph_2d: CMGDB MorseGraph object for 2D
        domain_bounds_3d: Domain bounds for 3D space
        latent_bounds_2d: Domain bounds for 2D latent space
        encoder: PyTorch encoder model
        device: PyTorch device
        z_data: Latent space data points for visualization
        output_path: Path to save figure
        title_prefix: Prefix for plot titles
        equilibria: Dict of equilibrium points in 3D space
        periodic_orbits: Dict of periodic orbits in 3D space
        equilibria_latent: Dict of equilibrium points in latent space
        labels: Dict with axis labels for 3D space
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(16, 14))
    
    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}

    cmap_3d = cm.cool
    cmap_2d = cm.viridis
    num_morse_sets_3d = morse_graph_3d.num_vertices()
    num_morse_sets_2d = morse_graph_2d.num_vertices()

    # Top-left: 3D Morse Graph Diagram
    ax1 = fig.add_subplot(2, 2, 1)
    _plot_morse_graph_diagram_subplot(
        ax1, morse_graph_3d, cmap_3d,
        f'{title_prefix}3D Morse Graph ({num_morse_sets_3d} sets)'
    )

    # Top-right: 2D Morse Graph Diagram
    ax2 = fig.add_subplot(2, 2, 2)
    _plot_morse_graph_diagram_subplot(
        ax2, morse_graph_2d, cmap_2d,
        f'{title_prefix}Learned Latent Dynamics (2D) ({num_morse_sets_2d} sets)'
    )
    
    # Bottom-left: 3D Scatter with Barycenters
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    _plot_3d_barycenters_with_equilibria(
        ax3, morse_graph_3d, domain_bounds_3d, cmap_3d,
        equilibria, periodic_orbits, labels, title_prefix
    )
    
    # Bottom-right: 2D Latent Space Scatter
    ax4 = fig.add_subplot(2, 2, 4)
    _plot_2d_latent_space_with_morse_sets(
        ax4, morse_graph_2d, z_data, latent_bounds_2d, cmap_2d,
        equilibria_latent, title_prefix
    )
    
    from .utils import finalize_plot
    finalize_plot(fig, ax=None, output_path=output_path, should_close=True)
    
    return {
        'num_morse_sets_3d': num_morse_sets_3d,
        'num_morse_sets_2d': num_morse_sets_2d,
        'num_edges_3d': _count_morse_graph_edges(morse_graph_3d),
        'num_edges_2d': _count_morse_graph_edges(morse_graph_2d)
    }


def _classify_points_to_morse_sets(encoder, device, X_sample, morse_graph_2d, 
                                   latent_bounds, subdiv_max):
    """Encode points and classify them to Morse sets."""
    import torch
    
    with torch.no_grad():
        z_sample = encoder(torch.FloatTensor(X_sample).to(device)).cpu().numpy()
    
    point_classifications = classify_points_to_morse_sets(
        z_sample, morse_graph_2d, latent_bounds, subdiv_max
    )
    
    return z_sample, point_classifications


def _organize_preimages_by_morse_set(X_sample, point_classifications, num_morse_sets):
    """Organize preimages by Morse set and identify sets with points."""
    preimages = {}
    morse_sets_with_points = []
    
    for morse_idx in range(num_morse_sets):
        mask = point_classifications == morse_idx
        if np.any(mask):
            preimages[morse_idx] = X_sample[mask]
            morse_sets_with_points.append(morse_idx)
        else:
            preimages[morse_idx] = np.array([])
    
    return preimages, morse_sets_with_points


def _plot_3d_preimage_views(fig, preimages, morse_sets_with_points, colors_map,
                           domain_bounds, labels, title_prefix, max_points_per_set):
    """Plot 3D preimages from three different viewing angles."""
    from mpl_toolkits.mplot3d import Axes3D
    
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    
    view_configs = [(ax1, 30, 45, "View 1"), 
                    (ax2, 30, 135, "View 2"), 
                    (ax3, 30, 225, "View 3")]
    
    for ax, elev, azim, view_name in view_configs:
        for morse_idx in morse_sets_with_points:
            points = preimages[morse_idx]
            if len(points) > max_points_per_set:
                indices = np.random.choice(len(points), max_points_per_set, replace=False)
                points = points[indices]
            
            color = colors_map[morse_idx]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=[color], s=10, alpha=0.4,
                      label=f'MS {morse_idx}' if ax == ax1 else None)
        
        ax.set_xlabel(labels['x'])
        ax.set_ylabel(labels['y'])
        ax.set_zlabel(labels['z'])
        ax.set_xlim(domain_bounds[0][0], domain_bounds[1][0])
        ax.set_ylim(domain_bounds[0][1], domain_bounds[1][1])
        ax.set_zlim(domain_bounds[0][2], domain_bounds[1][2])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title_prefix}Preimages in 3D ({view_name})")
    
    if morse_sets_with_points:
        ax1.legend(loc='upper left', fontsize=8, markerscale=2)


def _plot_latent_space_classification(ax, z_sample, point_classifications,
                                     morse_sets_with_points, colors_map,
                                     latent_bounds, title_prefix):
    """Plot 2D latent space with classified points."""
    for morse_idx in morse_sets_with_points:
        mask = point_classifications == morse_idx
        color = colors_map[morse_idx]
        ax.scatter(z_sample[mask, 0], z_sample[mask, 1],
                  c=[color], s=5, alpha=0.3, label=f'MS {morse_idx}')
    
    ax.set_xlabel('Latent Dim 0')
    ax.set_ylabel('Latent Dim 1')
    ax.set_xlim(latent_bounds[0][0], latent_bounds[1][0])
    ax.set_ylim(latent_bounds[0][1], latent_bounds[1][1])
    ax.set_title(f"{title_prefix}Latent Space - Classified Points")
    ax.grid(True, alpha=0.3)
    if morse_sets_with_points:
        ax.legend(loc='best', fontsize=8, markerscale=2)


def _plot_preimage_statistics(ax, X_sample, point_classifications, num_morse_sets,
                             morse_sets_with_points, method_name):
    """Plot statistics about preimage classification."""
    ax.axis('off')
    
    stats_text = f"Preimage Statistics ({method_name}):\n\n"
    stats_text += f"Total sample points: {len(X_sample)}\n"
    stats_text += f"Number of Morse sets: {num_morse_sets}\n"
    stats_text += f"Morse sets with points: {len(morse_sets_with_points)}\n\n"
    stats_text += "Points per Morse set:\n"
    
    for morse_idx in range(num_morse_sets):
        count = np.sum(point_classifications == morse_idx)
        percentage = 100 * count / len(X_sample) if len(X_sample) > 0 else 0
        marker = "✓" if morse_idx in morse_sets_with_points else "✗"
        stats_text += f"  {marker} MS {morse_idx}: {count:5d} ({percentage:5.1f}%)\n"
    
    unclassified = np.sum(point_classifications == -1)
    if unclassified > 0:
        percentage = 100 * unclassified / len(X_sample)
        stats_text += f"\n  Unclassified: {unclassified:5d} ({percentage:5.1f}%)\n"
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
           family='monospace')


def plot_preimage_classification(
    morse_graph_2d,
    encoder,
    decoder,
    device,
    X_sample,
    latent_bounds,
    domain_bounds,
    subdiv_max,
    output_path=None,
    title_prefix="",
    method_name="2D Latent",
    labels=None,
    max_points_per_set=1000
):
    """
    Visualize preimages of latent Morse sets in original 3D space.

    For each Morse set in latent space, shows which regions of the original
    3D space map to that set via the encoder. This helps understand the
    relationship between 3D dynamics and 2D latent structure.

    Creates a 2-row figure:
    - Top row: 3D preimages colored by Morse set membership
    - Bottom row: 2D latent space with Morse sets and corresponding points

    Args:
        morse_graph_2d: NetworkX DiGraph or CMGDB MorseGraph of 2D Morse graph
        encoder: PyTorch encoder model
        decoder: PyTorch decoder model
        device: PyTorch device
        X_sample: Sample points in original 3D space (N, 3)
        latent_bounds: [[z0_min, z1_min], [z0_max, z1_max]]
        domain_bounds: [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        subdiv_max: Maximum subdivision depth used in 2D computation
        output_path: Path to save figure
        title_prefix: Prefix for subplot titles
        method_name: Name of 2D method (e.g., "Data" or "Restricted")
        labels: Dict with 'x', 'y', 'z' keys for axis labels
        max_points_per_set: Maximum points to plot per Morse set (for performance)

    Returns:
        Dictionary mapping Morse set index to preimage points in 3D

    Example:
        >>> preimages = plot_preimage_classification(
        ...     morse_graph_2d, encoder, decoder, device,
        ...     X_large, latent_bounds, domain_bounds, subdiv_max,
        ...     output_path='results/preimages.png',
        ...     method_name='Data',
        ...     labels={'x': 'log(M)', 'y': 'log(A)', 'z': 'log(D)'}
        ... )
    """
    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}

    z_sample, point_classifications = _classify_points_to_morse_sets(
        encoder, device, X_sample, morse_graph_2d, latent_bounds, subdiv_max
    )

    num_morse_sets = get_num_morse_sets(morse_graph_2d)
    colors_map = get_morse_set_colors(num_morse_sets, colormap='viridis')

    preimages, morse_sets_with_points = _organize_preimages_by_morse_set(
        X_sample, point_classifications, num_morse_sets
    )

    fig = plt.figure(figsize=(21, 12))

    _plot_3d_preimage_views(
        fig, preimages, morse_sets_with_points, colors_map,
        domain_bounds, labels, title_prefix, max_points_per_set
    )

    ax4 = fig.add_subplot(2, 3, 4)
    _plot_latent_space_classification(
        ax4, z_sample, point_classifications, morse_sets_with_points,
        colors_map, latent_bounds, title_prefix
    )

    ax5 = fig.add_subplot(2, 3, 5)
    plot_morse_graph_diagram(
        morse_graph_2d,
        output_path=None,
        title=f"{title_prefix}Morse Graph ({method_name})",
        ax=ax5
    )

    ax6 = fig.add_subplot(2, 3, 6)
    _plot_preimage_statistics(
        ax6, X_sample, point_classifications, num_morse_sets,
        morse_sets_with_points, method_name
    )

    from .utils import finalize_plot
    finalize_plot(fig, ax=None, output_path=output_path, should_close=True)

    return preimages

