import os
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, Set, FrozenSet
import io

from .grids import AbstractGrid

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

    # Use colors from node attributes (assigned by compute_morse_graph)
    for morse_set in morse_sets:
        # Get color from node attribute, fallback to tab10 if not present
        if 'color' in morse_graph.nodes[morse_set]:
            color = morse_graph.nodes[morse_set]['color']
        else:
            # Fallback for backward compatibility
            num_sets = len(morse_sets)
            cmap = cm.get_cmap('tab10')
            color = cmap(list(morse_sets).index(morse_set) / max(num_sets, 10))

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
        # Get color from node attribute, fallback to tab10 if not present
        if 'color' in morse_graph.nodes[morse_set]:
            color = morse_graph.nodes[morse_set]['color']
        else:
            # Fallback for backward compatibility
            num_sets = len(morse_sets)
            cmap = cm.get_cmap('tab10')
            color = cmap(list(morse_sets).index(morse_set) / max(num_sets, 10))

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

def plot_basins_of_attraction(grid: AbstractGrid, basins,
                             morse_graph: nx.DiGraph = None, ax: plt.Axes = None,
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
    if ax is None:
        _, ax = plt.subplots()

    all_boxes = grid.get_boxes()

    # Plot each basin
    for attractor, basin in basins.items():
        # Get color from morse_graph node attributes if available
        if morse_graph and 'color' in morse_graph.nodes[attractor]:
            color = morse_graph.nodes[attractor]['color']
        else:
            # Fallback to viridis colormap for backward compatibility
            colors_cmap = plt.cm.get_cmap('viridis', len(basins))
            color = colors_cmap(list(basins.keys()).index(attractor))

        # Box-level basins: basin is a set of box indices
        # Plot basin boxes with lower opacity
        for box_index in basin:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, 0], box[0, 1]),
                               box[1, 0] - box[0, 0],
                               box[1, 1] - box[0, 1],
                               facecolor=color,
                               edgecolor='none',
                               alpha=0.3, **kwargs)
                ax.add_patch(rect)

        # Plot attractor itself with full opacity
        for box_index in attractor:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, 0], box[0, 1]),
                               box[1, 0] - box[0, 0],
                               box[1, 1] - box[0, 1],
                               facecolor=color,
                               edgecolor='none',
                               alpha=1.0, **kwargs)
                ax.add_patch(rect)

    # Optionally show boxes that don't belong to any basin (map outside domain)
    if show_outside:
        all_box_indices = set(range(len(all_boxes)))
        # For box basins, union all basin box sets
        basin_box_indices = set().union(*basins.values())
        outside_boxes = all_box_indices - basin_box_indices

        # Paint outside boxes black
        for box_index in outside_boxes:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, 0], box[0, 1]),
                               box[1, 0] - box[0, 0],
                               box[1, 1] - box[0, 1],
                               facecolor='black',
                               edgecolor='none',
                               alpha=0.5)
                ax.add_patch(rect)

    ax.set_xlim(grid.bounds[0, 0], grid.bounds[1, 0])
    ax.set_ylim(grid.bounds[0, 1], grid.bounds[1, 1])
    ax.set_aspect('equal', adjustable='box')

def plot_morse_graph(morse_graph: nx.DiGraph, ax: plt.Axes = None,
                    morse_sets_colors: dict = None, node_size: int = 300,
                    arrowsize: int = 20, font_size: int = 8):
    """
    Plots the Morse graph with hierarchical layout.

    :param morse_graph: The Morse graph to plot. Each node should have a 'color' attribute
                       (assigned by compute_morse_graph).
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param morse_sets_colors: Deprecated parameter, ignored. Colors are taken from node attributes.
    :param node_size: Size of the nodes.
    :param arrowsize: Size of the arrow heads.
    :param font_size: Font size for node labels.
    """
    if ax is None:
        _, ax = plt.subplots()

    morse_sets = list(morse_graph.nodes())

    # Use colors from node attributes or generate
    node_colors = []
    for morse_set in morse_sets:
        color = None
        if 'color' in morse_graph.nodes[morse_set]:
            color = morse_graph.nodes[morse_set]['color']
        else:
            # Generate color for backward compatibility
            num_sets = len(morse_sets)
            cmap = cm.get_cmap('tab10')
            color = cmap(morse_sets.index(morse_set) / max(num_sets, 10))

        # Convert numpy floats to python floats to avoid pygraphviz warning
        if hasattr(color, '__iter__'):
            color = tuple(float(c) for c in color)

        node_colors.append(color)
        # Update graph attribute for pygraphviz
        morse_graph.nodes[morse_set]['color'] = color

    # Create a mapping from frozenset to a shorter string representation
    node_labels = {node: str(i+1) for i, node in enumerate(morse_sets)} 
    
    # Try hierarchical layout, fallback to spring layout
    try:
        from networkx.drawing.nx_agraph import pygraphviz_layout
        pos = pygraphviz_layout(morse_graph, prog='dot')
    except (ImportError, Exception):
        pos = nx.spring_layout(morse_graph, seed=42)
    
    # Draw the graph components
    # Note: node_colors are RGBA tuples which matplotlib handles correctly
    nx.draw_networkx_nodes(morse_graph, pos, node_color=node_colors,
                          node_size=node_size, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(morse_graph, pos, edge_color='gray',
                          arrows=True, arrowsize=arrowsize, ax=ax, alpha=0.6)
    nx.draw_networkx_labels(morse_graph, pos, labels=node_labels,
                           font_size=font_size, ax=ax)

    ax.set_title("Morse Graph")

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
    cmap = plt.cm.get_cmap(colormap)
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


# =============================================================================
# Generalized Visualization Functions for 3D Pipeline
# =============================================================================

def plot_morse_graph_diagram(morse_graph, output_path=None, title="Morse Graph", cmap=None, figsize=(8, 8), ax=None):
    """
    Plot Morse graph structure using CMGDB's graphviz plotting.

    Args:
        morse_graph: CMGDB MorseGraph object or NetworkX DiGraph
        output_path: Path to save figure (if None, displays instead)
        title: Plot title
        cmap: Matplotlib colormap for nodes (default: cm.cool)
        figsize: Figure size tuple
        ax: Matplotlib axes to plot on (if None, creates new figure)

    Example:
        >>> plot_morse_graph_diagram(morse_graph, 'morse_graph.png')
    """
    try:
        import CMGDB
        import io
    except ImportError:
        raise ImportError("CMGDB is required for plot_morse_graph_diagram. Please install it.")

    if cmap is None:
        cmap = cm.cool

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        should_close = True
    else:
        fig = ax.get_figure()
        should_close = False

    if morse_graph is None or (hasattr(morse_graph, 'num_vertices') and morse_graph.num_vertices() == 0):
        ax.text(0.5, 0.5, 'Not computed', ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
    else:
        # Render as PNG using CMGDB's plotting
        try:
            gv_source = CMGDB.PlotMorseGraph(morse_graph, cmap=cmap)
            img_data = gv_source.pipe(format='png')
            img = plt.imread(io.BytesIO(img_data))
            
            # Display image
            ax.imshow(img, aspect='equal', interpolation='bilinear')
            ax.set_title(title)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f'Graphviz rendering failed: {e}', ha='center', va='center', wrap=True)
            ax.set_title(title)
            ax.axis('off')

    if ax is None:
        plt.tight_layout()
    
    if output_path and should_close:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    elif ax is None:
        plt.show()


def plot_training_curves(train_losses, val_losses, output_path=None):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: Dict with keys 'reconstruction', 'dynamics_recon', 'dynamics_consistency', 'total'
        val_losses: Dict with same keys as train_losses
        output_path: Path to save figure (if None, displays instead)

    Example:
        >>> plot_training_curves(train_losses, val_losses, 'training_curves.png')
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    loss_types = {
        'total': 'Total Loss',
        'reconstruction': 'Reconstruction: ||D(E(x)) - x||²',
        'dynamics_recon': 'Dynamics Recon: ||D(G(E(x))) - f(x)||²',
        'dynamics_consistency': 'Dynamics Cons: ||G(E(x)) - E(f(x))||²'
    }
    
    # a helper to zoom on the y-axis to the interesting part
    def set_dynamic_ylim(ax, train_loss_history, val_loss_history):
        num_epochs = len(train_loss_history)
        start_epoch = int(num_epochs * 0.2) if num_epochs > 10 else 0
        final_losses = train_loss_history[start_epoch:] + val_loss_history[start_epoch:]
        if final_losses:
            upper_limit = max(final_losses) * 1.2
            ax.set_ylim(bottom=0, top=upper_limit + 1e-6)
        else:
            ax.set_ylim(bottom=0)

    plot_order = ['total', 'reconstruction', 'dynamics_recon', 'dynamics_consistency']

    for i, key in enumerate(plot_order):
        if key in train_losses and key in val_losses:
            ax = axes[i]
            epochs = range(1, len(train_losses[key]) + 1)
            ax.plot(epochs, train_losses[key], label='Train')
            ax.plot(epochs, val_losses[key], label='Val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(loss_types.get(key, key))
            ax.legend()
            ax.grid(True, alpha=0.3)
            set_dynamic_ylim(ax, train_losses[key], val_losses[key])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


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
    fig = plt.figure(figsize=(12, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Get vertices using the CMGDB interface (more robust than range)
    vertices = morse_graph.vertices()
    num_morse_sets = len(vertices)

    # Color setup - use cool colormap
    cmap = cm.cool
    vertex_to_color_idx = {v: i for i, v in enumerate(vertices)}
    
    # Compute axis ranges for proper scaling
    domain_bounds_arr = np.array(domain_bounds)
    x_min, y_min, z_min = domain_bounds_arr[0]
    x_max, y_max, z_max = domain_bounds_arr[1]
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Collect all boxes and find size statistics for normalization
    all_boxes = []
    box_sizes_by_vertex = {}
    
    for v in vertices:
        boxes = morse_graph.morse_set_boxes(v)
        box_sizes_by_vertex[v] = []
        if boxes:
            for box in boxes:
                dim = len(box) // 2
                if dim == 3:
                    # Box format: [x_min, y_min, z_min, x_max, y_max, z_max]
                    sizes = np.array([box[d+dim] - box[d] for d in range(dim)])
                    box_sizes_by_vertex[v].append(sizes)
                    all_boxes.append((v, box, sizes))

    # Plot Morse sets
    for v in vertices:
        boxes = morse_graph.morse_set_boxes(v)
        if boxes:
            dim = len(boxes[0]) // 2
            color_idx = vertex_to_color_idx[v]
            color = cmap(color_idx / max(num_morse_sets - 1, 1))
            
            X, Y, Z, S = [], [], [], []
            
            for box in boxes:
                # Compute box center and dimensions
                center = np.array([(box[d] + box[d+dim]) / 2.0 for d in range(dim)])
                sizes = np.array([box[d+dim] - box[d] for d in range(dim)])
                
                X.append(center[0])
                Y.append(center[1])
                Z.append(center[2])
                
                # Marker size: scale by normalized box dimensions
                # Use the mean of normalized dimensions to get a representative size
                normalized_sizes = sizes / np.array([x_range, y_range, z_range])
                # Size in points^2, scaled by scale_factor
                marker_size = (scale_factor * 100 * np.mean(normalized_sizes)) ** 2
                S.append(marker_size)
            
            ax.scatter(X, Y, Z, s=S, c=[color] * len(X), alpha=0.5, marker='s',
                      label=f'Morse Set {v}', edgecolors='none')

    # Plot trajectory overlay
    if data_overlay is not None and len(data_overlay) > 0:
        ax.scatter(data_overlay[:, 0], data_overlay[:, 1], data_overlay[:, 2],
                  c='black', s=2, alpha=0.1, label='Data')

    # Plot equilibria
    if equilibria:
        for name, point in equilibria.items():
            ax.scatter(point[0], point[1], point[2],
                       c='red', marker='*', s=200,
                       label=name, zorder=100, edgecolors='darkred', linewidth=1)

    # Plot periodic orbits
    if periodic_orbits:
        for orbit_name, orbit_points in periodic_orbits.items():
            if orbit_points is not None and len(orbit_points) > 0:
                ax.scatter(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2],
                          c='orange', marker='o', s=80,
                          label=orbit_name, zorder=90, alpha=0.8, edgecolors='darkorange', linewidth=1)
                
                # Connect points to show the orbit
                orbit_closed = np.vstack([orbit_points, orbit_points[0:1]])
                ax.plot(orbit_closed[:, 0], orbit_closed[:, 1], orbit_closed[:, 2],
                       'orange', alpha=0.6, linewidth=1.5, zorder=85)

    # Set axis properties
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel(labels.get('x', 'X') if labels else 'X', fontsize=12)
    ax.set_ylabel(labels.get('y', 'Y') if labels else 'Y', fontsize=12)
    ax.set_zlabel(labels.get('z', 'Z') if labels else 'Z', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Only show legend if not too many sets
    if num_morse_sets <= 15:
        ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


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
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        should_close = True
    else:
        fig = ax.get_figure()
        should_close = False

    # Plot data points (light background)
    ax.scatter(z_data[:, 0], z_data[:, 1], c='lightgray', s=1, alpha=0.2, label='Data', rasterized=True, zorder=0)

    # Plot Morse sets if provided (2D Latent Morse Sets)
    # Use Viridis colormap for 2D sets
    if morse_graph is not None:
        try:
            num_morse_sets = morse_graph.num_vertices()
            cmap_2d = cm.viridis
            colors_2d = [cmap_2d(i / max(num_morse_sets - 1, 1)) for i in range(num_morse_sets)]

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
    handles, labels = ax.get_legend_handles_labels()
    # Filter duplicate labels if any
    by_label = dict(zip(labels, handles))
    # Sort logic could be added, but default order is usually fine
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small', framealpha=0.9)

    if ax is None:
        plt.tight_layout()
    
    if output_path and should_close:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    elif ax is None:
        plt.show()


# =============================================================================
# Enhanced Plotting Helper Functions
# =============================================================================

def get_morse_set_colors(num_morse_sets, colormap='tab10'):
    """
    Get consistent color scheme for Morse sets.

    Args:
        num_morse_sets: Number of Morse sets
        colormap: Matplotlib colormap name

    Returns:
        List of RGB color tuples
    """
    cmap = plt.get_cmap(colormap)
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

    encoded_barycenters = {}
    for morse_set_idx, barys in barycenters_3d.items():
        if len(barys) > 0:
            barys_array = np.array(barys)
            with torch.no_grad():
                encoded = encoder(torch.FloatTensor(barys_array).to(device)).cpu().numpy()
            encoded_barycenters[morse_set_idx] = [encoded[i] for i in range(len(encoded))]
        else:
            encoded_barycenters[morse_set_idx] = []
    return encoded_barycenters


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
    
    def _box_to_cuboid_faces(box):
        """Convert a 3D box to 6 faces for Poly3DCollection rendering."""
        x_min, y_min, z_min = box[0], box[1], box[2]
        x_max, y_max, z_max = box[3], box[4], box[5]

        # Define 8 vertices of the cuboid
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

        # Define 6 faces (each face has 4 vertices in counter-clockwise order)
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom (z=min)
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top (z=max)
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front (y=min)
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back (y=max)
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left (x=min)
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right (x=max)
        ]

        return faces

    start_time = time.time()
    
    # Count total boxes
    num_morse_sets = morse_graph.num_vertices()
    total_boxes = sum(len(morse_graph.morse_set_boxes(i)) for i in range(num_morse_sets))
    
    print(f"  Plotting {total_boxes} boxes across {num_morse_sets} Morse sets...")
    
    # Create 3D figure
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    morse_nodes = range(num_morse_sets)
    
    # Use cm.cool colormap (matching CMGDB default for 3D)
    cmap = cm.cool
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_morse_sets-1)

    # Plot each Morse set
    for morse_node in morse_nodes:
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"  WARNING: 3D cuboid rendering exceeded timeout ({timeout_seconds}s) after processing {morse_node}/{num_morse_sets} Morse sets.")
            print(f"  Skipping 3D cuboid plot. Use scatter plot instead for dense Morse sets.")
            plt.close(fig)
            return False
            
        morse_set_boxes = morse_graph.morse_set_boxes(morse_node)
        
        # Use morse_node as color index for consistency
        clr = matplotlib.colors.to_hex(cmap(cmap_norm(morse_node)), keep_alpha=True)

        # Collect all faces for this Morse set
        faces = []
        for box in morse_set_boxes:
            faces.extend(_box_to_cuboid_faces(box))

        if faces:
            # Create 3D polygon collection for this Morse set (matching CMGDB style)
            poly3d = Poly3DCollection(faces, facecolors=clr, alpha=alpha,
                                     edgecolors='none', linewidths=0)
            ax.add_collection3d(poly3d)

    # Set axis limits
    ax.set_xlim([domain_bounds[0][0], domain_bounds[1][0]])
    ax.set_ylim([domain_bounds[0][1], domain_bounds[1][1]])
    ax.set_zlim([domain_bounds[0][2], domain_bounds[1][2]])

    # Set viewing angle (matching CMGDB defaults)
    ax.view_init(elev=elev, azim=azim)

    # Add axis labels
    ax.set_xlabel(labels.get('x', '$x$') if labels else '$x$', fontsize=fontsize)
    ax.set_ylabel(labels.get('y', '$y$') if labels else '$y$', fontsize=fontsize)
    ax.set_zlabel(labels.get('z', '$z$') if labels else '$z$', fontsize=fontsize)
    
    # Set tick label size
    ax.tick_params(labelsize=fontsize)
    
    # Set title
    ax.set_title(title, fontsize=fontsize)

    plt.tight_layout()
    
    # Check timeout before saving
    elapsed = time.time() - start_time
    if elapsed > timeout_seconds:
        print(f"  WARNING: 3D cuboid rendering exceeded timeout ({timeout_seconds}s) during finalization.")
        print(f"  Skipping save. Use scatter plot instead.")
        plt.close(fig)
        return False
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    elapsed = time.time() - start_time
    print(f"  3D cuboid plot completed in {elapsed:.1f}s")
    return True


import os

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

    vertices = morse_graph.vertices()
    num_morse_sets = len(vertices)
    vertex_to_color_idx = {v: i for i, v in enumerate(vertices)}
    colors = [cmap(i / max(num_morse_sets - 1, 1)) for i in range(num_morse_sets)]

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
    fig = plt.figure(figsize=(13, 11), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    vertices = morse_graph.vertices()
    num_morse_sets = len(vertices)
    
    cmap = cm.cool
    vertex_to_color_idx = {v: i for i, v in enumerate(vertices)}

    # Compute axis ranges for proper scaling
    domain_bounds_arr = np.array(domain_bounds)
    x_min, y_min, z_min = domain_bounds_arr[0]
    x_max, y_max, z_max = domain_bounds_arr[1]
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Plot Morse sets
    for v in vertices:
        boxes = morse_graph.morse_set_boxes(v)
        if boxes:
            dim = len(boxes[0]) // 2
            color_idx = vertex_to_color_idx[v]
            color = cmap(color_idx / max(num_morse_sets - 1, 1))
            
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
            
            ax.scatter(X, Y, Z, s=S, c=[color] * len(X), alpha=0.5, marker='s',
                      label=f'Morse Set {v}', edgecolors='none')

    # Plot trajectory data
    if trajectory_data is not None:
        # Handle different input formats
        if isinstance(trajectory_data, dict):
            if 'Y_trajectories' in trajectory_data:
                traj_data = trajectory_data['Y_trajectories']
            elif 'X_trajectories' in trajectory_data:
                traj_data = trajectory_data['X_trajectories']
            else:
                traj_data = None
        else:
            traj_data = trajectory_data

        if traj_data is not None and len(traj_data) > 0:
            # Sample trajectories if needed
            if len(traj_data) > n_trajectories:
                indices = np.random.choice(len(traj_data), n_trajectories, replace=False)
                traj_data = traj_data[indices]
            
            # Extract tail if requested
            if use_tail_only and len(traj_data.shape) == 3:
                tail_start = int(traj_data.shape[1] * (1 - tail_fraction))
                traj_data = traj_data[:, tail_start:, :]
                traj_data = traj_data.reshape(-1, 3)
            elif len(traj_data.shape) == 3:
                traj_data = traj_data.reshape(-1, 3)
            
            # Plot trajectory tails
            ax.scatter(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2],
                      c='black', s=2, alpha=0.15, label=f'Trajectory Tails (n={len(traj_data)})',
                      edgecolors='none')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel(labels.get('x', 'X') if labels else 'X', fontsize=12)
    ax.set_ylabel(labels.get('y', 'Y') if labels else 'Y', fontsize=12)
    ax.set_zlabel(labels.get('z', 'Z') if labels else 'Z', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    if num_morse_sets <= 15:
        ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


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
    
    vertex_to_color_idx = {v: i for i, v in enumerate(vertices)}
    colors = [cmap(i / max(num_morse_sets - 1, 1)) for i in range(num_morse_sets)]

    # Process trajectory data
    traj_data = None
    if trajectory_data is not None:
        if isinstance(trajectory_data, dict):
            if 'Y_trajectories' in trajectory_data:
                traj_data = trajectory_data['Y_trajectories']
            elif 'X_trajectories' in trajectory_data:
                traj_data = trajectory_data['X_trajectories']
        else:
            traj_data = trajectory_data

    # Extract tail if needed
    if traj_data is not None and use_tail_only and len(traj_data.shape) == 3:
        tail_start = int(traj_data.shape[1] * (1 - tail_fraction))
        traj_data = traj_data[:, tail_start:, :]

    projections = [(0, 1, 'xy', 'X', 'Y'), (0, 2, 'xz', 'X', 'Z'), (1, 2, 'yz', 'Y', 'Z')]
    file_prefix = f"{prefix}_" if prefix else ""

    for dim1_idx, dim2_idx, proj_name, label1, label2 in projections:
        fig, ax = plt.subplots(figsize=(10, 9), dpi=100)
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
            # Sample trajectories if needed
            n_traj_to_plot = min(n_trajectories, len(traj_data))
            if n_traj_to_plot < len(traj_data):
                indices = np.random.choice(len(traj_data), n_traj_to_plot, replace=False)
                traj_subset = traj_data[indices]
            else:
                traj_subset = traj_data

            # Flatten and project trajectories
            traj_flat = traj_subset.reshape(-1, 3)
            ax.scatter(traj_flat[:, dim1_idx], traj_flat[:, dim2_idx],
                      c='black', s=2, alpha=0.12, label=f'Trajectory Tails',
                      edgecolors='none', zorder=5)

        if num_morse_sets <= 15:
            ax.legend(loc='best', fontsize=10)

        plt.tight_layout()
        filename = f"{file_prefix}morse_sets_3d_projection_{proj_name}_with_trajectories.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved 3D Morse set projections with trajectories to {output_dir}")


# =============================================================================
# Trajectory Analysis
# =============================================================================

def plot_trajectory_analysis(
    trajectories_3d,
    trajectories_latent,
    output_path=None,
    title_prefix="",
    labels=None,
    n_steps=None
):
    """
    Visualize trajectory simulations in both original and latent space.

    Creates a multi-panel figure showing:
    - Left column: Time series for each coordinate
    - Right column: Large phase portraits (3D original, 2D latent)

    This helps verify:
    - Numerical stability of dynamics
    - Trajectory behavior consistency
    - Correspondence between 3D and latent dynamics

    Args:
        trajectories_3d: List of 3D trajectories, each shape (n_steps, 3)
        trajectories_latent: List of 2D latent trajectories, each shape (n_steps, 2)
        output_path: Path to save figure (if None, displays instead)
        title_prefix: Prefix for subplot titles
        labels: Dict with keys 'x', 'y', 'z' for axis labels (optional)
        n_steps: Number of steps to plot (if None, plots all)

    Returns:
        None

    Example:
        >>> # Generate some trajectories
        >>> traj_3d = [trajectory1, trajectory2, trajectory3]  # Each shape (100, 3)
        >>> traj_latent = [latent_traj1, latent_traj2, latent_traj3]  # Each shape (100, 2)
        >>> plot_trajectory_analysis(
        ...     traj_3d, traj_latent,
        ...     output_path='results/trajectories.png',
        ...     title_prefix='Ives Model - ',
        ...     labels={'x': 'log(M)', 'y': 'log(A)', 'z': 'log(D)'}
        ... )
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec

    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}

    n_trajectories = len(trajectories_3d)
    colors = cm.viridis(np.linspace(0, 1, n_trajectories))

    # Truncate if needed
    if n_steps is not None:
        trajectories_3d = [traj[:n_steps] for traj in trajectories_3d]
        trajectories_latent = [traj[:n_steps] for traj in trajectories_latent]

    # Create figure with custom grid layout
    # Left column: 3 time series plots
    # Right column: 3D plot (spans 2 rows) + 2D plot (spans 1 row)
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1, 1])

    # Time series plots (left column: 3 rows)
    coord_names = [labels['x'], labels['y'], labels['z']]
    for i, coord_name in enumerate(coord_names):
        ax = fig.add_subplot(gs[i, 0])
        for j, traj in enumerate(trajectories_3d):
            steps = np.arange(len(traj))
            ax.plot(steps, traj[:, i], color=colors[j], alpha=0.7,
                   label=f'Trajectory {j+1}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(coord_name)
        ax.set_title(f'{title_prefix}{coord_name} vs Time')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='best', fontsize=8)

    # 3D phase portrait (right column, spans top 2 rows)
    ax_3d = fig.add_subplot(gs[0:2, 1], projection='3d')
    for j, traj in enumerate(trajectories_3d):
        ax_3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                  color=colors[j], alpha=0.7, linewidth=1.5)
        # Mark starting point
        ax_3d.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]],
                     color=colors[j], s=120, marker='o', edgecolors='white', linewidths=1.5)
        # Mark ending point with different marker
        ax_3d.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                     color=colors[j], s=120, marker='s', edgecolors='white', linewidths=1.5)
    ax_3d.set_xlabel(labels['x'], fontsize=11)
    ax_3d.set_ylabel(labels['y'], fontsize=11)
    ax_3d.set_zlabel(labels['z'], fontsize=11)
    ax_3d.set_title(f'{title_prefix}3D Phase Portrait', fontsize=12, fontweight='bold')

    # 2D latent phase portrait (right column, bottom row)
    ax_latent = fig.add_subplot(gs[2, 1])
    for j, traj in enumerate(trajectories_latent):
        ax_latent.plot(traj[:, 0], traj[:, 1],
                      color=colors[j], alpha=0.7, linewidth=1.5)
        # Mark starting point
        ax_latent.scatter([traj[0, 0]], [traj[0, 1]],
                         color=colors[j], s=120, marker='o', edgecolors='white', linewidths=1.5,
                         label=f'Traj {j+1}' if j < 3 else None)
        # Mark ending point
        ax_latent.scatter([traj[-1, 0]], [traj[-1, 1]],
                         color=colors[j], s=120, marker='s', edgecolors='white', linewidths=1.5)
    ax_latent.set_xlabel('Latent Dim 0', fontsize=11)
    ax_latent.set_ylabel('Latent Dim 1', fontsize=11)
    ax_latent.set_title(f'{title_prefix}2D Latent Phase Portrait', fontsize=12, fontweight='bold')
    ax_latent.grid(True, alpha=0.3)
    ax_latent.set_aspect('equal', adjustable='box')
    
    # Add legend to latent plot
    if n_trajectories <= 5:
        ax_latent.legend(loc='best', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# Comprehensive Morse Graph Comparison
# =============================================================================

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
    from mpl_toolkits.mplot3d import Axes3D

    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}

    # Extract statistics
    # CMGDB MorseGraph has num_vertices() and edges() methods
    # NetworkX graphs have nodes() and edges() methods

    # Helper function to extract stats from either type
    def get_graph_stats(mg):
        if hasattr(mg, 'num_vertices'):
            # CMGDB MorseGraph
            return mg.num_vertices(), sum(1 for v in range(mg.num_vertices()) for _ in mg.adjacencies(v))
        else:
            # NetworkX graph
            return len(mg.nodes()), len(mg.edges())

    num_morse_sets_3d, num_edges_3d = get_graph_stats(morse_graph_3d)
    num_morse_sets_2d_data, num_edges_2d_data = get_graph_stats(morse_graph_2d_data)
    num_morse_sets_2d_restricted, num_edges_2d_restricted = get_graph_stats(morse_graph_2d_restricted)

    # Create figure
    fig = plt.figure(figsize=(21, 12))

    # ========================================================================
    # Top Row: Morse Graph Diagrams
    # ========================================================================

    # Panel 1: 3D Morse graph diagram
    ax1 = fig.add_subplot(2, 3, 1)
    plot_morse_graph_diagram(
        morse_graph_3d,
        output_path=None,
        title=f"{title_prefix}3D Morse Graph\n({num_morse_sets_3d} sets, {num_edges_3d} edges)",
        ax=ax1
    )

    # Panel 2: 2D Data Morse graph diagram
    ax2 = fig.add_subplot(2, 3, 2)
    plot_morse_graph_diagram(
        morse_graph_2d_data,
        output_path=None,
        title=f"{title_prefix}Learned Latent Dynamics (2D) - Data\n({num_morse_sets_2d_data} sets, {num_edges_2d_data} edges)",
        ax=ax2
    )

    # Panel 3: 2D Restricted Morse graph diagram
    ax3 = fig.add_subplot(2, 3, 3)
    plot_morse_graph_diagram(
        morse_graph_2d_restricted,
        output_path=None,
        title=f"{title_prefix}Learned Latent Dynamics (2D) - Restricted\n({num_morse_sets_2d_restricted} sets, {num_edges_2d_restricted} edges)",
        ax=ax3
    )

    # ========================================================================
    # Bottom Row: Spatial Visualizations
    # ========================================================================

    # Panel 4: 3D Morse sets scatter
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    
    # Extract barycenters from 3D Morse graph
    if hasattr(morse_graph_3d, 'morse_set_boxes'):
        # CMGDB MorseGraph
        for morse_idx in range(num_morse_sets_3d):
            boxes = morse_graph_3d.morse_set_boxes(morse_idx)
            if boxes and morse_idx in barycenters_3d and barycenters_3d[morse_idx]:
                barys = np.array(barycenters_3d[morse_idx])
                if barys.ndim == 1:
                    barys = barys.reshape(1, -1)
                color = cm.cool(morse_idx / max(num_morse_sets_3d - 1, 1))
                ax4.scatter(barys[:, 0], barys[:, 1], barys[:, 2],
                           c=[color], s=50, alpha=0.6)
    
    # Plot equilibria if provided
    if equilibria is not None:
        for name, eq_point in equilibria.items():
            ax4.scatter([eq_point[0]], [eq_point[1]], [eq_point[2]],
                       c='red', marker='*', s=200, 
                       label=name, zorder=10)
    
    ax4.set_xlabel(labels['x'])
    ax4.set_ylabel(labels['y'])
    ax4.set_zlabel(labels['z'])
    ax4.set_title(f"{title_prefix}3D Morse Sets")
    ax4.set_xlim(domain_bounds[0][0], domain_bounds[1][0])
    ax4.set_ylim(domain_bounds[0][1], domain_bounds[1][1])
    ax4.set_zlim(domain_bounds[0][2], domain_bounds[1][2])
    if equilibria is not None:
        ax4.legend()

    # Panel 5: 2D latent space (Data method)
    ax5 = fig.add_subplot(2, 3, 5)
    plot_latent_space_2d(
        z_data, 
        latent_bounds,
        morse_graph=morse_graph_2d_data,
        output_path=None,
        title=f"{title_prefix}Latent Space (Data)",
        equilibria_latent=equilibria_latent,
        ax=ax5
    )

    # Panel 6: 2D latent space (Restricted method)
    ax6 = fig.add_subplot(2, 3, 6)
    plot_latent_space_2d(
        z_restricted,
        latent_bounds,
        morse_graph=morse_graph_2d_restricted,
        output_path=None,
        title=f"{title_prefix}Latent Space (Restricted)",
        equilibria_latent=equilibria_latent,
        ax=ax6
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return {
        'num_morse_sets_3d': num_morse_sets_3d,
        'num_morse_sets_2d_data': num_morse_sets_2d_data,
        'num_morse_sets_2d_restricted': num_morse_sets_2d_restricted,
        'num_edges_3d': num_edges_3d,
        'num_edges_2d_data': num_edges_2d_data,
        'num_edges_2d_restricted': num_edges_2d_restricted
    }


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
    from matplotlib.patches import Rectangle

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
    import torch
    from mpl_toolkits.mplot3d import Axes3D
    from io import BytesIO
    from PIL import Image
    import CMGDB
    
    fig = plt.figure(figsize=(16, 14)) 
    
    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}

    cmap_3d = cm.cool
    cmap_2d = cm.viridis

    # ===== Top-left: 3D Morse Graph Diagram (using CMGDB) =====
    ax1 = fig.add_subplot(2, 2, 1)
    
    num_morse_sets_3d = morse_graph_3d.num_vertices()
    
    # Use CMGDB.PlotMorseGraph to generate Graphviz diagram
    gv_source_3d = CMGDB.PlotMorseGraph(morse_graph_3d, cmap=cmap_3d)
    png_bytes_3d = gv_source_3d.pipe(format='png')
    img_3d = Image.open(BytesIO(png_bytes_3d))

    ax1.imshow(img_3d)
    ax1.axis('off')
    ax1.set_title(f'{title_prefix}3D Morse Graph ({num_morse_sets_3d} sets)',
                  fontsize=14, fontweight='bold')

    # ===== Top-right: 2D Morse Graph Diagram (using CMGDB) =====
    ax2 = fig.add_subplot(2, 2, 2)

    num_morse_sets_2d = morse_graph_2d.num_vertices()

    # Use CMGDB.PlotMorseGraph to generate Graphviz diagram
    gv_source_2d = CMGDB.PlotMorseGraph(morse_graph_2d, cmap=cmap_2d)
    png_bytes_2d = gv_source_2d.pipe(format='png')
    img_2d = Image.open(BytesIO(png_bytes_2d))
    
    ax2.imshow(img_2d)
    ax2.axis('off')
    ax2.set_title(f'{title_prefix}Learned Latent Dynamics (2D) ({num_morse_sets_2d} sets)',
                  fontsize=14, fontweight='bold')
    
    # ===== Bottom-left: 3D Scatter with Barycenters =====
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')

    # Plot 3D Morse set barycenters
    node_colors_3d = [cmap_3d(i / max(num_morse_sets_3d - 1, 1)) for i in range(num_morse_sets_3d)]
    
    for morse_idx in range(num_morse_sets_3d):
        boxes = morse_graph_3d.morse_set_boxes(morse_idx)
        if boxes:
            dim = len(boxes[0]) // 2
            centers = np.array([[(b[d] + b[d+dim]) / 2.0 for d in range(dim)] for b in boxes])
            ax3.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                       c=[node_colors_3d[morse_idx]], s=20, alpha=0.4, marker='s')
    
    # Plot equilibria
    if equilibria:
        for name, eq_point in equilibria.items():
            ax3.scatter([eq_point[0]], [eq_point[1]], [eq_point[2]],
                       c='red', marker='*', s=300,
                       label=name, zorder=10)
    
    # Plot periodic orbits
    if periodic_orbits:
        for orbit_name, orbit_points in periodic_orbits.items():
            if orbit_points is not None and len(orbit_points) > 0:
                # Plot orbit points
                ax3.scatter(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2],
                          c='orange', marker='o', s=100,
                          label=orbit_name, zorder=90, alpha=0.9)
                
                # Connect with lines
                orbit_closed = np.vstack([orbit_points, orbit_points[0:1]])
                ax3.plot(orbit_closed[:, 0], orbit_closed[:, 1], orbit_closed[:, 2],
                        'orange', linewidth=2.5, alpha=0.7, zorder=85)
    
    ax3.set_xlim(domain_bounds_3d[0][0], domain_bounds_3d[1][0])
    ax3.set_ylim(domain_bounds_3d[0][1], domain_bounds_3d[1][1])
    ax3.set_zlim(domain_bounds_3d[0][2], domain_bounds_3d[1][2])
    ax3.set_xlabel(labels['x'], fontsize=11)
    ax3.set_ylabel(labels['y'], fontsize=11)
    ax3.set_zlabel(labels['z'], fontsize=11)
    ax3.set_title(f'{title_prefix}3D Phase Space', fontsize=14, fontweight='bold')
    if equilibria or periodic_orbits:
        ax3.legend(fontsize=9, loc='upper right')
    
    # ===== Bottom-right: 2D Latent Space Scatter =====
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Plot latent space data (background)
    ax4.scatter(z_data[:, 0], z_data[:, 1], c='lightgray', s=1, alpha=0.3, rasterized=True, zorder=1)

    # Plot Morse sets using rectangle patches (low opacity, behind everything)
    node_colors_2d = [cmap_2d(i / max(num_morse_sets_2d - 1, 1))
                      for i in range(num_morse_sets_2d)]

    _plot_morse_sets_rectangles(
        ax4, morse_graph_2d, node_colors_2d
    )
    
    # Plot equilibria in latent space
    if equilibria_latent:
        for name, point_latent in equilibria_latent.items():
            ax4.scatter(point_latent[0], point_latent[1],
                       c='red', marker='*', s=300,
                       label=name, zorder=100, alpha=0.95)
        ax4.legend(fontsize=9, loc='upper right')
    
    ax4.set_xlim(latent_bounds_2d[0][0], latent_bounds_2d[1][0])
    ax4.set_ylim(latent_bounds_2d[0][1], latent_bounds_2d[1][1])
    ax4.set_xlabel('Latent Dim 0', fontsize=11)
    ax4.set_ylabel('Latent Dim 1', fontsize=11)
    ax4.set_title(f'{title_prefix}Learned Latent Dynamics (2D)', fontsize=14, fontweight='bold')
    ax4.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    # Build edge counts from adjacencies
    num_edges_3d = sum(len(list(morse_graph_3d.adjacencies(i))) for i in range(num_morse_sets_3d))
    num_edges_2d = sum(len(list(morse_graph_2d.adjacencies(i))) for i in range(num_morse_sets_2d))
    
    return {
        'num_morse_sets_3d': num_morse_sets_3d,
        'num_morse_sets_2d': num_morse_sets_2d,
        'num_edges_3d': num_edges_3d,
        'num_edges_2d': num_edges_2d
    }


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
    from mpl_toolkits.mplot3d import Axes3D
    import torch

    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}

    # Encode sample to latent space
    with torch.no_grad():
        z_sample = encoder(torch.FloatTensor(X_sample).to(device)).cpu().numpy()

    # Classify points to Morse sets
    point_classifications = classify_points_to_morse_sets(
        z_sample,
        morse_graph_2d,
        latent_bounds,
        subdiv_max
    )

    # Get number of Morse sets (handle both CMGDB and NetworkX)
    if hasattr(morse_graph_2d, 'num_vertices'):
        num_morse_sets = morse_graph_2d.num_vertices()
    else:
        num_morse_sets = len(morse_graph_2d.nodes())
    colors_map = get_morse_set_colors(num_morse_sets, colormap='viridis')

    # Organize preimages by Morse set and determine which sets have points
    preimages = {}
    morse_sets_with_points = []
    for morse_idx in range(num_morse_sets):
        mask = point_classifications == morse_idx
        if np.any(mask):
            preimages[morse_idx] = X_sample[mask]
            morse_sets_with_points.append(morse_idx)
        else:
            preimages[morse_idx] = np.array([])

    # Create figure with 2 rows
    fig = plt.figure(figsize=(21, 12))

    # ========================================================================
    # Top Row: 3D Preimages
    # ========================================================================

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')

    # Plot different views - ONLY morse sets with points
    for ax_idx, (ax, elev, azim) in enumerate([ 
        (ax1, 30, 45), 
        (ax2, 30, 135), 
        (ax3, 30, 225)
    ]):
        for morse_idx in morse_sets_with_points:  # Only plot sets with points
            points = preimages[morse_idx]
            # Subsample if too many points
            if len(points) > max_points_per_set:
                indices = np.random.choice(len(points), max_points_per_set, replace=False)
                points = points[indices]
            
            color = colors_map[morse_idx]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=[color], s=10, alpha=0.4,
                      label=f'MS {morse_idx}' if ax_idx == 0 else None)
        
        ax.set_xlabel(labels['x'])
        ax.set_ylabel(labels['y'])
        ax.set_zlabel(labels['z'])
        ax.set_xlim(domain_bounds[0][0], domain_bounds[1][0])
        ax.set_ylim(domain_bounds[0][1], domain_bounds[1][1])
        ax.set_zlim(domain_bounds[0][2], domain_bounds[1][2])
        ax.view_init(elev=elev, azim=azim)
        
        if ax_idx == 0:
            ax.set_title(f"{title_prefix}Preimages in 3D (View 1)")
        elif ax_idx == 1:
            ax.set_title(f"{title_prefix}Preimages in 3D (View 2)")
        else:
            ax.set_title(f"{title_prefix}Preimages in 3D (View 3)")

    # Add legend to first plot (only for sets with points)
    if morse_sets_with_points:
        ax1.legend(loc='upper left', fontsize=8, markerscale=2)

    # ========================================================================
    # Bottom Row: 2D Latent Space
    # ========================================================================

    # Show all classified points - ONLY morse sets with points
    ax4 = fig.add_subplot(2, 3, 4)
    for morse_idx in morse_sets_with_points:  # Only plot sets with points
        mask = point_classifications == morse_idx
        color = colors_map[morse_idx]
        ax4.scatter(z_sample[mask, 0], z_sample[mask, 1],
                   c=[color], s=5, alpha=0.3,
                   label=f'MS {morse_idx}')
    
    ax4.set_xlabel('Latent Dim 0')
    ax4.set_ylabel('Latent Dim 1')
    ax4.set_xlim(latent_bounds[0][0], latent_bounds[1][0])
    ax4.set_ylim(latent_bounds[0][1], latent_bounds[1][1])
    ax4.set_title(f"{title_prefix}Latent Space - Classified Points")
    ax4.grid(True, alpha=0.3)
    if morse_sets_with_points:
        ax4.legend(loc='best', fontsize=8, markerscale=2)

    # Show Morse graph diagram
    ax5 = fig.add_subplot(2, 3, 5)
    plot_morse_graph_diagram(
        morse_graph_2d,
        output_path=None,
        title=f"{title_prefix}Morse Graph ({method_name})",
        ax=ax5
    )

    # Show statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
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
    
    # Add count of unclassified points
    unclassified = np.sum(point_classifications == -1)
    if unclassified > 0:
        percentage = 100 * unclassified / len(X_sample)
        stats_text += f"\n  Unclassified: {unclassified:5d} ({percentage:5.1f}%)\n"
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
            family='monospace')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return preimages

def plot_encoder_decoder_roundtrip(X, encoder, decoder, output_path=None, num_samples=500, latent_grid_points=None):
    """
    Plot a 3x2 grid visualization of the encoder-decoder mappings.
    
    Rows:
    1. Original Grid (3D) | Encoded Grid (2D)
    2. Decoded Latent Grid (3D) | Latent Grid (2D)
    3. Reconstructed Original (3D) | Re-encoded Latent (2D)
    
    Args:
        X: Input data (N, D) - Original data sample
        encoder: Encoder model
        decoder: Decoder model
        output_path: Path to save figure
        num_samples: Number of points to use for original data sample
        latent_grid_points: Optional pre-generated latent grid points. If None, generated from bounds of E(X).
    """
    import torch
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 1. Prepare Data
    if len(X) > num_samples:
        indices = np.random.choice(len(X), num_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    try:
        device = next(encoder.parameters()).device
    except Exception:
        device = 'cpu'
        
    X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # E(X)
        Z_sample = encoder(X_tensor).cpu().numpy()
        
        # D(E(X)) - Reconstruction
        X_recon = decoder(encoder(X_tensor)).cpu().numpy()
        
        # Prepare Latent Grid
        if latent_grid_points is None:
            z_min = Z_sample.min(axis=0)
            z_max = Z_sample.max(axis=0)
            # Add slight padding
            padding = 0.1 * (z_max - z_min)
            z_min -= padding
            z_max += padding
            
            # Create grid (20x20 = 400 points)
            grid_res = 20
            x = np.linspace(z_min[0], z_max[0], grid_res)
            y = np.linspace(z_min[1], z_max[1], grid_res)
            xx, yy = np.meshgrid(x, y)
            latent_grid = np.column_stack([xx.ravel(), yy.ravel()])
        else:
            latent_grid = latent_grid_points

        latent_grid_tensor = torch.tensor(latent_grid, dtype=torch.float32).to(device)
        
        # D(Z_grid)
        X_latent_decoded = decoder(latent_grid_tensor).cpu().numpy()
        
        # E(D(Z_grid)) - Consistency
        Z_latent_reencoded = encoder(decoder(latent_grid_tensor)).cpu().numpy()

    # 2. Plotting
    fig = plt.figure(figsize=(16, 18))
    
    # Helper for 3D scatter
    def plot_3d(ax, data, title, color='blue', marker='o', alpha=0.5):
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, marker=marker, s=10, alpha=alpha)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

    # Helper for 2D scatter
    def plot_2d(ax, data, title, color='red', marker='o', alpha=0.5):
        ax.scatter(data[:, 0], data[:, 1], c=color, marker=marker, s=10, alpha=alpha)
        ax.set_xlabel('Latent 1')
        ax.set_ylabel('Latent 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # (1,1) Original Data (3D)
    ax1 = fig.add_subplot(3, 2, 1, projection='3d')
    plot_3d(ax1, X_sample, "Original Data (X)", color='blue')
    # (1,2) Encoded Data (2D)
    ax2 = fig.add_subplot(3, 2, 2)
    plot_2d(ax2, Z_sample, "Encoded Data E(X)", color='red')

    # (2,1) Decoded Latent Grid (3D)
    ax3 = fig.add_subplot(3, 2, 3, projection='3d')
    plot_3d(ax3, X_latent_decoded, "Decoded Latent Grid D(Z_grid)", color='green')

    # (2,2) Latent Grid (2D)
    ax4 = fig.add_subplot(3, 2, 4)
    plot_2d(ax4, latent_grid, "Latent Grid (Z_grid)", color='purple')

    # (3,1) Reconstructed Original (3D)
    ax5 = fig.add_subplot(3, 2, 5, projection='3d')
    plot_3d(ax5, X_recon, "Reconstructed D(E(X))", color='orange')

    # (3,2) Re-encoded Latent Grid (2D)
    ax6 = fig.add_subplot(3, 2, 6)
    plot_2d(ax6, Z_latent_reencoded, "Re-encoded Latent E(D(Z_grid))", color='brown')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
