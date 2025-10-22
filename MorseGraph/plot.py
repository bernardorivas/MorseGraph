import numpy as np
import networkx as nx
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
            # Use color from node attribute
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
                               marker_size_min: float = 1.0, marker_size_max: float = 100.0,
                               data_overlay: np.ndarray = None):
    """
    Plots the Morse sets from a CMGDB MorseGraph object on a 3D grid using a scatter plot of box centers.
    The size of each marker is proportional to the size of the corresponding box.

    :param morse_graph: The CMGDB.MorseGraph object.
    :param domain_bounds: [[lower_x, lower_y, lower_z], [upper_x, upper_y, upper_z]]
    :param output_path: Path to save figure (if None, displays instead)
    :param title: Plot title
    :param labels: Optional dict for axis labels, e.g. {'x': 'log(M)', 'y': 'log(A)', 'z': 'log(D)'}
    :param equilibria: Optional dict of equilibrium points to plot, e.g. {'EQ1': [x,y,z]}
    :param periodic_orbits: Optional dict of periodic orbits to plot, e.g. {'Period-12': array([[x,y,z], ...])}
    :param marker_size_min: Minimum marker size for the smallest boxes.
    :param marker_size_max: Maximum marker size for the largest boxes.
    :param data_overlay: Optional data to overlay (N x 3 array)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    num_morse_sets = morse_graph.num_vertices()
    cmap = cm.cool
    colors = [cmap(i / max(num_morse_sets - 1, 1)) for i in range(num_morse_sets)]

    all_box_sizes = []
    all_boxes_by_morse_set = []
    for morse_idx in range(num_morse_sets):
        boxes = morse_graph.morse_set_boxes(morse_idx)
        all_boxes_by_morse_set.append(boxes)
        if boxes:
            for box in boxes:
                dim = len(box) // 2
                size = np.mean([box[d + dim] - box[d] for d in range(dim)])
                all_box_sizes.append(size)
    
    min_box_size = min(all_box_sizes) if all_box_sizes else 0
    max_box_size = max(all_box_sizes) if all_box_sizes else 1

    def get_marker_size(box_size):
        if max_box_size == min_box_size:
            return marker_size_min
        # linear scaling
        return marker_size_min + (marker_size_max - marker_size_min) * (box_size - min_box_size) / (max_box_size - min_box_size)

    for morse_idx, boxes in enumerate(all_boxes_by_morse_set):
        if boxes:
            dim = len(boxes[0]) // 2
            centers = np.array([[(b[d] + b[d+dim]) / 2.0 for d in range(dim)] for b in boxes])
            sizes = [get_marker_size(np.mean([b[d + dim] - b[d] for d in range(dim)])) for b in boxes]
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                      c=[colors[morse_idx]], s=sizes, alpha=0.4, marker='s',
                      label=f'Morse Set {morse_idx}')

    if data_overlay is not None and len(data_overlay) > 0:
        ax.scatter(data_overlay[:, 0], data_overlay[:, 1], data_overlay[:, 2], c='black', s=1, alpha=0.1, label='Data')

    if equilibria:
        for name, point in equilibria.items():
            ax.scatter(point[0], point[1], point[2],
                       c='red', marker='*', s=200,
                       label=name, zorder=100)
    
    if periodic_orbits:
        for orbit_name, orbit_points in periodic_orbits.items():
            if orbit_points is not None and len(orbit_points) > 0:
                # Plot orbit points as circles
                ax.scatter(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2],
                          c='orange', marker='o', s=80,
                          label=orbit_name, zorder=90, alpha=0.9)
                
                # Connect points with lines to show the orbit
                # Close the orbit by connecting last point back to first
                orbit_closed = np.vstack([orbit_points, orbit_points[0:1]])
                ax.plot(orbit_closed[:, 0], orbit_closed[:, 1], orbit_closed[:, 2],
                       'orange', alpha=0.6, zorder=85)

    ax.set_xlim(domain_bounds[0][0], domain_bounds[1][0])
    ax.set_ylim(domain_bounds[0][1], domain_bounds[1][1])
    ax.set_zlim(domain_bounds[0][2], domain_bounds[1][2])

    ax.set_xlabel(labels.get('x', 'X') if labels else 'X')
    ax.set_ylabel(labels.get('y', 'Y') if labels else 'Y')
    ax.set_zlabel(labels.get('z', 'Z') if labels else 'Z')
    ax.set_title(title)
    ax.legend()

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
        fig, ax = plt.subplots(figsize=(8, 8))
        should_close = True
    else:
        fig = ax.get_figure()
        should_close = False

    # Plot data points
    ax.scatter(z_data[:, 0], z_data[:, 1], c='lightblue', s=1, alpha=0.5, label='Data')

    # Plot Morse sets if provided
    if morse_graph is not None:
        try:
            num_morse_sets = morse_graph.num_vertices()
            cmap = cm.viridis
            colors = [cmap(i / max(num_morse_sets - 1, 1)) for i in range(num_morse_sets)]

            for i in range(num_morse_sets):
                boxes = morse_graph.morse_set_boxes(i)
                if boxes:
                    # Add a label for the first box of each Morse set
                    label = f'Morse Set {i}'
                    for j, box in enumerate(boxes):
                        rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                       facecolor=colors[i], edgecolor='none', alpha=0.5,
                                       label=label if j == 0 else None)
                        ax.add_patch(rect)
        except AttributeError:
            # morse_graph might be NetworkX, not CMGDB
            pass

    # Plot encoded 3D barycenters if provided
    if barycenters_latent:
        # Get colors matching the Morse sets if morse_graph exists
        if morse_graph is not None:
            try:
                num_morse_sets = morse_graph.num_vertices()
                cmap = cm.viridis
                colors = [cmap(i / max(num_morse_sets - 1, 1)) for i in range(num_morse_sets)]
            except AttributeError:
                colors = None
        else:
            colors = None

        for morse_set_idx, barys in barycenters_latent.items():
            if barys:
                barys_array = np.array(barys)
                color = colors[morse_set_idx] if colors and morse_set_idx < len(colors) else 'orange'
                ax.scatter(barys_array[:, 0], barys_array[:, 1],
                          marker='X', s=100, c=[color],
                          label=f'3D Barycenter {morse_set_idx}' if morse_set_idx < 5 else None,
                          zorder=9)

    if equilibria_latent:
        for name, point in equilibria_latent.items():
            ax.plot(point[0], point[1],
                   marker='*', markersize=15, color='red',
                   label=name, zorder=10, linestyle='none')

    ax.set_xlim(latent_bounds[0][0], latent_bounds[1][0])
    ax.set_ylim(latent_bounds[0][1], latent_bounds[1][1])
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()

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


def plot_morse_sets_3d_cmgdb(morse_graph, domain_bounds, output_path=None, title="Morse Sets (3D)",
                             alpha: float = 0.3, labels: Dict[str, str] = None, **kwargs):
    """
    Plots the Morse sets from a CMGDB MorseGraph object on a 3D grid.

    :param morse_graph: The CMGDB.MorseGraph object.
    :param domain_bounds: [[lower_x, lower_y, lower_z], [upper_x, upper_y, upper_z]]
    :param output_path: Path to save figure (if None, displays instead)
    :param title: Plot title
    :param alpha: Transparency level for the boxes (0.0-1.0).
    :param labels: Optional dict for axis labels, e.g. {'x': 'log(M)', 'y': 'log(A)', 'z': 'log(D)'}
    :param kwargs: Additional keyword arguments to pass to Poly3DCollection.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def box_to_cuboid_faces(box):
        x_min, y_min, z_min = box[0], box[1], box[2]
        x_max, y_max, z_max = box[3], box[4], box[5]

        vertices = [
            [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max],
        ]
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]], [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]], [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]], [vertices[1], vertices[2], vertices[6], vertices[5]],
        ]
        return faces

    num_morse_sets = morse_graph.num_vertices()
    cmap = cm.cool
    colors = [cmap(i / max(num_morse_sets - 1, 1)) for i in range(num_morse_sets)]

    for morse_idx in range(num_morse_sets):
        boxes = morse_graph.morse_set_boxes(morse_idx)
        if boxes:
            faces = []
            for box in boxes:
                faces.extend(box_to_cuboid_faces(box))
            
            if faces:
                poly3d = Poly3DCollection(faces, facecolors=colors[morse_idx], alpha=alpha,
                                         edgecolors='none', **kwargs)
                ax.add_collection3d(poly3d)

    ax.set_xlim(domain_bounds[0][0], domain_bounds[1][0])
    ax.set_ylim(domain_bounds[0][1], domain_bounds[1][1])
    ax.set_zlim(domain_bounds[0][2], domain_bounds[1][2])

    ax.set_xlabel(labels.get('x', 'X') if labels else 'X')
    ax.set_ylabel(labels.get('y', 'Y') if labels else 'Y')
    ax.set_zlabel(labels.get('z', 'Z') if labels else 'Z')
    ax.set_title(title)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# =============================================================================
# Encoder/Decoder Round-Trip Analysis
# =============================================================================

def plot_encoder_decoder_roundtrip(
    encoder,
    decoder,
    device,
    original_bounds,
    latent_bounds,
    n_grid_points=50,
    output_path=None,
    title_prefix="",
    labels=None
):
    """
    Visualize encoder/decoder round-trip transformations.

    Creates a 2x3 multi-panel figure showing:
    - Top row: Original space (3D)
      - Uniform grid sample in original space
      - Decoded latent grid: D(latent_grid)
      - Round-trip: D(E(grid_sample))
    - Bottom row: Latent space (2D)
      - Encoded grid sample: E(grid_sample)
      - Regular latent grid
      - Re-encoded decoded grid: E(D(latent_grid))

    This visualization helps assess:
    - Encoder/decoder reconstruction quality
    - Coverage of latent space
    - Consistency of round-trip transformations

    Args:
        encoder: PyTorch encoder model (input_dim -> latent_dim)
        decoder: PyTorch decoder model (latent_dim -> input_dim)
        device: PyTorch device
        original_bounds: [[x_min, y_min, z_min], [x_max, y_max, z_max]] for original space
        latent_bounds: [[z0_min, z1_min], [z0_max, z1_max]] for latent space
        n_grid_points: Number of grid points per dimension
        output_path: Path to save figure (if None, displays instead)
        title_prefix: Prefix for subplot titles (e.g., "Ives Model - ")
        labels: Dict with keys 'x', 'y', 'z' for axis labels (optional)

    Returns:
        Dictionary with generated samples:
            - 'grid_sample_3d': Uniform grid in original space
            - 'encoded_grid': E(grid_sample)
            - 'roundtrip_3d': D(E(grid_sample))
            - 'latent_grid': Regular grid in latent space
            - 'decoded_grid': D(latent_grid)
            - 'reencoded_grid': E(D(latent_grid))

    Example:
        >>> samples = plot_encoder_decoder_roundtrip(
        ...     encoder, decoder, device,
        ...     original_bounds=[[-1, -4, -1], [2, 1, 1]],
        ...     latent_bounds=[[-3, -3], [3, 3]],
        ...     output_path='results/roundtrip.png',
        ...     title_prefix='Ives Model - ',
        ...     labels={'x': 'log(M)', 'y': 'log(A)', 'z': 'log(D)'}
        ... )
    """
    import torch
    from mpl_toolkits.mplot3d import Axes3D

    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}

    # Generate uniform grid in original 3D space
    x = np.linspace(original_bounds[0][0], original_bounds[1][0], n_grid_points)
    y = np.linspace(original_bounds[0][1], original_bounds[1][1], n_grid_points)
    z = np.linspace(original_bounds[0][2], original_bounds[1][2], n_grid_points)
    xx, yy, zz = np.meshgrid(x, y, z)
    grid_sample_3d = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # Generate regular grid in latent 2D space
    z0 = np.linspace(latent_bounds[0][0], latent_bounds[1][0], n_grid_points)
    z1 = np.linspace(latent_bounds[0][1], latent_bounds[1][1], n_grid_points)
    z0_grid, z1_grid = np.meshgrid(z0, z1)
    latent_grid = np.stack([z0_grid.ravel(), z1_grid.ravel()], axis=1)

    # Compute transformations
    with torch.no_grad():
        # Original -> Latent -> Original
        grid_tensor = torch.FloatTensor(grid_sample_3d).to(device)
        encoded_grid = encoder(grid_tensor).cpu().numpy()
        roundtrip_3d = decoder(torch.FloatTensor(encoded_grid).to(device)).cpu().numpy()

        # Latent -> Original -> Latent
        latent_tensor = torch.FloatTensor(latent_grid).to(device)
        decoded_grid = decoder(latent_tensor).cpu().numpy()
        reencoded_grid = encoder(torch.FloatTensor(decoded_grid).to(device)).cpu().numpy()

    # Create figure
    fig = plt.figure(figsize=(18, 12))

    # Top row: 3D visualizations
    # Panel 1: Original grid sample
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(grid_sample_3d[:, 0], grid_sample_3d[:, 1], grid_sample_3d[:, 2],
                c='blue', alpha=0.3, s=1)
    ax1.set_xlabel(labels['x'])
    ax1.set_ylabel(labels['y'])
    ax1.set_zlabel(labels['z'])
    ax1.set_title(f'{title_prefix}Original Grid Sample')

    # Panel 2: Decoded latent grid
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(decoded_grid[:, 0], decoded_grid[:, 1], decoded_grid[:, 2],
                c='green', alpha=0.3, s=1)
    ax2.set_xlabel(labels['x'])
    ax2.set_ylabel(labels['y'])
    ax2.set_zlabel(labels['z'])
    ax2.set_title(f'{title_prefix}D(Latent Grid)')

    # Panel 3: Round-trip
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.scatter(roundtrip_3d[:, 0], roundtrip_3d[:, 1], roundtrip_3d[:, 2],
                c='red', alpha=0.3, s=1)
    ax3.set_xlabel(labels['x'])
    ax3.set_ylabel(labels['y'])
    ax3.set_zlabel(labels['z'])
    ax3.set_title(f'{title_prefix}D(E(Grid Sample))')

    # Bottom row: 2D latent space visualizations
    # Panel 4: Encoded grid
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(encoded_grid[:, 0], encoded_grid[:, 1], c='blue', alpha=0.3, s=1)
    ax4.set_xlabel('Latent Dim 0')
    ax4.set_ylabel('Latent Dim 1')
    ax4.set_title(f'{title_prefix}E(Grid Sample)')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Latent grid
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(latent_grid[:, 0], latent_grid[:, 1], c='green', alpha=0.3, s=1)
    ax5.set_xlabel('Latent Dim 0')
    ax5.set_ylabel('Latent Dim 1')
    ax5.set_title(f'{title_prefix}Latent Grid')
    ax5.grid(True, alpha=0.3)

    # Panel 6: Re-encoded
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(reencoded_grid[:, 0], reencoded_grid[:, 1], c='red', alpha=0.3, s=1)
    ax6.set_xlabel('Latent Dim 0')
    ax6.set_ylabel('Latent Dim 1')
    ax6.set_title(f'{title_prefix}E(D(Latent Grid))')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return {
        'grid_sample_3d': grid_sample_3d,
        'encoded_grid': encoded_grid,
        'roundtrip_3d': roundtrip_3d,
        'latent_grid': latent_grid,
        'decoded_grid': decoded_grid,
        'reencoded_grid': reencoded_grid
    }


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
            return mg.num_vertices(), len(mg.edges())
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
                       c='red', s=200, marker='*',
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


def _plot_morse_sets_rectangles(ax, morse_graph, cmap_colors, alpha=0.5, zorder=2):
    """
    Plot morse sets as rectangles - faithful to actual box bounds.
    
    This approach directly draws each box as a Rectangle patch in data coordinates,
    ensuring the visual representation exactly matches the computational structure.
    
    Args:
        ax: Matplotlib axis to plot on
        morse_graph: CMGDB MorseGraph object
        cmap_colors: List of colors for each morse set
        alpha: Transparency of rectangles
        zorder: Z-order for layering
    """
    from matplotlib.patches import Rectangle
    
    num_morse_sets = morse_graph.num_vertices()
    
    for morse_idx in range(num_morse_sets):
        boxes = morse_graph.morse_set_boxes(morse_idx)
        if not boxes:
            continue
            
        for box in boxes:
            # Box format for 2D: [xmin, xmax, ymin, ymax]
            rect = Rectangle(
                (box[0], box[2]),           # lower-left corner (xmin, ymin)
                box[1] - box[0],            # width
                box[3] - box[2],            # height
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
        for name, point in equilibria.items():
            ax3.scatter(point[0], point[1], point[2],
                       c='red', marker='*', s=300,
                       label=name, zorder=100, alpha=0.95)
    
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

    # Plot Morse sets using rectangle patches
    node_colors_2d = [cmap_2d(i / max(num_morse_sets_2d - 1, 1))
                      for i in range(num_morse_sets_2d)]

    _plot_morse_sets_rectangles(
        ax4, morse_graph_2d, node_colors_2d,
        alpha=0.5, zorder=2
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


def plot_latent_space_flexible(
    z_data=None,
    morse_graph=None,
    latent_bounds=None,
    barycenters_latent=None,
    equilibrium_latent=None,
    period12_latent=None,
    output_path=None,
    title="Latent Space",
    show_data=False,
    show_morse_sets=False,
    show_barycenters=False,
    show_equilibrium=False,
    show_period12=False,
    barycenter_size=6,
    cmap_morse_sets='viridis',
    cmap_barycenters='cool'
):
    """
    Flexible latent space visualization with multiple display options.
    Uses rectangle patches for morse sets.

    Args:
        z_data: Latent space data points (N x 2)
        morse_graph: CMGDB MorseGraph object
        latent_bounds: [[xmin, ymin], [xmax, ymax]]
        barycenters_latent: Dict mapping morse_set_id -> array of barycenter points
        equilibrium_latent: Equilibrium point in latent space [x, y]
        period12_latent: Period-12 orbit points in latent space (12 x 2)
        output_path: Path to save figure
        title: Plot title
        show_data: Show gray data scatter
        show_morse_sets: Show morse sets (rectangle patches)
        show_barycenters: Show E(3D barycenters)
        show_equilibrium: Show equilibrium point
        show_period12: Show E(period-12 orbit)
        barycenter_size: Marker size for barycenters (4-9 recommended)
        cmap_morse_sets: Colormap name for 2D Morse sets (default: 'viridis')
        cmap_barycenters: Colormap name for E(3D barycenters) (default: 'cool')
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(10, 10))

    cmap_sets = getattr(cm, cmap_morse_sets)
    cmap_barys = getattr(cm, cmap_barycenters)
    
    # Determine number of morse sets for coloring
    if morse_graph is not None:
        num_morse_sets = morse_graph.num_vertices()
    elif barycenters_latent is not None:
        num_morse_sets = len(barycenters_latent)
    else:
        num_morse_sets = 1

    # Create color maps for both morse sets and barycenters
    num_colors = max(num_morse_sets, 1)
    colors_morse_sets = [cmap_sets(i / max(num_colors - 1, 1)) for i in range(num_colors)]
    colors_barycenters = [cmap_barys(i / max(num_colors - 1, 1)) for i in range(num_colors)]

    # 1. Show data (background, if requested)
    if show_data and z_data is not None:
        ax.scatter(z_data[:, 0], z_data[:, 1],
                  c='lightgray', s=1, alpha=0.3, rasterized=True, zorder=1)

    # 2. Show morse sets using rectangle patches (2D computed - viridis)
    if show_morse_sets and morse_graph is not None:
        _plot_morse_sets_rectangles(
            ax, morse_graph, colors_morse_sets,
            alpha=0.5, zorder=2
        )

    # 3. Show E(3D barycenters) (from 3D - cool colormap)
    if show_barycenters and barycenters_latent is not None:
        for morse_idx, barys in barycenters_latent.items():
            if len(barys) > 0:
                barys_array = np.array(barys)
                color_idx = morse_idx if morse_idx < len(colors_barycenters) else 0
                ax.scatter(barys_array[:, 0], barys_array[:, 1],
                          c=[colors_barycenters[color_idx]], s=barycenter_size,
                          alpha=0.8, marker='o', zorder=4,
                          label=f'E(Barycenters {morse_idx})')
    
    # 4. Show equilibrium
    if show_equilibrium and equilibrium_latent is not None:
        ax.scatter(equilibrium_latent[0], equilibrium_latent[1],
                  c='red', marker='*', s=200, alpha=0.95,
                  label='Equilibrium', zorder=5)
    
    # 5. Show E(period-12 orbit) with connecting lines
    if show_period12 and period12_latent is not None and len(period12_latent) > 0:
        # Plot orbit points
        ax.scatter(period12_latent[:, 0], period12_latent[:, 1],
                  c='orange', marker='o', s=80, alpha=0.9,
                  label='E(Period-12)', zorder=6)
        
        # Connect with lines (close the loop)
        orbit_closed = np.vstack([period12_latent, period12_latent[0:1]])
        ax.plot(orbit_closed[:, 0], orbit_closed[:, 1],
               'orange', linewidth=2.5, alpha=0.7, zorder=5)
    
    # Set limits
    if latent_bounds is not None:
        ax.set_xlim(latent_bounds[0][0], latent_bounds[1][0])
        ax.set_ylim(latent_bounds[0][1], latent_bounds[1][1])
    
    ax.set_xlabel('Latent Dim 0', fontsize=12)
    ax.set_ylabel('Latent Dim 1', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    
    # Add legend if any elements shown
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0 and len(handles) <= 8:  # Don't show legend if too many items
        ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


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
