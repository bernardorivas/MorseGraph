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
            linewidth=0.5,
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
            edgecolor='lightgray',
            linewidth=0.5
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
                fc='red', ec='red', alpha=0.3, linewidth=0.5)

    ax.set_xlim(grid.bounds[0, 0], grid.bounds[1, 0])
    ax.set_ylim(grid.bounds[0, 1], grid.bounds[1, 1])
    ax.set_aspect('equal')
    ax.set_title('Data Points and Transitions')
    ax.legend()


# =============================================================================
# Generalized Visualization Functions for 3D Pipeline
# =============================================================================

def plot_morse_graph_diagram(morse_graph, output_path=None, title="Morse Graph", cmap=None, figsize=(8, 8)):
    """
    Plot Morse graph structure using CMGDB's graphviz plotting.

    Args:
        morse_graph: CMGDB MorseGraph object
        output_path: Path to save figure (if None, displays instead)
        title: Plot title
        cmap: Matplotlib colormap for nodes (default: cm.cool)
        figsize: Figure size tuple

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

    fig, ax = plt.subplots(figsize=figsize)

    if morse_graph is None or morse_graph.num_vertices() == 0:
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

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
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
            ax.plot(epochs, train_losses[key], label='Train', linewidth=2)
            ax.plot(epochs, val_losses[key], label='Val', linewidth=2)
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
                       c='red', marker='*', s=100, edgecolors='black',
                       label=name, zorder=100)

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
                         equilibria_latent: Dict[str, np.ndarray] = None):
    """
    Plot 2D latent space with data points and optionally Morse sets.

    Args:
        z_data: Encoded data points in latent space (N x 2 array)
        latent_bounds: [[lower_x, lower_y], [upper_x, upper_y]]
        morse_graph: Optional CMGDB MorseGraph to overlay Morse sets
        output_path: Path to save figure (if None, displays instead)
        title: Plot title
        equilibria_latent: Optional dict of equilibrium points in latent space to plot, e.g. {'EQ1': [z1,z2]}

    Example:
        >>> plot_latent_space_2d(z_train, latent_bounds, morse_graph, 'latent_space.png',
        ...                      equilibria_latent={'EQ': [0.1, 0.2]})
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot data points
    ax.scatter(z_data[:, 0], z_data[:, 1], c='lightblue', s=1, alpha=0.5, label='Data')

    # Plot Morse sets if provided
    if morse_graph is not None:
        num_morse_sets = morse_graph.num_vertices()
        cmap = cm.cool
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

    if equilibria_latent:
        for name, point in equilibria_latent.items():
            ax.plot(point[0], point[1],
                   marker='*', markersize=15, color='red',
                   markeredgecolor='black',
                   label=name, zorder=10, linestyle='none')

    ax.set_xlim(latent_bounds[0][0], latent_bounds[1][0])
    ax.set_ylim(latent_bounds[0][1], latent_bounds[1][1])
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
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