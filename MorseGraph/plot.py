import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, Set, FrozenSet

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
        if 'color' in morse_graph.nodes[morse_set]:
            # Use color from node attribute
            node_colors.append(morse_graph.nodes[morse_set]['color'])
        else:
            # Generate color for backward compatibility
            num_sets = len(morse_sets)
            cmap = cm.get_cmap('tab10')
            color = cmap(morse_sets.index(morse_set) / max(num_sets, 10))
            node_colors.append(color)
    
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
