import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from typing import Dict, Set, FrozenSet

from .grids import AbstractGrid

def plot_morse_sets(grid: AbstractGrid, morse_graph: nx.DiGraph, ax: plt.Axes = None, **kwargs):
    """
    Plots the Morse sets on a grid.

    :param grid: The grid object used for the computation.
    :param morse_graph: The Morse graph (NetworkX DiGraph) containing the Morse sets as nodes.
                       Each node should have a 'color' attribute (assigned by compute_morse_graph).
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
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

    ax.set_xlim(grid.bounds[0, 0], grid.bounds[1, 0])
    ax.set_ylim(grid.bounds[0, 1], grid.bounds[1, 1])
    ax.set_aspect('equal', adjustable='box')

def plot_basins_of_attraction(grid: AbstractGrid, basins,
                             morse_graph: nx.DiGraph = None, ax: plt.Axes = None,
                             show_outside: bool = False, **kwargs):
    """
    Plots the basins of attraction with colors matching the Morse sets.

    :param grid: The grid object used for the computation.
    :param basins: Either:
                   - Dict[FrozenSet[int], Set[int]]: box-level basins (from compute_all_morse_set_basins)
                   - Dict[FrozenSet[int], Set[FrozenSet[int]]]: Morse-level basins (deprecated)
    :param morse_graph: The Morse graph with color attributes. If provided, uses colors from node attributes.
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param show_outside: If True, paint boxes not in any basin black (boxes mapping outside domain).
    :param kwargs: Additional keyword arguments to pass to Rectangle patches.
    """
    if ax is None:
        _, ax = plt.subplots()

    all_boxes = grid.get_boxes()

    # Detect basin type: check if values are sets of frozensets (Morse basins) or sets of ints (box basins)
    first_basin = next(iter(basins.values()))
    is_morse_basin = len(first_basin) > 0 and isinstance(next(iter(first_basin)), frozenset)

    # Plot each basin
    for attractor, basin in basins.items():
        # Get color from morse_graph node attributes if available
        if morse_graph and 'color' in morse_graph.nodes[attractor]:
            color = morse_graph.nodes[attractor]['color']
        else:
            # Fallback to viridis colormap for backward compatibility
            colors_cmap = plt.cm.get_cmap('viridis', len(basins))
            color = colors_cmap(list(basins.keys()).index(attractor))

        if is_morse_basin:
            # Morse-level basins: basin is a set of Morse sets (frozensets)
            # Color all boxes in each Morse set, with attractor at full opacity
            for morse_set in basin:
                alpha = 1.0 if morse_set == attractor else 0.7
                for box_index in morse_set:
                    if box_index < len(all_boxes):
                        box = all_boxes[box_index]
                        rect = Rectangle((box[0, 0], box[0, 1]),
                                       box[1, 0] - box[0, 0],
                                       box[1, 1] - box[0, 1],
                                       facecolor=color,
                                       edgecolor='none',
                                       alpha=alpha, **kwargs)
                        ax.add_patch(rect)
        else:
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
        if is_morse_basin:
            # For Morse basins, collect all box indices from all Morse sets in all basins
            basin_box_indices = set()
            for basin_morse_sets in basins.values():
                for morse_set in basin_morse_sets:
                    basin_box_indices.update(morse_set)
        else:
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
    :param morse_sets_colors: Optional dict mapping morse sets to colors. If None, uses colors
                             from node attributes. Deprecated - colors should come from graph.
    :param node_size: Size of the nodes.
    :param arrowsize: Size of the arrow heads.
    :param font_size: Font size for node labels.
    """
    if ax is None:
        _, ax = plt.subplots()

    morse_sets = list(morse_graph.nodes())

    # Use colors from node attributes first, fallback to provided dict or generate
    node_colors = []
    for morse_set in morse_sets:
        if 'color' in morse_graph.nodes[morse_set]:
            # Use color from node attribute (preferred)
            node_colors.append(morse_graph.nodes[morse_set]['color'])
        elif morse_sets_colors and morse_set in morse_sets_colors:
            # Fallback to provided color dict (deprecated)
            node_colors.append(morse_sets_colors[morse_set])
        else:
            # Last resort: generate color (backward compatibility)
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
