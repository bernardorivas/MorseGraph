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
    
    # Create a colormap for different Morse sets
    num_sets = len(morse_sets)
    if num_sets > 0:
        cmap = cm.get_cmap('tab10')
        
        for i, morse_set in enumerate(morse_sets):
            color = cmap(i / max(num_sets, 10))
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

def plot_basins_of_attraction(grid: AbstractGrid, basins: Dict[FrozenSet[int], Set[int]], ax: plt.Axes = None, **kwargs):
    """
    Plots the basins of attraction.

    :param grid: The grid object used for the computation.
    :param basins: A dictionary mapping attractors to their basins.
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param kwargs: Additional keyword arguments to pass to the PatchCollection.
    """
    if ax is None:
        _, ax = plt.subplots()

    colors = plt.cm.get_cmap('viridis', len(basins))
    
    for i, (_, basin) in enumerate(basins.items()):
        rects = []
        for box_index in basin:
            box = grid.get_boxes([box_index])[0]
            rect = Rectangle((box[0, 0], box[0, 1]), box[1, 0] - box[0, 0], box[1, 1] - box[0, 1])
            rects.append(rect)
        
        pc = PatchCollection(rects, facecolor=colors(i), **kwargs)
        ax.add_collection(pc)

    ax.set_xlim(grid.bounds[0, 0], grid.bounds[1, 0])
    ax.set_ylim(grid.bounds[0, 1], grid.bounds[1, 1])
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def plot_morse_graph(morse_graph: nx.DiGraph, ax: plt.Axes = None, 
                    morse_sets_colors: dict = None, node_size: int = 300,
                    arrowsize: int = 20, font_size: int = 8):
    """
    Plots the Morse graph with hierarchical layout.

    :param morse_graph: The Morse graph to plot.
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param morse_sets_colors: Optional dict mapping morse sets to colors for coordination with plot_morse_sets.
    :param node_size: Size of the nodes.
    :param arrowsize: Size of the arrow heads.
    :param font_size: Font size for node labels.
    """
    if ax is None:
        _, ax = plt.subplots()

    morse_sets = list(morse_graph.nodes())
    
    # Generate colors if not provided
    if morse_sets_colors is None:
        morse_sets_colors = {}
        num_sets = len(morse_sets)
        if num_sets > 0:
            cmap = cm.get_cmap('tab10')
            for i, morse_set in enumerate(morse_sets):
                color = cmap(i / max(num_sets, 10))
                morse_sets_colors[morse_set] = color
    
    # Create node colors list in the same order as morse_sets
    node_colors = [morse_sets_colors.get(morse_set, 'lightblue') for morse_set in morse_sets]
    
    # Create a mapping from frozenset to a shorter string representation
    node_labels = {node: str(i+1) for i, node in enumerate(morse_sets)}
    
    # Try hierarchical layout, fallback to spring layout
    try:
        from networkx.drawing.nx_agraph import pygraphviz_layout
        pos = pygraphviz_layout(morse_graph, prog='dot')
    except (ImportError, Exception):
        pos = nx.spring_layout(morse_graph, seed=42)
    
    # Draw the graph components
    nx.draw_networkx_nodes(morse_graph, pos, node_color=node_colors,
                          node_size=node_size, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(morse_graph, pos, edge_color='gray',
                          arrows=True, arrowsize=arrowsize, ax=ax, alpha=0.6)
    nx.draw_networkx_labels(morse_graph, pos, labels=node_labels,
                           font_size=font_size, ax=ax)
    
    ax.set_title("Morse Graph")
