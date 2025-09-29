import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from typing import Dict, Set, FrozenSet

from .grids import AbstractGrid

def plot_morse_sets(grid: AbstractGrid, morse_graph_or_sets, ax: plt.Axes = None, **kwargs):
    """
    Plots the Morse sets on a grid.

    :param grid: The grid object used for the computation.
    :param morse_graph_or_sets: Either a networkx DiGraph (morse graph) or a set of frozensets containing box indices
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param kwargs: Additional keyword arguments to pass to the PatchCollection.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Handle both morse graph (nx.DiGraph) and set of frozensets
    if hasattr(morse_graph_or_sets, 'nodes'):
        # It's a networkx graph
        morse_sets = morse_graph_or_sets.nodes()
    else:
        # It's already a set of frozensets
        morse_sets = morse_graph_or_sets

    # Get all boxes from the grid
    all_boxes = grid.get_boxes()
    
    rects = []
    colors = []
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
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
        fig, ax = plt.subplots()

    colors = plt.cm.get_cmap('viridis', len(basins))
    
    for i, (attractor, basin) in enumerate(basins.items()):
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

def plot_morse_graph(morse_graph: nx.DiGraph, ax: plt.Axes = None, **kwargs):
    """
    Plots the Morse graph.

    :param morse_graph: The Morse graph to plot.
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param kwargs: Additional keyword arguments to pass to networkx.draw.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Create a mapping from frozenset to a shorter string representation
    node_labels = {node: str(i) for i, node in enumerate(morse_graph.nodes())}
    
    pos = nx.spring_layout(morse_graph)
    nx.draw(morse_graph, pos, ax=ax, labels=node_labels, **kwargs)
    plt.show()
