'''
Visualization utilities for the MorseGraph library.

This module provides functions to plot:
- The Morse graph (Hasse diagram).
- The Morse sets (recurrent components) on the grid.
- The basins of attraction for attracting sets.
'''

from typing import Dict, List, Set, Optional, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Use try-except for type hinting to avoid circular imports
try:
    from .grids import AbstractGrid
except ImportError:
    from morsegraph.grids import AbstractGrid


def plot_morse_graph(morse_graph: nx.DiGraph, morse_sets: Dict[int, Set[int]], output_path: str = 'morse_graph.png') -> None:
    """
    Visualizes the Morse graph using pygraphviz and saves it to a file.

    Each node in the graph is labeled with its Morse set ID and the number of
    boxes it contains.

    :param morse_graph: A networkx.DiGraph representing the Morse graph.
    :param morse_sets: A mapping from Morse node ID to the set of boxes it contains.
    :param output_path: The path to save the output image file.
    """
    try:
        import pygraphviz as pgv
    except ImportError:
        print("Warning: pygraphviz is not installed. Cannot plot Morse graph.")
        print("Please install it via `pip install pygraphviz` (requires graphviz system library)." )
        return

    A = pgv.AGraph(directed=True, strict=True, rankdir='TB')

    for i, node_id in enumerate(morse_graph.nodes()):
        scc = morse_sets.get(node_id, set())
        node_label = f"M({node_id})\n({len(scc)} boxes)"
        A.add_node(node_id, label=node_label, shape="ellipse")

    for u, v in morse_graph.edges():
        A.add_edge(u, v)

    try:
        A.layout(prog='dot')
        A.draw(output_path)
        print(f"Morse graph saved to {output_path}")
    except Exception as e:
        print(f"Error drawing graph with pygraphviz: {e}")
        print("Please ensure the graphviz system package is installed (e.g., `brew install graphviz` or `sudo apt-get install graphviz`).")


def plot_morse_sets(
    grid: AbstractGrid,
    morse_sets: Dict[int, Set[int]],
    ax: Optional[plt.Axes] = None,
    cmap: str = 'viridis'
) -> None:
    """
    Plots the boxes corresponding to each Morse set in 2D.

    :param grid: The grid object used for the analysis.
    :param morse_sets: A mapping from Morse node ID to the set of boxes it contains.
    :param ax: A matplotlib axes object to plot on. If None, a new one is created.
    :param cmap: The colormap to use for coloring the Morse sets.
    """
    if grid.D != 2:
        raise ValueError("This plotting function only supports 2D grids.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.get_cmap(cmap, len(morse_sets))

    for i, (node_id, boxes) in enumerate(morse_sets.items()):
        color = colors(i)
        is_first_box = True
        for box_index in boxes:
            coords = grid.get_box_coordinates(box_index)
            x_min, y_min = coords[:, 0]
            width, height = coords[:, 1] - coords[:, 0]
            rect = Rectangle(
                (x_min, y_min), width, height,
                facecolor=color, alpha=0.7, edgecolor='k', linewidth=0.5
            )
            ax.add_patch(rect)

    ax.set_xlim(grid.bounds[0, :])
    ax.set_ylim(grid.bounds[1, :])
    ax.set_title("Morse Sets")
    ax.set_xlabel("State x_0")
    ax.set_ylabel("State x_1")
    ax.legend()


def plot_basins_of_attraction(
    grid: AbstractGrid,
    basins: Dict[int, Set[int]],
    attractor_sets: Dict[int, Set[int]],
    ax: Optional[plt.Axes] = None,
    cmap: str = 'viridis'
) -> None:
    """
    Plots the basins of attraction for each attractor in 2D.

    :param grid: The grid object.
    :param basins: A dictionary mapping attractor ID to its set of basin boxes.
    :param attractor_sets: A dictionary mapping attractor ID to its set of boxes (the Morse set).
    :param ax: A matplotlib axes object to plot on. If None, a new one is created.
    :param cmap: The colormap to use.
    """
    if grid.D != 2:
        raise ValueError("This plotting function only supports 2D grids.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.get_cmap(cmap, len(basins))

    # Plot basins with low alpha
    for i, (attractor_id, basin_boxes) in enumerate(basins.items()):
        color = colors(i)
        is_first_box = True
        for box_index in basin_boxes:
            coords = grid.get_box_coordinates(box_index)
            x_min, y_min = coords[:, 0]
            width, height = coords[:, 1] - coords[:, 0]
            rect = Rectangle(
                (x_min, y_min), width, height,
                facecolor=color, alpha=0.3, edgecolor=None
            )
            ax.add_patch(rect)

    # Plot attractors on top with higher alpha
    plot_morse_sets(grid, attractor_sets, ax=ax, cmap=cmap)

    ax.set_title("Basins of Attraction")
    ax.legend()
