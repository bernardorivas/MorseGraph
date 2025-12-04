"""
Morse graph diagram visualization functions.

This module provides functions for plotting Morse graph structures, including
NetworkX-based plotting and Graphviz rendering.
"""

import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, Optional, Any
import io

from ..analysis import identify_attractors, identify_repellers
from .utils import get_morse_set_color


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

    node_colors = []
    for i, morse_set in enumerate(morse_sets):
        color = get_morse_set_color(morse_set, morse_graph, i, len(morse_sets))
        
        # Convert numpy floats to python floats to avoid pygraphviz warning
        if hasattr(color, '__iter__') and not isinstance(color, str):
            color = tuple(float(c) for c in color)

        node_colors.append(color)
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

    # Add attractor/repeller information to title if available
    try:
        attractors = identify_attractors(morse_graph)
        repellers = identify_repellers(morse_graph)
        title = f"Morse Graph ({len(attractors)} attractors, {len(repellers)} repellers)"
    except:
        title = "Morse Graph"
    
    ax.set_title(title)


def _get_graphviz_colormap_normalizer(cmap, num_verts):
    """Get colormap normalization function for graphviz."""
    try:
        num_colors = len(cmap.colors) if hasattr(cmap, 'colors') else 0
    except:
        num_colors = 0
    
    if num_colors > 0 and num_colors < num_verts:
        return lambda k: k % num_colors
    else:
        return matplotlib.colors.Normalize(vmin=0, vmax=num_verts-1)


def _get_vertex_color(v, morse_graph, node_to_id, cmap, cmap_norm):
    """Return vertex color as hex string for graphviz."""
    node_idx = node_to_id[v]
    clr = matplotlib.colors.to_hex(cmap(cmap_norm(node_idx)), keep_alpha=True)
    return str(clr)


def _get_vertex_label(v, morse_graph, node_to_id):
    """Return vertex label for graphviz."""
    if 'label' in morse_graph.nodes[v]:
        return str(morse_graph.nodes[v]['label'])
    return str(node_to_id[v])


def _format_graphviz_nodes(morse_graph, node_to_id, cmap, cmap_norm, shape, margin):
    """Format graphviz node definitions."""
    gv_nodes = []
    for v in morse_graph.nodes():
        node_id = node_to_id[v]
        label = _get_vertex_label(v, morse_graph, node_to_id)
        color = _get_vertex_color(v, morse_graph, node_to_id, cmap, cmap_norm)
        gv_nodes.append(
            f'  {node_id} [label="{label}", '
            f'shape={shape}, style=filled, fillcolor="{color}", '
            f'margin="{margin}"];'
        )
    return '\n'.join(gv_nodes)


def _format_graphviz_ranks(attractors, repellers, node_to_id):
    """Format graphviz rank constraints for attractors and repellers."""
    gv_ranks = []
    
    if attractors:
        attractor_ids = ' '.join(str(node_to_id[v]) for v in attractors)
        gv_ranks.append(f'  {{rank=same; {attractor_ids}}};')
    
    if repellers:
        repeller_ids = ' '.join(str(node_to_id[v]) for v in repellers)
        gv_ranks.append(f'  {{rank=same; {repeller_ids}}};')
    
    return '\n'.join(gv_ranks) if gv_ranks else ''


def _format_graphviz_edges(morse_graph, node_to_id):
    """Format graphviz edge definitions."""
    gv_edges = []
    for u, v in morse_graph.edges():
        gv_edges.append(f'  {node_to_id[u]} -> {node_to_id[v]};')
    return '\n'.join(gv_edges)


def morse_graph_to_graphviz_string(morse_graph: nx.DiGraph, 
                                   cmap=None, 
                                   shape='ellipse', 
                                   margin='0.11, 0.055') -> str:
    """
    Generate a Graphviz DOT string representation of a Morse graph.
    
    Enhanced with attractor/repeller identification and ranking.
    Adapted from CMGDB_utils.PlotMorseGraph_new (MIT License 2025 Marcio Gameiro)
    
    Args:
        morse_graph: NetworkX DiGraph representing the Morse graph
        cmap: Matplotlib colormap (default: tab10)
        shape: Node shape for graphviz (default: 'ellipse')
        margin: Node margin for graphviz (default: '0.11, 0.055')
    
    Returns:
        Graphviz DOT format string
    
    Example:
        >>> gv_string = morse_graph_to_graphviz_string(morse_graph)
        >>> with open('morse_graph.dot', 'w') as f:
        ...     f.write(gv_string)
    """
    if cmap is None:
        import matplotlib.pyplot as plt
        cmap = plt.colormaps.get_cmap('tab10')
    
    num_verts = morse_graph.number_of_nodes()
    
    # Identify attractors and repellers
    attractors = identify_attractors(morse_graph)
    repellers = identify_repellers(morse_graph)
    
    # Create mapping from nodes to stable IDs
    node_to_id = {v: i for i, v in enumerate(morse_graph.nodes())}
    
    # Get colormap normalizer
    cmap_norm = _get_graphviz_colormap_normalizer(cmap, num_verts)
    
    # Build graphviz string
    gv_parts = ['digraph {']
    gv_parts.append(_format_graphviz_nodes(morse_graph, node_to_id, cmap, cmap_norm, shape, margin))
    
    ranks = _format_graphviz_ranks(attractors, repellers, node_to_id)
    if ranks:
        gv_parts.append(ranks)
    
    gv_parts.append(_format_graphviz_edges(morse_graph, node_to_id))
    gv_parts.append('}')
    
    return '\n'.join(gv_parts) + '\n'


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
    except ImportError:
        raise ImportError("CMGDB is required for plot_morse_graph_diagram. Please install it.")

    if cmap is None:
        cmap = cm.cool

    from .utils import setup_figure_and_axes, finalize_plot
    
    fig, ax, should_close = setup_figure_and_axes(ax, figsize=figsize)

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

    finalize_plot(fig, ax, output_path, should_close)

