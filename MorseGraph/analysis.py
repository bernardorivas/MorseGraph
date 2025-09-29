'''
Graph analysis algorithms for the MorseGraph library.

This module provides functions to analyze the map graph computed by `core.Model`.
It includes functions to:
- Compute the Morse graph (condensation of the map graph).
- Compute the basins of attraction for attracting Morse sets.
- Perform iterative refinement for adaptive grid analysis.
'''

from typing import Dict, List, Set, Tuple, Optional
import networkx as nx

# Use try-except for type hinting to avoid circular imports
try:
    from .core import Model
except ImportError:
    from morsegraph.core import Model


def compute_morse_graph(map_graph: nx.DiGraph) -> Tuple[nx.DiGraph, Dict[int, Set[int]]]:
    """
    Computes the Morse graph from the map graph.

    The Morse graph is the condensation graph of the map graph, where each node
    represents a Strongly Connected Component (SCC).

    :param map_graph: A networkx.DiGraph representing the state transitions on the grid.
    :return: A tuple containing:
        - morse_graph (nx.DiGraph): The condensation DAG, where nodes are integers.
        - morse_sets (dict): A mapping from Morse node ID (int) to the set of
                             box indices it contains.
    """
    # 1. Find Strongly Connected Components (SCCs)
    sccs = list(nx.strongly_connected_components(map_graph))

    # 2. Get the condensation graph. Nodes in this graph are integers from 0 to k-1,
    #    corresponding to the index in the `sccs` list.
    morse_graph = nx.condensation(map_graph, sccs)

    # 3. Store the mapping from Morse node ID back to the boxes they contain.
    morse_sets = {i: set(scc) for i, scc in enumerate(sccs)}

    return morse_graph, morse_sets


def compute_basins_of_attraction(
    map_graph: nx.DiGraph, morse_graph: nx.DiGraph, morse_sets: Dict[int, Set[int]]
) -> Dict[int, Set[int]]:
    """
    Computes the basins of attraction for the attracting Morse sets (attractors).

    Attractors are nodes in the Morse graph with an out-degree of 0 (sinks).
    The basin of an attractor is the set of all boxes in the grid that eventually
    flow into that attractor.

    :param map_graph: The original map graph.
    :param morse_graph: The condensation graph (Morse graph).
    :param morse_sets: A mapping from Morse node ID to the set of boxes it contains.
    :return: A dictionary mapping attractor ID to its set of basin boxes.
    """
    # 1. Identify attractors (sinks in the Morse graph)
    attractors = [node for node in morse_graph.nodes() if morse_graph.out_degree(node) == 0]

    basins = {}
    # Optimization: Pre-calculate the reversed map graph for efficient backward search
    reversed_map_graph = map_graph.reverse(copy=True)

    for attractor_id in attractors:
        attractor_boxes = morse_sets.get(attractor_id, set())
        if not attractor_boxes:
            continue

        # The basin is the set of all nodes that can reach the attractor.
        # In the reversed graph, this is the set of all descendants of the attractor nodes.
        basin_boxes = set(attractor_boxes)
        for box in attractor_boxes:
            if box in reversed_map_graph:
                basin_boxes.update(nx.descendants(reversed_map_graph, box))

        basins[attractor_id] = basin_boxes

    return basins


def iterative_morse_computation(
    model: Model, iterations: int, adaptive: bool = True
) -> Tuple[nx.DiGraph, nx.DiGraph, Dict[int, Set[int]]]:
    """
    Iteratively computes the Morse graph by refining the grid.

    This is essential for adaptive grid strategies, where the grid is refined
    only in areas of complex dynamics (the recurrent set).

    :param model: The Model instance, which contains the dynamics and the grid.
    :param iterations: The number of refinement iterations to perform.
    :param adaptive: If True, subdivides only the recurrent set (adaptive strategy).
                     If False, attempts uniform subdivision of the entire grid.
    :return: A tuple containing the final map_graph, morse_graph, and morse_sets.
    """
    map_graph, morse_graph, morse_sets = (nx.DiGraph(), nx.DiGraph(), {})

    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...")
        # 1. Compute Map Graph and Morse Graph at the current resolution
        map_graph = model.compute_map_graph()
        morse_graph, morse_sets = compute_morse_graph(map_graph)

        # 2. Determine which boxes to subdivide for the next iteration
        if i < iterations - 1:
            indices_to_subdivide: Optional[List[int]] = None

            if adaptive:
                # Adaptive strategy: Subdivide all boxes in the recurrent set (any SCC)
                recurrent_boxes = [box for scc in morse_sets.values() for box in scc]
                if not recurrent_boxes:
                    print("No recurrent behavior found. Stopping iterations.")
                    break
                indices_to_subdivide = recurrent_boxes
            
            # 3. Subdivide the grid. If not adaptive, indices_to_subdivide is None,
            # and the grid should perform a uniform subdivision.
            grid_changed = model.grid.subdivide(indices_to_subdivide)

            if not grid_changed:
                print("Grid could not be further refined (e.g., max depth reached). Stopping iterations.")
                break

    # Return the final results after the last iteration or early exit
    return map_graph, morse_graph, morse_sets
