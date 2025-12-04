"""
Morse graph analysis utilities.

This module provides functions for analyzing Morse graphs, including counting
attractors, extracting edge sets, and computing similarity metrics.
"""

from typing import Dict, Set, Tuple, Union


def count_attractors(morse_graph) -> int:
    """
    Count the number of attractors (sinks) in a Morse graph.

    An attractor is a node with no outgoing edges (out-degree = 0).
    Works with both CMGDB.MorseGraph and NetworkX DiGraph objects.

    Args:
        morse_graph: A Morse graph object (CMGDB.MorseGraph or NetworkX DiGraph)

    Returns:
        Number of attractor nodes. Returns 0 if graph is None or empty.

    Example:
        >>> num_attractors = count_attractors(morse_graph)
        >>> print(f"Found {num_attractors} attractors")
    """
    if morse_graph is None:
        return 0

    # Handle CMGDB.MorseGraph
    if hasattr(morse_graph, 'num_vertices'):
        return sum(1 for v in range(morse_graph.num_vertices())
                   if len(morse_graph.adjacencies(v)) == 0)
    
    # Handle NetworkX DiGraph
    if hasattr(morse_graph, 'nodes'):
        return sum(1 for node in morse_graph.nodes()
                   if morse_graph.out_degree(node) == 0)
    
    # Fallback for older interface with .vertices() method
    if hasattr(morse_graph, 'vertices'):
        vertices = morse_graph.vertices()
        if not vertices:
            return 0
        return sum(1 for v in vertices if not morse_graph.adjacencies(v))
    
    return 0


def extract_edge_set(morse_graph) -> Set[Tuple[int, int]]:
    """
    Extract the set of directed edges from a CMGDB Morse graph.

    Args:
        morse_graph: A CMGDB Morse graph object with methods:
                     - num_vertices() -> int
                     - adjacencies(vertex_id) -> list of adjacent vertex IDs

    Returns:
        Set of (source, target) tuples representing directed edges

    Example:
        >>> edges = extract_edge_set(morse_graph)
        >>> print(edges)
        {(3, 2), (2, 0), (2, 1)}
    """
    if morse_graph is None:
        return set()

    edges = set()
    num_vertices = morse_graph.num_vertices()

    for v in range(num_vertices):
        # Get all vertices that v has edges to
        adjacent_vertices = morse_graph.adjacencies(v)
        for target in adjacent_vertices:
            edges.add((v, target))

    return edges


def compute_similarity_vector(
    morse_graph,
    ground_truth: Dict[str, Union[int, Set[Tuple[int, int]]]]
) -> Dict[str, Union[int, float]]:
    """
    Compute a multi-dimensional topological similarity vector comparing
    a learned Morse graph to a ground truth structure.

    This function provides a quantitative alternative to visual comparison,
    measuring similarity across multiple topological features:
    - Number of attractors (sinks)
    - Total number of nodes (Morse sets)
    - Edge connectivity structure

    Args:
        morse_graph: A CMGDB Morse graph object (can be None)
        ground_truth: Dictionary containing:
            - 'num_nodes': Expected number of nodes (int)
            - 'edges': Expected edge set as {(source, target), ...} (set)
            - 'num_attractors': Expected number of attractors (int)

    Returns:
        Dictionary with keys:
            - 'attractor_diff': |attractors_learned - attractors_true|
            - 'node_diff': |nodes_learned - nodes_true|
            - 'connection_diff': Symmetric difference count |E_learned Δ E_true|
            - 'attractors_learned': Number of attractors in learned graph
            - 'attractors_true': Number of attractors in ground truth
            - 'nodes_learned': Number of nodes in learned graph
            - 'nodes_true': Number of nodes in ground truth
            - 'edges_learned': Set of edges in learned graph
            - 'edges_true': Set of edges in ground truth
            - 'missing_edges': Edges in ground truth but not learned
            - 'extra_edges': Edges learned but not in ground truth

    Example:
        >>> ground_truth = {
        ...     'num_nodes': 4,
        ...     'edges': {(3, 2), (2, 0), (2, 1)},
        ...     'num_attractors': 2
        ... }
        >>> similarity = compute_similarity_vector(morse_graph, ground_truth)
        >>> print(f"Node difference: {similarity['node_diff']}")
        >>> print(f"Connection difference: {similarity['connection_diff']}")
    """
    # Extract learned graph properties
    if morse_graph is None:
        nodes_learned = 0
        attractors_learned = 0
        edges_learned = set()
    else:
        nodes_learned = morse_graph.num_vertices()
        attractors_learned = count_attractors(morse_graph)
        edges_learned = extract_edge_set(morse_graph)

    # Extract ground truth properties
    nodes_true = ground_truth.get('num_nodes', 0)
    attractors_true = ground_truth.get('num_attractors', 0)
    edges_true = ground_truth.get('edges', set())

    # Compute differences
    node_diff = abs(nodes_learned - nodes_true)
    attractor_diff = abs(attractors_learned - attractors_true)

    # Edge set symmetric difference
    missing_edges = edges_true - edges_learned  # In ground truth but not learned
    extra_edges = edges_learned - edges_true     # Learned but not in ground truth
    connection_diff = len(missing_edges) + len(extra_edges)

    return {
        'attractor_diff': attractor_diff,
        'node_diff': node_diff,
        'connection_diff': connection_diff,
        'attractors_learned': attractors_learned,
        'attractors_true': attractors_true,
        'nodes_learned': nodes_learned,
        'nodes_true': nodes_true,
        'edges_learned': edges_learned,
        'edges_true': edges_true,
        'missing_edges': missing_edges,
        'extra_edges': extra_edges,
    }


def format_similarity_report(similarity: Dict[str, Union[int, float]],
                             title: str = "Topological Similarity") -> str:
    """
    Create a human-readable summary of topological similarity metrics.

    Args:
        similarity: Output from compute_similarity_vector()
        title: Optional title for the report

    Returns:
        Formatted string report

    Example:
        >>> report = format_similarity_report(similarity, title="Full Latent Graph")
        >>> print(report)
    """
    lines = [f"\n{title}:"]
    lines.append(f"  Nodes:      {similarity['nodes_learned']} vs {similarity['nodes_true']} " +
                 f"(diff: {similarity['node_diff']})")
    lines.append(f"  Attractors: {similarity['attractors_learned']} vs {similarity['attractors_true']} " +
                 f"(diff: {similarity['attractor_diff']})")
    lines.append(f"  Edges:      {len(similarity['edges_learned'])} vs {len(similarity['edges_true'])} " +
                 f"(diff: {similarity['connection_diff']})")

    if similarity['missing_edges']:
        lines.append(f"    Missing edges: {sorted(similarity['missing_edges'])}")
    if similarity['extra_edges']:
        lines.append(f"    Extra edges:   {sorted(similarity['extra_edges'])}")

    # Perfect match indicator
    if (similarity['node_diff'] == 0 and
        similarity['attractor_diff'] == 0 and
        similarity['connection_diff'] == 0):
        lines.append("  ✓ PERFECT MATCH!")

    return '\n'.join(lines)


def compute_train_val_divergence(similarity_train: Dict, similarity_val: Dict) -> Dict[str, int]:
    """
    Compute divergence between training and validation Morse graphs.

    This helps detect overfitting: if the training and validation graphs
    differ significantly, the model may not generalize well.

    Args:
        similarity_train: Similarity vector for training data graph
        similarity_val: Similarity vector for validation data graph

    Returns:
        Dictionary with:
            - 'node_divergence': |nodes_train - nodes_val|
            - 'attractor_divergence': |attractors_train - attractors_val|
            - 'edge_divergence': Symmetric difference of edge sets

    Example:
        >>> divergence = compute_train_val_divergence(sim_train, sim_val)
        >>> if divergence['node_divergence'] > 1:
        ...     print("WARNING: Train and validation graphs differ substantially")
    """
    node_div = abs(similarity_train['nodes_learned'] - similarity_val['nodes_learned'])
    attractor_div = abs(similarity_train['attractors_learned'] - similarity_val['attractors_learned'])

    edges_train = similarity_train['edges_learned']
    edges_val = similarity_val['edges_learned']
    edge_div = len((edges_train - edges_val) | (edges_val - edges_train))

    return {
        'node_divergence': node_div,
        'attractor_divergence': attractor_div,
        'edge_divergence': edge_div,
    }


__all__ = [
    'count_attractors',
    'extract_edge_set',
    'compute_similarity_vector',
    'format_similarity_report',
    'compute_train_val_divergence',
]

