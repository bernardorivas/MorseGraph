import networkx as nx
from typing import List, Set, Dict, FrozenSet

def compute_morse_graph(map_graph: nx.DiGraph) -> nx.DiGraph:
    """
    Compute the Morse graph from a map graph, including only non-trivial Morse sets.

    A non-trivial Morse set is one that either:
    1. Contains more than one node (multi-node SCC)
    2. Contains a single node with a self-loop

    :param map_graph: The map graph, where nodes are box indices.
    :return: A directed graph where each node is a non-trivial Morse set, 
             represented by a frozenset of the box indices it contains.
    """
    sccs = list(nx.strongly_connected_components(map_graph))
    
    # Filter to non-trivial SCCs only
    non_trivial_sccs = []
    for scc in sccs:
        if len(scc) > 1:
            # Multi-node SCC is always non-trivial
            non_trivial_sccs.append(scc)
        elif len(scc) == 1:
            # Single-node SCC is non-trivial only if it has a self-loop
            node = next(iter(scc))
            if map_graph.has_edge(node, node):
                non_trivial_sccs.append(scc)
    
    if not non_trivial_sccs:
        # Return empty graph if no non-trivial SCCs found
        return nx.DiGraph()
    
    # Build the condensation graph manually for non-trivial SCCs only
    morse_graph = nx.DiGraph()
    
    # Convert SCCs to frozensets and add as nodes
    non_trivial_frozensets = [frozenset(scc) for scc in non_trivial_sccs]
    morse_graph.add_nodes_from(non_trivial_frozensets)
    
    # Create a mapping from original nodes to their non-trivial SCC (if any)
    node_to_scc = {}
    for scc_frozenset in non_trivial_frozensets:
        for node in scc_frozenset:
            node_to_scc[node] = scc_frozenset
    
    # Add edges between non-trivial SCCs
    for u, v in map_graph.edges():
        scc_u = node_to_scc.get(u)
        scc_v = node_to_scc.get(v)
        
        # Only add edge if both nodes are in non-trivial SCCs and SCCs are different
        if scc_u is not None and scc_v is not None and scc_u != scc_v:
            morse_graph.add_edge(scc_u, scc_v)

    return morse_graph

def compute_basins_of_attraction(morse_graph: nx.DiGraph, map_graph: nx.DiGraph) -> Dict[FrozenSet[int], Set[int]]:
    """
    Computes the basin of attraction for each attractor in the Morse graph.

    :param morse_graph: The Morse graph.
    :param map_graph: The original map graph.
    :return: A dictionary mapping each attractor (a frozenset of box indices) to its basin of attraction (a set of box indices).
    """
    attractors = {node for node in morse_graph.nodes() if morse_graph.out_degree(node) == 0}
    
    reversed_map_graph = map_graph.reverse()
    
    basins = {}
    for attractor in attractors:
        basin = set(attractor)
        queue = list(attractor)
        
        visited = set(attractor)
        
        while queue:
            node = queue.pop(0)
            for predecessor in reversed_map_graph.successors(node):
                if predecessor not in visited:
                    visited.add(predecessor)
                    basin.add(predecessor)
                    queue.append(predecessor)
        basins[attractor] = basin
        
    return basins


def iterative_morse_computation(model, max_depth: int = 5, refinement_threshold: float = 0.1):
    """
    Iteratively compute Morse graphs with adaptive grid refinement.
    
    This function implements the adaptive refinement workflow:
    1. Compute Morse Graph -> 2. Identify "recurrent" boxes -> 3. Subdivide locally
    
    :param model: Model instance with dynamics and an adaptive grid
    :param max_depth: Maximum number of refinement iterations
    :param refinement_threshold: Threshold for determining which Morse sets to refine
    :return: Final Morse graph and refinement history
    """
    # Import here to avoid circular imports
    from .grids import AdaptiveGrid
    
    if not isinstance(model.grid, AdaptiveGrid):
        raise ValueError("iterative_morse_computation requires an AdaptiveGrid")
    
    refinement_history = []
    
    for iteration in range(max_depth):
        print(f"Refinement iteration {iteration + 1}/{max_depth}")
        
        # Step 1: Compute map graph for current grid
        map_graph = model.compute_map_graph()
        
        # Step 2: Compute Morse graph
        morse_graph = compute_morse_graph(map_graph)
        
        # Step 3: Identify recurrent boxes (those in non-trivial Morse sets)
        recurrent_indices = _identify_recurrent_boxes(morse_graph, refinement_threshold)
        
        # Record iteration info
        iteration_info = {
            'iteration': iteration,
            'num_boxes': len(model.grid.get_boxes()),
            'num_morse_sets': len(morse_graph.nodes()),
            'num_recurrent_boxes': len(recurrent_indices),
            'morse_graph': morse_graph.copy()
        }
        refinement_history.append(iteration_info)
        
        print(f"  Grid boxes: {iteration_info['num_boxes']}")
        print(f"  Morse sets: {iteration_info['num_morse_sets']}")
        print(f"  Recurrent boxes: {iteration_info['num_recurrent_boxes']}")
        
        # Step 4: Check termination conditions
        if len(recurrent_indices) == 0:
            print("  No recurrent boxes found - terminating refinement")
            break
        
        if iteration == max_depth - 1:
            print("  Maximum depth reached - terminating refinement")
            break
        
        # Step 5: Subdivide the grid locally at recurrent boxes
        print(f"  Subdividing {len(recurrent_indices)} boxes...")
        model.grid.subdivide(recurrent_indices)
        
        # Optional: Local map graph update optimization could be implemented here
        # For now, we recompute the entire map graph at each iteration
    
    # Return final results
    final_map_graph = model.compute_map_graph()
    final_morse_graph = compute_morse_graph(final_map_graph)
    
    return final_morse_graph, refinement_history


def _identify_recurrent_boxes(morse_graph: nx.DiGraph, threshold: float = 0.1) -> List[int]:
    """
    Identify boxes that should be refined based on Morse graph structure.
    
    A box is considered "recurrent" if it belongs to a Morse set that:
    1. Has more than one box (non-trivial)
    2. Is large enough relative to the total system (above threshold)
    
    :param morse_graph: The computed Morse graph
    :param threshold: Minimum relative size for a Morse set to be considered for refinement
    :return: List of box indices to refine
    """
    total_boxes = sum(len(morse_set) for morse_set in morse_graph.nodes())
    recurrent_indices = []
    
    for morse_set in morse_graph.nodes():
        set_size = len(morse_set)
        relative_size = set_size / total_boxes if total_boxes > 0 else 0
        
        # Criteria for refinement:
        # 1. Non-trivial (more than 1 box)
        # 2. Significant size relative to system
        if set_size > 1 and relative_size >= threshold:
            recurrent_indices.extend(list(morse_set))
    
    return recurrent_indices


def analyze_refinement_convergence(refinement_history: List[Dict]) -> Dict:
    """
    Analyze the convergence properties of an iterative refinement process.
    
    :param refinement_history: History from iterative_morse_computation
    :return: Analysis results including convergence metrics
    """
    if not refinement_history:
        return {}
    
    box_counts = [info['num_boxes'] for info in refinement_history]
    morse_counts = [info['num_morse_sets'] for info in refinement_history]
    recurrent_counts = [info['num_recurrent_boxes'] for info in refinement_history]
    
    analysis = {
        'total_iterations': len(refinement_history),
        'initial_boxes': box_counts[0],
        'final_boxes': box_counts[-1],
        'refinement_factor': box_counts[-1] / box_counts[0] if box_counts[0] > 0 else 0,
        'initial_morse_sets': morse_counts[0],
        'final_morse_sets': morse_counts[-1],
        'convergence_achieved': recurrent_counts[-1] == 0,
        'box_growth_rate': [],
        'morse_stability': []
    }
    
    # Compute growth rates
    for i in range(1, len(box_counts)):
        growth_rate = box_counts[i] / box_counts[i-1] if box_counts[i-1] > 0 else 0
        analysis['box_growth_rate'].append(growth_rate)
    
    # Compute Morse set stability (how much the structure changes)
    for i in range(1, len(morse_counts)):
        stability = abs(morse_counts[i] - morse_counts[i-1]) / morse_counts[i-1] if morse_counts[i-1] > 0 else 0
        analysis['morse_stability'].append(stability)
    
    return analysis
