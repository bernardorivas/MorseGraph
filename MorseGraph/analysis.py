import networkx as nx
import numpy as np
import matplotlib.cm as cm
from typing import List, Set, Dict, FrozenSet

def compute_morse_graph(box_map: nx.DiGraph, assign_colors: bool = True, cmap_name: str = 'tab10') -> nx.DiGraph:
    """
    Compute the Morse graph from a BoxMap, properly handling transient states.

    The Morse graph shows connectivity between non-trivial Morse sets, where:
    1. Multi-node SCCs (recurrent components)
    2. Single-node SCCs with self-loops (fixed points)

    Connectivity includes paths through transient states (trivial SCCs).

    :param box_map: The BoxMap (directed graph), where nodes are box indices.
    :param assign_colors: If True, assign colors to Morse sets as node attributes.
    :param cmap_name: Name of matplotlib colormap to use for coloring (default: 'tab10').
    :return: A directed graph where each node is a non-trivial Morse set,
             represented by a frozenset of the box indices it contains.
             If assign_colors=True, each node has a 'color' attribute.
    """
    # Get all strongly connected components
    all_sccs = list(nx.strongly_connected_components(box_map))
    
    # Identify non-trivial SCCs (the actual Morse sets)
    non_trivial_sccs = []
    for scc in all_sccs:
        if len(scc) > 1:
            # Multi-node SCC is always non-trivial
            non_trivial_sccs.append(scc)
        elif len(scc) == 1:
            # Single-node SCC is non-trivial only if it has a self-loop
            node = next(iter(scc))
            if box_map.has_edge(node, node):
                non_trivial_sccs.append(scc)
    
    if not non_trivial_sccs:
        # Return empty graph if no non-trivial SCCs found
        return nx.DiGraph()
    
    # Build the full condensation graph (includes all SCCs)
    condensation = nx.condensation(box_map, all_sccs)
    
    # Create mapping from SCC to condensation node
    scc_to_cnode = {}
    for i, scc in enumerate(all_sccs):
        scc_to_cnode[frozenset(scc)] = i
    
    # Convert non-trivial SCCs to frozensets
    non_trivial_frozensets = [frozenset(scc) for scc in non_trivial_sccs]
    
    # Create the Morse graph with non-trivial SCCs as nodes
    morse_graph = nx.DiGraph()
    morse_graph.add_nodes_from(non_trivial_frozensets)
    
    # Find connectivity between non-trivial SCCs through the condensation graph
    for scc1 in non_trivial_frozensets:
        for scc2 in non_trivial_frozensets:
            if scc1 != scc2:
                # Get corresponding condensation nodes
                cnode1 = scc_to_cnode[scc1]
                cnode2 = scc_to_cnode[scc2]
                
                # Check if there's a path in the condensation graph
                if nx.has_path(condensation, cnode1, cnode2):
                    morse_graph.add_edge(scc1, scc2)

    # Assign colors to Morse sets as node attributes
    if assign_colors:
        morse_sets = list(morse_graph.nodes())
        num_sets = len(morse_sets)
        if num_sets > 0:
            cmap = cm.get_cmap(cmap_name)
            for i, morse_set in enumerate(morse_sets):
                # Assign color as RGBA tuple
                # Note: pygraphviz may warn about RGBA tuples, but this is harmless
                # as the colors are used by matplotlib, not pygraphviz
                morse_graph.nodes[morse_set]['color'] = cmap(i / max(num_sets, 10))

    return morse_graph

def compute_all_morse_set_basins(morse_graph: nx.DiGraph, box_map: nx.DiGraph) -> Dict[FrozenSet[int], Set[int]]:
    """
    Compute the basin of attraction for each Morse set.

    Basin of a Morse set = all boxes in the BoxMap that eventually flow into any box in that Morse set.
    Uses efficient condensation-based algorithm for O(boxes + edges) complexity.

    When a transient box can reach multiple Morse sets, it is assigned to the basin of the
    "highest" Morse set in topological order (earliest in the DAG structure).

    :param morse_graph: The Morse graph containing all Morse sets as nodes
    :param box_map: The BoxMap (directed graph of box-to-box transitions)
    :return: Dictionary mapping each Morse set (frozenset of box indices) to its basin (set of box indices)
    """
    # Get all SCCs and build condensation
    all_sccs = list(nx.strongly_connected_components(box_map))

    # Identify non-trivial SCCs (the Morse sets)
    morse_sets_list = list(morse_graph.nodes())
    morse_sets_set = set(morse_sets_list)

    # Establish priority ranking based on topological order
    # Lower rank = higher in the DAG = takes priority for basin assignment
    morse_rank = {morse_set: rank for rank, morse_set in enumerate(nx.topological_sort(morse_graph))}

    # Create mapping from box to its SCC
    box_to_scc = {}
    for scc in all_sccs:
        scc_frozen = frozenset(scc)
        for box in scc:
            box_to_scc[box] = scc_frozen

    # Initialize basins: each Morse set contains itself
    basins = {morse_set: set(morse_set) for morse_set in morse_sets_list}

    # For each box, determine which Morse set(s) it can reach
    for box in box_map.nodes():
        box_scc = box_to_scc[box]

        # If box is already in a Morse set, skip (already in its own basin)
        if box_scc in morse_sets_set:
            continue

        # Box is transient (in trivial SCC) - find which Morse sets it can reach
        # Use BFS from this box to find reachable Morse sets
        visited = {box}
        queue = [box]
        reached_morse_sets = set()

        while queue:
            current = queue.pop(0)

            # Check if current box is in a Morse set
            current_scc = box_to_scc[current]
            if current_scc in morse_sets_set:
                reached_morse_sets.add(current_scc)
                # Don't explore beyond Morse sets
                continue

            # Explore successors
            for successor in box_map.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)

        # Add this box to the basin of the highest reachable Morse set (earliest in topological order)
        if reached_morse_sets:
            highest_morse_set = min(reached_morse_sets, key=lambda ms: morse_rank[ms])
            basins[highest_morse_set].add(box)

    return basins

def compute_basins_of_attraction(morse_graph: nx.DiGraph, box_map: nx.DiGraph) -> Dict[FrozenSet[int], Set[int]]:
    """
    Computes the basin of attraction for each attractor in the Morse graph.

    This uses backward reachability on the BoxMap to find all boxes that
    eventually reach each attractor. Note: This operates at the box level,
    not the Morse set level. For Morse-structure-aware basins, use compute_morse_basins().

    :param morse_graph: The Morse graph.
    :param box_map: The original BoxMap.
    :return: A dictionary mapping each attractor (a frozenset of box indices) to its basin of attraction (a set of box indices).
    """
    attractors = {node for node in morse_graph.nodes() if morse_graph.out_degree(node) == 0}
    
    reversed_box_map = box_map.reverse()
    
    basins = {}
    for attractor in attractors:
        basin = set(attractor)
        queue = list(attractor)
        
        visited = set(attractor)
        
        while queue:
            node = queue.pop(0)
            for predecessor in reversed_box_map.successors(node):
                if predecessor not in visited:
                    visited.add(predecessor)
                    basin.add(predecessor)
                    queue.append(predecessor)
        basins[attractor] = basin
        
    return basins


def _update_box_map(box_map: nx.DiGraph, model, boxes_to_refine: List[int], new_children_map: Dict[int, List[int]]):
    """
    Locally update the BoxMap after grid refinement.

    :param box_map: The existing BoxMap graph.
    :param model: The dynamics model.
    :param boxes_to_refine: List of box indices that were subdivided.
    :param new_children_map: Mapping from a subdivided box to its new children.
    """
    # 1. Find predecessors of refined boxes before they are removed
    predecessors_to_recompute = set()
    for box in boxes_to_refine:
        if box in box_map:
            for pred in box_map.predecessors(box):
                predecessors_to_recompute.add(pred)

    # Exclude predecessors that were also refined
    predecessors_to_recompute.difference_update(boxes_to_refine)

    # 2. Remove outgoing edges from predecessors
    for pred in predecessors_to_recompute:
        if pred in box_map:
            box_map.remove_edges_from(list(box_map.out_edges(pred)))

    # 3. Remove the refined boxes from the graph
    for box in boxes_to_refine:
        if box in box_map:
            box_map.remove_node(box)

    # 4. Get all new children and add them to the graph
    all_new_children = [child for children in new_children_map.values() for child in children]
    box_map.add_nodes_from(all_new_children)

    # 5. Recompute images for new children and predecessors
    nodes_to_recompute = set(all_new_children).union(predecessors_to_recompute)
    
    if not nodes_to_recompute:
        return box_map

    # Final validation: only recompute nodes that are actually in the new grid
    grid_indices = set(model.grid.leaf_map.keys())
    valid_nodes_to_recompute = nodes_to_recompute.intersection(grid_indices)

    if not valid_nodes_to_recompute:
        return box_map

    node_indices = sorted(list(valid_nodes_to_recompute))
    
    boxes_to_process = model.grid.get_boxes_by_index(node_indices)

    for i, box_idx in enumerate(node_indices):
        image_box = model.dynamics(boxes_to_process[i])
        adj = model.grid.box_to_indices(image_box)
        for image_idx in adj:
            if image_idx != -1 and box_map.has_node(image_idx):
                box_map.add_edge(box_idx, image_idx)

    return box_map


def iterative_morse_computation(model, max_depth: int = 5, refinement_threshold: float = 0.1):
    """
    Iteratively compute Morse graphs with adaptive grid refinement.
    
    This function implements the adaptive refinement workflow:
    1. Compute Morse Graph -> 2. Identify boxes to refine -> 3. Subdivide locally
    
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

    # Initial full BoxMap computation
    print("Performing initial full BoxMap computation...")
    box_map = model.compute_box_map()
    print(f"Initial BoxMap has {box_map.number_of_nodes()} nodes and {box_map.number_of_edges()} edges.")
    
    for iteration in range(max_depth):
        print(f"Refinement iteration {iteration + 1}/{max_depth}")
        
        # Step 2: Compute Morse graph
        morse_graph = compute_morse_graph(box_map)
        
        # Step 3: Identify boxes to refine
        boxes_to_refine = _identify_boxes_to_refine(morse_graph, model.grid, refinement_threshold)
        
        # Record iteration info
        iteration_info = {
            'iteration': iteration,
            'num_boxes': len(model.grid.get_boxes()),
            'num_morse_sets': len(morse_graph.nodes()),
            'num_boxes_to_refine': len(boxes_to_refine),
            'morse_graph': morse_graph.copy()
        }
        refinement_history.append(iteration_info)
        
        print(f"  Grid boxes: {iteration_info['num_boxes']}")
        print(f"  Morse sets: {iteration_info['num_morse_sets']}")
        print(f"  Boxes to refine: {iteration_info['num_boxes_to_refine']}")
        
        # Step 4: Check termination conditions
        if len(boxes_to_refine) == 0:
            print("  No further refinement needed - terminating.")
            break
        
        if iteration == max_depth - 1:
            print("  Maximum depth reached - terminating refinement")
            break
        
        # Step 5: Subdivide the grid locally
        print(f"  Subdividing {len(boxes_to_refine)} boxes...")
        new_children_map = model.grid.subdivide(boxes_to_refine)
        
        # Step 6: Update BoxMap locally
        print("  Updating BoxMap locally...")
        box_map = _update_box_map(box_map, model, boxes_to_refine, new_children_map)
        print(f"  Updated BoxMap has {box_map.number_of_nodes()} nodes and {box_map.number_of_edges()} edges.")

    # Return final results
    final_morse_graph = compute_morse_graph(box_map)
    
    return final_morse_graph, refinement_history


def _identify_boxes_to_refine(morse_graph: nx.DiGraph, grid: 'AdaptiveGrid', threshold: float = 0.1) -> List[int]:
    """
    Identify boxes that should be refined based on Morse graph structure.

    A box is selected for refinement if it belongs to a Morse set that:
    1. Has more than one box (non-trivial)
    2. Covers a large enough volume relative to the total domain (above threshold)

    :param morse_graph: The computed Morse graph
    :param grid: The adaptive grid, used to compute volumes
    :param threshold: Minimum relative volume for a Morse set to be considered for refinement
    :return: List of box indices to refine
    """
    total_volume = np.prod(grid.bounds[1] - grid.bounds[0])
    if total_volume == 0:
        # Avoid division by zero if the domain is flat
        return []
        
    leaf_map = grid.leaf_map
    boxes_to_refine = []

    for morse_set in morse_graph.nodes():
        set_size = len(morse_set)

        # Calculate volume of this Morse set
        morse_set_volume = sum(np.prod(leaf_map[i].bounds[1] - leaf_map[i].bounds[0]) for i in morse_set if i in leaf_map)
        relative_volume = morse_set_volume / total_volume

        # Criteria for refinement:
        # 1. Non-trivial (more than 1 box)
        # 2. Significant volume relative to domain
        if set_size > 1 and relative_volume >= threshold:
            boxes_to_refine.extend(list(morse_set))

    return boxes_to_refine


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
    refinement_counts = [info['num_boxes_to_refine'] for info in refinement_history]
    
    analysis = {
        'total_iterations': len(refinement_history),
        'initial_boxes': box_counts[0],
        'final_boxes': box_counts[-1],
        'refinement_factor': box_counts[-1] / box_counts[0] if box_counts[0] > 0 else 0,
        'initial_morse_sets': morse_counts[0],
        'final_morse_sets': morse_counts[-1],
        'convergence_achieved': refinement_counts[-1] == 0,
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
