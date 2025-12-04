import networkx as nx
import numpy as np
import matplotlib.cm as cm
from typing import List, Set, Dict, FrozenSet, Union, Any

def compute_morse_graph(box_map: Union[nx.DiGraph, Any], assign_colors: bool = True, cmap_name: str = 'tab10') -> nx.DiGraph:
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
    # Check if box_map is CMGDB.MorseGraph
    if hasattr(box_map, 'num_vertices') and hasattr(box_map, 'morse_set'):
        return _compute_morse_graph_from_cmgdb(box_map, assign_colors, cmap_name)

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
            import matplotlib.pyplot as plt
            cmap = plt.colormaps.get_cmap(cmap_name)
            for i, morse_set in enumerate(morse_sets):
                # Assign color as RGBA tuple
                # Note: pygraphviz may warn about RGBA tuples, but this is harmless
                # as the colors are used by matplotlib, not pygraphviz
                morse_graph.nodes[morse_set]['color'] = cmap(i / max(num_sets, 10))

    return morse_graph

def _compute_morse_graph_from_cmgdb(morse_graph_cmgdb, assign_colors, cmap_name):
    """Extract Morse graph directly from CMGDB object."""
    morse_graph_nx = nx.DiGraph()
    
    # CMGDB already computed the Morse graph!
    # We just need to convert it to NetworkX format
    # Nodes should be frozenset of MapGraph indices to match compute_morse_graph behavior
    
    morse_sets = {}
    for v in range(morse_graph_cmgdb.num_vertices()):
        # Get indices in the MapGraph (phase space boxes)
        # Note: These are CMGDB indices, not necessarily grid indices
        indices = morse_graph_cmgdb.morse_set(v)
        morse_set_frozen = frozenset(indices)
        morse_sets[v] = morse_set_frozen
        morse_graph_nx.add_node(morse_set_frozen)
    
    # Add edges from CMGDB adjacencies
    for v in range(morse_graph_cmgdb.num_vertices()):
        for adj_v in morse_graph_cmgdb.adjacencies(v):
            morse_graph_nx.add_edge(morse_sets[v], morse_sets[adj_v])
    
    # Assign colors
    if assign_colors:
        morse_sets_list = list(morse_graph_nx.nodes())
        num_sets = len(morse_sets_list)
        if num_sets > 0:
            cmap = cm.get_cmap(cmap_name)
            for i, morse_set in enumerate(morse_sets_list):
                morse_graph_nx.nodes[morse_set]['color'] = cmap(i / max(num_sets, 10))
    
    return morse_graph_nx


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


def _update_box_map(box_map: nx.DiGraph, model, boxes_to_refine: List[int], new_children_map: Dict[int, List[int]], nodes_to_recompute: Set[int]):
    """
    Update the BoxMap after grid refinement by re-computing a specified set of nodes.

    :param box_map: The existing BoxMap graph.
    :param model: The dynamics model.
    :param boxes_to_refine: List of box indices that were subdivided.
    :param new_children_map: Mapping from a subdivided box to its new children.
    :param nodes_to_recompute: The set of all nodes whose outgoing edges must be recomputed.
    """
    # 1. Remove outgoing edges from all nodes that will be recomputed
    for node in nodes_to_recompute:
        if box_map.has_node(node):
            box_map.remove_edges_from(list(box_map.out_edges(node)))

    # 2. Remove the refined boxes from the graph
    for box in boxes_to_refine:
        if box_map.has_node(box):
            box_map.remove_node(box)

    # 3. Get all new children and add them to the graph
    all_new_children = {child for children in new_children_map.values() for child in children}
    box_map.add_nodes_from(all_new_children)
    
    # 4. Recompute images for the specified nodes
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


def iterative_morse_computation(
    model,
    max_depth: int = 5,
    refinement_threshold: float = 0.1,
    neighborhood_radius: int = 1,
    criterion: str = 'volume',
    criterion_kwargs: dict = None
):
    """
    Iteratively compute Morse graphs with adaptive grid refinement.
    
    This function implements the adaptive refinement workflow:
    1. Compute Morse Graph -> 2. Identify boxes to refine -> 3. Subdivide locally
    
    :param model: Model instance with dynamics and an adaptive grid
    :param max_depth: Maximum number of refinement iterations
    :param refinement_threshold: Threshold for determining which Morse sets to refine (volume criterion)
    :param neighborhood_radius: Radius for neighborhood re-computation (k=0 for old behavior, k>=1 for more robust updates)
    :param criterion: Refinement criterion to use ('volume' or 'diameter')
    :param criterion_kwargs: Additional keyword arguments for the criterion functions
    :return: Final Morse graph and refinement history
    """
    # Import here to avoid circular imports
    from .grids import AdaptiveGrid
    
    if not isinstance(model.grid, AdaptiveGrid):
        raise ValueError("iterative_morse_computation requires an AdaptiveGrid")
    
    if criterion_kwargs is None:
        criterion_kwargs = {}
    
    refinement_history = []

    # Initial full BoxMap computation
    print("Performing initial full BoxMap computation...")
    box_map = model.compute_box_map()
    print(f"Initial BoxMap has {box_map.number_of_nodes()} nodes and {box_map.number_of_edges()} edges.")
    
    for iteration in range(max_depth):
        print(f"Refinement iteration {iteration + 1}/{max_depth}")
        
        # Step 2: Compute Morse graph
        morse_graph = compute_morse_graph(box_map)
        
        # Step 3: Identify boxes to refine using the specified criterion
        boxes_to_refine = _identify_boxes_to_refine(
            morse_graph,
            model.grid,
            threshold=refinement_threshold,
            criterion=criterion,
            model=model,
            criterion_kwargs=criterion_kwargs
        )
        
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
        if not boxes_to_refine:
            print("  No further refinement needed - terminating.")
            break
        
        if iteration == max_depth - 1:
            print("  Maximum depth reached - terminating refinement")
            break

        # Step 5: Identify the full set of nodes to recompute *before* subdividing
        # This includes new children, and predecessors of the neighborhood
        
        # Define the neighborhood around the boxes to be refined
        neighborhood = model.grid.dilate_indices(np.array(boxes_to_refine), radius=neighborhood_radius)
        
        # Find all predecessors of this neighborhood
        predecessors_to_recompute = set()
        for box_idx in neighborhood:
            if box_map.has_node(box_idx):
                predecessors_to_recompute.update(box_map.predecessors(box_idx))

        # We must recompute these predecessors
        nodes_to_recompute = predecessors_to_recompute.difference(boxes_to_refine)

        # Step 6: Subdivide the grid locally
        print(f"  Subdividing {len(boxes_to_refine)} boxes...")
        new_children_map = model.grid.subdivide(boxes_to_refine)
        
        # Add the new children to the set of nodes to recompute
        all_new_children = {child for children in new_children_map.values() for child in children}
        nodes_to_recompute.update(all_new_children)

        # Step 7: Update BoxMap
        print(f"  Updating BoxMap for {len(nodes_to_recompute)} nodes...")
        box_map = _update_box_map(box_map, model, boxes_to_refine, new_children_map, nodes_to_recompute)
        print(f"  Updated BoxMap has {box_map.number_of_nodes()} nodes and {box_map.number_of_edges()} edges.")

    # Return final results
    final_morse_graph = compute_morse_graph(box_map)
    
    return final_morse_graph, refinement_history


def diameter_criterion(box, image_box, expansion_threshold=2.0):
    """
    Refine if image diameter exceeds box diameter significantly.
    
    This criterion identifies boxes where the dynamics significantly expands
    the state space, indicating complex behavior that may benefit from higher resolution.
    
    :param box: Box coordinates as (2, D) array
    :param image_box: Image box coordinates as (2, D) array
    :param expansion_threshold: Minimum ratio of image diameter to box diameter for refinement
    :return: True if box should be refined, False otherwise
    """
    box_diam = np.linalg.norm(box[1] - box[0])
    image_diam = np.linalg.norm(image_box[1] - image_box[0])
    return image_diam > expansion_threshold * box_diam


def _identify_boxes_to_refine(
    morse_graph: nx.DiGraph,
    grid: 'AdaptiveGrid',
    threshold: float = 0.1,
    criterion: str = 'volume',
    model = None,
    criterion_kwargs: dict = None
) -> List[int]:
    """
    Identify boxes that should be refined based on various criteria.

    Available criteria:
    - 'volume': Volume-based heuristic (original method)
    - 'diameter': Refine boxes where dynamics expands significantly

    :param morse_graph: The computed Morse graph
    :param grid: The adaptive grid, used to compute volumes and get boxes
    :param threshold: Minimum relative volume for volume-based criterion
    :param criterion: Refinement criterion to use ('volume' or 'diameter')
    :param model: The Model instance (required for diameter criterion)
    :param criterion_kwargs: Additional keyword arguments for the criterion functions
    :return: List of box indices to refine
    """
    if criterion_kwargs is None:
        criterion_kwargs = {}
    
    # Volume-based criterion (original implementation)
    if criterion == 'volume':
        total_volume = np.prod(grid.bounds[1] - grid.bounds[0])
        if total_volume == 0:
            return []
            
        leaf_map = grid.leaf_map
        boxes_to_refine = []

        for morse_set in morse_graph.nodes():
            set_size = len(morse_set)

            # Calculate volume of this Morse set
            morse_set_volume = sum(
                np.prod(leaf_map[i].bounds[1] - leaf_map[i].bounds[0]) 
                for i in morse_set if i in leaf_map
            )
            relative_volume = morse_set_volume / total_volume

            # Criteria for refinement:
            # 1. Non-trivial (more than 1 box)
            # 2. Significant volume relative to domain
            if set_size > 1 and relative_volume >= threshold:
                boxes_to_refine.extend(list(morse_set))

        return boxes_to_refine
    
    # Diameter-based criterion
    elif criterion == 'diameter':
        if model is None:
            raise ValueError("model required for diameter criterion")
        
        expansion_threshold = criterion_kwargs.get('expansion_threshold', 2.0)
        boxes_to_refine = []
        
        # Check all boxes in all Morse sets
        for morse_set in morse_graph.nodes():
            for box_idx in morse_set:
                box = grid.get_boxes_by_index([box_idx])[0]
                image_box = model.dynamics(box)
                if diameter_criterion(box, image_box, expansion_threshold):
                    boxes_to_refine.append(box_idx)
        
        return boxes_to_refine
    
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Choose from: 'volume', 'diameter'")


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


# =============================================================================
# Attractor/Repeller Identification
# =============================================================================

def identify_attractors(morse_graph: Union[nx.DiGraph, Any]) -> List[Any]:
    """
    Identify attractors in a Morse graph.
    
    An attractor is a node with no outgoing edges (sink node).
    
    Args:
        morse_graph: Morse graph as NetworkX DiGraph or CMGDB.MorseGraph
    
    Returns:
        List of attractor nodes
    
    Example:
        >>> attractors = identify_attractors(morse_graph)
        >>> print(f"Found {len(attractors)} attractors")
    """
    if hasattr(morse_graph, 'num_vertices'):
        # CMGDB MorseGraph
        attractors = []
        for v in range(morse_graph.num_vertices()):
            if len(morse_graph.adjacencies(v)) == 0:
                attractors.append(v)
        return attractors
    else:
        # NetworkX DiGraph
        return [node for node in morse_graph.nodes() if morse_graph.out_degree(node) == 0]


def identify_repellers(morse_graph: Union[nx.DiGraph, Any]) -> List[Any]:
    """
    Identify repellers in a Morse graph.
    
    A repeller is a node with no incoming edges (source node).
    
    Args:
        morse_graph: Morse graph as NetworkX DiGraph or CMGDB.MorseGraph
    
    Returns:
        List of repeller nodes
    
    Example:
        >>> repellers = identify_repellers(morse_graph)
        >>> print(f"Found {len(repellers)} repellers")
    """
    if hasattr(morse_graph, 'num_vertices'):
        # CMGDB MorseGraph
        # Get set of nodes that have incoming edges
        nodes_with_incoming = set()
        for v in range(morse_graph.num_vertices()):
            for adj_v in morse_graph.adjacencies(v):
                nodes_with_incoming.add(adj_v)
        # Repellers are nodes without incoming edges
        repellers = [v for v in range(morse_graph.num_vertices()) 
                    if v not in nodes_with_incoming]
        return repellers
    else:
        # NetworkX DiGraph
        return [node for node in morse_graph.nodes() if morse_graph.in_degree(node) == 0]


# =============================================================================
# Attractor Lattice and Non-Trivial Filtering
# Extracted from CMGDB_utils (MIT License 2025 Marcio Gameiro)
# Adapted to use NetworkX instead of pychomp
# =============================================================================

def _transitive_closure_dag(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Compute the transitive closure of a directed acyclic graph.
    
    Args:
        graph: NetworkX DiGraph (assumed to be acyclic)
    
    Returns:
        New DiGraph with transitive closure (all reachable paths added as edges)
    """
    closure = graph.copy()
    
    # For each node, add edges to all descendants
    for node in graph.nodes():
        descendants = set()
        visited = set()
        stack = [node]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            for successor in graph.successors(current):
                if successor not in visited:
                    descendants.add(successor)
                    stack.append(successor)
                    # Add edge in closure if not already present
                    if not closure.has_edge(node, successor):
                        closure.add_edge(node, successor)
    
    return closure


def _transitive_reduction_dag(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Compute the transitive reduction of a directed acyclic graph.
    
    The transitive reduction removes redundant edges while preserving
    reachability relationships.
    
    Args:
        graph: NetworkX DiGraph (assumed to be acyclic)
    
    Returns:
        New DiGraph with transitive reduction applied
    """
    reduction = graph.copy()
    
    # For each edge (u, v), check if there's a path u -> ... -> v
    # If so, the edge (u, v) is redundant and can be removed
    edges_to_remove = []
    
    for u, v in list(graph.edges()):
        # Temporarily remove the edge
        if reduction.has_edge(u, v):
            reduction.remove_edge(u, v)
            # Check if there's still a path from u to v
            if nx.has_path(reduction, u, v):
                # Edge is redundant, mark for removal
                edges_to_remove.append((u, v))
            else:
                # Edge is necessary, restore it
                reduction.add_edge(u, v)
        else:
            # Edge doesn't exist, skip
            continue
    
    # Remove redundant edges (they should already be removed, but be safe)
    for u, v in edges_to_remove:
        if reduction.has_edge(u, v):
            reduction.remove_edge(u, v)
    
    return reduction


def compute_attractor_lattice(morse_graph: Union[nx.DiGraph, Any]) -> Dict[str, Any]:
    """
    Compute the lattice of attractors from a Morse graph.
    
    An attractor is a set of Morse sets that form a downset (closed under
    taking descendants). The lattice structure captures the partial order
    relationship between attractors.
    
    Adapted from CMGDB_utils.LatticeAttractors (MIT License 2025 Marcio Gameiro)
    
    Args:
        morse_graph: Morse graph as NetworkX DiGraph or CMGDB.MorseGraph
                    If NetworkX, nodes should be frozensets representing Morse sets
    
    Returns:
        Dictionary with keys:
        - 'attractors': Set of frozensets, each representing an attractor
        - 'lattice': NetworkX DiGraph representing the attractor lattice
        - 'elementary_attractors': Set of elementary attractors (downsets of single nodes)
    
    Example:
        >>> morse_graph = compute_morse_graph(box_map)
        >>> result = compute_attractor_lattice(morse_graph)
        >>> print(f"Found {len(result['attractors'])} attractors")
    """
    import functools
    import itertools
    
    # Convert CMGDB MorseGraph to NetworkX if needed
    if hasattr(morse_graph, 'num_vertices'):
        # Convert CMGDB to NetworkX
        nx_graph = nx.DiGraph()
        for v in range(morse_graph.num_vertices()):
            nx_graph.add_node(v)
        for v in range(morse_graph.num_vertices()):
            for w in morse_graph.adjacencies(v):
                nx_graph.add_edge(v, w)
        morse_graph = nx_graph
    
    # Compute transitive closure
    trans_closure = _transitive_closure_dag(morse_graph)
    
    # Find elementary attractors (downsets of each node)
    elementary_attractors = {frozenset()}  # Empty set is always an attractor
    
    for v in morse_graph.nodes():
        # Get all descendants of v (including v itself)
        descendants = set(trans_closure.successors(v))
        descendants.add(v)
        elementary_attractors.add(frozenset(descendants))
    
    # Compute all attractors by taking unions of elementary attractors
    # An attractor is the union of any set of elementary attractors
    attractors = elementary_attractors.copy()
    elementary_list = list(elementary_attractors)
    
    # Generate all combinations of elementary attractors
    for k in range(1, len(elementary_list)):
        for combo in itertools.combinations(elementary_list, k + 1):
            union_attractor = frozenset().union(*combo)
            attractors.add(union_attractor)
    
    # Build lattice graph (partial order on attractors)
    def _cmp_attractors(A1: frozenset, A2: frozenset) -> int:
        """Compare function for sorting attractors."""
        if len(A1) == len(A2):
            # Lexicographical order
            sorted_A1 = sorted(A1)
            sorted_A2 = sorted(A2)
            if sorted_A1 == sorted_A2:
                return 0
            return -1 if sorted_A1 < sorted_A2 else 1
        # Compare by length
        return -1 if len(A1) < len(A2) else 1
    
    sorted_attractors = sorted(attractors, key=functools.cmp_to_key(_cmp_attractors))
    
    # Create lattice graph
    lattice = nx.DiGraph()
    for i, attractor in enumerate(sorted_attractors):
        # Create label
        if attractor:
            label = '{' + ', '.join(map(str, sorted(attractor))) + '}'
        else:
            label = '{ }'
        lattice.add_node(i, attractor=attractor, label=label)
    
    # Add edges for partial order (subset relationship)
    # Edge direction: i -> j means A1 is a subset of A2 (A1 < A2)
    for i, A1 in enumerate(sorted_attractors):
        for j, A2 in enumerate(sorted_attractors):
            if A1.issubset(A2) and A1 != A2:
                lattice.add_edge(i, j)  # i -> j means A1 < A2
    
    # Apply transitive reduction
    lattice = _transitive_reduction_dag(lattice)
    
    return {
        'attractors': attractors,
        'lattice': lattice,
        'elementary_attractors': elementary_attractors
    }


def filter_trivial_morse_sets(morse_graph: Union[nx.DiGraph, Any], 
                               return_reduced: bool = True) -> nx.DiGraph:
    """
    Filter out trivial Morse sets from a Morse graph.
    
    A trivial Morse set has Conley index (0, 0, ..., 0), meaning it represents
    no interesting dynamics. This function creates a new graph containing only
    non-trivial Morse sets with transitively reduced edges.
    
    Adapted from CMGDB_utils.NonTrivialCMGraph (MIT License 2025 Marcio Gameiro)
    
    Args:
        morse_graph: Morse graph as NetworkX DiGraph or CMGDB.MorseGraph
        return_reduced: If True, return transitively reduced graph (default: True)
    
    Returns:
        NetworkX DiGraph containing only non-trivial Morse sets
    
    Example:
        >>> morse_graph = compute_morse_graph(box_map)
        >>> non_trivial = filter_trivial_morse_sets(morse_graph)
        >>> print(f"Reduced from {morse_graph.number_of_nodes()} to {non_trivial.number_of_nodes()} nodes")
    """
    # Convert CMGDB MorseGraph to NetworkX if needed
    if hasattr(morse_graph, 'num_vertices'):
        # For CMGDB objects, check annotations
        nx_graph = nx.DiGraph()
        non_trivial_nodes = []
        
        for v in range(morse_graph.num_vertices()):
            # Check if node is trivial (all annotations are '0')
            try:
                annotations = morse_graph.annotations(v)
                is_trivial = all(ann == '0' for ann in annotations)
            except (AttributeError, IndexError):
                # Fallback: assume non-trivial if we can't check
                is_trivial = False
            
            if not is_trivial:
                non_trivial_nodes.append(v)
                nx_graph.add_node(v)
        
        # Add edges between non-trivial nodes
        for v in non_trivial_nodes:
            for w in morse_graph.adjacencies(v):
                if w in non_trivial_nodes and v != w:
                    nx_graph.add_edge(v, w)
        
        result = nx_graph
    else:
        # For NetworkX graphs, we need to determine triviality differently
        # Since NetworkX graphs don't have annotations, we'll use a heuristic:
        # A node is trivial if it has no self-loop AND no outgoing edges AND no incoming edges
        # (A node with incoming edges might be part of a Morse set even without outgoing edges)
        # This is a simplified version - full implementation would need Conley index
        non_trivial_nodes = []
        for node in morse_graph.nodes():
            # Check if node has self-loop, outgoing edges, or incoming edges
            has_self_loop = morse_graph.has_edge(node, node)
            has_outgoing = any(morse_graph.successors(node))
            has_incoming = any(morse_graph.predecessors(node))
            # Node is trivial only if it has no self-loop, no outgoing, AND no incoming
            is_trivial = not (has_self_loop or has_outgoing or has_incoming)
            
            if not is_trivial:
                non_trivial_nodes.append(node)
        
        # Create subgraph with only non-trivial nodes
        # Need to preserve edges between non-trivial nodes
        result = nx.DiGraph()
        result.add_nodes_from(non_trivial_nodes)
        for u, v in morse_graph.edges():
            if u in non_trivial_nodes and v in non_trivial_nodes:
                result.add_edge(u, v)
    
    # Apply transitive reduction if requested
    if return_reduced and result.number_of_edges() > 0:
        result = _transitive_reduction_dag(result)
    
    return result
