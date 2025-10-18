import networkx as nx
import numpy as np
from joblib import Parallel, delayed

from .dynamics import Dynamics
from .grids import AbstractGrid


class Model:
    """
    The core engine that connects dynamics and grids.
    """

    def __init__(self, grid: AbstractGrid, dynamics: Dynamics,
                 dynamics_kwargs: dict = None):
        """
        :param grid: The grid that discretizes the state space.
        :param dynamics: The dynamical system.
        :param dynamics_kwargs: Additional kwargs to pass to dynamics.__call__
                               (optional, usually not needed)
        """
        self.grid = grid
        self.dynamics = dynamics
        self.dynamics_kwargs = dynamics_kwargs or {}

    def compute_box_map(self, n_jobs: int = -1) -> nx.DiGraph:
        """
        Compute the BoxMap.

        For each active box, computes its image under the dynamics and converts
        it to grid box indices. Note that for BoxMapData with output_enclosure='box_enclosure'
        (the default), grid.box_to_indices() creates a FILLED RECTANGULAR REGION of boxes,
        not just a sparse union. This implements the "cubical convex closure" of the output.

        :param n_jobs: The number of jobs to run in parallel. -1 means using all
                       available CPUs.
        :return: A directed graph representing the BoxMap, where nodes are box
                 indices and edges represent possible transitions.
        """
        boxes = self.grid.get_boxes()
        active_box_indices = self.dynamics.get_active_boxes(self.grid)

        # Check if grid supports batch mode for box_to_indices
        has_batch_mode = hasattr(self.grid, 'box_to_indices_batch')
        
        if has_batch_mode:
            # Batch mode: compute all image boxes in parallel, then process indices in batch
            def compute_image_box(i):
                return i, self.dynamics(boxes[i], **self.dynamics_kwargs)
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_image_box)(i) for i in active_box_indices
            )
            
            # Extract indices and image boxes
            indices = [i for i, _ in results]
            image_boxes = np.array([img_box for _, img_box in results])
            
            # Use batch mode to compute all adjacencies at once
            adjacencies = self.grid.box_to_indices_batch(image_boxes)
            
            # Build graph from batch results
            graph = nx.DiGraph()
            graph.add_nodes_from(active_box_indices)
            
            for i, adj in zip(indices, adjacencies):
                for j in adj:
                    if j in active_box_indices:
                        graph.add_edge(i, j)
        else:
            # Standard mode: compute adjacencies individually in parallel
            def compute_adjacencies(i):
                image_box = self.dynamics(boxes[i], **self.dynamics_kwargs)
                # Note: grid.box_to_indices() finds ALL boxes that intersect image_box.
                # For output_enclosure='box_enclosure', this creates a filled rectangle.
                adj = self.grid.box_to_indices(image_box)
                return i, adj

            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_adjacencies)(i) for i in active_box_indices
            )

            graph = nx.DiGraph()
            # Only add active boxes as nodes
            graph.add_nodes_from(active_box_indices)

            for i, adj in results:
                for j in adj:
                    # Only add edges to active boxes (j might be inactive)
                    if j in active_box_indices:
                        graph.add_edge(i, j)

        return graph


# =============================================================================
# CMGDB-based Morse Graph Computation
# =============================================================================

def compute_morse_graph_3d(
    map_func,
    domain_bounds,
    subdiv_min=30,
    subdiv_max=42,
    subdiv_init=0,
    subdiv_limit=10000,
    padding=True,
    cache_dir=None,
    use_cache=True,
    force_recompute=False,
    verbose=True
):
    """
    Compute 3D Morse graph using CMGDB for a discrete map.

    This function handles the full workflow for computing the Morse graph
    of a 3D map using CMGDB, including:
    - BoxMap computation with corner evaluation and bloating
    - Morse decomposition via strongly connected components
    - Barycenter computation for visualization
    - Optional caching to avoid redundant computation

    Args:
        map_func: Map function f: R^3 -> R^3, signature f(x) -> y
        domain_bounds: [[lower_x, lower_y, lower_z], [upper_x, upper_y, upper_z]]
        subdiv_min: Minimum subdivision depth
        subdiv_max: Maximum subdivision depth
        subdiv_init: Initial subdivision depth (0 for automatic)
        subdiv_limit: Maximum number of boxes
        padding: Whether to use padding in BoxMap (bloating)
        cache_dir: Directory for caching results (None to disable caching)
        use_cache: Whether to load from cache if available
        force_recompute: Force recomputation even if cached result exists
        verbose: Print progress messages

    Returns:
        Dictionary with:
            - 'morse_graph': CMGDB MorseGraph object
            - 'map_graph': CMGDB MapGraph object (None if loaded from cache)
            - 'num_morse_sets': Number of Morse sets
            - 'barycenters': Dict mapping Morse set index to list of barycenters
            - 'computation_time': Time in seconds
            - 'from_cache': Boolean indicating if result was loaded from cache

    Example:
        >>> from MorseGraph.systems import henon_map_3d
        >>> result = compute_morse_graph_3d(
        ...     henon_map_3d,
        ...     domain_bounds=[[-2, -2, -2], [2, 2, 2]],
        ...     subdiv_min=25,
        ...     subdiv_max=35,
        ...     cache_dir='./cache'
        ... )
        >>> print(f"Found {result['num_morse_sets']} Morse sets")
        >>> if result['from_cache']:
        ...     print("(loaded from cache)")
    """
    import CMGDB

    from functools import partial
    from itertools import product
    import time
    from .utils import compute_parameter_hash, load_morse_graph_cache, save_morse_graph_cache

    # Compute parameter hash for caching
    param_hash = compute_parameter_hash(
        map_func,
        domain_bounds,
        subdiv_min,
        subdiv_max,
        subdiv_init,
        subdiv_limit,
        padding
    )

    # Try to load from cache
    if cache_dir is not None and use_cache and not force_recompute:
        if verbose:
            print(f"Checking cache for 3D Morse graph (hash: {param_hash})...")
        cached = load_morse_graph_cache(cache_dir, param_hash, verbose=verbose)
        if cached is not None:
            # Return cached result with from_cache flag
            result = {
                'morse_graph': cached['morse_graph'],
                'map_graph': None,  # MapGraph not cached
                'num_morse_sets': cached['num_morse_sets'],
                'barycenters': cached['barycenters'],
                'computation_time': cached['metadata'].get('computation_time', 0.0),
                'from_cache': True,
            }
            return result

    # If we reach here, we need to compute
    if verbose:
        print(f"Computing 3D Morse graph...")
        print(f"  Domain: {domain_bounds[0]} to {domain_bounds[1]}")
        print(f"  Subdivisions: min={subdiv_min}, max={subdiv_max}, init={subdiv_init}")

    # Define BoxMap function
    def F(rect):
        dim = 3
        # Evaluate map at all corners of the box
        corners = list(product(*[(rect[d], rect[d+dim]) for d in range(dim)]))
        corners = [list(c) for c in corners]
        corners_next = np.array([map_func(np.array(c)) for c in corners])

        # Compute bounding box of images
        if padding:
            # Add bloating: box width in each dimension
            padding_size = [(rect[d + dim] - rect[d]) for d in range(dim)]
        else:
            padding_size = [0] * dim

        Y_l_bounds = [corners_next[:, d].min() - padding_size[d] for d in range(dim)]
        Y_u_bounds = [corners_next[:, d].max() + padding_size[d] for d in range(dim)]

        return Y_l_bounds + Y_u_bounds

    # Build CMGDB model
    model = CMGDB.Model(
        subdiv_min,
        subdiv_max,
        subdiv_init,
        subdiv_limit,
        domain_bounds[0],
        domain_bounds[1],
        F
    )

    # Compute Morse graph
    start_time = time.time()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    computation_time = time.time() - start_time

    if verbose:
        print(f"  Completed in {computation_time:.2f}s")
        print(f"  Found {morse_graph.num_vertices()} Morse sets")

    # Compute barycenters
    barycenters = {}
    for i in range(morse_graph.num_vertices()):
        morse_set_boxes = morse_graph.morse_set_boxes(i)
        barycenters[i] = []
        if morse_set_boxes:
            dim = len(morse_set_boxes[0]) // 2
            for box in morse_set_boxes:
                barycenter = np.array([(box[j] + box[j + dim]) / 2.0 for j in range(dim)])
                barycenters[i].append(barycenter)

    # Save to cache if requested
    if cache_dir is not None:
        metadata = {
            'domain_bounds': domain_bounds,
            'subdiv_min': subdiv_min,
            'subdiv_max': subdiv_max,
            'subdiv_init': subdiv_init,
            'subdiv_limit': subdiv_limit,
            'padding': padding,
            'computation_time': computation_time,
            'num_morse_sets': morse_graph.num_vertices(),
        }
        save_morse_graph_cache(
            morse_graph,
            map_graph,
            barycenters,
            metadata,
            cache_dir,
            param_hash,
            verbose=verbose
        )

    return {
        'morse_graph': morse_graph,
        'map_graph': map_graph,
        'num_morse_sets': morse_graph.num_vertices(),
        'barycenters': barycenters,
        'computation_time': computation_time,
        'from_cache': False,
    }


def compute_morse_graph_2d_data(
    latent_dynamics,
    device,
    z_data,
    latent_bounds,
    subdiv_min=20,
    subdiv_max=28,
    subdiv_init=0,
    subdiv_limit=10000,
    cache_dir=None,
    model_hash=None,
    use_cache=True,
    force_recompute=False,
    verbose=True
):
    """
    Compute 2D latent Morse graph using BoxMapData on encoded samples.

    This method uses CMGDB.BoxMapData to compute the Morse graph based on
    data pairs (z, G(z)) where z are encoded samples and G is the learned
    latent dynamics.

    Args:
        latent_dynamics: Trained latent dynamics model (PyTorch)
        device: torch device
        z_data: Encoded data points in latent space (N x 2 array)
        latent_bounds: [[lower_x, lower_y], [upper_x, upper_y]]
        subdiv_min, subdiv_max, subdiv_init, subdiv_limit: CMGDB parameters
        cache_dir: Directory for caching results (None to disable)
        model_hash: Optional hash string identifying the trained model
                   (required for caching, should be based on training config or model file)
        use_cache: Whether to load from cache if available
        force_recompute: Force recomputation even if cached
        verbose: Print progress

    Returns:
        Dictionary with morse_graph, map_graph, num_morse_sets, from_cache

    Example:
        >>> result = compute_morse_graph_2d_data(
        ...     latent_dynamics, device, z_train,
        ...     latent_bounds=[[-3, -2], [3, 2]],
        ...     cache_dir='./cache',
        ...     model_hash='abc123'  # Based on training config
        ... )
    """
    try:
        import CMGDB
        import torch
    except ImportError as e:
        raise ImportError(f"Required package not available: {e}")

    import time
    from .utils import compute_parameter_hash, load_morse_graph_cache, save_morse_graph_cache

    # Compute parameter hash for caching (include model_hash if provided)
    # Use a dummy function for hashing since BoxMapData doesn't depend on function
    dummy_func = lambda x: x  # Placeholder
    extra_params = {
        'method': 'BoxMapData',
        'model_hash': model_hash,
        'data_size': len(z_data),
    }
    param_hash = compute_parameter_hash(
        dummy_func,
        latent_bounds,
        subdiv_min,
        subdiv_max,
        subdiv_init,
        subdiv_limit,
        False,  # padding not applicable for BoxMapData
        extra_params=extra_params
    )

    # Try to load from cache
    if cache_dir is not None and use_cache and not force_recompute and model_hash is not None:
        if verbose:
            print(f"Checking cache for 2D Morse graph (BoxMapData, hash: {param_hash})...")
        cached = load_morse_graph_cache(cache_dir, param_hash, verbose=verbose)
        if cached is not None:
            result = {
                'morse_graph': cached['morse_graph'],
                'map_graph': None,
                'num_morse_sets': cached['num_morse_sets'],
                'computation_time': cached['metadata'].get('computation_time', 0.0),
                'from_cache': True,
            }
            return result

    if verbose:
        print(f"Computing 2D Morse graph (BoxMapData)...")
        print(f"  Latent bounds: {latent_bounds[0]} to {latent_bounds[1]}")
        print(f"  Data points: {len(z_data)}")

    # Compute G(z) for all data points
    with torch.no_grad():
        G_z = latent_dynamics(torch.FloatTensor(z_data).to(device)).cpu().numpy()

    # Create BoxMapData
    box_map_data = CMGDB.BoxMapData(
        X=z_data,
        Y=G_z,
        map_empty='outside',
        lower_bounds=latent_bounds[0],
        upper_bounds=latent_bounds[1]
    )

    def F_data(rect):
        return box_map_data.compute(rect)

    # Build model
    model = CMGDB.Model(
        subdiv_min,
        subdiv_max,
        subdiv_init,
        subdiv_limit,
        latent_bounds[0],
        latent_bounds[1],
        F_data
    )

    # Compute Morse graph
    start_time = time.time()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    computation_time = time.time() - start_time

    if verbose:
        print(f"  Completed in {computation_time:.2f}s")
        print(f"  Found {morse_graph.num_vertices()} Morse sets")

    # Compute barycenters for caching
    barycenters = {}
    for i in range(morse_graph.num_vertices()):
        morse_set_boxes = morse_graph.morse_set_boxes(i)
        barycenters[i] = []
        if morse_set_boxes:
            dim = len(morse_set_boxes[0]) // 2
            for box in morse_set_boxes:
                barycenter = np.array([(box[j] + box[j + dim]) / 2.0 for j in range(dim)])
                barycenters[i].append(barycenter)

    # Save to cache if requested
    if cache_dir is not None and model_hash is not None:
        metadata = {
            'method': 'BoxMapData',
            'latent_bounds': latent_bounds,
            'subdiv_min': subdiv_min,
            'subdiv_max': subdiv_max,
            'subdiv_init': subdiv_init,
            'subdiv_limit': subdiv_limit,
            'model_hash': model_hash,
            'data_size': len(z_data),
            'computation_time': computation_time,
            'num_morse_sets': morse_graph.num_vertices(),
        }
        save_morse_graph_cache(
            morse_graph,
            map_graph,
            barycenters,
            metadata,
            cache_dir,
            param_hash,
            verbose=verbose
        )

    return {
        'morse_graph': morse_graph,
        'map_graph': map_graph,
        'num_morse_sets': morse_graph.num_vertices(),
        'computation_time': computation_time,
        'from_cache': False,
    }


def compute_morse_graph_2d_restricted(
    latent_dynamics,
    device,
    z_data,
    latent_bounds,
    subdiv_min=20,
    subdiv_max=28,
    subdiv_init=0,
    subdiv_limit=10000,
    include_neighbors=True,
    padding=True,
    cache_dir=None,
    model_hash=None,
    use_cache=True,
    force_recompute=False,
    verbose=True
):
    """
    Compute 2D latent Morse graph with BoxMap restricted to data-containing boxes.

    This method computes the full BoxMap for the latent dynamics G, but only
    on boxes that contain data points (and optionally their neighbors). Boxes
    without data are mapped outside the domain.

    Args:
        latent_dynamics: Trained latent dynamics model (PyTorch)
        device: torch device
        z_data: Encoded data points in latent space (N x 2 array)
        latent_bounds: [[lower_x, lower_y], [upper_x, upper_y]]
        subdiv_min, subdiv_max, subdiv_init, subdiv_limit: CMGDB parameters
        include_neighbors: If True, include neighboring boxes of data-containing boxes
        padding: Whether to use padding in BoxMap
        cache_dir: Directory for caching results (None to disable)
        model_hash: Optional hash string identifying the trained model
        use_cache: Whether to load from cache if available
        force_recompute: Force recomputation even if cached
        verbose: Print progress

    Returns:
        Dictionary with morse_graph, map_graph, num_morse_sets, from_cache

    Example:
        >>> result = compute_morse_graph_2d_restricted(
        ...     latent_dynamics, device, z_large,
        ...     latent_bounds=[[-3, -2], [3, 2]],
        ...     include_neighbors=True,
        ...     cache_dir='./cache',
        ...     model_hash='abc123'
        ... )
    """
    try:
        import CMGDB
        import torch
    except ImportError as e:
        raise ImportError(f"Required package not available: {e}")

    from itertools import product
    import time
    from .utils import compute_parameter_hash, load_morse_graph_cache, save_morse_graph_cache

    # Compute parameter hash for caching
    dummy_func = lambda x: x  # Placeholder
    extra_params = {
        'method': 'domain_restricted',
        'model_hash': model_hash,
        'data_size': len(z_data),
        'include_neighbors': include_neighbors,
    }
    param_hash = compute_parameter_hash(
        dummy_func,
        latent_bounds,
        subdiv_min,
        subdiv_max,
        subdiv_init,
        subdiv_limit,
        padding,
        extra_params=extra_params
    )

    # Try to load from cache
    if cache_dir is not None and use_cache and not force_recompute and model_hash is not None:
        if verbose:
            print(f"Checking cache for 2D Morse graph (restricted, hash: {param_hash})...")
        cached = load_morse_graph_cache(cache_dir, param_hash, verbose=verbose)
        if cached is not None:
            result = {
                'morse_graph': cached['morse_graph'],
                'map_graph': None,
                'num_morse_sets': cached['num_morse_sets'],
                'computation_time': cached['metadata'].get('computation_time', 0.0),
                'from_cache': True,
            }
            return result

    if verbose:
        print(f"Computing 2D Morse graph (domain-restricted)...")
        print(f"  Latent bounds: {latent_bounds[0]} to {latent_bounds[1]}")
        print(f"  Data points: {len(z_data)}")
        print(f"  Include neighbors: {include_neighbors}")

    # Find boxes containing data at max subdivision depth
    lower = np.array(latent_bounds[0])
    upper = np.array(latent_bounds[1])
    dim = len(lower)
    n_boxes = 2 ** subdiv_max
    box_width = (upper - lower) / n_boxes

    # Find unique boxes containing data
    data_boxes = set()
    for point in z_data:
        box_idx = np.floor((point - lower) / box_width).astype(int)
        box_idx = np.clip(box_idx, 0, n_boxes - 1)
        data_boxes.add(tuple(box_idx))

    # Optionally expand to neighbors
    if include_neighbors:
        offsets = list(product([-1, 0, 1], repeat=dim))
        expanded_boxes = set()
        for box_idx in data_boxes:
            for offset in offsets:
                neighbor_idx = tuple(box_idx[d] + offset[d] for d in range(dim))
                if all(0 <= neighbor_idx[d] < n_boxes for d in range(dim)):
                    expanded_boxes.add(neighbor_idx)
        allowed_box_indices = expanded_boxes
    else:
        allowed_box_indices = data_boxes

    # Convert box indices to rectangles
    allowed_boxes = set()
    for idx in allowed_box_indices:
        box_lower = lower + np.array(idx) * box_width
        box_upper = lower + (np.array(idx) + 1) * box_width
        box_rect = tuple(list(box_lower) + list(box_upper))
        allowed_boxes.add(box_rect)

    if verbose:
        print(f"  Allowed boxes: {len(allowed_boxes)} (data: {len(data_boxes)})")

    # Define latent dynamics function
    def latent_map_func_batched(points):
        with torch.no_grad():
            points_tensor = torch.FloatTensor(points).to(device)
            points_next = latent_dynamics(points_tensor).cpu().numpy()
        return points_next

    # Define restricted BoxMap
    def F_restricted(rect):
        # Check if this box is in the allowed set
        rect_lower = np.array(rect[:dim])
        rect_center = rect_lower + (np.array(rect[dim:]) - rect_lower) / 2

        # Find box index at max subdivision
        box_idx = np.floor((rect_center - lower) / box_width).astype(int)
        box_idx = np.clip(box_idx, 0, n_boxes - 1)

        # Convert to box rectangle
        box_lower = lower + box_idx * box_width
        box_upper = lower + (box_idx + 1) * box_width
        box_rect = tuple(list(box_lower) + list(box_upper))

        # If not in allowed boxes, map outside domain
        if box_rect not in allowed_boxes:
            return list(upper + 1) + list(upper + 2)

        # Otherwise, compute standard BoxMap
        if padding:
            corners = list(product(*[(rect[d], rect[d+dim]) for d in range(dim)]))
            corners = [list(c) for c in corners]
        else:
            corners = [[(rect[d] + rect[d+dim])/2 for d in range(dim)]]

        corners_next = latent_map_func_batched(corners)

        padding_size = [(rect[d + dim] - rect[d]) for d in range(dim)] if padding else [0] * dim
        Y_l_bounds = [corners_next[:, d].min() - padding_size[d] for d in range(dim)]
        Y_u_bounds = [corners_next[:, d].max() + padding_size[d] for d in range(dim)]

        return Y_l_bounds + Y_u_bounds

    # Build model
    model = CMGDB.Model(
        subdiv_min,
        subdiv_max,
        subdiv_init,
        subdiv_limit,
        latent_bounds[0],
        latent_bounds[1],
        F_restricted
    )

    # Compute Morse graph
    start_time = time.time()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    computation_time = time.time() - start_time

    if verbose:
        print(f"  Completed in {computation_time:.2f}s")
        print(f"  Found {morse_graph.num_vertices()} Morse sets")

    # Compute barycenters for caching
    barycenters = {}
    for i in range(morse_graph.num_vertices()):
        morse_set_boxes = morse_graph.morse_set_boxes(i)
        barycenters[i] = []
        if morse_set_boxes:
            dim = len(morse_set_boxes[0]) // 2
            for box in morse_set_boxes:
                barycenter = np.array([(box[j] + box[j + dim]) / 2.0 for j in range(dim)])
                barycenters[i].append(barycenter)

    # Save to cache if requested
    if cache_dir is not None and model_hash is not None:
        metadata = {
            'method': 'domain_restricted',
            'latent_bounds': latent_bounds,
            'subdiv_min': subdiv_min,
            'subdiv_max': subdiv_max,
            'subdiv_init': subdiv_init,
            'subdiv_limit': subdiv_limit,
            'include_neighbors': include_neighbors,
            'padding': padding,
            'model_hash': model_hash,
            'data_size': len(z_data),
            'computation_time': computation_time,
            'num_morse_sets': morse_graph.num_vertices(),
        }
        save_morse_graph_cache(
            morse_graph,
            map_graph,
            barycenters,
            metadata,
            cache_dir,
            param_hash,
            verbose=verbose
        )

    return {
        'morse_graph': morse_graph,
        'map_graph': map_graph,
        'num_morse_sets': morse_graph.num_vertices(),
        'computation_time': computation_time,
        'from_cache': False,
    }