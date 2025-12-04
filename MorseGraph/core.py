import networkx as nx
import numpy as np
from joblib import Parallel, delayed
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from .dynamics import Dynamics
from .grids import AbstractGrid

# =============================================================================
# Architecture: CMGDB vs Python Implementations
# =============================================================================
#
# This module follows a clear separation of concerns:
#
# 1. CMGDB (Production Backend):
#    - CMGDB is the REQUIRED computation backend for all production code
#    - Uses fast C++ implementation for Morse graph computation
#    - Located in ./cmgdb/ directory
#    - All user-facing code MUST use CMGDB
#
# 2. Python Implementations (Internal Verification/Testing):
#    - Pure Python implementations exist for internal verification and testing
#    - Methods prefixed with `_compute_*_python` are for:
#      * Unit testing without CMGDB dependency
#      * Verification of CMGDB results
#      * Development/debugging
#    - These are NOT used in production code paths
#    - These are kept separate and clearly marked
#
# 3. CMGDB_utils (External Tools - DO NOT MODIFY):
#    - Located in ./CMGDB_utils/ directory
#    - Contains tools from another author that extend CMGDB functionality
#    - DO NOT MODIFY - these are external dependencies maintained separately
#    - May be used by MorseGraph but kept completely separate
#    - Any integration should be done through clean interfaces, not by modifying
#      the CMGDB_utils code itself
#
# =============================================================================

# Check for CMGDB availability
try:
    import CMGDB
    _CMGDB_AVAILABLE = True
except ImportError:
    _CMGDB_AVAILABLE = False
    CMGDB = None


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

    def compute_box_map(self, 
                       subdiv_min: int = None,
                       subdiv_max: int = None,
                       subdiv_init: int = 0,
                       subdiv_limit: int = 10000,
                       return_format: str = 'networkx',
                       n_jobs: int = -1,
                       **cmgdb_kwargs) -> Union[nx.DiGraph, Any]:
        """
        Compute the BoxMap using CMGDB backend.

        :param subdiv_min: Minimum subdivision depth (if None, auto-selected based on dim)
        :param subdiv_max: Maximum subdivision depth (if None, auto-selected based on dim)
        :param subdiv_init: Initial subdivision depth
        :param subdiv_limit: Maximum number of boxes
        :param return_format: 'networkx' (default) or 'cmgdb'
        :param n_jobs: Number of jobs (ignored if using CMGDB, kept for compatibility)
        :param cmgdb_kwargs: Additional arguments for CMGDB
        :return: BoxMap as nx.DiGraph or CMGDB.MorseGraph (if return_format='cmgdb')
        """
        # CMGDB is required - no fallback to Python implementation
        # This enforces the "Always-Use-CMGDB" architecture
        if not _CMGDB_AVAILABLE:
            raise ImportError(
                "CMGDB is required for MorseGraph. "
                "Please install it via: pip install -e ./cmgdb\n"
                "The pure Python fallback has been removed as part of the architecture refactoring."
            )
        
        # Auto-select parameters
        dim = self.grid.dim
        if subdiv_min is None:
            # Default values based on dimension
            if dim == 2: subdiv_min = 20
            elif dim == 3: subdiv_min = 30
            else: subdiv_min = 15
            
        if subdiv_max is None:
            if dim == 2: subdiv_max = 28
            elif dim == 3: subdiv_max = 42
            else: subdiv_max = 20
        
        # Get bounds
        domain_bounds = [self.grid.bounds[0].tolist(), 
                         self.grid.bounds[1].tolist()]
        
        # Run CMGDB computation
        # We use _run_cmgdb_compute helper which handles the Dynamics adapter
        morse_graph, morse_sets, barycenters, map_graph = _run_cmgdb_compute(
            self.dynamics, domain_bounds,
            subdiv_min, subdiv_max, subdiv_init, subdiv_limit,
            verbose=False # Suppress output for internal call
        )
        
        # Store for later use/inspection
        self._morse_graph_cmgdb = morse_graph
        self._map_graph_cmgdb = map_graph
        
        if return_format == 'cmgdb':
            return morse_graph
        
        # Convert to NetworkX
        return self._cmgdb_to_networkx_boxmap(map_graph, morse_graph)

    def _compute_box_map_python(self, n_jobs: int = -1) -> nx.DiGraph:
        """
        Pure Python implementation of BoxMap computation for internal verification.
        
        PURPOSE:
        - Internal testing and verification (unit tests, integration tests)
        - Development/debugging without CMGDB dependency
        - Cross-validation of CMGDB results
        
        ARCHITECTURE:
        - This is NOT used in production code paths
        - Production code MUST use CMGDB via compute_box_map()
        - This method is kept separate and clearly marked as verification-only
        
        USAGE:
        - Called explicitly from tests: model._compute_box_map_python()
        - Never called automatically - CMGDB is required for production
        
        NOTE:
        - This implements the same algorithm as CMGDB but in pure Python
        - Useful for understanding the computation without C++ complexity
        - Performance is much slower than CMGDB - not for production use
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

    def _cmgdb_to_networkx_boxmap(self, map_graph_cmgdb, morse_graph_cmgdb=None) -> nx.DiGraph:
        """
        Convert CMGDB MapGraph to NetworkX BoxMap.
        
        Note: This returns a graph where nodes are CMGDB vertex indices.
        Mapping these back to self.grid indices is complex if grids don't match.
        For now, we return the graph structure.
        """
        graph = nx.DiGraph()
        
        # We need to iterate over vertices in map_graph
        # map_graph.num_vertices() gives total count
        # map_graph.adjacencies(v) gives neighbors
        
        # Since map_graph can be huge, we might want to be careful.
        # But if user requested 'networkx', they expect the full graph.
        
        num_vertices = map_graph_cmgdb.num_vertices()
        
        # Add nodes
        graph.add_nodes_from(range(num_vertices))
        
        # Add edges
        # This loop might be slow in Python for millions of nodes
        for v in range(num_vertices):
            try:
                adj = map_graph_cmgdb.adjacencies(v)
                for neighbor in adj:
                    graph.add_edge(v, neighbor)
            except (IndexError, RuntimeError) as e:
                import warnings
                warnings.warn(f"Failed to get adjacencies for vertex {v}: {e}")
                
        return graph


# =============================================================================
# CMGDB-based Morse Graph Computation
# =============================================================================

def _check_cmgdb_available():
    """Check if CMGDB is available, raise ImportError if not."""
    if not _CMGDB_AVAILABLE:
        raise ImportError(
            "CMGDB is required. MorseGraph uses CMGDB as the computation backend.\n"
            "Please install it via: pip install -e ./cmgdb"
        )


def extract_cmgdb_to_pipeline_format(
    morse_graph_cmgdb: 'CMGDB.MorseGraph',
    map_graph_cmgdb: Optional['CMGDB.MapGraph'] = None,
    config: Optional[Any] = None,
    method: Optional[str] = None,
    computation_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Extract CMGDB data in pipeline's expected format.
    
    Converts CMGDB.MorseGraph output to dict format compatible with:
    - save_morse_graph_data() / load_morse_graph_data()
    - Pipeline's morse_graph_3d_data / morse_graph_2d_data
    - Visualization functions (plot_morse_sets_3d_scatter, etc.)
    
    Args:
        morse_graph_cmgdb: CMGDB.MorseGraph object from CMGDB.ComputeMorseGraph()
        map_graph_cmgdb: CMGDB.MapGraph object (optional, for future use)
        config: ExperimentConfig object or dict (optional, for reproducibility)
        method: Method string like 'data', 'restricted', 'full', 'enclosure' (optional)
        computation_time: Computation time in seconds (optional)
    
    Returns:
        Dict with keys:
        - 'morse_graph': CMGDB.MorseGraph object
        - 'morse_sets': Dict[int, List] mapping vertex index to list of boxes
        - 'morse_set_barycenters': Dict[int, List[np.ndarray]] mapping vertex index to list of barycenters
        - 'config': dict (if config provided)
        - 'method': str (if method provided)
        - 'computation_time': float (if computation_time provided)
    
    Example:
        >>> morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
        >>> data = extract_cmgdb_to_pipeline_format(
        ...     morse_graph, map_graph, 
        ...     config=config, method='data', computation_time=45.2
        ... )
        >>> save_morse_graph_data('cache_dir', data)
    """
    morse_sets = {}
    barycenters = {}
    
    # Extract Morse sets (boxes)
    for v in range(morse_graph_cmgdb.num_vertices()):
        boxes = morse_graph_cmgdb.morse_set_boxes(v)
        morse_sets[v] = boxes  # List of boxes (CMGDB format: flat lists)
        
        # Compute barycenters (pipeline expects this)
        barycenters[v] = []
        if boxes:
            dim = len(boxes[0]) // 2
            for box in boxes:
                barycenter = np.array([(box[j] + box[j + dim]) / 2.0 for j in range(dim)])
                barycenters[v].append(barycenter)
    
    result = {
        'morse_graph': morse_graph_cmgdb,
        'morse_sets': morse_sets,
        'morse_set_barycenters': barycenters,
    }
    
    # Add optional fields
    if config is not None:
        if hasattr(config, 'to_dict'):
            result['config'] = config.to_dict()
        else:
            result['config'] = config
    if method is not None:
        result['method'] = method
    if computation_time is not None:
        result['computation_time'] = computation_time
    
    return result


def _run_cmgdb_compute(dynamics, domain_bounds, subdiv_min, subdiv_max, subdiv_init, subdiv_limit, verbose=True):
    """Helper to run CMGDB computation given a Dynamics object."""
    _check_cmgdb_available()
    
    # Define F for CMGDB: adapts Dynamics (box->box) to CMGDB (rect->rect)
    def F(rect):
        # CMGDB passes rect as [min_x, min_y, ..., max_x, max_y, ...] (flat list)
        dim = len(rect) // 2
        box = np.array([rect[:dim], rect[dim:]])
        
        # Call dynamics
        res = dynamics(box)
        
        # Return as flat list [min_x, ..., max_x, ...]
        return list(res[0]) + list(res[1])

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
                
    # Extract Morse sets (as list of boxes)
    morse_sets = {}
    for i in range(morse_graph.num_vertices()):
        morse_sets[i] = morse_graph.morse_set_boxes(i)

    return morse_graph, morse_sets, barycenters, map_graph


def compute_morse_graph_3d(
    dynamics,
    domain_bounds,
    subdiv_min=30,
    subdiv_max=42,
    subdiv_init=0,
    subdiv_limit=10000,
    verbose=True
) -> Tuple['CMGDB.MorseGraph', Dict[int, List], Dict[int, List[np.ndarray]]]:
    """
    Compute 3D Morse graph using CMGDB and a Dynamics object.

    Returns format compatible with pipeline's expected structure:
    - morse_graph: CMGDB.MorseGraph object (for advanced use and visualization)
    - morse_sets: Dict[int, List] mapping vertex index to list of boxes
    - morse_set_barycenters: Dict[int, List[np.ndarray]] mapping vertex index to list of barycenters
    
    This matches what pipeline expects in morse_graph_3d_data dict.

    Args:
        dynamics: Dynamics object (e.g. BoxMapFunction)
        domain_bounds: [[lower_x, ...], [upper_x, ...]]
        subdiv_min: Minimum subdivision depth
        subdiv_max: Maximum subdivision depth
        subdiv_init: Initial subdivision depth
        subdiv_limit: Maximum number of boxes
        verbose: Whether to print progress messages

    Returns:
        Tuple of (morse_graph, morse_sets, morse_set_barycenters) where:
        - morse_graph: CMGDB.MorseGraph object
        - morse_sets: Dict[int, List] - vertex index -> list of boxes (CMGDB format: flat lists)
        - morse_set_barycenters: Dict[int, List[np.ndarray]] - vertex index -> list of barycenters
    
    Example:
        >>> morse_graph, morse_sets, barycenters = compute_morse_graph_3d(
        ...     box_map_function, domain_bounds, subdiv_min=30, subdiv_max=42
        ... )
        >>> # Use in pipeline:
        >>> morse_graph_data = {
        ...     'morse_graph': morse_graph,
        ...     'morse_sets': morse_sets,
        ...     'morse_set_barycenters': barycenters,
        ...     'config': config.to_dict()
        ... }
    """
    if verbose:
        print(f"Computing 3D Morse graph...")
        print(f"  Domain: {domain_bounds[0]} to {domain_bounds[1]}")
        
    morse_graph, morse_sets, barycenters, _ = _run_cmgdb_compute(
        dynamics, domain_bounds, 
        subdiv_min, subdiv_max, subdiv_init, subdiv_limit, 
        verbose
    )
    return morse_graph, morse_sets, barycenters


def compute_morse_graph_3d_for_pipeline(
    dynamics,
    domain_bounds,
    config: Any,
    verbose: bool = True,
    computation_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute 3D Morse graph and return in pipeline's expected format.
    
    This is a convenience wrapper that reads parameters from config and returns
    a dict compatible with save_morse_graph_data() and the pipeline.

    Args:
        dynamics: Dynamics object (e.g. BoxMapFunction)
        domain_bounds: [[lower_x, ...], [upper_x, ...]]
        config: ExperimentConfig object with subdiv_min, subdiv_max, etc.
        verbose: Whether to print progress messages
        computation_time: Optional computation time (will be measured if not provided)

    Returns:
        Dict compatible with save_morse_graph_data():
        {
            'morse_graph': CMGDB.MorseGraph,
            'morse_sets': Dict[int, List],
            'morse_set_barycenters': Dict[int, List[np.ndarray]],
            'config': dict
        }
    
    Example:
        >>> config = load_experiment_config('configs/ives_default.yaml')
        >>> morse_graph_data = compute_morse_graph_3d_for_pipeline(
        ...     box_map_function, domain_bounds, config
        ... )
        >>> save_morse_graph_data('cache_dir', morse_graph_data)
    """
    morse_graph, morse_sets, barycenters = compute_morse_graph_3d(
        dynamics, domain_bounds, 
        subdiv_min=config.subdiv_min,
        subdiv_max=config.subdiv_max,
        subdiv_init=config.subdiv_init,
        subdiv_limit=config.subdiv_limit,
        verbose=verbose
    )
    
    return extract_cmgdb_to_pipeline_format(
        morse_graph,
        config=config,
        computation_time=computation_time
    )


def compute_morse_graph_2d_data(
    dynamics,
    domain_bounds,
    subdiv_min=20,
    subdiv_max=28,
    subdiv_init=0,
    subdiv_limit=10000,
    verbose=True
) -> Tuple['CMGDB.MorseGraph', Dict[int, List], Dict[int, List[np.ndarray]]]:
    """
    Compute 2D Morse graph using CMGDB and a Dynamics object (e.g. BoxMapData).

    Returns format compatible with pipeline's expected structure:
    - morse_graph: CMGDB.MorseGraph object
    - morse_sets: Dict[int, List] mapping vertex index to list of boxes
    - morse_set_barycenters: Dict[int, List[np.ndarray]] mapping vertex index to list of barycenters

    Args:
        dynamics: Dynamics object (e.g. BoxMapData)
        domain_bounds: [[lower_x, ...], [upper_x, ...]]
        subdiv_min: Minimum subdivision depth
        subdiv_max: Maximum subdivision depth
        subdiv_init: Initial subdivision depth
        subdiv_limit: Maximum number of boxes
        verbose: Whether to print progress messages

    Returns:
        Tuple of (morse_graph, morse_sets, morse_set_barycenters)
    """
    if verbose:
        print(f"Computing 2D Morse graph (Data)...")
        print(f"  Domain: {domain_bounds[0]} to {domain_bounds[1]}")

    morse_graph, morse_sets, barycenters, _ = _run_cmgdb_compute(
        dynamics, domain_bounds, 
        subdiv_min, subdiv_max, subdiv_init, subdiv_limit, 
        verbose
    )
    return morse_graph, morse_sets, barycenters


def compute_morse_graph_2d_for_pipeline(
    method: str,
    dynamics,
    domain_bounds,
    config: Any,
    latent_bounds: Optional[np.ndarray] = None,
    verbose: bool = True,
    computation_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute 2D Morse graph using specified method and return in pipeline's expected format.
    
    This is a convenience wrapper that reads parameters from config and returns
    a dict compatible with save_morse_graph_data() and the pipeline.

    Args:
        method: Method string ('data', 'restricted', 'full', 'enclosure')
        dynamics: Dynamics object (e.g. BoxMapData or BoxMapLearnedLatent)
        domain_bounds: [[lower_x, ...], [upper_x, ...]]
        config: ExperimentConfig object with latent_subdiv_min, latent_subdiv_max, etc.
        latent_bounds: Latent space bounds (optional, for some methods)
        verbose: Whether to print progress messages
        computation_time: Optional computation time (will be measured if not provided)

    Returns:
        Dict compatible with save_morse_graph_data():
        {
            'morse_graph': CMGDB.MorseGraph,
            'morse_sets': Dict[int, List],
            'morse_set_barycenters': Dict[int, List[np.ndarray]],
            'config': dict,
            'method': str
        }
    
    Note:
        This function handles the 'data' method. For 'restricted', 'full', 'enclosure',
        use the pipeline's _compute_method_learned() or call CMGDB directly.
    """
    if method == 'data':
        morse_graph, morse_sets, barycenters = compute_morse_graph_2d_data(
            dynamics, domain_bounds,
            subdiv_min=config.latent_subdiv_min,
            subdiv_max=config.latent_subdiv_max,
            subdiv_init=config.latent_subdiv_init,
            subdiv_limit=config.latent_subdiv_limit,
            verbose=verbose
        )
        
        return extract_cmgdb_to_pipeline_format(
            morse_graph,
            config=config,
            method=method,
            computation_time=computation_time
        )
    else:
        raise ValueError(
            f"Method '{method}' not supported by compute_morse_graph_2d_for_pipeline(). "
            f"For 'restricted', 'full', 'enclosure', use pipeline's _compute_method_learned() "
            f"or call CMGDB directly."
        )


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
    DEPRECATED: Use MorseGraphPipeline._compute_method_learned(method='restricted') instead.
    
    This function was used for restricted domain computation with manual box checking.
    The functionality has been integrated into the pipeline's _compute_method_learned method
    which uses BoxMapLearnedLatent with allowed_indices computed from training data.
    """
    raise NotImplementedError(
        "compute_morse_graph_2d_restricted is deprecated. "
        "Use MorseGraphPipeline with method='restricted' instead, which properly computes "
        "allowed_indices from training data and passes them to BoxMapLearnedLatent."
    )
