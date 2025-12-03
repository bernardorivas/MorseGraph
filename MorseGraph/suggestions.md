# MorseGraph Performance Optimization Suggestions

This document outlines potential improvements to MorseGraph in terms of speed, memory efficiency, and scalability, with a focus on leveraging CMGDB whenever possible.

---

## Table of Contents

1. [CMGDB Integration Improvements](#cmgdb-integration-improvements)
2. [Memory Optimization](#memory-optimization)
3. [Computational Efficiency](#computational-efficiency)
4. [Parallelization Enhancements](#parallelization-enhancements)
5. [Caching and Persistence](#caching-and-persistence)
6. [Algorithmic Improvements](#algorithmic-improvements)
7. [Code Organization](#code-organization)

---

## CMGDB Integration Improvements

### 1. **ALWAYS USE CMGDB** - Core Refactoring

**Current State:** MorseGraph has dual code paths - pure Python (`Model.compute_box_map()`) and CMGDB (`compute_morse_graph_3d()`, etc.). Users must choose which to use.

**Proposed Change:** **Make CMGDB the PRIMARY and ONLY computation backend.** Refactor `Model.compute_box_map()` to always use CMGDB internally.

**Architecture:**

```python
# In core.py Model class
def compute_box_map(self, 
                    subdiv_min: int = None,
                    subdiv_max: int = None,
                    subdiv_init: int = 0,
                    subdiv_limit: int = 10000,
                    return_format: str = 'networkx') -> Union[nx.DiGraph, 'CMGDB.MorseGraph']:
    """
    Compute BoxMap using CMGDB backend.
    
    CMGDB is now the primary computation engine for all dimensions.
    This provides significant performance improvements and consistency.
    
    Args:
        subdiv_min: Minimum subdivision depth (auto-selected if None)
        subdiv_max: Maximum subdivision depth (auto-selected if None)
        subdiv_init: Initial subdivision depth
        subdiv_limit: Maximum number of boxes
        return_format: 'networkx' (default) or 'cmgdb'
    
    Returns:
        If return_format='networkx': NetworkX DiGraph (backward compatible)
        If return_format='cmgdb': CMGDB.MorseGraph object (for advanced usage)
    
    Raises:
        ImportError: If CMGDB is not installed
    """
    try:
        import CMGDB
    except ImportError:
        raise ImportError(
            "CMGDB is required. Please install it via:\n"
            "  pip install -e ./cmgdb\n"
            "MorseGraph now uses CMGDB as the primary computation backend."
        )
    
    # Auto-select subdivision parameters based on dimension
    dim = self.grid.dim
    if subdiv_min is None:
        subdiv_min = {2: 20, 3: 30, 4: 15}.get(dim, 15)
    if subdiv_max is None:
        subdiv_max = {2: 28, 3: 42, 4: 20}.get(dim, 20)
    
    # Get domain bounds from grid
    domain_bounds = [self.grid.bounds[0].tolist(), 
                     self.grid.bounds[1].tolist()]
    
    # Adapt Dynamics to CMGDB interface
    def F(rect):
        """CMGDB function adapter."""
        dim = len(rect) // 2
        box = np.array([rect[:dim], rect[dim:]])
        res = self.dynamics(box, **self.dynamics_kwargs)
        return list(res[0]) + list(res[1])
    
    # Build CMGDB model
    cmgdb_model = CMGDB.Model(
        subdiv_min, subdiv_max, subdiv_init, subdiv_limit,
        domain_bounds[0], domain_bounds[1], F
    )
    
    # Compute Morse graph
    morse_graph_cmgdb, map_graph_cmgdb = CMGDB.ComputeMorseGraph(cmgdb_model)
    
    if return_format == 'cmgdb':
        return morse_graph_cmgdb
    
    # Convert to NetworkX for backward compatibility
    return self._cmgdb_to_networkx(morse_graph_cmgdb, map_graph_cmgdb)

def _cmgdb_to_networkx(self, morse_graph_cmgdb, map_graph_cmgdb) -> nx.DiGraph:
    """
    Convert CMGDB output to NetworkX DiGraph format.
    
    This maintains backward compatibility with existing code that expects
    NetworkX graphs.
    """
    graph = nx.DiGraph()
    
    # Extract all boxes and their transitions from map_graph
    # CMGDB's map_graph contains the full box-to-box transitions
    # We need to reconstruct the NetworkX graph from this
    
    # Get all boxes at maximum subdivision
    # This is a simplified version - full implementation would need
    # to traverse CMGDB's internal structure
    for i in range(map_graph_cmgdb.num_vertices()):
        # Get boxes in this vertex
        boxes = map_graph_cmgdb.morse_set_boxes(i)
        # Extract transitions from map_graph
        # ... (implementation details)
        pass
    
    return graph
```

**Benefits:**
- **Single code path** - eliminates dual implementation complexity
- **Consistent performance** - CMGDB's C++ backend is always used
- **Better scalability** - CMGDB handles large grids efficiently
- **Access to CMGDB features** - Conley index, advanced visualization, etc.
- **Backward compatible** - can still return NetworkX format

**Migration Path:**
1. Make CMGDB a required dependency (or check at import)
2. Refactor `Model.compute_box_map()` to use CMGDB
3. Add conversion utilities CMGDB ↔ NetworkX
4. Update `compute_morse_graph()` to work with CMGDB objects
5. Deprecate pure Python path (remove after transition period)

**Implementation Priority:** **CRITICAL** - This is the foundation for all other optimizations

---

### 2. CMGDB ↔ NetworkX Conversion Utilities

**Current State:** CMGDB returns `CMGDB.MorseGraph` objects, but many functions expect NetworkX graphs.

**Suggestion:** Create robust conversion utilities to maintain backward compatibility.

```python
def cmgdb_morse_graph_to_networkx(morse_graph_cmgdb, map_graph_cmgdb) -> nx.DiGraph:
    """
    Convert CMGDB MorseGraph to NetworkX DiGraph format.
    
    This allows existing code that expects NetworkX graphs to work seamlessly
    with CMGDB backend.
    
    Args:
        morse_graph_cmgdb: CMGDB.MorseGraph object
        map_graph_cmgdb: CMGDB.MapGraph object (contains box-to-box transitions)
    
    Returns:
        NetworkX DiGraph where:
        - Nodes are frozensets of box indices (Morse sets)
        - Edges represent connectivity between Morse sets
    """
    # Build box-to-morse-set mapping
    box_to_morse_set = {}
    for v in range(morse_graph_cmgdb.num_vertices()):
        boxes = morse_graph_cmgdb.morse_set_boxes(v)
        for box in boxes:
            # Convert box to index (requires grid information)
            box_idx = self._box_to_index(box)
            box_to_morse_set[box_idx] = v
    
    # Build NetworkX graph
    morse_graph_nx = nx.DiGraph()
    
    # Add nodes (Morse sets as frozensets)
    morse_sets = {}
    for v in range(morse_graph_cmgdb.num_vertices()):
        boxes = morse_graph_cmgdb.morse_set_boxes(v)
        box_indices = frozenset(self._box_to_index(box) for box in boxes)
        morse_sets[v] = box_indices
        morse_graph_nx.add_node(box_indices)
    
    # Add edges based on CMGDB adjacencies
    for v in range(morse_graph_cmgdb.num_vertices()):
        for adj_v in morse_graph_cmgdb.adjacencies(v):
            morse_graph_nx.add_edge(morse_sets[v], morse_sets[adj_v])
    
    return morse_graph_nx

def networkx_to_cmgdb_format(morse_graph_nx: nx.DiGraph) -> Dict:
    """
    Convert NetworkX MorseGraph to format compatible with CMGDB utilities.
    
    Useful for functions that need CMGDB-style data structures.
    """
    result = {
        'morse_sets': {},
        'barycenters': {},
        'adjacencies': {}
    }
    
    for i, morse_set in enumerate(morse_graph_nx.nodes()):
        result['morse_sets'][i] = list(morse_set)
        # Compute barycenters from boxes
        # ... (implementation)
    
    return result
```

**Benefits:**
- Seamless backward compatibility
- Can use CMGDB performance with NetworkX API
- Gradual migration path

**Implementation Priority:** High (required for #1)

---

### 3. Unified CMGDB Interface for All Dimensions

**Current State:** ~~Separate functions for 2D and 3D CMGDB computations.~~ (Now unified via Model.compute_box_map())

**Suggestion:** ~~Create a unified interface that works for 2D, 3D, and higher dimensions.~~ **Already achieved by making Model.compute_box_map() always use CMGDB.**

```python
def compute_morse_graph_cmgdb(
    dynamics,
    domain_bounds,
    subdiv_min=20,
    subdiv_max=28,
    subdiv_init=0,
    subdiv_limit=10000,
    verbose=True
):
    """
    Unified CMGDB interface for any dimension >= 2.
    
    Automatically selects optimal parameters based on dimension:
    - 2D: subdiv_min=20, subdiv_max=28 (default)
    - 3D: subdiv_min=30, subdiv_max=42 (default)
    - 4D+: subdiv_min=15, subdiv_max=20 (conservative)
    """
    dim = len(domain_bounds[0])
    
    # Auto-adjust parameters based on dimension
    if dim == 2:
        subdiv_min = subdiv_min or 20
        subdiv_max = subdiv_max or 28
    elif dim == 3:
        subdiv_min = subdiv_min or 30
        subdiv_max = subdiv_max or 42
    else:
        # Higher dimensions: more conservative
        subdiv_min = subdiv_min or 15
        subdiv_max = subdiv_max or 20
    
    return _run_cmgdb_compute(dynamics, domain_bounds, 
                              subdiv_min, subdiv_max, 
                              subdiv_init, subdiv_limit, verbose)
```

**Benefits:**
- Single API for all dimensions
- Automatic parameter tuning
- Easier to use

**Implementation Priority:** Medium

---

### 4. ~~CMGDB for Large 2D Grids~~ (OBSOLETE)

**Current State:** ~~CMGDB is primarily used for 3D systems.~~

**Status:** **OBSOLETE** - With "always use CMGDB" refactoring, CMGDB is now used for ALL dimensions including 2D. No special handling needed.

```python
def compute_box_map(self, n_jobs: int = -1, force_cmgdb: bool = False):
    """
    Automatically use CMGDB for high-resolution 2D grids.
    """
    total_boxes = np.prod(self.grid.divisions)
    use_cmgdb = (
        force_cmgdb or
        (self.grid.dim >= 3) or
        (self.grid.dim == 2 and total_boxes > 2**24)  # > 16M boxes
    )
    
    if use_cmgdb:
        try:
            import CMGDB
            return self._compute_box_map_cmgdb()
        except ImportError:
            pass
    
    return self._compute_box_map_python(n_jobs)
```

**Benefits:**
- Better performance for high-resolution 2D systems
- Transparent to user
- Handles memory-intensive cases better

**Implementation Priority:** Medium

---

### 5. Batch CMGDB Computations

**Current State:** Each CMGDB computation is independent.

**Suggestion:** Support batch processing for multiple parameter sets or systems.

```python
def compute_morse_graphs_batch(
    dynamics_list,
    domain_bounds_list,
    subdiv_params_list,
    n_jobs: int = -1
):
    """
    Compute multiple Morse graphs in parallel using CMGDB.
    
    Useful for parameter sweeps or comparing multiple systems.
    """
    from joblib import Parallel, delayed
    
    def compute_one(args):
        dynamics, bounds, params = args
        return compute_morse_graph_cmgdb(dynamics, bounds, **params)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_one)(args)
        for args in zip(dynamics_list, domain_bounds_list, subdiv_params_list)
    )
    
    return results
```

**Benefits:**
- Efficient parameter sweeps
- Parallel processing of independent computations
- Useful for sensitivity analysis

**Implementation Priority:** Low

---

## Memory Optimization

### 5. Sparse BoxMap Representation

**Current State:** NetworkX DiGraph stores all edges explicitly.

**Suggestion:** Use sparse matrix representation for very large BoxMaps.

```python
from scipy.sparse import csr_matrix

class SparseBoxMap:
    """
    Sparse matrix representation of BoxMap for memory efficiency.
    """
    def __init__(self, n_boxes: int):
        self.n_boxes = n_boxes
        self.rows = []
        self.cols = []
    
    def add_edge(self, i: int, j: int):
        self.rows.append(i)
        self.cols.append(j)
    
    def to_csr(self):
        """Convert to Compressed Sparse Row format."""
        data = np.ones(len(self.rows), dtype=bool)
        return csr_matrix((data, (self.rows, self.cols)), 
                         shape=(self.n_boxes, self.n_boxes))
    
    def to_networkx(self):
        """Convert to NetworkX graph when needed for analysis."""
        G = nx.DiGraph()
        G.add_edges_from(zip(self.rows, self.cols))
        return G
```

**Benefits:**
- Significant memory savings for sparse BoxMaps
- Faster construction
- Can still convert to NetworkX for analysis

**Implementation Priority:** High

---

### 6. Streaming BoxMap Construction

**Current State:** All boxes processed in memory simultaneously.

**Suggestion:** Stream box processing for very large grids, writing to disk incrementally.

```python
def compute_box_map_streaming(self, output_file: str, chunk_size: int = 10000):
    """
    Compute BoxMap in chunks, writing to disk incrementally.
    
    Useful for grids with millions of boxes.
    """
    import h5py
    
    boxes = self.grid.get_boxes()
    n_boxes = len(boxes)
    
    with h5py.File(output_file, 'w') as f:
        edges_dset = f.create_dataset('edges', (0, 2), maxshape=(None, 2), 
                                       dtype=np.int32, chunks=True)
        
        for i in range(0, n_boxes, chunk_size):
            chunk_boxes = boxes[i:i+chunk_size]
            chunk_indices = np.arange(i, min(i+chunk_size, n_boxes))
            
            # Process chunk
            edges = []
            for idx, box in zip(chunk_indices, chunk_boxes):
                image_box = self.dynamics(box)
                target_indices = self.grid.box_to_indices(image_box)
                for j in target_indices:
                    edges.append([idx, j])
            
            # Append to dataset
            if edges:
                edges_array = np.array(edges, dtype=np.int32)
                edges_dset.resize((edges_dset.shape[0] + len(edges_array), 2))
                edges_dset[-len(edges_array):] = edges_array
```

**Benefits:**
- Handle grids that don't fit in memory
- Can resume interrupted computations
- Process data in manageable chunks

**Implementation Priority:** Medium

---

### 7. Lazy Box Evaluation

**Current State:** All boxes evaluated upfront.

**Suggestion:** Evaluate boxes on-demand, especially for adaptive refinement.

```python
class LazyBoxMap:
    """
    Lazy evaluation of box images - only compute when needed.
    """
    def __init__(self, model):
        self.model = model
        self._cache = {}
        self._computed = set()
    
    def get_image_indices(self, box_idx: int):
        """Get image indices for a box, computing if necessary."""
        if box_idx not in self._computed:
            box = self.model.grid.get_boxes()[box_idx]
            image_box = self.model.dynamics(box)
            self._cache[box_idx] = self.model.grid.box_to_indices(image_box)
            self._computed.add(box_idx)
        return self._cache[box_idx]
    
    def compute_subset(self, box_indices: np.ndarray):
        """Compute images for a subset of boxes."""
        for idx in box_indices:
            if idx not in self._computed:
                self.get_image_indices(idx)
```

**Benefits:**
- Only compute what's needed
- Useful for adaptive refinement
- Can prioritize important boxes

**Implementation Priority:** Medium

---

## Computational Efficiency

### 8. Vectorized Box Operations

**Current State:** Box operations are mostly element-wise.

**Suggestion:** Vectorize common operations using NumPy broadcasting.

```python
def box_to_indices_vectorized(self, boxes: np.ndarray) -> np.ndarray:
    """
    Vectorized version for multiple boxes.
    
    boxes: shape (N, 2, D)
    Returns: list of N arrays of indices
    """
    N = boxes.shape[0]
    
    # Vectorized clipping
    clipped = np.clip(boxes, self.bounds[0], self.bounds[1])
    
    # Vectorized index calculation
    min_indices = np.floor((clipped[:, 0, :] - self.bounds[0]) / self.box_size).astype(int)
    max_indices = np.ceil((clipped[:, 1, :] - self.bounds[0]) / self.box_size).astype(int) - 1
    
    # Clip to valid range
    min_indices = np.clip(min_indices, 0, self.divisions - 1)
    max_indices = np.clip(max_indices, 0, self.divisions - 1)
    
    # Process each box (hard to fully vectorize due to variable output sizes)
    # But can use numba JIT for inner loop
    return self._process_indices_vectorized(min_indices, max_indices)
```

**Benefits:**
- Faster for batch operations
- Better cache locality
- Can use Numba JIT for further speedup

**Implementation Priority:** High

---

### 9. Numba JIT Compilation

**Current State:** Pure Python implementation.

**Suggestion:** Use Numba JIT for hot paths.

```python
from numba import jit

@jit(nopython=True)
def box_to_indices_numba(box, bounds, box_size, divisions):
    """
    Numba-compiled version of box_to_indices.
    """
    dim = len(box[0])
    
    # Clip box
    min_bounds = np.maximum(box[0], bounds[0])
    max_bounds = np.minimum(box[1], bounds[1])
    
    # Calculate indices
    min_idx = np.floor((min_bounds - bounds[0]) / box_size).astype(np.int32)
    max_idx = np.ceil((max_bounds - bounds[0]) / box_size).astype(np.int32) - 1
    
    # Clip to valid range
    min_idx = np.maximum(min_idx, 0)
    max_idx = np.minimum(max_idx, divisions - 1)
    
    # Generate all indices (simplified version)
    # ... implementation ...
    
    return indices
```

**Benefits:**
- 10-100x speedup for numerical operations
- No external dependencies (Numba is optional)
- Can fall back to Python if Numba unavailable

**Implementation Priority:** Medium

---

### 10. GPU Acceleration for Data-Driven Dynamics

**Current State:** CPU-only computation.

**Suggestion:** Use GPU for nearest neighbor searches in BoxMapData.

```python
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

class BoxMapDataGPU(BoxMapData):
    """
    GPU-accelerated version of BoxMapData.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if HAS_CUPY:
            self.X_gpu = cp.asarray(self.X)
            self.Y_gpu = cp.asarray(self.Y)
    
    def _get_boxes_in_epsilon_neighborhood_gpu(self, point, epsilon, metric):
        """GPU-accelerated neighborhood search."""
        if not HAS_CUPY:
            return super()._get_boxes_in_epsilon_neighborhood(point, epsilon, metric)
        
        point_gpu = cp.asarray(point)
        distances = cp.linalg.norm(self.X_gpu - point_gpu, axis=1)
        mask = distances <= cp.linalg.norm(epsilon)
        return cp.asnumpy(cp.where(mask)[0])
```

**Benefits:**
- Significant speedup for large datasets
- Transparent fallback to CPU
- Optional dependency

**Implementation Priority:** Low (requires CuPy)

---

## Parallelization Enhancements

### 11. Distributed Computing Support

**Current State:** Single-machine parallelization via joblib.

**Suggestion:** Support distributed computing for very large problems.

```python
def compute_box_map_distributed(self, n_workers: int = 4, backend: str = 'dask'):
    """
    Compute BoxMap using distributed computing.
    
    Supports Dask, Ray, or MPI backends.
    """
    if backend == 'dask':
        from dask.distributed import Client
        client = Client(n_workers=n_workers)
        
        boxes = self.grid.get_boxes()
        futures = []
        for i, box in enumerate(boxes):
            future = client.submit(self._compute_box_image, box, i)
            futures.append(future)
        
        results = client.gather(futures)
        # Build graph from results
        ...
    
    elif backend == 'ray':
        import ray
        ray.init(num_cpus=n_workers)
        
        @ray.remote
        def compute_image(box, idx):
            return idx, self.dynamics(box)
        
        boxes = self.grid.get_boxes()
        futures = [compute_image.remote(box, i) for i, box in enumerate(boxes)]
        results = ray.get(futures)
        ...
```

**Benefits:**
- Scale to clusters
- Handle extremely large grids
- Utilize multiple machines

**Implementation Priority:** Low (advanced use case)

---

### 12. Async Box Processing

**Current State:** Synchronous parallel processing.

**Suggestion:** Use async/await for I/O-bound operations and better resource utilization.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def compute_box_map_async(self, n_workers: int = 4):
    """
    Async version of compute_box_map for better I/O handling.
    """
    executor = ThreadPoolExecutor(max_workers=n_workers)
    loop = asyncio.get_event_loop()
    
    boxes = self.grid.get_boxes()
    active_indices = self.dynamics.get_active_boxes(self.grid)
    
    async def process_box(i):
        box = boxes[i]
        image_box = await loop.run_in_executor(executor, self.dynamics, box)
        indices = await loop.run_in_executor(executor, 
                                             self.grid.box_to_indices, 
                                             image_box)
        return i, indices
    
    tasks = [process_box(i) for i in active_indices]
    results = await asyncio.gather(*tasks)
    
    # Build graph
    graph = nx.DiGraph()
    for i, indices in results:
        graph.add_node(i)
        for j in indices:
            graph.add_edge(i, j)
    
    return graph
```

**Benefits:**
- Better I/O handling
- More efficient resource usage
- Can interleave computation and I/O

**Implementation Priority:** Low

---

## Caching and Persistence

### 13. Incremental BoxMap Updates

**Current State:** Full recomputation on grid changes.

**Suggestion:** Incrementally update BoxMap when grid is refined.

```python
class IncrementalBoxMap:
    """
    Maintains BoxMap and updates incrementally on grid changes.
    """
    def __init__(self, model):
        self.model = model
        self.box_map = nx.DiGraph()
        self._box_to_image = {}  # Cache: box_idx -> image_indices
    
    def update_on_subdivision(self, subdivided_boxes: Dict[int, List[int]]):
        """
        Update BoxMap when boxes are subdivided.
        
        subdivided_boxes: mapping from parent_idx to list of child_indices
        """
        # Remove old boxes
        for parent_idx in subdivided_boxes:
            self.box_map.remove_node(parent_idx)
            del self._box_to_image[parent_idx]
        
        # Add new boxes
        for parent_idx, child_indices in subdivided_boxes.items():
            for child_idx in child_indices:
                self.box_map.add_node(child_idx)
                # Compute image for new box
                box = self.model.grid.get_boxes()[child_idx]
                image_box = self.model.dynamics(box)
                image_indices = self.model.grid.box_to_indices(image_box)
                self._box_to_image[child_idx] = image_indices
                
                # Add edges
                for j in image_indices:
                    self.box_map.add_edge(child_idx, j)
        
        # Update predecessors (if needed)
        self._update_predecessors(subdivided_boxes)
```

**Benefits:**
- Faster adaptive refinement
- Only recompute what's necessary
- Significant speedup for iterative methods

**Implementation Priority:** High

---

### 14. Persistent BoxMap Storage

**Current State:** BoxMap recomputed each time.

**Suggestion:** Save BoxMap to disk for reuse.

```python
def save_box_map(self, box_map: nx.DiGraph, filepath: str):
    """
    Save BoxMap to disk in efficient format.
    """
    import pickle
    import gzip
    
    # Convert to edge list for compact storage
    edges = list(box_map.edges())
    nodes = list(box_map.nodes())
    
    data = {
        'nodes': nodes,
        'edges': edges,
        'metadata': {
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        }
    }
    
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_box_map(self, filepath: str) -> nx.DiGraph:
    """Load BoxMap from disk."""
    import pickle
    import gzip
    
    with gzip.open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    graph = nx.DiGraph()
    graph.add_nodes_from(data['nodes'])
    graph.add_edges_from(data['edges'])
    
    return graph
```

**Benefits:**
- Avoid recomputation
- Share results between runs
- Faster iteration during development

**Implementation Priority:** Medium

---

## Algorithmic Improvements

### 15. Early Termination for Basin Computation

**Current State:** Full BFS for each transient box.

**Suggestion:** Use early termination when all Morse sets are reached.

```python
def compute_all_morse_set_basins_optimized(morse_graph, box_map):
    """
    Optimized basin computation with early termination.
    """
    morse_sets = list(morse_graph.nodes())
    morse_sets_set = set(morse_sets)
    
    # Precompute which boxes are in which Morse set
    box_to_morse_set = {}
    for morse_set in morse_sets:
        for box in morse_set:
            box_to_morse_set[box] = morse_set
    
    basins = {morse_set: set(morse_set) for morse_set in morse_sets}
    
    # Process transient boxes
    all_morse_sets_reached = set()
    for box in box_map.nodes():
        if box in box_to_morse_set:
            continue  # Already in Morse set
        
        # BFS with early termination
        visited = {box}
        queue = [box]
        reached_morse_sets = set()
        
        while queue and len(reached_morse_sets) < len(morse_sets):
            current = queue.pop(0)
            current_scc = box_to_scc[current]
            
            if current_scc in morse_sets_set:
                reached_morse_sets.add(current_scc)
                # Early termination: if we've reached all Morse sets, stop
                if len(reached_morse_sets) == len(morse_sets):
                    break
                continue
            
            for successor in box_map.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)
        
        # Assign to highest Morse set
        if reached_morse_sets:
            highest = min(reached_morse_sets, key=lambda ms: morse_rank[ms])
            basins[highest].add(box)
    
    return basins
```

**Benefits:**
- Faster basin computation
- Early termination reduces unnecessary exploration
- Better for systems with many Morse sets

**Implementation Priority:** Medium

---

### 16. Hierarchical Grid Refinement

**Current State:** Uniform or adaptive refinement.

**Suggestion:** Use hierarchical/multiresolution approach.

```python
class HierarchicalGrid(AbstractGrid):
    """
    Multi-resolution grid with hierarchical refinement.
    """
    def __init__(self, bounds, base_resolution, max_levels=5):
        self.bounds = bounds
        self.base_resolution = base_resolution
        self.max_levels = max_levels
        self.levels = {}  # level -> grid
        
        # Initialize base level
        self.levels[0] = UniformGrid(bounds, base_resolution)
    
    def refine_hierarchically(self, boxes_to_refine, level=0):
        """
        Refine boxes at a specific level, creating children at level+1.
        """
        if level >= self.max_levels:
            return
        
        parent_grid = self.levels[level]
        if level + 1 not in self.levels:
            # Create next level grid with 2x resolution
            next_resolution = parent_grid.divisions * 2
            self.levels[level + 1] = UniformGrid(self.bounds, next_resolution)
        
        # Subdivide boxes at this level
        children = {}
        for box_idx in boxes_to_refine:
            parent_box = parent_grid.get_boxes()[box_idx]
            # Find corresponding boxes in next level
            child_indices = self.levels[level + 1].box_to_indices(parent_box)
            children[box_idx] = child_indices
        
        return children
```

**Benefits:**
- More efficient memory usage
- Better for systems with localized complexity
- Natural for multiscale analysis

**Implementation Priority:** Low (research direction)

---

## Code Organization

### 17. Plugin Architecture for Dynamics

**Current State:** Dynamics classes are hardcoded.

**Suggestion:** Plugin system for custom dynamics implementations.

```python
class DynamicsRegistry:
    """
    Registry for dynamics implementations.
    """
    _dynamics_classes = {}
    
    @classmethod
    def register(cls, name: str, dynamics_class):
        cls._dynamics_classes[name] = dynamics_class
    
    @classmethod
    def create(cls, name: str, *args, **kwargs):
        if name not in cls._dynamics_classes:
            raise ValueError(f"Unknown dynamics: {name}")
        return cls._dynamics_classes[name](*args, **kwargs)

# Usage
DynamicsRegistry.register('custom', CustomDynamics)
dynamics = DynamicsRegistry.create('custom', param1=value1)
```

**Benefits:**
- Easier to extend
- Better code organization
- Supports user-defined dynamics

**Implementation Priority:** Low

---

### 18. Configuration Validation and Optimization

**Current State:** Basic configuration validation.

**Suggestion:** Automatic parameter optimization and validation.

```python
def optimize_cmgdb_parameters(
    dynamics,
    domain_bounds,
    target_morse_sets: int = None,
    max_computation_time: float = 3600.0
):
    """
    Automatically find optimal CMGDB parameters.
    
    Uses binary search or grid search to find parameters that:
    - Produce desired number of Morse sets (if specified)
    - Complete within time budget
    - Balance resolution vs. computation time
    """
    # Start with conservative parameters
    subdiv_min = 20
    subdiv_max = 30
    
    best_params = None
    best_score = float('inf')
    
    for subdiv_max_candidate in range(30, 50, 2):
        try:
            start_time = time.time()
            morse_graph, _, _ = compute_morse_graph_cmgdb(
                dynamics, domain_bounds,
                subdiv_min=subdiv_min,
                subdiv_max=subdiv_max_candidate,
                ...
            )
            elapsed = time.time() - start_time
            
            if elapsed > max_computation_time:
                break
            
            num_sets = morse_graph.num_vertices()
            score = abs(num_sets - target_morse_sets) if target_morse_sets else elapsed
            
            if score < best_score:
                best_score = score
                best_params = {
                    'subdiv_min': subdiv_min,
                    'subdiv_max': subdiv_max_candidate
                }
        except Exception as e:
            continue
    
    return best_params
```

**Benefits:**
- Automatic parameter tuning
- Better user experience
- Optimal performance out of the box

**Implementation Priority:** Low

---

## Summary of Priorities

### CRITICAL Priority (Foundation)

1. **ALWAYS USE CMGDB** - Core refactoring to make CMGDB the primary backend
   - Refactor `Model.compute_box_map()` to always use CMGDB
   - Add CMGDB ↔ NetworkX conversion utilities
   - Make CMGDB a required dependency (or fail gracefully with clear error)
   - Update all functions to work with CMGDB objects
   - **This enables all other optimizations**

### High Priority (Immediate Impact)

2. **CMGDB ↔ NetworkX Conversion Utilities** - Required for backward compatibility
3. **Sparse BoxMap Representation** - Significant memory savings (if still needed after CMGDB refactor)
4. **Incremental BoxMap Updates** - Faster adaptive refinement (using CMGDB's internal structures)

### Medium Priority (Significant Benefits)

5. **Batch CMGDB Computations** - Parameter sweeps
6. **Streaming BoxMap Construction** - Handle very large grids (if CMGDB doesn't handle it)
7. **Lazy Box Evaluation** - Efficient adaptive refinement (using CMGDB's incremental capabilities)
8. **Persistent BoxMap Storage** - Avoid recomputation (cache CMGDB results)
9. **Early Termination for Basins** - Faster basin computation (using CMGDB's MapGraph)
10. **CMGDB Parameter Auto-tuning** - Automatically find optimal subdivision parameters

### Low Priority (Advanced Features)

11. **GPU Acceleration** - For very large datasets (if CMGDB doesn't support)
12. **Distributed Computing** - Cluster support (CMGDB is single-machine)
13. **Async Processing** - Better I/O handling
14. **Hierarchical Grid Refinement** - Research direction (may leverage CMGDB's subdivision)
15. **Plugin Architecture** - Extensibility
16. **Numba JIT Compilation** - Speedup for conversion utilities (less critical with CMGDB)

---

## Implementation Notes

### Backward Compatibility

All suggestions maintain backward compatibility:
- New features are opt-in via parameters
- Default behavior unchanged
- Existing code continues to work

### Testing Strategy

For each optimization:
1. Unit tests for correctness
2. Benchmark tests for performance
3. Integration tests with existing code
4. Memory profiling for memory optimizations

### Documentation

Each optimization should include:
- Performance benchmarks
- Memory usage comparisons
- Usage examples
- When to use vs. when not to use

---

## Implementation Details: Always-Use-CMGDB Refactoring

### Step-by-Step Implementation Plan

#### Step 1: Add CMGDB Dependency Check

```python
# In MorseGraph/__init__.py or core.py
try:
    import CMGDB
    _CMGDB_AVAILABLE = True
except ImportError:
    _CMGDB_AVAILABLE = False
    
    if not _CMGDB_AVAILABLE:
        import warnings
        warnings.warn(
            "CMGDB is not installed. MorseGraph requires CMGDB.\n"
            "Please install it via: pip install -e ./cmgdb",
            ImportWarning
        )
```

#### Step 2: Refactor Model.compute_box_map()

```python
# In core.py
class Model:
    def compute_box_map(self, 
                       subdiv_min: int = None,
                       subdiv_max: int = None,
                       subdiv_init: int = 0,
                       subdiv_limit: int = 10000,
                       return_format: str = 'networkx',
                       **cmgdb_kwargs) -> Union[nx.DiGraph, 'CMGDB.MorseGraph']:
        """
        Compute BoxMap using CMGDB backend.
        
        This is now the ONLY computation path - CMGDB is always used.
        """
        if not _CMGDB_AVAILABLE:
            raise ImportError(
                "CMGDB is required. Please install it via:\n"
                "  pip install -e ./cmgdb"
            )
        
        # Auto-select parameters
        dim = self.grid.dim
        if subdiv_min is None:
            subdiv_min = self._get_default_subdiv_min(dim)
        if subdiv_max is None:
            subdiv_max = self._get_default_subdiv_max(dim)
        
        # Get bounds
        domain_bounds = [self.grid.bounds[0].tolist(), 
                         self.grid.bounds[1].tolist()]
        
        # Create CMGDB adapter function
        def F(rect):
            """Adapt Dynamics interface to CMGDB."""
            dim = len(rect) // 2
            box = np.array([rect[:dim], rect[dim:]])
            res = self.dynamics(box, **self.dynamics_kwargs)
            return list(res[0]) + list(res[1])
        
        # Build CMGDB model
        cmgdb_model = CMGDB.Model(
            subdiv_min, subdiv_max, subdiv_init, subdiv_limit,
            domain_bounds[0], domain_bounds[1], F,
            **cmgdb_kwargs
        )
        
        # Compute
        morse_graph_cmgdb, map_graph_cmgdb = CMGDB.ComputeMorseGraph(cmgdb_model)
        
        # Store for later use
        self._morse_graph_cmgdb = morse_graph_cmgdb
        self._map_graph_cmgdb = map_graph_cmgdb
        
        if return_format == 'cmgdb':
            return morse_graph_cmgdb
        
        # Convert to NetworkX
        return self._cmgdb_to_networkx_boxmap(map_graph_cmgdb)
    
    def _cmgdb_to_networkx_boxmap(self, map_graph_cmgdb) -> nx.DiGraph:
        """
        Convert CMGDB MapGraph to NetworkX BoxMap.
        
        MapGraph contains the full box-to-box transition information.
        We need to extract this and build a NetworkX graph.
        """
        graph = nx.DiGraph()
        
        # CMGDB's MapGraph has methods to access box transitions
        # This is a simplified version - actual implementation would need
        # to traverse CMGDB's internal data structures
        
        # For now, we can use CMGDB's existing utilities
        # or access the underlying graph structure
        
        # Option 1: Use CMGDB's SaveMorseGraphData to get edge list
        # Option 2: Traverse CMGDB's internal structures directly
        
        # Placeholder - needs actual CMGDB API exploration
        # This would involve:
        # 1. Getting all boxes at max subdivision
        # 2. For each box, getting its image boxes
        # 3. Converting box coordinates to indices
        # 4. Building NetworkX graph
        
        return graph
```

#### Step 3: Update compute_morse_graph() to Accept CMGDB Objects

```python
# In analysis.py
def compute_morse_graph(box_map_or_cmgdb, assign_colors: bool = True, 
                        cmap_name: str = 'tab10') -> nx.DiGraph:
    """
    Compute Morse graph from BoxMap (NetworkX) or CMGDB MorseGraph.
    
    Now accepts both formats for flexibility.
    """
    # Check if input is CMGDB object
    if hasattr(box_map_or_cmgdb, 'num_vertices'):
        # It's a CMGDB.MorseGraph
        return _compute_morse_graph_from_cmgdb(box_map_or_cmgdb, assign_colors, cmap_name)
    
    # Otherwise, assume NetworkX DiGraph (existing code)
    return _compute_morse_graph_from_networkx(box_map_or_cmgdb, assign_colors, cmap_name)

def _compute_morse_graph_from_cmgdb(morse_graph_cmgdb, assign_colors, cmap_name):
    """Extract Morse graph directly from CMGDB object."""
    morse_graph_nx = nx.DiGraph()
    
    # CMGDB already computed the Morse graph!
    # We just need to convert it to NetworkX format
    
    morse_sets = {}
    for v in range(morse_graph_cmgdb.num_vertices()):
        boxes = morse_graph_cmgdb.morse_set_boxes(v)
        # Convert boxes to indices (requires grid info)
        box_indices = frozenset(_box_to_index(box) for box in boxes)
        morse_sets[v] = box_indices
        morse_graph_nx.add_node(box_indices)
    
    # Add edges from CMGDB adjacencies
    for v in range(morse_graph_cmgdb.num_vertices()):
        for adj_v in morse_graph_cmgdb.adjacencies(v):
            morse_graph_nx.add_edge(morse_sets[v], morse_sets[adj_v])
    
    # Assign colors
    if assign_colors:
        morse_sets_list = list(morse_graph_nx.nodes())
        cmap = cm.get_cmap(cmap_name)
        for i, morse_set in enumerate(morse_sets_list):
            morse_graph_nx.nodes[morse_set]['color'] = cmap(i / max(len(morse_sets_list), 10))
    
    return morse_graph_nx
```

#### Step 4: Update compute_all_morse_set_basins() to Work with CMGDB

```python
# In analysis.py
def compute_all_morse_set_basins(morse_graph, box_map_or_cmgdb):
    """
    Compute basins, accepting either NetworkX or CMGDB objects.
    """
    # If CMGDB objects, convert or use CMGDB's MapGraph directly
    if hasattr(box_map_or_cmgdb, 'num_vertices'):
        # Use CMGDB's MapGraph for efficient basin computation
        return _compute_basins_from_cmgdb(morse_graph, box_map_or_cmgdb)
    
    # Existing NetworkX code
    return _compute_basins_from_networkx(morse_graph, box_map_or_cmgdb)
```

### Migration Guide for Users

**Before (old code):**
```python
model = Model(grid, dynamics)
box_map = model.compute_box_map()  # Pure Python
morse_graph = compute_morse_graph(box_map)  # NetworkX
```

**After (new code - backward compatible):**
```python
model = Model(grid, dynamics)
box_map = model.compute_box_map()  # Now uses CMGDB internally, returns NetworkX
morse_graph = compute_morse_graph(box_map)  # Works as before
```

**After (new code - using CMGDB objects directly):**
```python
model = Model(grid, dynamics)
morse_graph_cmgdb = model.compute_box_map(return_format='cmgdb')  # CMGDB object
morse_graph = compute_morse_graph(morse_graph_cmgdb)  # Converts automatically
```

### Benefits of This Approach

1. **Zero Breaking Changes**: Existing code continues to work
2. **Automatic Performance**: Users get CMGDB speed without code changes
3. **Flexibility**: Can use CMGDB objects directly for advanced features
4. **Single Code Path**: Eliminates maintenance burden of dual implementation
5. **Future-Proof**: All optimizations leverage CMGDB's capabilities

### Open Questions / Implementation Notes

1. **CMGDB MapGraph API**: Need to explore CMGDB's MapGraph API to extract box-to-box transitions
2. **Box Index Mapping**: Need efficient way to map CMGDB box coordinates to grid indices
3. **Memory**: CMGDB may store data differently - need to verify memory efficiency
4. **Error Handling**: How to handle CMGDB-specific errors gracefully

## Conclusion

**Key Architectural Decision: ALWAYS USE CMGDB**

The primary refactoring is to make CMGDB the **only** computation backend. This:
- Eliminates dual code paths and complexity
- Provides consistent high performance across all dimensions
- Simplifies maintenance (one codebase to optimize)
- Enables access to CMGDB's advanced features (Conley index, etc.)

**Migration Strategy:**
1. **Phase 1**: Add CMGDB ↔ NetworkX conversion utilities
2. **Phase 2**: Refactor `Model.compute_box_map()` to use CMGDB internally
3. **Phase 3**: Update `compute_morse_graph()` to work with CMGDB objects directly
4. **Phase 4**: Deprecate pure Python path (with clear migration guide)
5. **Phase 5**: Remove pure Python implementation

**Benefits:**
- Users get CMGDB performance automatically
- No code changes required (backward compatible via conversion)
- Single, optimized code path
- Foundation for all future optimizations

The remaining optimizations focus on:
- Better integration with CMGDB's internal structures
- Caching and persistence of CMGDB results
- Parameter auto-tuning for CMGDB
- Advanced features leveraging CMGDB capabilities

