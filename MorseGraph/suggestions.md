# MorseGraph Performance Optimization Suggestions

This document outlines potential improvements to MorseGraph in terms of speed, memory efficiency, and scalability, with a focus on leveraging CMGDB whenever possible.

---

## Implementation Status (Updated 2025-12-03)

### Summary

The CMGDB integration has progressed significantly. Critical bugs identified in the initial review have been fixed.

**Overall Quality Score: 8/10** - Core functionality working, ready for testing.

### Critical Bugs (FIXED 2025-12-03)

| Bug | Location | Status | Fix Applied |
|-----|----------|--------|-------------|
| Tuple unpacking error | `pipeline.py:586` | FIXED | Changed to unpack 3 values |
| Incomplete function | `core.py:549-598` | FIXED | Raises NotImplementedError with deprecation message |
| 'restricted' not restricting | `pipeline.py:687-709` | FIXED | Computes allowed_indices from training data with dilation |
| Silent error swallowing | `core.py:187-189` | FIXED | Added specific exception handling with warnings |

### Implementation Status Table

| Feature | Status | File | Notes |
|---------|--------|------|-------|
| **CMGDB Integration** | | | |
| Model.compute_box_map() CMGDB backend | Done | core.py | With Python fallback |
| extract_cmgdb_to_pipeline_format() | Done | core.py | Works correctly |
| _run_cmgdb_compute() adapter | Done | core.py | Dynamics to CMGDB rect conversion |
| compute_morse_graph_3d() | Done | core.py | Returns pipeline-compatible format |
| compute_morse_graph_2d_data() | Done | core.py | Returns pipeline-compatible format |
| compute_morse_graph_2d_restricted() | Deprecated | core.py | Raises NotImplementedError, use pipeline instead |
| compute_morse_graph_2d_for_pipeline() | Partial | core.py | Only 'data' method |
| **Pipeline Integration** | | | |
| Stage 1 (3D Morse Graph) | Done | pipeline.py | CMGDB integration working |
| Stage 2 (Trajectory Generation) | Done | pipeline.py | Working |
| Stage 3 (Autoencoder Training) | Done | pipeline.py | Working |
| Stage 4 (Latent Encoding) | Done | pipeline.py | Working |
| Stage 5 - 'data' method | Done | pipeline.py | Fixed tuple unpacking |
| Stage 5 - 'restricted' method | Done | pipeline.py | Computes allowed_indices from training data |
| Stage 5 - 'full' method | Done | pipeline.py | Working |
| Stage 5 - 'enclosure' method | Done | pipeline.py | Same as 'full' with padding |
| Hash-based caching | Done | pipeline.py | Working |
| Config YAML integration | Done | pipeline.py | Reading from cmgdb_3d/cmgdb_2d sections |
| **Analysis Integration** | | | |
| compute_morse_graph() CMGDB input | Done | analysis.py | Handles both NetworkX and CMGDB |
| _compute_morse_graph_from_cmgdb() | Done | analysis.py | Conversion working |
| compute_all_morse_set_basins() CMGDB | Missing | analysis.py | Only handles NetworkX |
| iterative_morse_computation() | Partial | analysis.py | Doesn't leverage CMGDB adaptive refinement |
| **Testing** | | | |
| Test coverage for CMGDB integration | Missing | tests/ | Critical gap |

### Code Quality Issues

| Issue | Location | Description |
|-------|----------|-------------|
| DRY violation | core.py:253-259, 318-326 | Barycenter computation duplicated |
| DRY violation | analysis.py:73-83, 110-116 | Color assignment duplicated |
| Print vs warnings | core.py, pipeline.py | Uses print() instead of warnings.warn() |
| No format validation | core.py:286-295 | Box format conversion not validated |
| Magic numbers | core.py:66-75 | Hardcoded subdiv defaults without justification |
| Missing docstrings | core.py:281-333 | _run_cmgdb_compute() minimal documentation |

### Revised Priority List

**CRITICAL (COMPLETED 2025-12-03):**

1. ~~Fix pipeline.py:586 tuple unpacking~~ - DONE
2. ~~Fix or remove compute_morse_graph_2d_restricted()~~ - DONE (deprecated with NotImplementedError)
3. ~~Implement 'restricted' method properly~~ - DONE (computes allowed_indices from training data with dilation)
4. ~~Fix silent error handling~~ - DONE (specific exception handling with warnings)

**HIGH (For production readiness):**

5. **Add test coverage** - Create tests/test_core_cmgdb.py, tests/test_pipeline_cmgdb.py, tests/test_analysis_cmgdb.py
6. **Add CMGDB support to compute_all_morse_set_basins()** - Check input type and handle CMGDB objects
7. **Add config validation** - Validate required YAML sections exist on pipeline init
8. **Use warnings.warn()** - Replace print() statements with proper warnings

**MEDIUM (Code quality):**

9. **Extract barycenter helper** - Create `_compute_barycenters_from_boxes()` function
10. **Extract color assignment helper** - Create `_assign_morse_set_colors()` function
11. **Add format validation** - Validate box format in _run_cmgdb_compute() adapter
12. **Document index semantics** - Clarify CMGDB indices vs grid indices

**LOW (Future improvements):**

13. **Leverage CMGDB adaptive refinement** - Use CMGDB's native capabilities in iterative_morse_computation()
14. **Add logging system** - Replace print/warnings with proper logging
15. **Improve type hints** - Replace `Any` with specific CMGDB types

---

## Architectural Vision

**MorseGraph's Role:**
- **Python Wrapper**: Provides Pythonic interface to CMGDB's C++ backend
- **MORALS Extension**: Integrates MORALS approach (learned latent dynamics, autoencoder workflows)
- **Extended Tools**: Provides additional analysis, visualization, and computation utilities beyond CMGDB and MORALS

**Key Principle**: MorseGraph wraps and extends CMGDB, making it easier to use while adding high-level functionality.

### What MorseGraph Is:

1. **Python Wrapper Around CMGDB**
   - Wraps CMGDB's C++ backend with Pythonic interfaces
   - Handles data format conversions (CMGDB ↔ NumPy ↔ NetworkX)
   - Provides automatic parameter selection and optimization
   - Makes CMGDB easier to use

2. **MORALS Workflow Integration**
   - Complete autoencoder + latent dynamics pipeline
   - Learned latent space analysis
   - Comparison between full-space and latent-space dynamics
   - Preimage analysis and barycenter projection
   - Research-ready workflows for learned dynamics

3. **Extended Analysis Tools**
   - Basin analysis beyond CMGDB's basics
   - Stability and robustness analysis
   - Parameter sensitivity analysis
   - Comparison utilities for multiple systems
   - Statistical analysis and metrics

4. **Python Ecosystem Integration**
   - NumPy/SciPy for numerical operations
   - PyTorch for machine learning (MORALS)
   - NetworkX for graph analysis
   - Matplotlib for visualization
   - Seamless integration with Python ML workflows

### What MorseGraph Is NOT:

- **Not a replacement for CMGDB**: CMGDB remains the core computation engine
- **Not just MORALS**: Provides tools beyond MORALS workflows
- **Not a pure Python reimplementation**: Uses CMGDB for all heavy computation
- **Not a separate tool**: Extends and wraps CMGDB, not competes with it

### Value Proposition:

**For Users:**
- Get CMGDB's performance automatically
- Use Pythonic interfaces instead of C++ API
- Access MORALS workflows out of the box
- Extended analysis tools for research

**For Developers:**
- Single codebase to maintain (wrapper + extensions)
- Leverage CMGDB's optimized C++ code
- Focus on high-level functionality
- Python ecosystem integration

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

### 1. **CMGDB as Computation Backend - Pipeline-Compatible Wrapper**

**Current State:** MorseGraph has dual code paths - pure Python (`Model.compute_box_map()`) and CMGDB (`compute_morse_graph_3d()`, etc.). The pipeline expects specific data structures that work with your caching and visualization systems.

**Problem with CMGDB:** CMGDB is very structured and opinionated - it's hard to extract data and use it with custom tools/visualizations. Your pipeline expects specific dict formats with keys like `'morse_graph'`, `'morse_sets'`, `'morse_set_barycenters'`, `'config'`, `'method'`.

**Proposed Change:** **Use CMGDB as computation backend, but ensure compatibility with your pipeline's expected data structures:**
- **Wrap CMGDB for computation** - Use CMGDB's C++ backend for speed
- **Extract data in pipeline format** - Return dicts compatible with `save_morse_graph_data()` / `load_morse_graph_data()`
- **Preserve pipeline structure** - Keep YAML configs, caching system, visualization tools
- **Support all pipeline methods** - Ensure 'data', 'restricted', 'full', 'enclosure' methods work seamlessly
- **Compatible with existing functions** - `compute_morse_graph_3d()`, `compute_morse_graph_2d_data()`, etc. should return pipeline-compatible formats

**Architecture - Pipeline-Compatible Data Extraction:**

```python
# In core.py - Update existing functions to return pipeline-compatible formats
def compute_morse_graph_3d(
    dynamics,
    domain_bounds,
    subdiv_min=30,
    subdiv_max=42,
    subdiv_init=0,
    subdiv_limit=10000,
    verbose=True
) -> Tuple['CMGDB.MorseGraph', Dict[int, List], Dict[int, List]]:
    """
    Compute 3D Morse graph using CMGDB.
    
    Returns format compatible with pipeline's expected structure:
    - morse_graph: CMGDB.MorseGraph object (for advanced use)
    - morse_sets: Dict[int, List] mapping vertex index to list of boxes
    - morse_set_barycenters: Dict[int, List] mapping vertex index to list of barycenters
    
    This matches what pipeline expects in morse_graph_3d_data dict.
    """
    # Use CMGDB for computation (existing code)
    morse_graph_cmgdb, map_graph_cmgdb = _run_cmgdb_compute(...)
    
    # Extract in pipeline-compatible format
    morse_sets = {}
    barycenters = {}
    
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
    
    return morse_graph_cmgdb, morse_sets, barycenters

# Pipeline wrapper that ensures correct format
def compute_morse_graph_3d_for_pipeline(
    dynamics,
    domain_bounds,
    config: ExperimentConfig,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute 3D Morse graph and return in pipeline's expected format.
    
    Returns dict compatible with save_morse_graph_data():
    {
        'morse_graph': CMGDB.MorseGraph,
        'morse_sets': Dict[int, List],
        'morse_set_barycenters': Dict[int, List],
        'config': config.to_dict()
    }
    """
    morse_graph, morse_sets, barycenters = compute_morse_graph_3d(
        dynamics, domain_bounds, 
        subdiv_min=config.subdiv_min,
        subdiv_max=config.subdiv_max,
        subdiv_init=config.subdiv_init,
        subdiv_limit=config.subdiv_limit,
        **kwargs
    )
    
    return {
        'morse_graph': morse_graph,
        'morse_sets': morse_sets,
        'morse_set_barycenters': barycenters,
        'config': config.to_dict()
    }
```

**Key Principle:** 
- **Use CMGDB for computation** (it's fast)
- **Extract data in pipeline format** (compatible with your caching/visualization)
- **Preserve your tools** (YAML configs, pipeline, visualizations work unchanged)

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
- **CMGDB Performance**: Use fast C++ backend for computation
- **Flexibility**: Extract data your way, not forced into CMGDB's structure
- **Your Pipeline Preserved**: YAML configs, pipeline structure, visualization tools all work
- **Easy Data Extraction**: Get what you need for your tools/figures
- **Not Opinionated**: Don't force CMGDB's rigid structure on users

**MorseGraph's Value Proposition:**
1. **Computation Backend**: Use CMGDB for fast computation
2. **Flexible Data Access**: Extract data in formats YOU need
3. **Your Tools Work**: YAML configs, pipeline, visualizations preserved
4. **MORALS Integration**: Complete workflows with your config system
5. **Freedom from CMGDB Structure**: Work with data your way

**Migration Path:**
1. Make CMGDB a required dependency (MorseGraph wraps it)
2. Refactor `Model.compute_box_map()` to use CMGDB internally
3. Add conversion utilities CMGDB ↔ NetworkX (wrapper functionality)
4. Enhance MORALS workflows (learned latent dynamics)
5. Add extended analysis tools (basins, visualization, etc.)

**Implementation Priority:** **CRITICAL** - This defines MorseGraph's core architecture

---

### 2. Preserve Your YAML Config System and Pipeline Integration

**Current State:** You have a well-designed YAML config system with sections for `cmgdb_3d`, `cmgdb_2d`, `training`, `model`, `loss_weights`, etc. The pipeline reads from these configs and expects specific data structures.

**Suggestion:** **Ensure CMGDB integration reads directly from your YAML configs and produces pipeline-compatible outputs.**

```python
# Your config system stays the same - CMGDB integration reads from it
config = load_experiment_config('configs/ives_default.yaml')

# Pipeline's run_stage_1_3d() expects this format:
def compute_morse_graph_3d_for_pipeline(
    dynamics,
    domain_bounds,
    config: ExperimentConfig
) -> Dict[str, Any]:
    """
    Compute 3D Morse graph using CMGDB, reading from YOUR config.
    
    Returns dict compatible with pipeline's morse_graph_3d_data:
    {
        'morse_graph': CMGDB.MorseGraph,
        'morse_sets': Dict[int, List],  # vertex -> boxes
        'morse_set_barycenters': Dict[int, List],  # vertex -> barycenters
        'config': config.to_dict()  # For reproducibility
    }
    """
    # Read from YOUR config structure (cmgdb_3d section)
    morse_graph, morse_sets, barycenters = compute_morse_graph_3d(
        dynamics,
        domain_bounds,
        subdiv_min=config.subdiv_min,  # From cmgdb_3d section
        subdiv_max=config.subdiv_max,
        subdiv_init=config.subdiv_init,
        subdiv_limit=config.subdiv_limit,
        verbose=True
    )
    
    # Return in pipeline's expected format
    return {
        'morse_graph': morse_graph,
        'morse_sets': morse_sets,
        'morse_set_barycenters': barycenters,
        'config': config.to_dict()
    }

# Similarly for 2D computations
def compute_morse_graph_2d_for_pipeline(
    method: str,  # 'data', 'restricted', 'full', 'enclosure'
    config: ExperimentConfig,
    latent_bounds: np.ndarray,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute 2D Morse graph using CMGDB, reading from YOUR config.
    
    Returns dict compatible with pipeline's morse_graph_2d_data:
    {
        'morse_graph': CMGDB.MorseGraph,
        'morse_sets': Dict[int, List],
        'morse_set_barycenters': Dict[int, List],
        'config': config.to_dict(),
        'method': method  # Important for pipeline
    }
    """
    # Read from YOUR config structure (cmgdb_2d section)
    if method == 'data':
        return compute_morse_graph_2d_data(...)
    elif method in ['full', 'restricted']:
        return compute_morse_graph_2d_restricted(...)
    # ... etc
```

**Benefits:**
- **Your Configs Work**: No need to change YAML structure
- **Pipeline Compatible**: Works seamlessly with `MorseGraphPipeline.run_stage_1_3d()` and `run_stage_5_latent_morse()`
- **Caching Compatible**: Returns format compatible with `save_morse_graph_data()` / `load_morse_graph_data()`
- **Visualization Compatible**: Data format matches what your plot functions expect

**Implementation Priority:** High - This preserves your workflow and ensures pipeline compatibility

---

### 3. Pipeline-Compatible Data Extraction Utilities

**Problem:** CMGDB's data structures are rigid, but your pipeline expects specific dict formats with keys like `'morse_graph'`, `'morse_sets'`, `'morse_set_barycenters'`, `'config'`, `'method'`. Your caching system (`save_morse_graph_data()` / `load_morse_graph_data()`) expects these formats.

**Suggestion:** Provide extraction utilities that convert CMGDB data to your pipeline's expected formats, ensuring compatibility with caching and visualization.

```python
# In core.py or utils.py
def extract_cmgdb_to_pipeline_format(
    morse_graph_cmgdb: 'CMGDB.MorseGraph',
    map_graph_cmgdb: 'CMGDB.MapGraph' = None,
    config: 'ExperimentConfig' = None,
    method: str = None,
    computation_time: float = None
) -> Dict[str, Any]:
    """
    Extract CMGDB data in pipeline's expected format.
    
    Returns dict compatible with:
    - save_morse_graph_data() / load_morse_graph_data()
    - Pipeline's morse_graph_3d_data / morse_graph_2d_data
    - Visualization functions (plot_morse_sets_3d_scatter, etc.)
    
    Format matches what pipeline expects:
    {
        'morse_graph': CMGDB.MorseGraph,  # Raw object for advanced use
        'morse_sets': Dict[int, List],  # vertex -> list of boxes
        'morse_set_barycenters': Dict[int, List],  # vertex -> list of barycenters
        'config': dict,  # Config dict for reproducibility
        'method': str,  # Optional: 'data', 'restricted', 'full', 'enclosure'
        'computation_time': float,  # Optional: timing info
    }
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
        result['config'] = config.to_dict()
    if method is not None:
        result['method'] = method
    if computation_time is not None:
        result['computation_time'] = computation_time
    
    return result

# Helper for backward compatibility with NetworkX code
def extract_cmgdb_to_networkx(morse_graph_cmgdb, map_graph_cmgdb) -> nx.DiGraph:
    """
    Convert CMGDB MorseGraph to NetworkX format for existing code.
    
    Useful for functions that still expect NetworkX graphs.
    """
    # Implementation converts CMGDB format to NetworkX
    # ... (see suggestion #6 for details)
    pass
```

**Benefits:**
- **Pipeline Compatible**: Returns format that works with your caching system
- **Visualization Compatible**: Data format matches what your plot functions expect
- **Caching Compatible**: Can be saved/loaded with `save_morse_graph_data()` / `load_morse_graph_data()`
- **Backward Compatible**: Can also convert to NetworkX for existing code

**Implementation Priority:** High - This ensures pipeline compatibility

---

### 4. Preserve Your Visualization Tools and Ensure Data Compatibility

**Current State:** You have extensive visualization tools (`plot.py`) that expect specific data formats. Your pipeline calls functions like `plot_morse_sets_3d_scatter()`, `plot_morse_graph_diagram()`, `plot_2x2_morse_comparison()`, etc., which expect CMGDB.MorseGraph objects and specific dict structures.

**Suggestion:** **Ensure CMGDB integration provides data in formats your visualization functions expect.**

```python
# Your plot functions stay the same - they expect:
# - morse_graph: CMGDB.MorseGraph object
# - morse_set_barycenters: Dict[int, List[np.ndarray]]
# - domain_bounds: np.ndarray

# Pipeline's visualization calls (from run_stage_1_3d):
plot_morse_graph_diagram(
    morse_graph_data['morse_graph'],  # CMGDB.MorseGraph object
    output_path,
    title
)

plot_morse_sets_3d_scatter(
    morse_graph_data['morse_graph'],  # CMGDB.MorseGraph object
    domain_bounds,
    output_path,
    title=...,
    labels=...
)

plot_morse_sets_3d_projections(
    morse_graph_data['morse_graph'],  # CMGDB.MorseGraph object
    morse_graph_data['morse_set_barycenters'],  # Dict[int, List]
    output_dir,
    system_name,
    domain_bounds,
    prefix="03"
)

# CMGDB integration must provide data in this format
# The extract_cmgdb_to_pipeline_format() function ensures this
```

**Key Compatibility Points:**
1. **CMGDB.MorseGraph objects**: Your plots use `morse_graph.num_vertices()`, `morse_graph.morse_set_boxes()`, etc.
2. **Barycenters format**: `Dict[int, List[np.ndarray]]` where each list contains barycenters for that Morse set
3. **Domain bounds**: `np.ndarray` shape `(2, D)` with `[[min_vals], [max_vals]]`
4. **Comparison plots**: Expect both 3D and 2D Morse graphs in CMGDB format

**Benefits:**
- **Your Visualizations Work**: No need to rewrite plot functions
- **Pipeline Compatible**: Data format matches what pipeline passes to plots
- **CMGDB Integration**: Can use CMGDB's native objects directly
- **Flexible**: Add new visualizations using CMGDB objects

**Implementation Priority:** High - Ensures visualization workflow works unchanged

---

### 5. Support All Pipeline Computation Methods

**Current State:** Your pipeline supports multiple methods for 2D latent Morse graph computation: `'data'`, `'restricted'`, `'full'`, `'enclosure'`. Each method has different requirements and produces different results.

**Suggestion:** **Ensure CMGDB integration supports all methods your pipeline uses, with proper data format conversion.**

```python
# Pipeline's run_stage_5_latent_morse() supports these methods:
def compute_morse_graph_2d_for_pipeline(
    method: str,  # 'data', 'restricted', 'full', 'enclosure'
    config: ExperimentConfig,
    latent_bounds: np.ndarray,
    encoder, decoder, latent_dynamics,  # For learned methods
    Z_grid_encoded=None, G_Z_grid_encoded=None,  # For data method
    device=None
) -> Dict[str, Any]:
    """
    Compute 2D Morse graph using specified method, returning pipeline format.
    
    Methods:
    - 'data': Uses BoxMapData with encoded grid points
    - 'restricted': Uses BoxMapLearnedLatent with restricted domain
    - 'full': Uses BoxMapLearnedLatent on full domain
    - 'enclosure': Uses BoxMapLearnedLatent with corner evaluation + padding
    
    Returns pipeline-compatible dict with 'method' key.
    """
    if method == 'data':
        # Uses BoxMapData - needs encoded grid points
        box_map_2d = BoxMapData(
            Z_grid_encoded, G_Z_grid_encoded,
            grid=data_grid,
            map_empty='outside',
            output_enclosure='box_enclosure'
        )
        morse_graph, morse_sets, barycenters = compute_morse_graph_2d_data(
            box_map_2d, latent_bounds,
            config.latent_subdiv_min,
            config.latent_subdiv_max,
            config.latent_subdiv_init,
            config.latent_subdiv_limit
        )
    
    elif method in ['full', 'restricted']:
        # Uses BoxMapLearnedLatent
        dynamics = BoxMapLearnedLatent(
            latent_dynamics, device,
            padding=config.latent_padding if config.latent_padding else 0.0,
            allowed_indices=None if method == 'full' else allowed_indices
        )
        result = _compute_method_learned(...)
        morse_graph = result['morse_graph']
        morse_sets = {i: morse_graph.morse_set_boxes(i) 
                     for i in range(morse_graph.num_vertices())}
        barycenters = result['barycenters']
    
    elif method == 'enclosure':
        # Similar to 'full' but with specific evaluation method
        # (corner evaluation + padding)
        result = _compute_method_learned(...)
        morse_graph = result['morse_graph']
        morse_sets = {i: morse_graph.morse_set_boxes(i) 
                     for i in range(morse_graph.num_vertices())}
        barycenters = result['barycenters']
    
    # Return in pipeline format
    return {
        'morse_graph': morse_graph,
        'morse_sets': morse_sets,
        'morse_set_barycenters': barycenters,
        'config': config.to_dict(),
        'method': method  # Important for pipeline's generate_comparisons()
    }
```

**Benefits:**
- **All Methods Supported**: Pipeline can use any method seamlessly
- **Consistent Format**: All methods return same pipeline-compatible format
- **Method Tracking**: 'method' key allows pipeline to handle method-specific logic
- **Caching Compatible**: All methods can be cached using same system

**Implementation Priority:** High - Required for pipeline's method comparison features

---

### 6. Enhanced MORALS Workflow Integration

**Current State:** MORALS workflows exist but are somewhat separate from core MorseGraph functionality.

**Suggestion:** Deeply integrate MORALS approach as a first-class feature of MorseGraph.

```python
# In core.py or new morals.py module
class MORALSWorkflow:
    """
    MORALS (Morse Graph-aided discovery of Regions of Attraction in Learned Latent Space)
    workflow integrated into MorseGraph.
    
    This extends CMGDB with:
    - Autoencoder training for dimensionality reduction
    - Latent space Morse graph computation
    - Comparison between full-space and latent-space dynamics
    """
    
    def __init__(self, config):
        self.config = config
        self.encoder = None
        self.decoder = None
        self.latent_dynamics = None
    
    def train_autoencoder(self, X, Y):
        """
        Train autoencoder and latent dynamics models.
        
        Uses MORALS-style training with:
        - Reconstruction loss
        - Dynamics reconstruction loss
        - Dynamics consistency loss
        """
        # Train encoder, decoder, latent_dynamics
        # Using MorseGraph's training utilities
        pass
    
    def compute_latent_morse_graph(self, latent_bounds, method='data'):
        """
        Compute Morse graph in latent space using CMGDB.
        
        Methods:
        - 'data': Use BoxMapData with encoded trajectories
        - 'learned': Use BoxMapLearnedLatent with neural network
        - 'restricted': Restricted domain evaluation
        """
        # Use CMGDB via MorseGraph wrapper
        # Integrates with learned models
        pass
    
    def compare_3d_vs_latent(self):
        """
        Compare 3D ground truth with 2D latent approximation.
        
        MORALS-specific analysis:
        - Preimage classification
        - Barycenter projection
        - Agreement metrics
        """
        pass
```

**Benefits:**
- **Integrated Workflow**: MORALS becomes core MorseGraph feature
- **CMGDB Integration**: Uses CMGDB for both 3D and latent computations
- **Extended Analysis**: Adds comparison tools, preimage analysis, etc.
- **Research-Ready**: Complete pipeline for learned dynamics analysis

**Implementation Priority:** High - This is a key differentiator for MorseGraph

---

### 7. Caching System Integration

**Current State:** Your pipeline has sophisticated hash-based caching (`compute_cmgdb_3d_hash()`, `compute_cmgdb_2d_hash()`, etc.) that saves/loads data using `save_morse_graph_data()` / `load_morse_graph_data()`. The caching system expects specific data formats.

**Suggestion:** **Ensure CMGDB integration works seamlessly with your caching system.**

```python
# Your caching functions expect this format:
def save_morse_graph_data(directory: str, data: Dict[str, Any]) -> None:
    """
    Save Morse graph data in pipeline's format.
    
    Expects data dict with keys:
    - 'morse_graph': CMGDB.MorseGraph (will be pickled)
    - 'morse_sets': Dict[int, List] (will be saved)
    - 'morse_set_barycenters': Dict[int, List] (will be saved as .npz)
    - 'config': dict (will be saved as JSON)
    - 'method': str (optional, for 2D computations)
    """
    # Save CMGDB object using CMGDB's SaveMorseGraphData
    # Save barycenters as .npz
    # Save config as JSON
    # Save metadata
    pass

# CMGDB integration should produce data compatible with this
def compute_and_cache_morse_graph_3d(
    dynamics,
    domain_bounds,
    config: ExperimentConfig,
    cache_dir: str,
    force_recompute: bool = False
) -> Tuple[Dict[str, Any], bool]:
    """
    Compute 3D Morse graph using CMGDB, with caching support.
    
    Returns:
        (morse_graph_data, was_cached) tuple
        morse_graph_data is compatible with save_morse_graph_data()
    """
    # Check cache
    if not force_recompute:
        cached = load_morse_graph_data(cache_dir)
        if cached is not None:
            return cached, True
    
    # Compute using CMGDB
    morse_graph, morse_sets, barycenters = compute_morse_graph_3d(
        dynamics, domain_bounds,
        subdiv_min=config.subdiv_min,
        subdiv_max=config.subdiv_max,
        subdiv_init=config.subdiv_init,
        subdiv_limit=config.subdiv_limit
    )
    
    # Format for pipeline
    morse_graph_data = extract_cmgdb_to_pipeline_format(
        morse_graph,
        config=config,
        computation_time=...
    )
    
    # Save to cache
    save_morse_graph_data(cache_dir, morse_graph_data)
    
    return morse_graph_data, False
```

**Benefits:**
- **Caching Works**: CMGDB computations can be cached using your existing system
- **Pipeline Compatible**: Cache format matches what pipeline expects
- **Efficient**: Avoid recomputation when parameters haven't changed
- **Reproducible**: Config saved with results for reproducibility

**Implementation Priority:** High - Required for pipeline's caching workflow

**Note on Current Caching Implementation:**
Your `save_morse_graph_data()` currently converts CMGDB.MorseGraph to NetworkX for saving. Consider:
- **Option 1**: Keep conversion (current approach) - simpler, but loses CMGDB-specific features
- **Option 2**: Use CMGDB's `SaveMorseGraphData()` - preserves CMGDB format, but need to adapt loading
- **Option 3**: Save both formats - NetworkX for compatibility, CMGDB format for advanced use

Recommendation: Option 3 for maximum flexibility while maintaining compatibility.

---

### 8. CMGDB ↔ NetworkX Conversion Utilities (For Backward Compatibility)

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
- Seamless backward compatibility with existing code
- Can use CMGDB performance with NetworkX API
- Gradual migration path
- Works with `Model.compute_box_map()` returning NetworkX format

**Implementation Priority:** High (required for backward compatibility)

**Note:** Your `save_morse_graph_data()` already converts CMGDB to NetworkX. The conversion utilities should ensure this works correctly and can be reversed if needed.

---

### 9. Support Direct Usage Patterns (Examples)

**Current State:** Your examples show both pipeline usage (`7_ives_model.py`) and direct usage (`6_map_learned_dynamics.py`, `1_map_dynamics.py`, etc.). Direct usage patterns use `Model.compute_box_map()` and `compute_morse_graph()`.

**Suggestion:** **Ensure CMGDB integration supports both pipeline and direct usage patterns.**

```python
# Direct usage pattern (from examples/1_map_dynamics.py):
model = Model(grid, dynamics)
box_map = model.compute_box_map()  # Currently pure Python
morse_graph = compute_morse_graph(box_map)  # NetworkX format
basins = compute_all_morse_set_basins(morse_graph, box_map)

# With CMGDB integration, this should still work:
# Option 1: Model.compute_box_map() uses CMGDB internally, returns NetworkX
model = Model(grid, dynamics)
box_map = model.compute_box_map()  # Uses CMGDB, returns NetworkX (backward compatible)
morse_graph = compute_morse_graph(box_map)  # Works as before

# Option 2: Direct CMGDB usage for advanced users
morse_graph_cmgdb, morse_sets, barycenters = compute_morse_graph_3d(
    dynamics, domain_bounds, subdiv_min=20, subdiv_max=28
)
# Use CMGDB objects directly for advanced features

# Option 3: Pipeline usage (unchanged)
pipeline = MorseGraphPipeline(config_path='configs/ives_default.yaml')
pipeline.run()  # Uses CMGDB internally, everything works
```

**Benefits:**
- **Backward Compatible**: Existing example scripts continue to work
- **Pipeline Compatible**: Pipeline workflow unchanged
- **Flexible**: Advanced users can use CMGDB objects directly
- **Gradual Migration**: Can migrate code incrementally

**Implementation Priority:** High - Ensures all usage patterns work

---

### 10. Extended Analysis Tools Beyond CMGDB

**Current State:** CMGDB provides core Morse graph computation. MorseGraph adds some analysis.

**Suggestion:** Expand MorseGraph's analysis toolkit to provide comprehensive tools.

```python
# In analysis.py - Extended tools
class MorseGraphAnalyzer:
    """
    Extended analysis tools that build on CMGDB's core computation.
    
    Provides:
    - Basin analysis (beyond CMGDB's basic functionality)
    - Stability analysis
    - Parameter sensitivity
    - Comparison tools
    - Statistical analysis
    """
    
    def analyze_basins(self, morse_graph_cmgdb, map_graph_cmgdb):
        """
        Comprehensive basin analysis using CMGDB data.
        
        Extends CMGDB with:
        - Basin volume computation
        - Basin boundary analysis
        - Transient time analysis
        - Basin stability metrics
        """
        pass
    
    def compare_morse_graphs(self, mg1, mg2):
        """
        Compare two Morse graphs (e.g., 3D vs latent).
        
        MORALS-specific:
        - Agreement metrics
        - Preimage analysis
        - Correspondence mapping
        """
        pass
    
    def parameter_sensitivity(self, dynamics_func, param_ranges):
        """
        Analyze how Morse graph changes with parameters.
        
        Uses CMGDB for each parameter configuration.
        """
        pass
    
    def stability_analysis(self, morse_graph_cmgdb):
        """
        Analyze stability properties of Morse sets.
        
        Extends CMGDB with:
        - Lyapunov-like analysis
        - Stability indices
        - Robustness metrics
        """
        pass
```

**Benefits:**
- **Extended Functionality**: Tools beyond what CMGDB provides
- **Research Tools**: Comprehensive analysis for research workflows
- **MORALS Integration**: Specific tools for learned dynamics analysis
- **Python Ecosystem**: Leverages NumPy, SciPy, NetworkX, etc.

**Implementation Priority:** Medium - Adds value beyond core CMGDB functionality

---

### 11. Unified CMGDB Interface for All Dimensions

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

### 12. Batch CMGDB Computations

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

### CRITICAL Priority (Foundation - Pipeline Compatibility)

1. **CMGDB as Computation Backend - Pipeline-Compatible Wrapper**
   - Use CMGDB for computation (fast C++ backend)
   - Extract data in pipeline's expected format (dicts with 'morse_graph', 'morse_sets', 'morse_set_barycenters', 'config', 'method')
   - **Compatible with caching system** (`save_morse_graph_data()` / `load_morse_graph_data()`)
   - **Compatible with visualization** (plot functions expect CMGDB.MorseGraph objects)
   - **Preserve pipeline structure** (MorseGraphPipeline works unchanged)
   - **This ensures pipeline compatibility**

2. **Pipeline-Compatible Data Extraction**
   - Extract CMGDB data in format pipeline expects
   - Return dicts compatible with `save_morse_graph_data()` / `load_morse_graph_data()`
   - Ensure format matches what visualization functions expect
   - Support all pipeline methods ('data', 'restricted', 'full', 'enclosure')
   - **This ensures seamless pipeline integration**

3. **YAML Config System Integration**
   - CMGDB integration reads from YOUR configs (cmgdb_3d, cmgdb_2d sections)
   - No changes to config structure needed
   - Works with `load_experiment_config()` as-is
   - Parameters read from config (subdiv_min, subdiv_max, etc.)
   - **This preserves your configuration workflow**

4. **Visualization Compatibility**
   - CMGDB integration provides CMGDB.MorseGraph objects (what plots expect)
   - Barycenters in format `Dict[int, List[np.ndarray]]`
   - Domain bounds in format `np.ndarray` shape `(2, D)`
   - All plot functions work unchanged
   - **This preserves your visualization workflow**

5. **Caching System Integration**
   - CMGDB computations compatible with hash-based caching
   - Data format works with `save_morse_graph_data()` / `load_morse_graph_data()`
   - Cache keys include all relevant parameters
   - **This preserves your caching workflow**

6. **Support All Pipeline Methods**
   - Ensure 'data', 'restricted', 'full', 'enclosure' methods all work
   - Each method returns pipeline-compatible format
   - Method tracking via 'method' key in result dict
   - **This enables pipeline's method comparison features**

### High Priority (Key Differentiators)

7. **Enhanced MORALS Workflow Integration**
   - Integrate MORALS with YOUR pipeline structure
   - Works with YOUR YAML configs
   - Autoencoder + latent dynamics workflows
   - Comparison tools (3D vs latent)
   - Preimage classification (when BoxMapData available)
   - **This extends your pipeline, doesn't replace it**

8. **Support Direct Usage Patterns**
   - Ensure `Model.compute_box_map()` works with CMGDB backend
   - Backward compatible with existing example scripts
   - Can return NetworkX format for existing code
   - Advanced users can access CMGDB objects directly
   - **This ensures all usage patterns work**

9. **Extended Analysis Tools**
   - Basin analysis beyond CMGDB basics
   - Stability analysis
   - Parameter sensitivity
   - Comparison utilities (3D vs 2D, method comparison)
   - **Extends CMGDB functionality while preserving your tools**

### Medium Priority (Significant Benefits)

10. **CMGDB ↔ NetworkX Conversion Utilities** - For backward compatibility with existing code that uses NetworkX
11. **Batch CMGDB Computations** - Parameter sweeps using YOUR config system
12. **CMGDB Parameter Auto-tuning** - Auto-tune but save to YOUR config format
13. **Incremental Updates** - Leverage CMGDB for refinement, extract data in pipeline format
14. **Streaming for Very Large Grids** - Handle edge cases if needed (if CMGDB doesn't cover)
15. **Preimage Classification Support** - Store BoxMapData objects in cache for preimage analysis

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

**Key Architectural Vision: MorseGraph Respects Your Pipeline**

MorseGraph's role is to:
1. **Use CMGDB for Computation**: Fast C++ backend for heavy lifting
2. **Extract Data Flexibly**: Don't force CMGDB's rigid structure
3. **Preserve Your Tools**: YAML configs, pipeline, visualizations all work
4. **Extend with MORALS**: Integrate learned dynamics with YOUR workflow
5. **Add Analysis Tools**: Beyond CMGDB, compatible with YOUR tools

**Architecture - Respecting Your Pipeline:**
```
┌─────────────────────────────────────────┐
│      YOUR Pipeline & Tools               │
│  - YAML Configs (preserved)              │
│  - MorseGraphPipeline (preserved)       │
│  - plot.py visualizations (preserved)    │
│  - Your data formats (preserved)          │
└──────────────┬──────────────────────────┘
               │ Uses
┌──────────────▼──────────────────────────┐
│         MorseGraph (Python)              │
│  - Flexible CMGDB wrapper                 │
│  - Data extraction utilities               │
│  - MORALS workflows                       │
│  - Extended analysis                      │
└──────────────┬──────────────────────────┘
               │ Uses for computation
┌──────────────▼──────────────────────────┐
│         CMGDB (C++ Backend)              │
│  - Fast computation                       │
│  - But data extracted flexibly            │
└─────────────────────────────────────────┘
```

**Migration Strategy - Pipeline-First Approach:**
1. **Phase 1**: Update `compute_morse_graph_3d()` and `compute_morse_graph_2d_data()` to return pipeline-compatible formats
2. **Phase 2**: Ensure data extraction utilities produce format compatible with `save_morse_graph_data()` / `load_morse_graph_data()`
3. **Phase 3**: Verify pipeline's `run_stage_1_3d()` and `run_stage_5_latent_morse()` work with CMGDB data
4. **Phase 4**: Ensure visualization functions receive correct data formats (CMGDB.MorseGraph objects, barycenters dicts)
5. **Phase 5**: Update `Model.compute_box_map()` to use CMGDB internally (backward compatible NetworkX output)
6. **Phase 6**: Add NetworkX conversion utilities for existing code that needs it
7. **Phase 7**: Enhance MORALS workflows (uses YOUR pipeline structure)
8. **Phase 8**: Add extended tools (compatible with YOUR formats)

**Value Proposition:**
- **CMGDB Performance**: Fast computation when you need it
- **Your Tools Work**: YAML configs, pipeline, visualizations unchanged
- **Flexible Data**: Extract CMGDB data in formats YOU need
- **Not Forced**: Don't have to use CMGDB's rigid structure
- **MORALS Integration**: Complete workflows with YOUR config system

**Key Principles:**
- **Pipeline Compatibility First**: Ensure CMGDB integration works seamlessly with your pipeline
- **Data Format Compatibility**: Extract CMGDB data in format your caching and visualization systems expect
- **Config System Integration**: Read parameters from your YAML configs, don't force new structure
- **Backward Compatibility**: Existing code (examples, direct usage) continues to work
- **Your Tools First**: CMGDB serves your tools (pipeline, configs, visualizations), not the other way around
- **Best of Both**: CMGDB's speed + your flexibility + pipeline compatibility

**Specific Pipeline Compatibility Requirements:**

1. **Data Structures**: 
   - Return dicts with keys: `'morse_graph'` (CMGDB.MorseGraph), `'morse_sets'` (Dict[int, List]), `'morse_set_barycenters'` (Dict[int, List[np.ndarray]]), `'config'` (dict), `'method'` (str, optional)
   - Format matches what `save_morse_graph_data()` / `load_morse_graph_data()` expect

2. **Caching Compatibility**: 
   - Data format works with hash-based caching (`compute_cmgdb_3d_hash()`, `compute_cmgdb_2d_hash()`)
   - Can be saved/loaded using existing caching utilities
   - Config included for reproducibility

3. **Visualization Compatibility**: 
   - Provide CMGDB.MorseGraph objects (plots use `morse_graph.num_vertices()`, `morse_graph.morse_set_boxes()`, etc.)
   - Barycenters in format `Dict[int, List[np.ndarray]]` (what plots expect)
   - Domain bounds as `np.ndarray` shape `(2, D)`

4. **Config Integration**: 
   - Read parameters from YAML configs: `config.subdiv_min`, `config.subdiv_max`, `config.subdiv_init`, `config.subdiv_limit`, `config.padding`
   - Support both `cmgdb_3d` and `cmgdb_2d` sections
   - No changes to config structure needed

5. **Method Support**: 
   - All methods ('data', 'restricted', 'full', 'enclosure') return consistent format
   - Method tracked via 'method' key for pipeline's `generate_comparisons()`
   - Each method works with pipeline's `run_stage_5_latent_morse()`

6. **Pipeline Stages**: 
   - `run_stage_1_3d()`: Works with CMGDB data format
   - `run_stage_5_latent_morse()`: Supports all methods seamlessly
   - `generate_comparisons()`: Can compare 3D vs 2D using CMGDB objects

7. **Example Scripts**: 
   - Direct usage (`Model.compute_box_map()`) works with CMGDB backend
   - Pipeline usage (`MorseGraphPipeline.run()`) works unchanged
   - Both patterns supported simultaneously

