# MorseGraph - Complete Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Python API Reference](#python-api-reference)
5. [Data Structures](#data-structures)
6. [Algorithms](#algorithms)
7. [File Structure](#file-structure)
8. [Usage Patterns](#usage-patterns)
9. [Integration with CMGDB](#integration-with-cmgdb)

---

## Overview

**MorseGraph** is a lightweight Python library for computing and analyzing Morse graphs of dynamical systems using computational topology methods. It provides tools to study the global structure of dynamical systems through discrete abstractions on grids, implementing outer approximations of dynamics and using graph-theoretic methods to identify attractors, basins of attraction, and connecting orbits.

### Mathematical Foundation

- **State Space Discretization**: The state space $X$ is partitioned into boxes using a grid
- **Box Map**: For each box $B$, compute the image $F(B)$ under the dynamics
- **Outer Approximation**: Use epsilon-bloating to ensure rigorous outer approximation
- **Morse Decomposition**: Identify strongly connected components (SCCs) in the transition graph
- **Morse Sets**: Non-trivial SCCs (multi-node or single-node with self-loops) represent recurrent dynamics
- **Basins of Attraction**: All boxes that eventually flow into a Morse set

### Key Features

- Support for discrete maps, ODEs, data-driven and latent dynamics
- Outer approximation using epsilon-enlargement of boxes
- Parallel computation of box-to-box transitions
- Morse decomposition via strongly connected components
- Computation of basin of attraction
- ML integration for learning low-dimensional representations
- Built-in visualization utilities
- Integration with CMGDB for high-performance 3D computations

---

## Architecture

MorseGraph follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  (Examples, Pipeline, Config Management)                    │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│   Analysis     │  │     Plot       │  │   Pipeline     │
│  (Morse Graph) │  │  (Visualize)   │  │  (Workflow)    │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│                    Core Computation Layer                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │  Model   │  │  Grids   │  │ Dynamics │                │
│  └──────────┘  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│   NetworkX     │  │   NumPy/SciPy  │  │   CMGDB        │
│  (Graph Ops)   │  │  (Computation)  │  │  (C++ Backend) │
└────────────────┘  └────────────────┘  └────────────────┘
```

### Component Layers

1. **Core Layer**: `Model`, `Grid`, `Dynamics` - Fundamental abstractions
2. **Analysis Layer**: Morse graph computation, basin analysis
3. **Visualization Layer**: Plotting utilities for Morse sets, graphs, basins
4. **Pipeline Layer**: High-level workflow automation
5. **Integration Layer**: CMGDB for 3D computations, PyTorch for ML

---

## Core Components

### Grids (`grids.py`)

#### `AbstractGrid` (ABC)

Abstract base class for grid discretization.

**Methods:**
- `get_boxes() -> np.ndarray`: Return all boxes in the grid (shape: `(N, 2, D)`)
- `box_to_indices(box: np.ndarray) -> np.ndarray`: Find grid box indices intersecting a given box
- `subdivide(indices: np.ndarray = None)`: Subdivide grid (global or local)
- `dilate_indices(indices: np.ndarray, radius: int = 1) -> np.ndarray`: Expand indices to include spatial neighbors

#### `UniformGrid`

Uniform rectangular grid with subdivision support.

**Initialization:**
```python
grid = UniformGrid(bounds=np.array([[x_min, y_min], [x_max, y_max]]), 
                    divisions=np.array([nx, ny]))
```

**Attributes:**
- `bounds`: Domain bounds (shape: `(2, D)`)
- `divisions`: Number of divisions per dimension (shape: `(D,)`)
- `dim`: Dimension of state space
- `box_size`: Size of each box (shape: `(D,)`)

**Methods:**
- `box_to_indices_batch(boxes: np.ndarray) -> list`: Vectorized index computation for multiple boxes
- `dilate_indices()`: Grid dilation using spatial indexing

#### `AdaptiveGrid` (EXPERIMENTAL)

Tree-based adaptive grid supporting local refinement.

**Initialization:**
```python
grid = AdaptiveGrid(bounds=np.array([[x_min, y_min], [x_max, y_max]]), 
                     max_depth=10)
```

**Features:**
- Tree structure with leaf nodes representing active boxes
- Efficient spatial indexing via leaf cache
- Local subdivision support
- **Warning**: Experimental, use `UniformGrid` for production

### Dynamics (`dynamics.py`)

#### `Dynamics` (ABC)

Abstract base class for dynamical systems.

**Methods:**
- `__call__(box: np.ndarray) -> np.ndarray`: Apply dynamics to a box, return image box
- `get_active_boxes(grid) -> np.ndarray`: Return indices of boxes with meaningful dynamics

#### `BoxMapFunction`

Explicit function-based dynamics with interval arithmetic.

**Initialization:**
```python
dynamics = BoxMapFunction(map_f=henon_map, 
                         epsilon=0.01,
                         evaluation_method="corners",  # or "center", "random"
                         num_random_points=10)
```

**Features:**
- Epsilon-bloating for rigorous outer approximation
- Multiple evaluation methods: corners, center, random sampling
- Works with any callable function `f: R^D -> R^D`

**Example:**
```python
from MorseGraph.systems import henon_map
dynamics = BoxMapFunction(henon_map, epsilon=0.01)
image_box = dynamics(box)  # box shape: (2, D)
```

#### `BoxMapODE`

ODE-based dynamics using scipy integration.

**Initialization:**
```python
dynamics = BoxMapODE(ode_f=van_der_pol_ode, 
                     tau=1.0,  # integration time
                     epsilon=0.0)  # bloating factor
```

**Features:**
- Integrates ODE from box corners and center
- Computes bounding box of resulting points
- Supports any ODE function compatible with `scipy.integrate.solve_ivp`

**Example:**
```python
from MorseGraph.systems import van_der_pol_ode
dynamics = BoxMapODE(van_der_pol_ode, tau=1.0, epsilon=0.01)
```

#### `BoxMapData`

Data-driven dynamics optimized for uniform grids.

**Initialization:**
```python
dynamics = BoxMapData(X, Y, grid,
                     input_distance_metric='L1',  # or 'L2'
                     output_distance_metric='L1',
                     input_epsilon=None,  # defaults to grid.box_size
                     output_epsilon=None,
                     map_empty='interpolate',  # or 'outside', 'terminate'
                     k_neighbors=5,
                     force_interpolation=False,
                     output_enclosure='box_enclosure')  # or 'box_union'
```

**Features:**
- Pre-assigns data points to grid boxes for performance
- Supports L1 (Manhattan) and L2 (Euclidean) neighborhoods
- Multiple strategies for empty boxes: interpolate, map outside, terminate
- Two output enclosure strategies: box_enclosure (filled rectangle) or box_union (sparse)

**Distance Metrics:**
- `L1`: Axis-aligned rectangular neighborhoods (face-adjacent boxes)
- `L2`: Ball-based neighborhoods (includes corner-touching boxes)

**Empty Box Handling:**
- `interpolate`: Use k-nearest neighbors to estimate dynamics
- `outside`: Map empty boxes outside the domain
- `terminate`: Raise error if empty boxes encountered

**Example:**
```python
X, Y, _ = generate_trajectory_data(ode_func, params, n_samples, ...)
dynamics = BoxMapData(X, Y, grid, 
                     input_epsilon=grid.box_size,
                     output_epsilon=grid.box_size)
```

#### `BoxMapLearnedLatent` (Optional PyTorch)

Learned latent dynamics using neural networks.

**Initialization:**
```python
dynamics = BoxMapLearnedLatent(latent_dynamics_model=G,
                               device=device,
                               padding=0.0,  # epsilon for output bloating
                               allowed_indices=None)  # restrict to subset of boxes
```

**Features:**
- Supports full domain evaluation (Method 2)
- Supports restricted domain evaluation (Method 3) via `allowed_indices`
- Rigorous padding using L-infinity ball
- Samples corners, center, and face midpoints for better coverage

**Example:**
```python
import torch
encoder = Encoder(...)
decoder = Decoder(...)
latent_dynamics = LatentDynamics(...)
# Train models...

# Full domain
dynamics = BoxMapLearnedLatent(latent_dynamics, device, padding=0.01)

# Restricted domain
allowed = set(range(100, 200))  # Only compute for specific boxes
dynamics = BoxMapLearnedLatent(latent_dynamics, device, 
                               padding=0.01, allowed_indices=allowed)
```

### Core (`core.py`)

#### `Model`

The core engine connecting grids and dynamics.

**Initialization:**
```python
model = Model(grid=grid, 
              dynamics=dynamics,
              dynamics_kwargs=None)  # optional kwargs for dynamics.__call__
```

**Methods:**
- `compute_box_map(n_jobs: int = -1) -> nx.DiGraph`: Compute box-to-box transition graph
  - Uses parallel processing via `joblib`
  - Supports batch mode if grid has `box_to_indices_batch`
  - Returns NetworkX directed graph

**Example:**
```python
model = Model(grid, dynamics)
box_map = model.compute_box_map(n_jobs=-1)  # Use all CPUs
```

#### CMGDB Integration Functions

**`compute_morse_graph_3d()`**

Compute 3D Morse graph using CMGDB backend.

```python
morse_graph, morse_sets, barycenters = compute_morse_graph_3d(
    dynamics=dynamics,
    domain_bounds=[[x_min, y_min, z_min], [x_max, y_max, z_max]],
    subdiv_min=30,
    subdiv_max=42,
    subdiv_init=0,
    subdiv_limit=10000,
    verbose=True
)
```

**Returns:**
- `morse_graph`: CMGDB MorseGraph object
- `morse_sets`: Dict mapping vertex index to list of boxes
- `barycenters`: Dict mapping vertex index to list of barycenter coordinates

**`compute_morse_graph_2d_data()`**

Compute 2D Morse graph using CMGDB for data-driven dynamics.

```python
morse_graph, morse_sets, barycenters = compute_morse_graph_2d_data(
    dynamics=box_map_data,
    domain_bounds=[[x_min, y_min], [x_max, y_max]],
    subdiv_min=20,
    subdiv_max=28,
    ...
)
```

---

## Python API Reference

### Analysis (`analysis.py`)

#### `compute_morse_graph(box_map: nx.DiGraph, assign_colors: bool = True, cmap_name: str = 'tab10') -> nx.DiGraph`

Compute Morse graph from BoxMap.

**Algorithm:**
1. Find all strongly connected components (SCCs)
2. Identify non-trivial SCCs:
   - Multi-node SCCs (recurrent components)
   - Single-node SCCs with self-loops (fixed points)
3. Build condensation graph
4. Extract connectivity between non-trivial SCCs

**Returns:** NetworkX DiGraph where nodes are frozensets of box indices (Morse sets)

**Example:**
```python
morse_graph = compute_morse_graph(box_map)
for morse_set in morse_graph.nodes():
    print(f"Morse set: {morse_set}, size: {len(morse_set)}")
```

#### `compute_all_morse_set_basins(morse_graph: nx.DiGraph, box_map: nx.DiGraph) -> Dict[FrozenSet[int], Set[int]]`

Compute basin of attraction for each Morse set.

**Algorithm:**
- Uses condensation-based algorithm for O(boxes + edges) complexity
- For each transient box, finds reachable Morse sets
- Assigns to basin of highest Morse set (earliest in topological order)

**Returns:** Dictionary mapping Morse set (frozenset) to basin (set of box indices)

**Example:**
```python
basins = compute_all_morse_set_basins(morse_graph, box_map)
for morse_set, basin in basins.items():
    print(f"Morse set {morse_set}: basin size {len(basin)}")
```

#### `compute_basins_of_attraction(morse_graph: nx.DiGraph, box_map: nx.DiGraph) -> Dict[FrozenSet[int], Set[int]]`

Compute basins for attractors only (Morse sets with out_degree == 0).

**Note:** For Morse-structure-aware basins, use `compute_all_morse_set_basins()`.

#### `iterative_morse_computation(model, max_depth: int = 5, refinement_threshold: float = 0.1, neighborhood_radius: int = 1, criterion: str = 'volume', criterion_kwargs: dict = None)`

Iteratively compute Morse graphs with adaptive grid refinement.

**Workflow:**
1. Compute initial Morse graph
2. Identify boxes to refine using criterion
3. Subdivide grid locally
4. Update BoxMap for affected nodes
5. Repeat until convergence or max_depth

**Refinement Criteria:**
- `'volume'`: Refine Morse sets with relative volume >= threshold
- `'diameter'`: Refine boxes where image diameter > expansion_threshold * box diameter

**Returns:** Tuple of (final_morse_graph, refinement_history)

**Example:**
```python
from MorseGraph.grids import AdaptiveGrid
grid = AdaptiveGrid(bounds, max_depth=10)
model = Model(grid, dynamics)
morse_graph, history = iterative_morse_computation(
    model, 
    max_depth=5,
    refinement_threshold=0.1,
    criterion='volume'
)
```

#### `diameter_criterion(box, image_box, expansion_threshold=2.0) -> bool`

Refinement criterion: refine if image diameter exceeds box diameter significantly.

#### `analyze_refinement_convergence(refinement_history: List[Dict]) -> Dict`

Analyze convergence properties of iterative refinement.

**Returns:** Dictionary with:
- `total_iterations`, `initial_boxes`, `final_boxes`
- `refinement_factor`, `convergence_achieved`
- `box_growth_rate`, `morse_stability`

### Plotting (`plot.py`)

#### 2D Plotting Functions

**`plot_morse_sets(grid, morse_graph, ax=None, box_map=None, show_outside=False, **kwargs)`**

Plot Morse sets on 2D grid.

**`plot_basins_of_attraction(grid, basins, morse_graph=None, ax=None, show_outside=False, **kwargs)`**

Plot basins of attraction with colors matching Morse sets.

**`plot_morse_graph(morse_graph, ax=None, node_size=300, arrowsize=20, font_size=8)`**

Plot Morse graph structure with hierarchical layout.

**`plot_data_coverage(grid, box_map_data, ax=None, colormap='viridis')`**

Visualize data point density per box (2D only).

**`plot_data_points_overlay(grid, X, Y, ax=None, max_arrows=500)`**

Overlay data points (X as dots, Y as arrows) on grid (2D only).

#### 3D Plotting Functions

**`plot_morse_sets_3d(grid, morse_graph, ax=None, box_map=None, show_outside=False, alpha=0.3, **kwargs)`**

Plot Morse sets on 3D grid using cuboid visualization.

**`plot_morse_graph_diagram(morse_graph, output_path=None, title="Morse Graph", cmap=None, figsize=(8, 8), ax=None)`**

Plot Morse graph using CMGDB's graphviz plotting.

**`plot_morse_sets_3d_scatter(morse_graph, domain_bounds, output_path=None, title="Morse Sets (3D)", labels=None, equilibria=None, periodic_orbits=None, scale_factor=1.0, data_overlay=None)`**

Plot Morse sets as 3D scatter plot with box centers and marker sizes proportional to box dimensions.

**`plot_morse_sets_3d_projections(morse_graph, barycenters_3d, output_dir, system_name, domain_bounds, cmap=cm.cool, prefix="")`**

Plot 2D projections (xy, xz, yz) of 3D Morse sets.

**`plot_latent_space_2d(z_data, latent_bounds, morse_graph=None, output_path=None, title="Latent Space (2D)", equilibria_latent=None, barycenters_latent=None, ax=None)`**

Plot 2D latent space with data points and optionally Morse sets.

**`plot_2x2_morse_comparison(morse_graph_3d, morse_graph_2d, domain_bounds_3d, latent_bounds_2d, encoder, device, z_data, output_path=None, title_prefix="", equilibria=None, periodic_orbits=None, equilibria_latent=None, labels=None)`**

Create 2x2 comparison figure: 3D/2D Morse graphs and 3D/2D phase portraits.

**`plot_preimage_classification(morse_graph_2d, encoder, decoder, device, X_sample, latent_bounds, domain_bounds, subdiv_max, output_path=None, title_prefix="", method_name="2D Latent", labels=None, max_points_per_set=1000)`**

Visualize preimages of latent Morse sets in original 3D space.

**`plot_training_curves(train_losses, val_losses, output_path=None)`**

Plot training and validation loss curves.

**`plot_trajectory_analysis(trajectories_3d, trajectories_latent, output_path=None, title_prefix="", labels=None, n_steps=None)`**

Visualize trajectory simulations in both original and latent space.

### Systems (`systems.py`)

Pre-defined dynamical systems for testing and examples.

#### Discrete Maps

**`henon_map(x: np.ndarray, a: float = 1.4, b: float = 0.3) -> np.ndarray`**

Hénon map: canonical chaotic 2D discrete system.

**`leslie_map(x: np.ndarray, th1: float = 19.6, th2: float = 23.68, mortality: float = 0.7) -> np.ndarray`**

Leslie matrix population model (2D).

**`leslie_map_3d(x: np.ndarray, theta_1: float = 37.5, theta_2: float = 37.5, theta_3: float = 37.5, survival_1: float = 0.8, survival_2: float = 0.6) -> np.ndarray`**

Three-dimensional Leslie population model.

**`ives_model(x: np.ndarray, r1: float = 3.873, r2: float = 11.746, c: float = 10**-6.435, d: float = 0.5517, p: float = 0.06659, q: float = 0.9026) -> np.ndarray`**

Ives model: 3D ecological system.

**`ives_model_log(x: np.ndarray, ...) -> np.ndarray`**

Ives model in log-space coordinates.

#### ODEs

**`van_der_pol_ode(t: float, y: np.ndarray, mu: float = 1.0) -> list`**

Van der Pol oscillator.

**`toggle_switch_ode(t: float, y: np.ndarray, alpha1: float = 156.25, alpha2: float = 156.25, beta: float = 2.5, gamma: float = 2.0, n: int = 4) -> list`**

Genetic toggle switch (bistable system).

**`lorenz_ode(t: float, state: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0) -> list`**

Lorenz system (chaotic attractor).

### Utilities (`utils.py`)

#### Trajectory Data Generation

**`generate_trajectory_data(ode_func, ode_params, n_samples, total_time, n_points, sampling_domain, random_seed=42, timeskip=0.0, n_jobs=-1)`**

Generate trajectory data from ODE by sampling random initial conditions.

**Returns:** `(X, Y, trajectories)` where:
- `X`: Current states (shape: `(N, D)`)
- `Y`: Next states (shape: `(N, D)`)
- `trajectories`: List of trajectory arrays

**`generate_map_trajectory_data(map_func, n_trajectories, n_points, sampling_domain, random_seed=42, skip_initial=0, n_jobs=-1)`**

Generate trajectory data from discrete map.

#### Configuration

**`ExperimentConfig`**

Configuration class for experiments with attributes:
- `system_type`, `dynamics_name`
- `domain_bounds`
- `subdiv_min`, `subdiv_max`, `subdiv_init`, `subdiv_limit`, `padding`
- `latent_subdiv_min`, `latent_subdiv_max`, etc.
- `n_trajectories`, `n_points`, `skip_initial`, `random_seed`
- `input_dim`, `latent_dim`, `hidden_dim`, `num_layers`
- `num_epochs`, `batch_size`, `learning_rate`
- `w_recon`, `w_dyn_recon`, `w_dyn_cons` (loss weights)

#### Caching Utilities

**`compute_cmgdb_3d_hash(...)`**: Compute hash for 3D CMGDB parameters

**`compute_cmgdb_2d_hash(...)`**: Compute hash for 2D CMGDB parameters

**`compute_trajectory_data_hash(...)`**: Compute hash for trajectory data parameters

**`compute_training_hash(...)`**: Compute hash for training parameters

**`load_morse_graph_data(cache_dir)`**: Load cached Morse graph data

**`save_morse_graph_data(cache_dir, data)`**: Save Morse graph data to cache

**`load_trajectory_data(cache_dir)`**: Load cached trajectory data

**`save_trajectory_data(cache_dir, data)`**: Save trajectory data to cache

**`load_models(cache_dir)`**: Load trained PyTorch models

**`save_models(cache_dir, encoder, decoder, latent_dynamics)`**: Save trained models

### Configuration (`config.py`)

#### `load_experiment_config(config_path: str, system_type: str = 'map', base_config_path: Optional[str] = None, verbose: bool = True) -> ExperimentConfig`

Load ExperimentConfig from YAML file.

**YAML Structure:**
```yaml
system:
  type: map  # or 'ode'
  name: henon_map

domain:
  bounds: [[-2.5, -0.5], [2.5, 0.5]]

cmgdb_3d:
  subdiv_min: 30
  subdiv_max: 42
  subdiv_init: 0
  subdiv_limit: 10000
  padding: true

cmgdb_2d:
  subdiv_min: 20
  subdiv_max: 28
  padding: 0.01
  method: data  # or 'learned'

training:
  n_trajectories: 1000
  n_points: 50
  num_epochs: 300
  batch_size: 64
  learning_rate: 0.001

model:
  input_dim: 3
  latent_dim: 2
  hidden_dim: 64
  num_layers: 3
```

#### System Registry

**`get_system_dynamics(system_type, dynamics_name, **kwargs) -> Callable`**

Retrieve callable dynamics function with parameters applied.

**`get_system_bounds(system_type, dynamics_name) -> np.ndarray`**

Retrieve default domain bounds.

**`get_system_parameters(system_type, dynamics_name) -> Dict[str, Any]`**

Retrieve default parameters.

### Models (`models.py`) - Optional PyTorch

#### `Encoder(input_dim, latent_dim, hidden_dim=64, num_layers=3, output_activation=None)`

Neural network encoder: high-dimensional → low-dimensional.

**Activations:** `None`, `'tanh'`, `'sigmoid'`

#### `Decoder(latent_dim, output_dim, hidden_dim=64, num_layers=3, output_activation=None)`

Neural network decoder: low-dimensional → high-dimensional.

#### `LatentDynamics(latent_dim, hidden_dim=64, num_layers=3, output_activation=None)`

Neural network for latent space dynamics: `G: R^d -> R^d`.

### Training (`training.py`) - Optional PyTorch

#### `train_autoencoder_dynamics(encoder, decoder, latent_dynamics, train_loader, val_loader, num_epochs, device, w_recon=1.0, w_dyn_recon=1.0, w_dyn_cons=1.0, learning_rate=0.001, early_stopping_patience=10, min_delta=1e-6, verbose=True)`

Train autoencoder and latent dynamics models.

**Loss Function:**
```
L = w_recon * ||D(E(x)) - x||² 
  + w_dyn_recon * ||D(G(E(x))) - f(x)||²
  + w_dyn_cons * ||G(E(x)) - E(f(x))||²
```

**Returns:** Dictionary with training history and best models.

### Pipeline (`pipeline.py`)

#### `MorseGraphPipeline`

High-level pipeline for end-to-end workflow.

**Initialization:**
```python
pipeline = MorseGraphPipeline(config_path='configs/ives_default.yaml',
                              output_dir='output')
```

**Methods:**
- `run_stage_1_3d(force_recompute=False)`: Compute 3D Morse graph (ground truth)
- `run_stage_2_trajectories(force_recompute=False)`: Generate trajectory data
- `run_stage_3_training(force_retrain=False)`: Train autoencoder models
- `run_stage_4_latent_bounds()`: Compute latent space bounds
- `run_stage_5_latent_morse(method='data', force_recompute=False)`: Compute 2D Morse graph
- `generate_comparisons()`: Generate comparison visualizations

**Workflow:**
```
Stage 1: 3D Morse Graph (CMGDB)
    ↓
Stage 2: Trajectory Data Generation
    ↓
Stage 3: Autoencoder Training
    ↓
Stage 4: Latent Bounds Computation
    ↓
Stage 5: 2D Latent Morse Graph (CMGDB)
    ↓
Comparisons & Analysis
```

**Example:**
```python
pipeline = MorseGraphPipeline('configs/ives_default.yaml')
pipeline.run_stage_1_3d()
pipeline.run_stage_2_trajectories()
pipeline.run_stage_3_training()
pipeline.run_stage_4_latent_bounds()
pipeline.run_stage_5_latent_morse(method='data')
pipeline.generate_comparisons()
```

---

## Data Structures

### Box Representation

All boxes are numpy arrays of shape `(2, D)` where:
- `box[0]`: Lower bounds (shape: `(D,)`)
- `box[1]`: Upper bounds (shape: `(D,)`)

### Grid Box Array

Grid boxes are stored as numpy array of shape `(N, 2, D)` where:
- `N`: Number of boxes
- Each box is `(2, D)` as above

### BoxMap (NetworkX DiGraph)

- **Nodes**: Box indices (integers)
- **Edges**: `(i, j)` if box `i` maps to box `j`
- **Attributes**: None by default

### Morse Graph (NetworkX DiGraph)

- **Nodes**: Frozensets of box indices (Morse sets)
- **Edges**: `(ms1, ms2)` if there's a path from `ms1` to `ms2` through transient states
- **Node Attributes**: `'color'` (RGBA tuple) assigned by `compute_morse_graph()`

### CMGDB MorseGraph Object

CMGDB returns a C++ object with methods:
- `num_vertices() -> int`: Number of Morse sets
- `vertices() -> list`: List of vertex indices
- `morse_set_boxes(vertex: int) -> list`: List of boxes (as flat lists)
- `adjacencies(vertex: int) -> iterator`: Adjacent vertices

---

## Algorithms

### Box Map Computation

1. **For each active box**:
   - Compute image under dynamics: `image_box = dynamics(box)`
   - Find intersecting grid boxes: `indices = grid.box_to_indices(image_box)`
   - Add edges: `box_map.add_edges_from([(box_idx, j) for j in indices])`

2. **Parallelization**: Uses `joblib.Parallel` to process boxes in parallel

3. **Batch Mode**: If grid supports `box_to_indices_batch()`, uses vectorized computation

### Morse Graph Computation

1. **Find SCCs**: Use NetworkX `strongly_connected_components()`
2. **Identify Non-trivial SCCs**:
   - Multi-node SCCs → always non-trivial
   - Single-node SCCs → non-trivial if self-loop exists
3. **Build Condensation**: Use NetworkX `condensation()`
4. **Extract Connectivity**: For each pair of non-trivial SCCs, check if path exists in condensation

### Basin Computation

**Algorithm (`compute_all_morse_set_basins`):**
1. Build condensation graph
2. For each box:
   - If in Morse set → already in its own basin
   - If transient → BFS to find reachable Morse sets
   - Assign to basin of highest Morse set (topological order)

**Complexity:** O(boxes + edges)

### Adaptive Refinement

**Algorithm (`iterative_morse_computation`):**
1. Compute initial Morse graph
2. Identify boxes to refine using criterion:
   - Volume: Morse sets with relative volume >= threshold
   - Diameter: Boxes where image expands significantly
3. Subdivide selected boxes
4. Identify nodes to recompute:
   - New children
   - Predecessors of neighborhood (radius-based)
5. Update BoxMap for affected nodes
6. Repeat until convergence or max_depth

---

## File Structure

```
MorseGraph/
├── __init__.py          # Package exports
├── core.py              # Model, CMGDB integration
├── grids.py             # Grid implementations
├── dynamics.py           # Dynamics implementations
├── analysis.py          # Morse graph computation, basins
├── plot.py              # Visualization utilities
├── systems.py           # Pre-defined dynamical systems
├── utils.py             # Data generation, caching, config
├── config.py            # Configuration management
├── models.py            # Neural network models (optional)
├── training.py          # Training utilities (optional)
└── pipeline.py          # High-level workflow automation
```

---

## Usage Patterns

### Basic Workflow (2D Map)

```python
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapFunction
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph, compute_all_morse_set_basins
from MorseGraph.plot import plot_morse_sets, plot_basins_of_attraction
from MorseGraph.systems import henon_map

# Setup
domain = np.array([[-2.5, -0.5], [2.5, 0.5]])
grid = UniformGrid(domain, divisions=np.array([256, 256]))
dynamics = BoxMapFunction(henon_map, epsilon=0.01)
model = Model(grid, dynamics)

# Compute
box_map = model.compute_box_map()
morse_graph = compute_morse_graph(box_map)
basins = compute_all_morse_set_basins(morse_graph, box_map)

# Visualize
plot_morse_sets(grid, morse_graph)
plot_basins_of_attraction(grid, basins, morse_graph=morse_graph)
```

### Data-Driven Workflow

```python
from MorseGraph.dynamics import BoxMapData
from MorseGraph.utils import generate_trajectory_data

# Generate data
X, Y, _ = generate_trajectory_data(ode_func, params, n_samples=5000, ...)

# Setup
grid = UniformGrid(domain, divisions=[128, 128])
dynamics = BoxMapData(X, Y, grid, 
                     input_epsilon=grid.box_size,
                     output_epsilon=grid.box_size,
                     map_empty='interpolate')

# Compute
model = Model(grid, dynamics)
box_map = model.compute_box_map()
morse_graph = compute_morse_graph(box_map)
```

### 3D CMGDB Workflow

```python
from MorseGraph.core import compute_morse_graph_3d
from MorseGraph.dynamics import BoxMapFunction

# Setup dynamics
dynamics = BoxMapFunction(ives_model, epsilon=1e-6)

# Compute using CMGDB
morse_graph, morse_sets, barycenters = compute_morse_graph_3d(
    dynamics,
    domain_bounds=[[-7, -7, -7], [7, 7, 7]],
    subdiv_min=30,
    subdiv_max=42,
    subdiv_init=0,
    subdiv_limit=10000
)

# Visualize
from MorseGraph.plot import plot_morse_sets_3d_scatter
plot_morse_sets_3d_scatter(morse_graph, domain_bounds, ...)
```

### Pipeline Workflow

```python
from MorseGraph.pipeline import MorseGraphPipeline

# Run full pipeline
pipeline = MorseGraphPipeline('configs/ives_default.yaml', 'output')
pipeline.run_stage_1_3d()
pipeline.run_stage_2_trajectories()
pipeline.run_stage_3_training()
pipeline.run_stage_4_latent_bounds()
pipeline.run_stage_5_latent_morse(method='data')
pipeline.generate_comparisons()
```

---

## Integration with CMGDB

MorseGraph integrates with CMGDB for high-performance 3D computations:

### When to Use CMGDB

- **3D systems**: CMGDB is optimized for 3D and higher dimensions
- **High resolution**: CMGDB handles large subdivision depths efficiently
- **Performance**: C++ backend provides significant speedup

### When to Use Pure Python

- **2D systems**: Pure Python is sufficient and more flexible
- **Rapid prototyping**: Easier to debug and modify
- **Custom dynamics**: More straightforward to implement custom logic

### CMGDB Integration Points

1. **`compute_morse_graph_3d()`**: Wraps CMGDB for 3D computations
2. **`compute_morse_graph_2d_data()`**: Uses CMGDB for 2D data-driven dynamics
3. **`_run_cmgdb_compute()`**: Internal helper that adapts Dynamics interface to CMGDB

### Adapter Pattern

MorseGraph's `Dynamics` objects are adapted to CMGDB's interface:

```python
def F(rect):
    # CMGDB passes rect as [min_x, min_y, ..., max_x, max_y, ...]
    dim = len(rect) // 2
    box = np.array([rect[:dim], rect[dim:]])
    
    # Call MorseGraph dynamics
    res = dynamics(box)
    
    # Return as flat list [min_x, ..., max_x, ...]
    return list(res[0]) + list(res[1])
```

This allows seamless use of CMGDB with any MorseGraph `Dynamics` object.

---

## Summary

MorseGraph provides a comprehensive Python toolkit for Morse graph analysis:

- **Core Abstractions**: Grids, Dynamics, Model
- **Analysis Tools**: Morse graph computation, basin analysis, adaptive refinement
- **Visualization**: Extensive plotting utilities for 2D and 3D
- **ML Integration**: Autoencoder framework for dimensionality reduction
- **Pipeline**: High-level workflow automation
- **CMGDB Integration**: High-performance 3D computations via C++ backend

The library is designed for both rapid prototyping (pure Python) and production use (CMGDB integration), making it suitable for a wide range of applications in dynamical systems analysis.

