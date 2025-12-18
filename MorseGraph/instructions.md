# MorseGraph - User Instructions

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Basic Usage Patterns](#basic-usage-patterns)
6. [Advanced Features](#advanced-features)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## Introduction

**MorseGraph** is a Python library for computing and analyzing Morse graphs of dynamical systems. It provides tools to study the global structure of dynamical systems through discrete abstractions on grids, identifying attractors, basins of attraction, and connecting orbits.

### What is a Morse Graph?

A Morse graph captures the recurrent dynamics (attractors, periodic orbits) and their connectivity in a dynamical system. It's computed by:

1. **Discretizing** the state space into boxes (grid cells)
2. **Computing** how boxes map to other boxes under the dynamics
3. **Identifying** strongly connected components (Morse sets) that represent recurrent behavior
4. **Analyzing** connectivity between Morse sets

### Key Features

- ✅ Support for discrete maps, ODEs, and data-driven dynamics
- ✅ Rigorous outer approximation using epsilon-bloating
- ✅ Parallel computation for performance
- ✅ Integration with CMGDB for high-performance 3D computations
- ✅ Machine learning integration for dimensionality reduction
- ✅ Built-in visualization utilities

---

## Installation

### Prerequisites

- Python >= 3.8
- CMGDB (C++ backend) - **Required for production use**

### Basic Installation

From the repository root:

```bash
# Install MorseGraph
pip install -e .

# Install CMGDB (required)
pip install -e ./cmgdb
```

### Optional Dependencies

For machine learning features (autoencoder training):

```bash
pip install -e .[ml]  # Adds torch, scikit-learn
```

For running tests:

```bash
pip install -e .[test]  # Adds pytest
```

### Verify Installation

```python
import MorseGraph
print(MorseGraph.__version__)

# Check CMGDB availability
from MorseGraph.core import _CMGDB_AVAILABLE
print(f"CMGDB available: {_CMGDB_AVAILABLE}")
```

---

## Quick Start

### Example 1: Analyzing a Discrete Map

```python
import numpy as np
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapFunction
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph, compute_all_morse_set_basins
from MorseGraph.plot import plot_morse_sets, plot_basins_of_attraction
from MorseGraph.systems import henon_map

# 1. Define domain and create grid
domain = np.array([[-2.5, -0.5], [2.5, 0.5]])
grid = UniformGrid(bounds=domain, divisions=np.array([256, 256]))

# 2. Create dynamics
dynamics = BoxMapFunction(henon_map, epsilon=0.01)

# 3. Build model and compute box map
model = Model(grid, dynamics)
box_map = model.compute_box_map(subdiv_min=16, subdiv_max=16)

# 4. Compute Morse graph
morse_graph = compute_morse_graph(box_map)
print(f"Found {len(morse_graph.nodes())} Morse sets")

# 5. Compute basins
basins = compute_all_morse_set_basins(morse_graph, box_map)

# 6. Visualize
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
plot_morse_sets(grid, morse_graph, ax=ax)
plt.savefig("morse_sets.png")
```

### Example 2: Analyzing an ODE

```python
from MorseGraph.dynamics import BoxMapODE
from MorseGraph.systems import van_der_pol_ode

# Create ODE dynamics
dynamics = BoxMapODE(ode_f=van_der_pol_ode, tau=1.0, epsilon=0.01)

# Rest is the same as Example 1
model = Model(grid, dynamics)
box_map = model.compute_box_map()
morse_graph = compute_morse_graph(box_map)
```

### Example 3: Data-Driven Analysis

```python
from MorseGraph.dynamics import BoxMapData
from MorseGraph.utils import generate_trajectory_data

# Generate trajectory data
X, Y, _ = generate_trajectory_data(
    ode_func=van_der_pol_ode,
    ode_params={},
    n_samples=5000,
    total_time=10.0,
    n_points=100,
    sampling_domain=domain
)

# Create data-driven dynamics
grid = UniformGrid(bounds=domain, divisions=[128, 128])
dynamics = BoxMapData(
    X, Y, grid,
    input_epsilon=grid.box_size,
    output_epsilon=grid.box_size,
    map_empty='interpolate'  # Use k-NN for empty boxes
)

# Compute Morse graph
model = Model(grid, dynamics)
box_map = model.compute_box_map()
morse_graph = compute_morse_graph(box_map)
```

---

## Core Concepts

### 1. Grids (`grids.py`)

Grids discretize the continuous state space into boxes.

**UniformGrid**: Rectangular uniform grid
```python
grid = UniformGrid(
    bounds=np.array([[x_min, y_min], [x_max, y_max]]),
    divisions=np.array([nx, ny])  # Number of divisions per dimension
)
```

**Key Methods:**
- `get_boxes()`: Get all boxes as `(N, 2, D)` array
- `box_to_indices(box)`: Find grid boxes intersecting a given box
- `subdivide(indices)`: Refine grid locally or globally

### 2. Dynamics (`dynamics.py`)

Dynamics objects define how boxes map to other boxes.

**BoxMapFunction**: For explicit functions
```python
dynamics = BoxMapFunction(
    map_f=my_function,  # Function: R^D -> R^D
    epsilon=0.01,        # Bloating factor
    evaluation_method="corners"  # or "center", "random"
)
```

**BoxMapODE**: For continuous-time systems
```python
dynamics = BoxMapODE(
    ode_f=my_ode_function,  # ODE: (t, y) -> dy/dt
    tau=1.0,               # Integration time
    epsilon=0.01
)
```

**BoxMapData**: For data-driven dynamics
```python
dynamics = BoxMapData(
    X, Y, grid,           # Training data (X -> Y)
    input_epsilon=None,   # Defaults to grid.box_size
    output_epsilon=None,
    map_empty='interpolate',  # Strategy for empty boxes
    k_neighbors=5
)
```

### 3. Model (`core.py`)

The `Model` class connects grids and dynamics.

```python
model = Model(grid, dynamics)
box_map = model.compute_box_map(
    subdiv_min=20,    # Minimum subdivision depth
    subdiv_max=28,    # Maximum subdivision depth
    subdiv_init=0,    # Initial depth
    subdiv_limit=10000 # Max boxes
)
```

**Note**: `compute_box_map()` uses CMGDB backend by default. The returned `box_map` is a NetworkX DiGraph where:
- Nodes = box indices
- Edges = transitions between boxes

### 4. Analysis (`analysis.py`)

**Compute Morse Graph:**
```python
from MorseGraph.analysis import compute_morse_graph

morse_graph = compute_morse_graph(box_map)
# Returns NetworkX DiGraph where nodes are Morse sets (frozensets of box indices)
```

**Compute Basins:**
```python
from MorseGraph.analysis import compute_all_morse_set_basins

basins = compute_all_morse_set_basins(morse_graph, box_map)
# Returns dict: {morse_set: set of box indices in basin}
```

### 5. Visualization (`plot.py`)

**2D Plotting:**
```python
from MorseGraph.plot import (
    plot_morse_sets,
    plot_basins_of_attraction,
    plot_morse_graph,
    plot_data_coverage
)

plot_morse_sets(grid, morse_graph, ax=ax)
plot_basins_of_attraction(grid, basins, morse_graph=morse_graph, ax=ax)
plot_morse_graph(morse_graph, ax=ax)
```

**3D Plotting (requires CMGDB):**
```python
from MorseGraph.plot import plot_morse_sets_3d_scatter

plot_morse_sets_3d_scatter(
    morse_graph_cmgdb,
    domain_bounds,
    output_path="morse_3d.png"
)
```

---

## Basic Usage Patterns

### Pattern 1: Simple Map Analysis

```python
# 1. Setup
domain = np.array([[-2, -2], [2, 2]])
grid = UniformGrid(domain, divisions=[128, 128])
dynamics = BoxMapFunction(my_map, epsilon=0.01)

# 2. Compute
model = Model(grid, dynamics)
box_map = model.compute_box_map()
morse_graph = compute_morse_graph(box_map)

# 3. Analyze
for morse_set in morse_graph.nodes():
    print(f"Morse set size: {len(morse_set)}")
```

### Pattern 2: ODE with Custom Time Horizon

```python
dynamics = BoxMapODE(
    ode_f=my_ode,
    tau=2.0,  # Longer integration time
    epsilon=0.01
)
```

### Pattern 3: Adaptive Grid Refinement

```python
from MorseGraph.analysis import iterative_morse_computation
from MorseGraph.grids import AdaptiveGrid

grid = AdaptiveGrid(bounds=domain, max_depth=10)
model = Model(grid, dynamics)

morse_graph, history = iterative_morse_computation(
    model,
    max_depth=5,
    refinement_threshold=0.1,
    criterion='volume'
)
```

### Pattern 4: Using Pre-defined Systems

```python
from MorseGraph.systems import (
    henon_map,           # 2D chaotic map
    leslie_map,          # 2D population model
    leslie_map_3d,       # 3D population model
    ives_model,          # 3D ecological model
    van_der_pol_ode,     # Nonlinear oscillator
    toggle_switch_ode,   # Bistable system
    lorenz_ode           # Chaotic attractor
)

dynamics = BoxMapFunction(henon_map, epsilon=0.01)
```

---

## Advanced Features

### 1. CMGDB Integration (3D Systems)

For 3D systems, use CMGDB directly for better performance:

```python
from MorseGraph.core import compute_morse_graph_3d

morse_graph, morse_sets, barycenters = compute_morse_graph_3d(
    dynamics=dynamics,
    domain_bounds=[[x_min, y_min, z_min], [x_max, y_max, z_max]],
    subdiv_min=30,
    subdiv_max=42,
    subdiv_init=0,
    subdiv_limit=10000
)
```

### 2. Machine Learning Pipeline

For dimensionality reduction with autoencoders:

```python
from MorseGraph.pipeline import MorseGraphPipeline

# Run full pipeline: 3D -> data -> training -> 2D latent
pipeline = MorseGraphPipeline(
    config_path='configs/ives_default.yaml',
    output_dir='output'
)

pipeline.run_stage_1_3d()              # Compute 3D Morse graph
pipeline.run_stage_2_trajectories()     # Generate data
pipeline.run_stage_3_training()        # Train autoencoder
pipeline.run_stage_4_encoding()        # Compute latent bounds
pipeline.run_stage_5_latent_morse(method='data')  # Compute 2D Morse graph
pipeline.generate_comparisons()        # Visualize
```

### 3. Custom Dynamics

Implement your own dynamics:

```python
from MorseGraph.dynamics import Dynamics
import numpy as np

class MyCustomDynamics(Dynamics):
    def __call__(self, box: np.ndarray) -> np.ndarray:
        """
        Map a box to its image box.
        
        :param box: Shape (2, D) - [lower_bounds, upper_bounds]
        :return: Shape (2, D) - image box
        """
        # Your custom logic here
        lower = box[0]
        upper = box[1]
        
        # Example: simple linear map
        image_lower = 2 * lower
        image_upper = 2 * upper
        
        return np.array([image_lower, image_upper])
    
    def get_active_boxes(self, grid):
        # Optionally filter active boxes
        return np.arange(len(grid.get_boxes()))
```

### 4. Configuration Files

Use YAML configs for reproducible experiments:

```yaml
# configs/my_experiment.yaml
system:
  type: map
  name: henon_map

domain:
  bounds: [[-2.5, -0.5], [2.5, 0.5]]

cmgdb_3d:
  subdiv_min: 20
  subdiv_max: 28
  subdiv_init: 0
  subdiv_limit: 10000
  padding: true
```

Load config:
```python
from MorseGraph.config import load_experiment_config

config = load_experiment_config('configs/my_experiment.yaml')
```

---

## Examples

The `examples/` directory contains complete working examples:

1. **`1_map_dynamics.py`** - Hénon map analysis
2. **`2_ode_dynamics.py`** - Toggle switch (bistable ODE)
3. **`3_ode_dynamics.py`** - Van der Pol oscillator
4. **`4_data_driven.py`** - Data-driven Morse graph
5. **`5_ode_learned_dynamics.py`** - Learned latent dynamics (requires ML)
6. **`6_map_learned_dynamics.py`** - Learned map dynamics
7. **`7_ives_model.py`** - 3D Ives model with pipeline
8. **`8_leslie_map_3d.py`** - 3D Leslie model

Run examples:
```bash
cd examples
python 1_map_dynamics.py
```

---

## Troubleshooting

### CMGDB Not Found

**Error**: `ImportError: CMGDB is required`

**Solution**: Install CMGDB:
```bash
pip install -e ./cmgdb
```

### Out of Memory

**Problem**: Large grids cause memory issues

**Solutions**:
- Reduce `subdiv_max` (lower resolution)
- Increase `subdiv_limit` to prevent excessive refinement
- Use CMGDB for 3D systems (more memory efficient)

### Empty Morse Graph

**Problem**: `compute_morse_graph()` returns empty graph

**Possible Causes**:
- Epsilon too small (dynamics not captured)
- Grid resolution too coarse
- Domain bounds don't contain dynamics

**Solutions**:
- Increase `epsilon` (e.g., 0.01 → 0.1)
- Increase grid resolution
- Check domain bounds cover relevant region

### Slow Computation

**Problem**: `compute_box_map()` is slow

**Solutions**:
- Use CMGDB backend (default, faster)
- Reduce grid resolution
- Use `subdiv_limit` to cap refinement
- For 2D, consider pure Python if debugging

### Data-Driven: Empty Boxes

**Problem**: `BoxMapData` encounters empty boxes

**Solutions**:
- Set `map_empty='interpolate'` (uses k-NN)
- Increase `k_neighbors`
- Generate more training data
- Use `map_empty='outside'` to map outside domain

---

## API Reference

### Core Classes

#### `Model(grid, dynamics, dynamics_kwargs=None)`
Connects grid and dynamics for computation.

**Methods:**
- `compute_box_map(subdiv_min, subdiv_max, ...) -> nx.DiGraph`

#### `UniformGrid(bounds, divisions)`
Uniform rectangular grid.

**Attributes:**
- `bounds`: Domain bounds `(2, D)`
- `divisions`: Divisions per dimension `(D,)`
- `dim`: Dimension
- `box_size`: Box size `(D,)`

#### `BoxMapFunction(map_f, epsilon=1e-6, evaluation_method="corners")`
Function-based dynamics.

#### `BoxMapODE(ode_f, tau, epsilon=0.0)`
ODE-based dynamics.

#### `BoxMapData(X, Y, grid, ...)`
Data-driven dynamics.

### Analysis Functions

#### `compute_morse_graph(box_map, assign_colors=True) -> nx.DiGraph`
Compute Morse graph from box map.

#### `compute_all_morse_set_basins(morse_graph, box_map) -> Dict`
Compute basins for all Morse sets.

#### `compute_basins_of_attraction(morse_graph, box_map) -> Dict`
Compute basins for attractors only.

### Plotting Functions

#### `plot_morse_sets(grid, morse_graph, ax=None, ...)`
Plot Morse sets on 2D grid.

#### `plot_basins_of_attraction(grid, basins, morse_graph=None, ax=None, ...)`
Plot basins of attraction.

#### `plot_morse_graph(morse_graph, ax=None, ...)`
Plot Morse graph structure.

### Utility Functions

#### `generate_trajectory_data(ode_func, ode_params, n_samples, ...)`
Generate trajectory data from ODE.

#### `load_experiment_config(config_path) -> ExperimentConfig`
Load experiment configuration from YAML.

---

## Additional Resources

- **Detailed API Reference**: See existing `instructions.md` (966 lines) for comprehensive API documentation
- **Architecture Notes**: See `CLAUDE.md` for architectural details
- **CMGDB Documentation**: See `cmgdb/README.md` for CMGDB-specific information
- **Examples**: See `examples/` directory for complete working code

---

## Getting Help

1. Check the examples in `examples/`
2. Review the detailed API reference in the existing `instructions.md`
3. Check error messages - they often include helpful suggestions
4. Verify CMGDB installation: `from MorseGraph.core import _CMGDB_AVAILABLE`

---

## License

See LICENSE file in repository root.
