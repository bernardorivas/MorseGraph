There are 4 packages in this repo:

1. morsegraph (from /MorseGraph/) - installed via root pyproject.toml
2. CMGDB (from /cmgdb/) - C++ package with Python bindings (@marciogameiro)
3. hybrid-dynamics (from /hybrid_boxmap/) - imported as hybrid_dynamics (@bernardorivas)
4. MORALS (from /morals/) - depends on CMGDB (@ewertonvieira)

# Installation Options
From repo root
pip install -e .                  # Installs morsegraph
pip install -e ./cmgdb            # Installs CMGDB (requires cmake)
pip install -e ./hybrid_boxmap    # Installs hybrid-dynamics  
pip install -e ./morals           # Installs MORALS

Once installed, you import them by their package names (not directory names):

# MorseGraph package
from MorseGraph import Model

# CMGDB package (C++)
import CMGDB

# Hybrid dynamics package
from hybrid_dynamics import HybridSystem

# MORALS package
import MORALS


## About MorseGraph

MorseGraph is a lightweight Python library for computing and analyzing Morse graphs of dynamical systems using computational topology methods. It's mostly useful for toy problems. For faster computations, we recommend @cmgdb

## Overview

MorseGraph provides tools to study the global structure of dynamical systems through discrete abstractions on grids. The library implements outer approximations of dynamics and uses graph-theoretic methods to identify attractors, basins of attraction, and connecting orbits.

**Key Features:**
- Support for discrete maps, ODEs, data-driven and latent dynamics
- Outer approximation using epsilon-enlargement of boxes
- Parallel computation of box-to-box transitions
- Morse decomposition via (non-trivial) strongly connected components
- Computation of basin of attraction
- ML integration for learning low-dimensional representations
- Built-in visualization utilities

## Installation

### Basic Installation

Install the core library with standard dependencies:

```bash
pip install -e .
```

This includes: `numpy`, `scipy`, `networkx`, `matplotlib`, `joblib`

### Installation with ML Support

For machine learning features (Example 5 - learned latent dynamics):

```bash
pip install -e .[ml]
```

This adds: `torch`, `scikit-learn`

### Installation for Development

For running tests:

```bash
pip install -e .[test]
```

This adds: `pytest`

## Quick Start

```python
import numpy as np
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapFunction
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph
from MorseGraph.systems import henon_map

# Define domain and grid
domain = np.array([[-2.5, -0.5], [2.5, 0.5]])
grid = UniformGrid(domain, depth=8)

# Create dynamics
dynamics = BoxMapFunction(henon_map, epsilon=0.01)

# Build model and compute Morse graph
model = Model(grid, dynamics)
morse_graph = compute_morse_graph(model)
```

## Library Structure

### Core Components

- **`grids.py`** - Grid discretization
  - `UniformGrid`: Rectangular uniform grid with subdivision support

- **`dynamics.py`** - Dynamics implementations
  - `BoxMapFunction`: Explicit function-based dynamics with interval arithmetic
  - `BoxMapODE`: ODE-based dynamics using scipy integration
  - `BoxMapData`: Data-driven dynamics with spatial indexing (cKDTree)
  - `BoxMapLearnedLatent`: Learned latent dynamics for dimensionality reduction

- **`core.py`** - Main computation engine
  - `Model`: Connects grids and dynamics
  - `compute_box_map()`: Parallel computation of box-to-box transitions

- **`analysis.py`** - Graph analysis
  - `compute_morse_graph()`: Morse decomposition via SCC analysis
  - `compute_all_morse_set_basins()`: Basin of attraction computation

- **`plot.py`** - Visualization utilities
  - Morse sets, Morse graphs, basins of attraction, data coverage

### Utility Modules

- **`systems.py`** - Pre-defined dynamical systems
  - `henon_map`: Classic chaotic 2D map
  - `toggle_switch_ode`: Bistable genetic regulatory network
  - `van_der_pol_ode`: Nonlinear oscillator
  - `lorenz_ode`: Chaotic 3D attractor

- **`utils.py`** - Helper functions
  - Trajectory data generation and I/O
  - Latent space utilities for ML workflows

- **`models.py`** - Neural network models (requires `torch`)
  - `Encoder`, `Decoder`, `LatentDynamics`: Autoencoder framework

- **`training.py`** - Training utilities for ML models

## Examples

All examples are located in the `examples/` directory with well-organized configuration sections for easy customization.

### 1. Map Dynamics (`1_map_dynamics.py`)
Analyzes the **Hénon map**, a classic chaotic 2D discrete dynamical system. Demonstrates:
- `BoxMapFunction` for discrete maps
- Epsilon-bloating for rigorous outer approximation
- Morse graph computation and visualization

### 2. ODE Dynamics - Toggle Switch (`2_ode_dynamics.py`)
Studies a **genetic toggle switch** - a bistable biological system with mutual inhibition. Demonstrates:
- `BoxMapODE` for continuous-time systems
- Analysis of multistable dynamics
- Basin of attraction computation

### 3. ODE Dynamics - Van der Pol (`3_ode_dynamics.py`)
Analyzes the **Van der Pol oscillator**, a nonlinear system with limit cycle behavior. Demonstrates:
- ODE integration with configurable time horizon
- Visualization of periodic attractors

### 4. Data-Driven Dynamics (`4_data_driven.py`)
Computes Morse graphs from **trajectory data** (Van der Pol). Demonstrates:
- `BoxMapData` with k-nearest neighbor interpolation
- Data coverage visualization
- Working with sampled trajectories instead of analytical models

### 5. Learned Latent Dynamics (`5_learned_dynamics.py`)
Learns a **2D latent representation** of the 3D **Lorenz system** using autoencoders. Demonstrates:
- `BoxMapLearnedLatent` for ML-based dynamics
- Encoder-decoder framework for dimensionality reduction
- Comparing Morse graphs in full vs. latent space
- Requires ML dependencies (`pip install -e .[ml]`)

## Running Examples

```bash
cd examples
python 1_map_dynamics.py
python 2_ode_dynamics.py
python 3_ode_dynamics.py
python 4_data_driven.py
python 5_learned_dynamics.py  # Requires torch
```

Figures are saved to `examples/figures/`.

## Testing

Run the test suite:

```bash
pytest                              # Run all tests
pytest tests/test_dynamics.py       # Run specific test file
pytest -v                           # Verbose output
```

## Requirements

- Python >= 3.8
- Core: `numpy`, `scipy`, `networkx`, `matplotlib`, `joblib`
- Optional (ML): `torch`, `scikit-learn`
- Testing: `pytest`

## Design Philosophy

**Box Representation:** All boxes are numpy arrays of shape `(2, D)` where `D` is dimension.

**Rigorous Approximation:** The library uses epsilon-bloating to compute rigorous outer approximations of dynamics, ensuring that all true dynamics are captured.

**Modularity:** Clean separation between grids, dynamics, computation, analysis, and visualization layers.

**Performance:** Parallel processing via `joblib` and optimized spatial indexing with scipy's `cKDTree`.

## Documentation

For detailed architectural information, see `CLAUDE.md` in the repository.

## License

(License information to be added)

## Citation

(Citation information to be added)
