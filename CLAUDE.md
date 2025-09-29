# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MorseGraph is a Python library for computing and analyzing Morse graphs of dynamical systems. The library implements computational topology methods to study the structure of dynamical systems through discrete abstractions on grids.

## Development Commands

### Installation and Setup
```bash
pip install -e .                    # Install in development mode
pip install -e .[ml]                # Install with optional ML dependencies (PyTorch)
pip install -e .[test]              # Install with testing dependencies
```

### Testing
```bash
pytest                              # Run all tests
pytest tests/test_dynamics.py       # Run specific test file
pytest -v                          # Verbose output
pytest --tb=short                   # Short traceback format
```

### Package Building
```bash
pip install build                   # Install build tool if needed
python -m build                     # Build wheel and source distribution
```

## Core Architecture

The library follows a modular design with clear separation of concerns:

### Core Components (`MorseGraph/`)

1. **`dynamics.py`** - Dynamics abstraction layer
   - `Dynamics(ABC)`: Abstract base for all dynamical systems
   - `BoxMapFunction`: Explicit function-based dynamics with bloating mechanism
   - `BoxMapData`: Data-driven dynamics using spatial indexing (cKDTree)
   - `BoxMapODE`: ODE-based dynamics using scipy integration

2. **`grids.py`** - Grid discretization layer
   - `AbstractGrid(ABC)`: Interface for all grid types
   - `UniformGrid`: Rectangular uniform grid implementation
   - Key methods: `get_boxes()`, `box_to_indices()`, `subdivide()`

3. **`core.py`** - Main computation engine
   - `Model`: Connects dynamics and grids
   - `compute_map_graph()`: Parallel computation of box-to-box transitions using joblib

4. **`analysis.py`** - Graph analysis and Morse decomposition
   - `compute_morse_graph()`: NetworkX-based strongly connected components analysis
   - `compute_basins_of_attraction()`: Backward reachability computation

5. **`plot.py`** - Visualization utilities (matplotlib-based)

### Key Design Patterns

- **Box Representation**: All boxes are numpy arrays of shape `(2, D)` where `D` is dimension
- **Bloating Mechanism**: Dynamics implementations use epsilon-bloating for rigorous outer approximation
- **Parallel Processing**: Core computations use joblib for box image calculations
- **Spatial Indexing**: Data-driven dynamics leverage scipy.spatial.cKDTree for efficient queries
- **Graph-based Analysis**: Heavy reliance on NetworkX for graph operations and SCC decomposition

### Implementation Status

Based on `Morse Graph.md`:
- **Milestone 1**: Complete (core workflow with uniform grids)
- **Milestone 2**: Expanded capabilities (data/ODE dynamics, basins, visualization)
- **Milestone 3**: Adaptive grids and iterative refinement
- **Milestone 4**: ML integration with PyTorch models

## Testing Strategy

Tests are organized by module:
- `test_grids.py`: Grid indexing, coordinate conversion, subdivision
- `test_dynamics.py`: Dynamics implementations with known maps
- `test_milestone_2.py`: Integration tests for M2 features

## Dependencies

Core dependencies: numpy, scipy, networkx, matplotlib, joblib
Optional: torch (for ML features), pytest (for testing)

The library maintains backwards compatibility with Python >=3.8.