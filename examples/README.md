# MorseGraph Python Examples

This directory contains Python script versions of the Jupyter notebooks in the `examples/` directory. Each script demonstrates different aspects of the MorseGraph library and saves generated figures to the same directory.

## Examples Overview

### 1. `1_map_dynamics.py` - Basic Map Dynamics
- **Purpose**: Demonstrates MorseGraph with explicit function dynamics
- **System**: Hénon map (classic 2D chaotic map)
- **Key Features**: 
  - `BoxMapFunction` dynamics class
  - `UniformGrid` discretization
  - Basic Morse graph computation and visualization

### 2. `2_data_driven.py` - Data-Driven Dynamics  
- **Purpose**: Shows how to compute Morse graphs from data pairs (X, Y)
- **System**: Hénon map (data-driven approach)
- **Key Features**:
  - `BoxMapData` dynamics class using spatial indexing
  - Training data generation and validation
  - Data coverage analysis

### 3. `3_ode_dynamics.py` - ODE Integration
- **Purpose**: Demonstrates Morse graph computation for ODE systems
- **System**: Bistable system with two attractors and one saddle
- **Key Features**:
  - `BoxMapODE` dynamics with numerical integration
  - Vector field visualization
  - Fixed point analysis

### 4. `4_adaptive_refinement.py` - Iterative Refinement
- **Purpose**: Shows adaptive grid refinement for detailed analysis
- **System**: Hénon map with adaptive grid
- **Key Features**:
  - `AdaptiveGrid` with tree-based structure
  - `iterative_morse_computation` function
  - Refinement convergence analysis
  - Comparison of initial vs refined results

### 5. `5_learned_dynamics.py` - Machine Learning Integration
- **Purpose**: Complete ML pipeline for learning dynamics in latent space
- **System**: Hénon map with neural network dynamics
- **Key Features**:
  - PyTorch neural networks (Encoder/Decoder/LatentDynamics)
  - Joint training with reconstruction + dynamics losses
  - `LearnedMapDynamics` wrapper
  - Model evaluation and comparison

## Running the Examples

### Prerequisites
- Python 3.8+
- MorseGraph library installed (`pip install -e .` from repository root)
- Required dependencies: `numpy`, `scipy`, `networkx`, `matplotlib`, `joblib`

### Optional Dependencies
- **For ML example (5)**: `torch`, `scikit-learn`
- **For enhanced features**: Install with `pip install -e .[ml]`

### Execution
```bash
# Run individual examples
python 1_map_dynamics.py
python 2_data_driven.py
python 3_ode_dynamics.py
python 4_adaptive_refinement.py
python 5_learned_dynamics.py  # Requires PyTorch

# Or run all examples (skip ML if PyTorch unavailable)
for script in *.py; do python "$script"; done
```

## Generated Figures

Each example saves figures to the same directory:

- **Example 1**: `henon_morse_sets.png`
- **Example 2**: `henon_data.png`, `henon_data_morse_sets.png`  
- **Example 3**: `bistable_vector_field.png`, `bistable_morse_sets.png`
- **Example 4**: `refinement_progress.png`, `adaptive_refinement_final.png`, `refinement_comparison.png`
- **Example 5**: `learned_dynamics_data.png`, `learned_dynamics_results.png`

## Key Concepts Demonstrated

### Core Dynamics Classes
- **`BoxMapFunction`**: Explicit mathematical functions with rigorous bloating
- **`BoxMapData`**: Data-driven dynamics using spatial indexing (cKDTree)
- **`BoxMapODE`**: ODE integration using scipy's solve_ivp
- **`LearnedMapDynamics`**: Neural network dynamics in latent space

### Grid Types
- **`UniformGrid`**: Regular rectangular discretization
- **`AdaptiveGrid`**: Tree-based adaptive refinement (Quadtree/Octree)

### Analysis Tools
- **`compute_box_map`**: Compute the discrete dynamical system on grid boxes (BoxMap)
- **`compute_morse_graph`**: Extract non-trivial strongly connected components from BoxMap
- **`iterative_morse_computation`**: Adaptive refinement workflow
- **`plot_morse_sets`**: Visualization of Morse decomposition

### Rigorous Approximation Strategies
- **Geometric Bloating**: Epsilon expansion in continuous space
- **Grid Dilation**: Neighbor expansion in discrete grid space

## Performance Notes

- **Grid Resolution**: Examples use moderate resolutions for reasonable runtime
- **Data Points**: Examples use 5K-10K points for good coverage without excessive computation
- **ML Training**: 50 epochs sufficient for demonstration (increase for production)
- **Refinement Depth**: 4-5 iterations typically sufficient for most systems

## Troubleshooting

**Import Errors**: Ensure MorseGraph is installed in development mode from repository root
**PyTorch Missing**: Example 5 will skip gracefully if PyTorch unavailable
**Figure Display**: Scripts show plots interactively and save to files
**Memory Issues**: Reduce grid resolution or data points if needed

For more details, see the original Jupyter notebooks in `examples/` or the main documentation.