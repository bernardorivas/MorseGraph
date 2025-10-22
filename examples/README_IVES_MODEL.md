# Ives Ecological Model - MorseGraph Analysis Pipeline

## Overview

`7_ives_model.py` implements a comprehensive pipeline for analyzing the Ives midge-algae-detritus ecological model using Morse Graph theory combined with autoencoder-based dimensionality reduction.

**Reference**: Ives et al. (2008) "High-amplitude fluctuations and alternative dynamical states of midges in Lake Myvatn", Nature 452: 84-87

The pipeline:

1. Computes 3D Morse graph using CMGDB
2. Trains an autoencoder to project 3D dynamics to 2D latent space
3. Computes 2D Morse graphs in the learned latent space
4. Compares the learned 2D representation against the 3D ground truth

## Color Convention

### 3D Space (Original Dynamics)

- **Colormap**: `cm.cool` (blue → purple → magenta)
- **Used for**:
  - 3D Morse graph diagrams
  - 3D Morse set scatter plots
  - 3D projection plots (morse_sets_proj_*.png)
  - **E(3D barycenters)** when shown in latent space

### 2D Space (Learned Latent Dynamics)

- **Colormap**: `cm.viridis` (purple → green → yellow)
- **Used for**:
  - 2D Morse graph diagrams (computed in latent space)
  - 2D Morse set rectangles in latent space plots
  - Latent dynamics visualizations

### Rationale

E(3D barycenters) retain the `cm.cool` colormap even when displayed in latent space to visually indicate their origin from the 3D computation. This helps distinguish:

- **Cool colors** = Derived from 3D original space
- **Viridis colors** = Computed in 2D latent space

## Directory Structure

The pipeline uses **3-level hash-based caching** to avoid redundant computations:

```
examples/ives_model_output/
├── cmgdb_3d/                    # Level 1: Cached 3D Morse graphs
│   ├── {3d_hash1}/              # Configuration 1 (e.g., subdiv_min=30)
│   │   ├── morse_graph_data.pkl    # NetworkX DiGraph
│   │   ├── barycenters.npz         # 3D barycenter coordinates
│   │   ├── metadata.json           # Parameter configuration
│   │   └── results/                # 3D visualizations
│   │       ├── morse_graph_3d.png
│   │       ├── morse_sets_3d.png
│   │       └── morse_sets_proj_*.png
│   └── {3d_hash2}/              # Configuration 2 (e.g., subdiv_min=36)
│       └── ...
├── training/                    # Level 2: Cached training results
│   ├── {training_hash1}/        # Training config 1
│   │   ├── models/
│   │   │   ├── encoder.pt          # Trained encoder
│   │   │   ├── decoder.pt          # Trained decoder
│   │   │   └── latent_dynamics.pt  # Trained latent dynamics
│   │   ├── training_losses.pkl     # Loss curves (train + val)
│   │   ├── latent_bounds.npz       # Computed latent bounds
│   │   └── metadata.json           # Training configuration
│   └── {training_hash2}/
│       └── ...
├── cmgdb_2d/                    # Level 3: Cached 2D Morse graphs
│   ├── {2d_hash1}/              # 2D CMGDB config 1
│   │   ├── morse_graph_padded.pkl      # 2D Morse graph (padded)
│   │   ├── barycenters_padded.npz      # Barycenters (padded)
│   │   ├── morse_graph_unpadded.pkl    # 2D Morse graph (unpadded)
│   │   ├── barycenters_unpadded.npz    # Barycenters (unpadded)
│   │   └── metadata.json               # 2D CMGDB configuration
│   └── {2d_hash2}/
│       └── ...
└── run_NNN/                     # Individual execution runs (for visualizations)
    ├── config.yaml              # Saved configuration for this run
    ├── models/                  # Symlinked/copied models for reference
    │   ├── encoder.pt
    │   ├── decoder.pt
    │   └── latent_dynamics.pt
    └── results/                 # Run-specific visualizations
        ├── training_curves.png
        ├── encoder_decoder_roundtrip.png
        ├── trajectory_analysis.png
        ├── latent_*.png         # Latent space visualizations
        ├── morse_graph_2d_*.png
        └── morse_*_comparison_*.png
```

### 3-Level Hash-Based Caching

The pipeline implements **dependency-chained caching** at three levels to maximize reuse:

#### Level 1: 3D CMGDB Cache

- **Hash based on**: `map_func`, `domain_bounds`, `subdiv_min`, `subdiv_max`, `subdiv_init`, `subdiv_limit`, `padding`
- **Computation time**: Hours for high resolution
- **Cache directory**: `cmgdb_3d/{3d_hash}/`
- **Force recompute**: `--force-recompute-3d`

#### Level 2: Training Cache

- **Hash based on**: `3d_hash` + all training parameters (architecture, hyperparameters, data generation)
- **Dependencies**: Depends on Level 1 (3D hash included in training hash)
- **Computation time**: Minutes to hours depending on architecture
- **Cache directory**: `training/{training_hash}/`
- **Force retrain**: `--force-retrain`

#### Level 3: 2D CMGDB Cache

- **Hash based on**: `training_hash` + 2D CMGDB parameters (latent subdivision, padding, etc.)
- **Dependencies**: Depends on Level 2 (training hash included in 2D hash)
- **Computation time**: Seconds to minutes
- **Cache directory**: `cmgdb_2d/{2d_hash}/`
- **Force recompute**: `--force-recompute-2d`

**Dependency Chain**:

```
3D CMGDB → Training → 2D CMGDB
```

Example workflow:

```bash
# First run: computes all three levels
python 7_ives_model.py --config configs/ives_default.yaml
# → Computes 3D (2 hours), trains (30 min), computes 2D (2 min)

# Second run with same config: uses all caches
python 7_ives_model.py --config configs/ives_default.yaml
# → Loads 3D (instant), loads training (instant), loads 2D (instant)

# Change only training params: reuses 3D, retrains, recomputes 2D
# Edit config: num_epochs: 2000
python 7_ives_model.py --config configs/ives_default.yaml
# → Loads 3D (instant), trains (30 min), computes 2D (2 min)

# Change only 2D params: reuses 3D and training, recomputes 2D
# Edit config: latent_subdiv_min: 18
python 7_ives_model.py --config configs/ives_default.yaml
# → Loads 3D (instant), loads training (instant), computes 2D (2 min)

# Force recompute specific level
python 7_ives_model.py --force-retrain          # Re-train (keeps 3D cache)
python 7_ives_model.py --force-recompute-2d     # Re-compute 2D (keeps 3D + training)
python 7_ives_model.py --force-all              # Recompute everything
```

## Configuration

All parameters are defined in YAML configuration files in `examples/configs/`.

### Key Configuration Sections

#### Dynamics Parameters (Required)

```yaml
dynamics:
  r1: 3.873           # Midge reproduction rate
  r2: 11.746          # Algae growth rate
  c: 3.67e-07         # Constant input
  d: 0.5517           # Detritus decay rate
  p: 0.06659          # Relative palatability
  q: 0.9026           # Consumption exponent
  log_offset: 0.001   # Offset for log transform (< 1e-7)
```

**Note**: All dynamics parameters are **required**. The script will fail fast with a clear error message if any are missing.

#### Domain (Optional Visualization Elements)

```yaml
domain:
  bounds: [[-2.5, -7.0, -2.5], [1.5, 1.5, 1.5]]  # 3D domain
  equilibrium: [0.792107, 0.209010, 0.376449]    # Optional
  period_12_orbit:                                # Optional
    - [0.391, 0.912, 1.046]
    - ...
```

#### 3D CMGDB Parameters

```yaml
cmgdb_3d:
  subdiv_min: 30       # Subdivision depth (2^30 ≈ 1B boxes total)
  subdiv_max: 36       # Maximum depth for adaptive refinement
  subdiv_init: 0       # Start at subdiv_min
  subdiv_limit: 100000 # Max boxes before stopping
  padding: false       # Boundary padding
```

**Computation time scales exponentially with subdivision:**

- subdiv_min=15 (~33k boxes): seconds
- subdiv_min=30 (~1B boxes): hours
- subdiv_min=36 (~68B boxes): many hours to days

#### 2D CMGDB Parameters

```yaml
cmgdb_2d:
  subdiv_min: 16       # Latent space subdivision
  subdiv_max: 20
  subdiv_init: 0
  subdiv_limit: 100000
  padding: true        # Enable padding in latent space
  bounds_padding: 1.01
  original_grid_subdiv: 18  # Resolution for encoding 3D grid
```

#### Neural Network Architecture

```yaml
model:
  input_dim: 3          # 3D original space
  latent_dim: 2         # 2D latent space
  hidden_dim: 32        # Hidden layer size
  num_layers: 3         # Number of hidden layers

  # Output activations (null = linear)
  encoder_activation: null
  decoder_activation: null
  latent_dynamics_activation: null
```

#### Training Parameters

```yaml
training:
  n_trajectories: 10000  # Number of random trajectories
  n_points: 20           # Points per trajectory
  skip_initial: 0        # Transient to skip
  random_seed: 42

  num_epochs: 1500
  batch_size: 1024
  learning_rate: 0.001
  early_stopping_patience: 50
  min_delta: 1.0e-5
```

#### Loss Weights

```yaml
loss_weights:
  w_recon: 500.0        # Reconstruction loss
  w_dyn_recon: 1.0      # Dynamics reconstruction
  w_dyn_cons: 1.0       # Dynamics consistency
```

## Usage

### Basic Usage

```bash
# Run with default configuration
python examples/7_ives_model.py

# Run with specific config
python examples/7_ives_model.py --config examples/configs/ives_high_res.yaml

# Force recomputation at specific levels
python examples/7_ives_model.py --force-recompute-3d    # Force 3D CMGDB recompute
python examples/7_ives_model.py --force-retrain         # Force autoencoder retrain
python examples/7_ives_model.py --force-recompute-2d    # Force 2D CMGDB recompute
python examples/7_ives_model.py --force-all             # Force all computations
```

### Command-Line Flags

- `--config PATH`: Path to YAML configuration file (default: `examples/configs/ives_default.yaml`)
- `--force-recompute-3d`: Ignore 3D CMGDB cache and recompute from scratch
- `--force-retrain`: Ignore training cache and retrain autoencoder models
- `--force-recompute-2d`: Ignore 2D CMGDB cache and recompute in latent space
- `--force-all`: Equivalent to using all three force flags together
- `--help`: Show help message with all options

### Typical Workflow

1. **Quick test** (subdiv_min=15, fast)

   ```bash
   python 7_ives_model.py --config configs/ives_fast.yaml
   ```

2. **Default resolution** (subdiv_min=30, hours)

   ```bash
   python 7_ives_model.py --config configs/ives_default.yaml
   ```

3. **High resolution** (subdiv_min=36, many hours)

   ```bash
   python 7_ives_model.py --config configs/ives_high_res.yaml
   ```

4. **Parameter exploration** (leveraging caches)

   ```bash
   # Run with default config → caches all levels
   python 7_ives_model.py

   # Edit config to change only training params (e.g., hidden_dim: 64)
   # → Reuses 3D cache, retrains, recomputes 2D
   python 7_ives_model.py

   # Edit config to change only 2D params (e.g., latent_subdiv_min: 18)
   # → Reuses 3D and training caches, only recomputes 2D
   python 7_ives_model.py
   ```

## Output Visualizations

### 3D Visualizations (in `cmgdb_3d/{hash}/results/`)

Shared across runs with identical 3D parameters:

- **`morse_graph_3d.png`**: Morse graph diagram (nodes = Morse sets, edges = transitions)
- **`morse_sets_3d.png`**: 3D scatter plot of barycenters with equilibrium/orbit
- **`morse_sets_proj_01.png`**: Projection onto dimensions 0-1 (Midge-Algae)
- **`morse_sets_proj_02.png`**: Projection onto dimensions 0-2 (Midge-Detritus)
- **`morse_sets_proj_12.png`**: Projection onto dimensions 1-2 (Algae-Detritus)

### Run-Specific Visualizations (in `run_NNN/results/`)

#### Training Quality

- **`training_curves.png`**: Loss curves during training
- **`encoder_decoder_roundtrip.png`**: E/D transformation quality (6-panel grid)
- **`trajectory_analysis.png`**: Trajectory simulations (3D + latent)

#### Latent Space Views

**Basic views:**

- `latent_morse_sets_only.png`: Just 2D Morse sets
- `latent_barycenters_only.png`: Just E(3D barycenters)
- `latent_equilibrium_only.png`: Just equilibrium

**Combined views:**

- `latent_data_morse_sets.png`: Data + 2D Morse sets
- `latent_data_barycenters.png`: Data + E(barycenters) + equilibrium
- `latent_morse_barycenters.png`: 2D Morse sets + E(barycenters) + equilibrium

**Period-12 orbit views** (if provided in config):

- `latent_period12_only.png`: Just E(period-12 orbit)
- `latent_data_period12.png`: Data + E(period-12 orbit)
- `latent_morse_period12.png`: Morse sets + E(period-12 orbit)

#### Morse Graph Comparisons

- **`morse_graph_2d_padded.png`**: 2D Morse graph (padded method)
- **`morse_graph_2d_unpadded.png`**: 2D Morse graph (unpadded method)
- **`morse_graph_comparison.png`**: 6-panel comparison (3D vs 2D methods)
- **`morse_2x2_comparison_padded.png`**: Clean 4-panel comparison (padded)
- **`morse_2x2_comparison_unpadded.png`**: Clean 4-panel comparison (unpadded)

#### Preimage Analysis

- **`preimage_classification_padded.png`**: Preimage analysis (padded method)
- **`preimage_classification_unpadded.png`**: Preimage analysis (unpadded method)

Shows how 3D space maps to 2D Morse sets under E(·).

## Dependencies

### Required

- **Python** ≥ 3.8
- **CMGDB**: C++ library for Morse graph computation
- **PyTorch**: Neural network training
- **NumPy, SciPy, NetworkX**: Numerical/graph operations
- **Matplotlib**: Visualization
- **joblib**: Parallel computation

### Optional

- **Graphviz**: For Morse graph diagram rendering

Install:

```bash
pip install -e .           # Core dependencies
pip install -e .[ml]       # Add PyTorch
pip install -e .[test]     # Add testing dependencies
```

## Technical Details

### Caching System

The caching implementation uses NetworkX DiGraph instead of CMGDB's C++ DirectedGraph:

**Advantages:**

- No CMGDB dependency for loading cached results
- Pure Python for easy analysis
- Full NetworkX graph algorithm support

**Cached Data:**

- `morse_graph_data.pkl`: NetworkX DiGraph with node attributes
  - Nodes: Morse set indices
  - Node attributes: `morse_set_boxes` (list of boxes)
  - Edges: Transitions between Morse sets
- `barycenters.npz`: NumPy arrays of barycenter coordinates
- `metadata.json`: Parameters and timestamp

**Access pattern:**

```python
result_3d, was_cached = load_or_compute_3d_morse_graph(...)
morse_graph = result_3d['morse_graph']

# CMGDB-compatible interface
num_sets = morse_graph.num_vertices()
edges = morse_graph.edges()
boxes = morse_graph.morse_set_boxes(node_id)

# Or access NetworkX graph directly
nx_graph = morse_graph.graph
sinks = [n for n in nx_graph.nodes() if nx_graph.out_degree(n) == 0]
```

### Parameter Hash Computation

Hash includes:

- Function source code (detects map changes)
- Domain bounds
- All CMGDB subdivision parameters
- Padding flag

Returns first 16 chars of SHA256 for readability.

## Common Issues

### 1. Long computation time

**Solution**: Start with `subdiv_min=15` for testing, increase gradually

### 2. Out of memory

**Solution**: Reduce `subdiv_limit` or decrease `subdiv_max`

### 3. Cache not found

**Solution**: Check that CMGDB parameters match exactly. Use `--force-recompute` to rebuild.

### 4. Missing dynamics parameters

**Error**: `ValueError: Missing required dynamics parameters in config: ['p', 'q']`
**Solution**: Add all required parameters to `dynamics:` section in YAML

## Extending the Pipeline

### Adding a New Ecological Model

1. Define map function in `MorseGraph/systems.py`
2. Create config file in `examples/configs/`
3. Set domain bounds to contain attractors
4. Start with low subdivision for testing

### Custom Visualizations

Access the data directly:

```python
from MorseGraph.utils import load_morse_graph_cache, get_cache_path

cache_paths = get_cache_path('cmgdb_3d', param_hash)
cached = load_morse_graph_cache('cmgdb_3d', param_hash)

morse_graph = cached['morse_graph']
barycenters = cached['barycenters']

# Your custom analysis here
import networkx as nx
scc = list(nx.strongly_connected_components(morse_graph.graph))
```

## References

- Ives et al. (2008). Nature 452: 84-87
- CMGDB: <https://github.com/shaunharker/CMGDB>
- MorseGraph documentation: See main README.md
