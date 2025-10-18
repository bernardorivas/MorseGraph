# 7_ives_model.py - Implementation Notes

## Overview
Complete adaptation of `ives_model/` to the new generalized MorseGraph API, demonstrating how to use the unified pipeline for a real-world ecological model.

## Key Features Implemented

### 1. Exact Parameter Preservation
All parameters from `ives_model/ives_modules/config.py`:
```python
R1 = 3.873          # Midge reproduction rate
R2 = 11.746         # Algae growth rate
C = 10**-6.435      # Constant input
D = 0.5517          # Detritus decay rate
P = 0.06659         # Palatability
Q = 0.9026          # Consumption exponent
LOG_OFFSET = 0.001  # Log transform offset

DOMAIN_BOUNDS = [[-1,-4,-1], [2,1,1]]  # Log₁₀ scale
EQUILIBRIUM_POINT = [0.792107, 0.209010, 0.376449]

# CMGDB 3D
subdiv_min=36, subdiv_max=50, subdiv_init=24

# CMGDB 2D (latent)
latent_subdiv_min=30, latent_subdiv_max=40

# Architecture
latent_dim=2, hidden_dim=32, num_layers=3

# Training
num_epochs=1500, batch_size=1024
n_trajectories=5000, n_points=20
```

### 2. Generalized API Usage
```python
# Configuration
config = ExperimentConfig(...)
ives_map = partial(ives_model_log, r1=R1, r2=R2, ...)
config.set_map_func(ives_map)

# 3D Morse graph
result_3d = compute_morse_graph_3d(config.map_func, config.domain_bounds, ...)

# Training
training_result = train_autoencoder_dynamics(x_train, y_train, x_val, y_val, config)

# 2D Morse graphs
result_2d_data = compute_morse_graph_2d_data(latent_dynamics, device, z_train, ...)
result_2d_restricted = compute_morse_graph_2d_restricted(latent_dynamics, device, z_large, ...)
```

### 3. Enhanced Visualizations

#### Custom Functions Added
- `plot_morse_sets_3d_with_equilibrium()`: 3D scatter with red star for equilibrium
- `plot_morse_sets_projection_2d()`: 2D projections with equilibrium marked
- `plot_latent_space_with_equilibrium()`: Latent space with projected equilibrium

#### Ecological Labels
- X-axis: "log₁₀(Midge)"
- Y-axis: "log₁₀(Algae)"
- Z-axis: "log₁₀(Detritus)"

#### Projections Created
1. Midge vs Algae (dims 0-1)
2. Midge vs Detritus (dims 0-2)
3. Algae vs Detritus (dims 1-2)

### 4. Complete Output Structure
```
examples/ives_model_output/
├── metadata.json                      # Config + results + Ives parameters
├── training_data/
│   └── trajectory_data.npz
├── models/
│   ├── encoder.pt
│   ├── decoder.pt
│   └── latent_dynamics.pt
└── results/
    ├── morse_graph_3d.png             # 3D graph diagram
    ├── morse_sets_3d.png              # 3D scatter with equilibrium
    ├── morse_sets_proj_01.png         # Midge vs Algae
    ├── morse_sets_proj_02.png         # Midge vs Detritus
    ├── morse_sets_proj_12.png         # Algae vs Detritus
    ├── training_curves.png            # 4-panel loss plots
    ├── morse_graph_2d_data.png        # BoxMapData diagram
    ├── latent_space_data.png          # BoxMapData latent space
    ├── morse_graph_2d_restricted.png  # Restricted diagram
    └── latent_space_restricted.png    # Restricted latent space
```

## Differences from Original ives_model/

### Advantages of New Version
1. **Unified API**: Uses same pattern as generic_3d_pipeline
2. **Less Code**: ~500 lines vs ~800+ across multiple modules
3. **Clearer Flow**: Single file, sequential steps
4. **Maintainable**: Leverages library functions, no duplication
5. **Comparable**: Easy to compare with Leslie or other models

### Original Features Not Included
- Custom multi-panel comparison figures (can add if needed)
- Multiple runs with parameter sweeps (separate script: `3_run_multiple.py`)
- Pre-computation caching (not needed for single runs)
- CSV exports of Morse sets (can add if needed)

### Features Added
- Two 2D methods side-by-side (BoxMapData + domain-restricted)
- Projected equilibrium in latent space
- Uses generalized plotting functions
- Automatic GPU/MPS detection
- Modern PyTorch training loop with early stopping

## Usage Comparison

### Old Way (ives_model/)
```python
from ives_modules import Config, run_learning_experiment

result = run_learning_experiment(
    run_name="my_run",
    output_dir=None,  # Uses Config paths
    config_overrides={'NUM_EPOCHS': 2000},
    verbose=True
)
```

### New Way (7_ives_model.py)
```python
# Just run the script
python 7_ives_model.py

# Or customize by editing config in main()
config = ExperimentConfig(
    domain_bounds=[[-1,-4,-1], [2,1,1]],
    num_epochs=2000,
    # ... all parameters explicit
)
```

## Validation

### Parameter Parity Check
✓ All R1, R2, C, D, P, Q, LOG_OFFSET match
✓ Domain bounds match: [[-1,-4,-1], [2,1,1]]
✓ Equilibrium point matches: [0.792107, 0.209010, 0.376449]
✓ Subdivision parameters match: 36-50 (3D), 30-40 (2D)
✓ Architecture matches: 3→2→3 with hidden_dim=32
✓ Training matches: 1500 epochs, 5000 trajectories

### Expected Results
Should produce similar Morse graphs to original implementation:
- 3D: Multiple Morse sets (typically 3-5)
- 2D: Simplified structure in latent space
- Equilibrium contained within a Morse set
- Topological features preserved

## Performance Notes

### Computation Time (estimated)
- 3D Morse graph: ~2-5 minutes (subdiv 36-50)
- Data generation: ~30 seconds (5000 trajectories)
- Training: ~5-10 minutes (1500 epochs, CPU)
- 2D Morse graphs: ~1-2 minutes each
- **Total**: ~15-25 minutes on CPU

### Memory Usage
- Training data: ~500 MB
- Model: ~1 MB
- Large sample (10k): ~800 MB
- Peak RAM: ~2-3 GB

### GPU Acceleration
Training time can be reduced to ~2-3 minutes on GPU/MPS

## Future Enhancements

### Potential Additions
1. Parameter sweep functionality (like `3_run_multiple.py`)
2. Comparison with precomputed ground truth
3. Additional biological metrics (stability, bifurcations)
4. Interactive visualization (plotly)
5. Export to formats compatible with biology tools

### Integration Opportunities
- Could be imported as module: `from examples.ives_model_7 import create_ives_config`
- Could be base class for Ives variants
- Could be template for other ecological models

## References

**Original Implementation**: `ives_model/`
**Model Paper**: Ives, A.R., et al., Nature 452: 84-87 (2008)
**Generalized Pipeline**: `examples/generic_3d_pipeline/README.md`
**Library Docs**: `CLAUDE.md`

## Questions/Support

For questions about:
- **Ives model specifics**: See `ives_model/` directory
- **Generalized pipeline**: See `generic_3d_pipeline/README.md`
- **API usage**: See `examples/README.md`
- **Library architecture**: See `CLAUDE.md`
