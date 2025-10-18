# MorseGraph Examples

This directory contains examples demonstrating various features of the MorseGraph library.

## Quick Reference

| Example | Description | Concepts |
|---------|-------------|----------|
| `1_map_dynamics.py` | Basic discrete map BoxMap | Maps, grids, box mapping |
| `2_ode_dynamics.py` | ODE-based dynamics (Van der Pol) | ODEs, continuous systems |
| `3_ode_dynamics.py` | ODE dynamics with Morse graph | ODE integration, Morse decomposition |
| `4_data_driven.py` | Data-driven BoxMap computation | Empirical data, BoxMapData |
| `5_ode_learned_dynamics.py` | Learning dynamics from ODE data | Autoencoder, latent dynamics, ODE |
| `6_map_learned_dynamics.py` | Learning dynamics from map data | Autoencoder, latent dynamics, maps |
| `7_ives_model.py` | **Ives ecological model (NEW)** | Autoencoder, latent dynamics, maps |

### Output of `7_ives_model.py`

```
examples/ives_model_output/
├── metadata.json                      # All parameters and results
├── models/                            # Trained neural networks
├── training_data/                     # Generated trajectories
└── results/                           # 10+ visualization files
    ├── morse_graph_3d.png            # 3D Morse graph structure
    ├── morse_sets_3d.png             # 3D scatter with equilibrium
    ├── morse_sets_proj_01.png        # Midge vs Algae
    ├── morse_sets_proj_02.png        # Midge vs Detritus
    ├── morse_sets_proj_12.png        # Algae vs Detritus
    ├── training_curves.png           # Loss curves
    └── latent_space_*.png            # 2D latent space plots
```

### Parameters

- Domain: `[[-1,-4,-1], [2,1,1]]` (log scale)
- Subdivisions: 36-50 (high resolution)
- Training: 1500 epochs, 5000 trajectories
- Architecture: 3D→2D→3D (latent dim = 2)

### Reference

Ives, A.R., et al., "High-amplitude fluctuations and alternative dynamical states of midges in Lake Myvatn", Nature 452: 84-87 (2008)

## Example Categories

### Basic Examples (1-4)

Foundational examples demonstrating core library features:

- Grid discretization
- BoxMap computation (function-based and data-driven)
- ODE integration
- Morse graph analysis

### Learning Examples (5-6)

Neural network-based dynamics learning:

- Autoencoder for dimensionality reduction
- Latent dynamics learning
- Model comparison and validation

### Application Examples (7)

Real-world scientific applications:

- Ecological modeling (Ives model)
- Full analysis pipeline
- Domain-specific visualizations

## Generic Pipeline Template

For creating your own examples with any 3D map, see:

```
generic_3d_pipeline/README.md
```

This provides a complete template and documentation for:

1. Defining a custom 3D map f: R³ → R³
2. Configuring the experiment
3. Running the full 3D→2D learning pipeline
4. Generating all visualizations

## Common Patterns

### Using ExperimentConfig (NEW)

```python
from MorseGraph.utils import ExperimentConfig

config = ExperimentConfig(
    domain_bounds=[[-5,-5,-5], [5,5,5]],
    latent_dim=2,
    hidden_dim=32,
    num_epochs=1500
)
config.set_map_func(my_map_function)
```

### Computing 3D Morse Graphs

```python
from MorseGraph.core import compute_morse_graph_3d

result = compute_morse_graph_3d(
    config.map_func,
    config.domain_bounds,
    subdiv_min=30,
    subdiv_max=42
)
morse_graph = result['morse_graph']
```

### Training Models (NEW)

```python
from MorseGraph.training import train_autoencoder_dynamics

result = train_autoencoder_dynamics(
    x_train, y_train,
    x_val, y_val,
    config
)
encoder = result['encoder']
latent_dynamics = result['latent_dynamics']
```

### 2D Morse Graphs (NEW)

```python
from MorseGraph.core import compute_morse_graph_2d_data, compute_morse_graph_2d_restricted

# Method 1: BoxMapData
result_data = compute_morse_graph_2d_data(
    latent_dynamics, device, z_train, latent_bounds
)

# Method 2: Domain-restricted
result_restricted = compute_morse_graph_2d_restricted(
    latent_dynamics, device, z_large, latent_bounds,
    include_neighbors=True
)
```

## Running Examples

All examples are standalone Python scripts:

```bash
# Run any example
python 1_map_dynamics.py
python 7_ives_model.py

# Or make executable and run directly
chmod +x 7_ives_model.py
./7_ives_model.py
```

## Dependencies

Core examples (1-4):

- numpy
- scipy
- matplotlib
- networkx
- CMGDB

Learning examples (5-7):

- torch (PyTorch)
- All core dependencies

## Tips

1. **Start with**: `1_map_dynamics.py` for basics, `7_ives_model.py` for full pipeline
2. **GPU/MPS**: Learning examples automatically use GPU if available
3. **Memory**: Large samples in domain-restricted methods can use significant RAM
4. **Computation Time**: High subdivision levels (>42) can be slow in CMGDB
5. **Customization**: Copy `7_ives_model.py` and adapt for your own system

## Related Directories

- `ives_model/` - Original Ives model implementation (more features, custom plots)
- `leslie_map_3d/` - Leslie population model implementation
- `generic_3d_pipeline/` - Template and documentation for custom maps

## Contributing

When adding new examples:

1. Number sequentially (8_, 9_, etc.)
2. Include docstring with description and usage
3. Make executable: `chmod +x example.py`
4. Update this README
5. Add to Quick Reference table

## Support

For issues or questions:

- Check `generic_3d_pipeline/README.md` for detailed documentation
- Review `CLAUDE.md` in project root for architecture details
- See specific model directories (`ives_model/`, `leslie_map_3d/`) for advanced usage
