# MorseGraph Configuration Files

This directory contains YAML configuration files for MorseGraph experiments. Using configuration files makes it easy to:

- **Change parameters** without editing Python code
- **Version control** exact experimental settings
- **Share configurations** with collaborators
- **Reproduce results** by reusing exact configs
- **Run parameter sweeps** by creating config variants

## Quick Start

### Using a config file

```bash
# Run with default config
python 7_ives_model.py --config configs/ives_default.yaml

# Run with high-resolution config
python 7_ives_model.py --config configs/ives_high_res.yaml

# Quick test run
python 7_ives_model.py --config configs/ives_fast.yaml

# Force recompute 3D CMGDB with custom config
python 7_ives_model.py --config my_custom.yaml --force-recompute
```

## Available Configurations

### `ives_default.yaml` - Recommended

Standard configuration for the Ives ecological model with balanced quality and computation time.

- **3D CMGDB**: subdiv 33 (uniform grid, ~77 minutes)
- **Training**: 1000 trajectories, 1500 epochs (~2-3 minutes)
- **Use for**: Regular research experiments and analysis

### `ives_high_res.yaml` - Publication Quality

High-resolution settings for detailed analysis and publication figures.

- **3D CMGDB**: subdiv 36-50 (adaptive refinement, extended computation time)
- **Training**: 2000 trajectories, 2500 epochs (~5-7 minutes)
- **Use for**: Final results, publication figures, detailed structure

**Note**: Computationally expensive, intended for final analyses.

### `ives_fast.yaml` - Development Testing

Lightweight configuration for rapid iteration during development.

- **3D CMGDB**: subdiv 18-24 (adaptive refinement, ~1-2 minutes)
- **Training**: 200 trajectories, 300 epochs (~30 seconds)
- **Use for**: Quick tests, debugging, prototyping

**Note**: Not suitable for research results.

## Configuration File Structure

```yaml
# System metadata
system:
  name: "Ives Ecological Model"
  type: "map"
  description: "..."
  reference: "..."

# System-specific parameters (ecological model)
dynamics:
  r1: 3.873           # Model parameters
  r2: 11.746
  ...

# Computational domain
domain:
  bounds: [[-1, -4, -1], [2, 1, 1]]
  equilibrium: [0.792107, 0.209010, 0.376449]

# 3D CMGDB parameters
cmgdb_3d:
  subdiv_min: 33      # Grid resolution parameters
  subdiv_max: 39
  subdiv_limit: 100000
  padding: false

# 2D CMGDB parameters (latent space)
cmgdb_2d:
  subdiv_min: 22
  subdiv_max: 26
  ...

# Training data generation
training:
  n_trajectories: 1000
  n_points: 20
  num_epochs: 1500
  ...

# Neural network architecture
model:
  latent_dim: 2
  hidden_dim: 32
  num_layers: 3
  ...

# Loss function weights
loss_weights:
  w_recon: 1.0
  w_dyn_recon: 1.0
  w_dyn_cons: 1.0
```

## Creating Custom Configurations

### Option 1: Copy and modify an existing config

```bash
cp configs/ives_default.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml with your parameters
python 7_ives_model.py --config configs/my_experiment.yaml
```

### Option 2: Minimal config (inherits from default)

Create a config with only the parameters you want to change:

```yaml
# my_custom.yaml - only override specific parameters
cmgdb_3d:
  subdiv_min: 30
  subdiv_max: 42

training:
  num_epochs: 2000
```

## Key Parameters to Adjust

### For Resolution/Quality

- **`cmgdb_3d.subdiv_min/max`**: Controls 3D grid resolution
  - Lower (15-25): Fast, coarse
  - Medium (30-40): Good quality
  - Higher (45-55): Very fine, slow

- **`cmgdb_2d.subdiv_min/max`**: Controls latent space resolution
  - Similar trade-offs as 3D

### For Training Quality

- **`training.n_trajectories`**: More data = better learning
  - 100-500: Quick testing
  - 1000-2000: Good quality
  - 5000+: Publication quality

- **`training.num_epochs`**: Training duration
  - 300-500: Fast
  - 1000-2000: Standard
  - 3000+: Thorough

### For Model Capacity

- **`model.hidden_dim`**: Network size
  - 16-32: Fast, simple
  - 64-128: More capacity
  - 256+: Very expressive

- **`model.num_layers`**: Network depth
  - 2: Shallow
  - 3-4: Standard
  - 5+: Deep

## Config Management Tips

### 1. Version Control Your Configs

```bash
git add configs/my_experiment_v1.yaml
git commit -m "Add experiment config for parameter sweep"
```

### 2. Configs are Automatically Saved

Each run saves the exact config used:

```
ives_model_output/
├── run_001/
│   ├── config.yaml     # ← Exact config used for this run
│   └── results/
```

### 3. Parameter Sweeps

Create multiple configs for systematic exploration:

```
configs/
├── sweep_subdiv_30.yaml
├── sweep_subdiv_35.yaml
├── sweep_subdiv_40.yaml
└── sweep_subdiv_45.yaml
```

Then run:

```bash
for config in configs/sweep_*.yaml; do
    python 7_ives_model.py --config $config
done
```

### 4. Sharing Configurations

To share exact experimental settings with collaborators:

```bash
# Send them the config file
git push origin my-experiment-configs

# They can reproduce your results
python 7_ives_model.py --config configs/your_config.yaml
```

## Troubleshooting

### "Missing required config section: 'domain'"

Your config file is missing required sections. Check the structure against `ives_default.yaml`.

### "Config file not found"

Make sure to provide the full path relative to where you run the script:

```bash
python 7_ives_model.py --config configs/ives_default.yaml  # ✓ Correct
python 7_ives_model.py --config ives_default.yaml          # ✗ Won't find it
```

### "subdiv_min must be <= subdiv_max"

Check your CMGDB parameters - min should always be less than or equal to max.

## Template for New Systems

When creating configs for a new dynamical system, use this template:

```yaml
system:
  name: "My System"
  type: "map"  # or "ode"

dynamics:
  # Your system-specific parameters here
  param1: value1
  param2: value2

domain:
  bounds: [[x_min, y_min, z_min], [x_max, y_max, z_max]]

cmgdb_3d:
  subdiv_min: 30
  subdiv_max: 40
  subdiv_limit: 100000
  padding: false

# ... rest follows ives_default.yaml structure
```

## Further Reading

- See `7_ives_model.py` for how configs are loaded
- See `MorseGraph/config.py` for config utilities
- See `TODO.md` for planned config features
