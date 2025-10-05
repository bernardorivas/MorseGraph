# Leslie Map Learned Dynamics Experiments

## Overview

This directory contains a comprehensive experimental framework for studying learned dynamics of the Leslie map using different neural network architectures and hyperparameters.

## Files

- **`6_map_learned_dynamics.py`** - Base experiment script that runs 50 independent trials with a given configuration
- **`7_multiple_runs.py`** - Orchestrator that runs example 6 with 11 different configurations

## Experiment Grid (11 Experiments)

### 1. Baseline
- 3 layers, 128 hidden units, no output activation
- Loss weights: (1, 1, 1)

### 2-3. Output Activation Variations
- **tanh_output**: Tanh activation on all network outputs
- **sigmoid_output**: Sigmoid activation on all network outputs

### 4-5. Network Depth Variations
- **shallow_2layer**: 2-layer networks
- **deep_4layer**: 4-layer networks

### 6-7. Network Width Variations
- **narrow_64**: 64 hidden units
- **wide_256**: 256 hidden units

### 8-11. Loss Weight Variations
- **recon_heavy**: (100, 1, 1) - Emphasize reconstruction
- **dyn_recon_heavy**: (1, 100, 1) - Emphasize dynamics prediction
- **dyn_cons_heavy**: (1, 1, 100) - Emphasize dynamics consistency
- **balanced_10**: (10, 10, 1) - Balanced emphasis

## Running Experiments

### Quick Test (single experiment, reduced runs)
```bash
cd examples
python 6_map_learned_dynamics.py
```

### Full Experiment Suite
```bash
cd examples
python 7_multiple_runs.py
```

**Expected Runtime**: ~44-88 hours (11 experiments × 50 runs × ~5-10 min/run)

## Output Structure

```
examples/leslie_experiments/
├── experiment_baseline/
│   ├── run_001/
│   │   ├── figures/
│   │   │   ├── 6_learned_dynamics_data.png
│   │   │   ├── 6_learned_dynamics_losses.png
│   │   │   ├── 6_learned_dynamics_trajectories.png
│   │   │   └── 6_morsegraph_comparison.png
│   │   └── data/
│   │       └── learned_models/
│   ├── run_002/
│   └── ...
│   └── experiment_results.json
├── experiment_tanh_output/
├── experiment_sigmoid_output/
├── ...
├── all_experiments_summary.json
├── comparison_boxplots.png
├── comparison_barplots.png
└── best_performers.txt
```

## Results Analysis

After running `7_multiple_runs.py`, you'll get:

1. **Individual Experiment Results** - JSON file per experiment with all 50 runs
2. **Aggregated Summary** - `all_experiments_summary.json` with complete results
3. **Comparison Visualizations**:
   - Box plots showing error distributions across experiments
   - Bar plots showing mean ± std for each metric
4. **Best Performers Summary** - Text file identifying best configs for each metric

## Key Metrics Tracked

For each run:
- Reconstruction error (MSE)
- Dynamics prediction error (MSE)
- Dynamics consistency error (MSE)
- Number of Morse sets (Leslie map 3D)
- Number of Morse sets (latent 2D full)
- Number of Morse sets (latent 2D restricted)

## Customizing Experiments

To add new experiments, edit `EXPERIMENTS` list in `7_multiple_runs.py`:

```python
EXPERIMENTS.append({
    "name": "custom_config",
    "description": "Description of your experiment",
    "config": {
        "HIDDEN_DIM": 128,
        "NUM_LAYERS": 3,
        "OUTPUT_ACTIVATION": None,  # or "tanh", "sigmoid"
        "W_RECON": 1.0,
        "W_DYN_RECON": 1.0,
        "W_DYN_CONS": 1.0,
        "NUM_RUNS": 50,
        # Can also override other Config parameters
        "NUM_EPOCHS": 300,
        "LEARNING_RATE": 0.001,
        # etc.
    }
})
```

## Model Architecture

All networks use:
- **Encoder**: 3D input → 2D latent
- **Decoder**: 2D latent → 3D output
- **Latent Dynamics**: 2D → 2D latent map

Architecture is controlled by:
- `HIDDEN_DIM`: Width of hidden layers
- `NUM_LAYERS`: Number of hidden layers (not counting input/output)
- `OUTPUT_ACTIVATION`: Final activation function (None, "tanh", "sigmoid")

All networks use ReLU activations in hidden layers.
