# Generic 3D Pipeline Example

This example demonstrates how to use the generalized MorseGraph tools for any 3D map f: R³ → R³.

## Overview

The pipeline consists of:

1. **3D Morse Graph Computation**: Compute ground truth Morse graph using CMGDB
2. **Learning Pipeline**: Train autoencoder + latent dynamics
3. **2D Morse Graphs**: Compute Morse graphs in learned 2D latent space
4. **Visualization**: Generate figures for all components

## Quick Start

```python
import numpy as np
from MorseGraph.utils import ExperimentConfig, generate_map_trajectory_data, setup_experiment_dirs, save_experiment_metadata
from MorseGraph.core import compute_morse_graph_3d, compute_morse_graph_2d_data, compute_morse_graph_2d_restricted
from MorseGraph.training import train_autoencoder_dynamics
from MorseGraph.plot import (plot_morse_graph_diagram, plot_training_curves,
                              plot_morse_sets_barycenters_3d, plot_latent_space_2d)
import torch

# ============================================================================
# Step 1: Define your 3D map
# ============================================================================

def my_map(x):
    """
    Example: A simple nonlinear 3D map
    Replace this with your own dynamics!
    """
    x1, x2, x3 = x[0], x[1], x[2]
    y1 = 0.9 * x1 + 0.1 * x2
    y2 = 0.1 * x1 + 0.8 * x2 + 0.1 * x3
    y3 = 0.9 * x3 + 0.05 * (x1**2 + x2**2)
    return np.array([y1, y2, y3])

# ============================================================================
# Step 2: Configure experiment
# ============================================================================

config = ExperimentConfig(
    # Domain
    domain_bounds=[[-5, -5, -5], [5, 5, 5]],

    # 3D CMGDB parameters
    subdiv_min=30,
    subdiv_max=42,
    subdiv_init=0,
    subdiv_limit=10000,
    padding=True,

    # Data generation
    n_trajectories=1000,
    n_points=20,
    skip_initial=0,
    random_seed=42,

    # Neural network architecture
    input_dim=3,
    latent_dim=2,
    hidden_dim=32,
    num_layers=3,
    output_activation=None,

    # Training
    num_epochs=500,  # Reduced for quick testing
    batch_size=1024,
    learning_rate=0.001,
    early_stopping_patience=50,

    # Latent space parameters
    latent_subdiv_min=20,
    latent_subdiv_max=28,
    latent_padding=True,
)

config.set_map_func(my_map)

# ============================================================================
# Step 3: Setup directories
# ============================================================================

base_dir = "./experiment_output"
dirs = setup_experiment_dirs(base_dir)

# ============================================================================
# Step 4: Compute 3D Morse Graph (Ground Truth)
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Computing 3D Morse Graph")
print("="*80)

result_3d = compute_morse_graph_3d(
    config.map_func,
    config.domain_bounds,
    subdiv_min=config.subdiv_min,
    subdiv_max=config.subdiv_max,
    subdiv_init=config.subdiv_init,
    subdiv_limit=config.subdiv_limit,
    padding=config.padding,
    verbose=True
)

morse_graph_3d = result_3d['morse_graph']
barycenters_3d = result_3d['barycenters']

# Visualize 3D Morse graph
plot_morse_graph_diagram(morse_graph_3d,
                         output_path=f"{dirs['results']}/morse_graph_3d.png",
                         title="3D Morse Graph")
plot_morse_sets_barycenters_3d(barycenters_3d, config.domain_bounds,
                               output_path=f"{dirs['results']}/morse_sets_3d.png")

# ============================================================================
# Step 5: Generate Training Data
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Generating Training Data")
print("="*80)

X, Y, trajectories = generate_map_trajectory_data(
    config.map_func,
    config.n_trajectories,
    config.n_points,
    np.array(config.domain_bounds),
    random_seed=config.random_seed,
    skip_initial=config.skip_initial
)

# Train/val split
split = int(0.8 * len(X))
x_train, y_train = X[:split], Y[:split]
x_val, y_val = X[split:], Y[split:]

print(f"Train: {len(x_train)} samples")
print(f"Val:   {len(x_val)} samples")

# ============================================================================
# Step 6: Train Autoencoder + Latent Dynamics
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Training Autoencoder + Latent Dynamics")
print("="*80)

training_result = train_autoencoder_dynamics(
    x_train, y_train,
    x_val, y_val,
    config,
    verbose=True,
    progress_interval=50
)

encoder = training_result['encoder']
decoder = training_result['decoder']
latent_dynamics = training_result['latent_dynamics']
device = training_result['device']

# Save models
torch.save(encoder.state_dict(), f"{dirs['models']}/encoder.pt")
torch.save(decoder.state_dict(), f"{dirs['models']}/decoder.pt")
torch.save(latent_dynamics.state_dict(), f"{dirs['models']}/latent_dynamics.pt")

# Plot training curves
plot_training_curves(training_result['train_losses'],
                     training_result['val_losses'],
                     output_path=f"{dirs['results']}/training_curves.png")

# ============================================================================
# Step 7: Compute Latent Bounds and Encode Data
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Computing Latent Space Bounds")
print("="*80)

from MorseGraph.utils import compute_latent_bounds

with torch.no_grad():
    z_train = encoder(torch.FloatTensor(x_train).to(device)).cpu().numpy()
    z_val = encoder(torch.FloatTensor(x_val).to(device)).cpu().numpy()
    z_all = encoder(torch.FloatTensor(X).to(device)).cpu().numpy()

latent_bounds = compute_latent_bounds(z_train, padding_factor=config.latent_bounds_padding)
print(f"Latent bounds: {latent_bounds.tolist()}")

# ============================================================================
# Step 8: Compute 2D Morse Graphs (Latent Space)
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Computing 2D Morse Graphs (Latent Space)")
print("="*80)

# Method 1: BoxMapData on E(X_train) and G(E(X_train))
print("\nMethod 1: BoxMapData...")
result_2d_data = compute_morse_graph_2d_data(
    latent_dynamics,
    device,
    z_train,
    latent_bounds.tolist(),
    subdiv_min=config.latent_subdiv_min,
    subdiv_max=config.latent_subdiv_max,
    subdiv_init=config.latent_subdiv_init,
    subdiv_limit=config.latent_subdiv_limit,
    verbose=True
)

morse_graph_2d_data = result_2d_data['morse_graph']

# Method 2: Domain-restricted with neighbors
print("\nMethod 2: Domain-restricted (with neighbors)...")

# Generate large sample for better coverage
X_large, _, _ = generate_map_trajectory_data(
    config.map_func,
    n_trajectories=min(10000, config.n_trajectories * 5),
    n_points=10,
    sampling_domain=np.array(config.domain_bounds),
    random_seed=None,  # Different seed
    skip_initial=0
)

with torch.no_grad():
    z_large = encoder(torch.FloatTensor(X_large).to(device)).cpu().numpy()

result_2d_restricted = compute_morse_graph_2d_restricted(
    latent_dynamics,
    device,
    z_large,
    latent_bounds.tolist(),
    subdiv_min=config.latent_subdiv_min,
    subdiv_max=config.latent_subdiv_max,
    subdiv_init=config.latent_subdiv_init,
    subdiv_limit=config.latent_subdiv_limit,
    include_neighbors=True,
    padding=config.latent_padding,
    verbose=True
)

morse_graph_2d_restricted = result_2d_restricted['morse_graph']

# ============================================================================
# Step 9: Visualize 2D Morse Graphs
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Generating Visualizations")
print("="*80)

# BoxMapData Morse graph
plot_morse_graph_diagram(morse_graph_2d_data,
                         output_path=f"{dirs['results']}/morse_graph_2d_data.png",
                         title="2D Morse Graph (BoxMapData)")
plot_latent_space_2d(z_train, latent_bounds.tolist(), morse_graph_2d_data,
                     output_path=f"{dirs['results']}/latent_space_data.png",
                     title="Latent Space (BoxMapData)")

# Domain-restricted Morse graph
plot_morse_graph_diagram(morse_graph_2d_restricted,
                         output_path=f"{dirs['results']}/morse_graph_2d_restricted.png",
                         title="2D Morse Graph (Domain-Restricted)")
plot_latent_space_2d(z_large, latent_bounds.tolist(), morse_graph_2d_restricted,
                     output_path=f"{dirs['results']}/latent_space_restricted.png",
                     title="Latent Space (Domain-Restricted)")

# ============================================================================
# Step 10: Save Results Summary
# ============================================================================

results = {
    'num_morse_sets_3d': result_3d['num_morse_sets'],
    'computation_time_3d': result_3d['computation_time'],
    'num_morse_sets_2d_data': result_2d_data['num_morse_sets'],
    'num_morse_sets_2d_restricted': result_2d_restricted['num_morse_sets'],
    'training_time': training_result['training_time'],
    'final_train_loss': training_result['train_losses']['total'][-1],
    'final_val_loss': training_result['val_losses']['total'][-1],
}

save_experiment_metadata(f"{dirs['base']}/results_summary.json", config, results)

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print(f"\nResults Summary:")
print(f"  3D Morse Sets:              {results['num_morse_sets_3d']}")
print(f"  2D Morse Sets (Data):       {results['num_morse_sets_2d_data']}")
print(f"  2D Morse Sets (Restricted): {results['num_morse_sets_2d_restricted']}")
print(f"  Training Time:              {results['training_time']:.2f}s")
print(f"  Final Train Loss:           {results['final_train_loss']:.6f}")
print(f"  Final Val Loss:             {results['final_val_loss']:.6f}")
print(f"\nAll results saved to: {base_dir}")
print("="*80)
```

## Output Structure

```
experiment_output/
├── results_summary.json          # Metadata and results
├── training_data/                # Generated trajectory data
├── models/                       # Trained model weights
│   ├── encoder.pt
│   ├── decoder.pt
│   └── latent_dynamics.pt
└── results/                      # Visualizations
    ├── morse_graph_3d.png        # 3D Morse graph diagram
    ├── morse_sets_3d.png         # 3D scatter plot
    ├── training_curves.png       # Training loss curves
    ├── morse_graph_2d_data.png   # 2D Morse graph (BoxMapData)
    ├── latent_space_data.png     # 2D latent space (BoxMapData)
    ├── morse_graph_2d_restricted.png  # 2D Morse graph (restricted)
    └── latent_space_restricted.png    # 2D latent space (restricted)
```

## Customization

### Using Your Own Map

Simply replace `my_map` with your dynamical system:

```python
def my_map(x):
    """Your custom 3D map: f: R³ → R³"""
    # Your dynamics here
    return np.array([...])
```

### Adjusting Parameters

All parameters can be configured via `ExperimentConfig`:

```python
config = ExperimentConfig(
    domain_bounds=[...],      # Your domain
    subdiv_min=30,            # CMGDB fineness
    latent_dim=2,             # Latent space dimension
    hidden_dim=64,            # Network capacity
    num_epochs=1500,          # Training duration
    # ... many more options
)
```

## Advanced Usage

### Custom Training Loop

The `train_autoencoder_dynamics` function returns trained models that you can use directly:

```python
training_result = train_autoencoder_dynamics(x_train, y_train, x_val, y_val, config)
encoder = training_result['encoder']

# Use the encoder
with torch.no_grad():
    z = encoder(torch.FloatTensor(x).to(device)).cpu().numpy()
```

### Multiple Runs

```python
for run_id in range(10):
    config = ExperimentConfig(random_seed=run_id)
    config.set_map_func(my_map)

    # Run full pipeline...
    # Save to f"experiment_output/run_{run_id:03d}/"
```

## Integration with Existing Examples

This generalized pipeline is used by:

- `ives_model/` - Ives ecological model
- `leslie_map_3d/` - Leslie population model

You can look at those directories for more complex examples with custom plotting and analysis.

## Notes

- **Memory**: Large samples for domain-restricted computation can use significant RAM
- **Computation Time**: CMGDB can be slow for fine subdivisions (high `subdiv_max`)
- **GPU**: Training automatically uses GPU/MPS if available
- **Normalization**: Automatically applied when using `tanh` or `sigmoid` activations
