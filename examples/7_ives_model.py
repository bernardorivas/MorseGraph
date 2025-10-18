#!/usr/bin/env python3
"""
Ives Ecological Model - Using Generalized MorseGraph Pipeline

This example demonstrates the Ives midge-algae-detritus ecological model
using the generalized MorseGraph API. The model operates in log10 scale
to handle many orders of magnitude in population abundances.

Model: Ives et al. (2008) - "High-amplitude fluctuations and alternative
       dynamical states of midges in Lake Myvatn", Nature 452: 84-87

The system models:
- Midge larvae (M)
- Algae (A)
- Detritus (D)

with nonlinear interactions including midge consumption of algae/detritus.

Usage:
    python 7_ives_model.py

Output:
    All results saved to examples/ives_model_output/
"""

import numpy as np
import os
import sys
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import torch

# Add parent directory to path for MorseGraph imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.utils import (
    ExperimentConfig,
    generate_map_trajectory_data,
    setup_experiment_dirs,
    save_experiment_metadata,
    compute_latent_bounds
)
from MorseGraph.core import (
    compute_morse_graph_3d,
    compute_morse_graph_2d_data,
    compute_morse_graph_2d_restricted
)
from MorseGraph.training import train_autoencoder_dynamics
from MorseGraph.plot import (
    plot_morse_graph_diagram,
    plot_training_curves,
    plot_morse_sets_3d_scatter,
    plot_latent_space_2d
)
from MorseGraph.systems import ives_model_log


# ============================================================================
# Ives Model Parameters 
# ============================================================================

# Ecological parameters
R1 = 3.873          # Midge reproduction rate
R2 = 11.746         # Algae growth rate
C = 10**-6.435      # Constant input of algae and detritus
D = 0.5517          # Detritus decay rate
P = 0.06659         # Relative palatability of detritus
Q = 0.9026          # Exponent in midge consumption
LOG_OFFSET = 0.001  # Offset for log transform to avoid log0

# Equilibrium point in log10 scale - Computed numerically
EQUILIBRIUM_POINT = np.array([0.792107, 0.209010, 0.376449])

# Domain bounds in log10 scale
# These bounds focus on the region containing the computed Morse sets
DOMAIN_BOUNDS = [[-1, -4, -1], [2, 1, 1]]  # [[log(M), log(A), log(D)], ...]


# ============================================================================
# CMGDB / ML Configuration
# ============================================================================

def create_ives_config():
    """Create configuration with all Ives model parameters."""
    config = ExperimentConfig(
        # Domain (log10 scale)
        domain_bounds=DOMAIN_BOUNDS,

        # 3D CMGDB parameters - original space
        subdiv_min=15, #36
        subdiv_max=18, #50 
        subdiv_init=0, # 24
        subdiv_limit=50000,
        padding=True,

        # 2D CMGDB parameters - latent space
        latent_subdiv_min=14, # 24
        latent_subdiv_max=20, # 30 
        latent_subdiv_init=0,
        latent_subdiv_limit=50000,
        latent_padding=True,
        latent_bounds_padding=1.01,

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
        output_activation=None,  # tanh, sigmoid

        # Training parameters
        num_epochs=1500,
        batch_size=1024,
        learning_rate=0.001,
        early_stopping_patience=50,
        min_delta=1e-5,

        # Loss weights
        w_recon=1.0,
        w_dyn_recon=1.0,
        w_dyn_cons=1.0,
    )

    # Create ives_map
    ives_map = partial(
        ives_model_log,
        r1=R1, r2=R2, c=C, d=D, p=P, q=Q, offset=LOG_OFFSET
    )
    config.set_map_func(ives_map)

    return config, ives_map



# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Run complete Ives model analysis pipeline."""

    print("\n" + "="*80)
    print("IVES ECOLOGICAL MODEL - MORSE GRAPH ANALYSIS")
    print("="*80)
    print("\nModel: Midge-Algae-Detritus dynamics (Ives et al. 2008)")
    print(f"Parameters: R1={R1:.3f}, R2={R2:.3f}, C={C:.2e}, D={D:.4f}, P={P:.5f}, Q={Q:.3f}")
    print(f"Domain (log): {DOMAIN_BOUNDS[0]} to {DOMAIN_BOUNDS[1]}")
    print(f"Equilibrium (log): [{EQUILIBRIUM_POINT[0]:.4f}, {EQUILIBRIUM_POINT[1]:.4f}, {EQUILIBRIUM_POINT[2]:.4f}]")

    # Configuration
    config, ives_map = create_ives_config()

    # Setup directories
    base_dir = os.path.join(os.path.dirname(__file__), "ives_model_output")
    dirs = setup_experiment_dirs(base_dir)

    # ========================================================================
    # Step 1: Compute 3D Morse Graph (Ground Truth)
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 1: Computing 3D Morse Graph (Log Scale)")
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
    plot_morse_graph_diagram(
        morse_graph_3d,
        output_path=f"{dirs['results']}/morse_graph_3d.png",
        title="Ives Model - 3D Morse Graph Diagram"
    )

    plot_morse_sets_3d_scatter(
        morse_graph_3d,
        config.domain_bounds,
        output_path=f"{dirs['results']}/morse_sets_3d.png",
        title="Ives Model - Morse Sets (3D)",
        equilibria={'Equilibrium': EQUILIBRIUM_POINT},
        labels={'x': 'log(Midge)', 'y': 'log(Algae)', 'z': 'log(Detritus)'}
    )

    # ========================================================================
    # Step 2: Generate Training Data
    # ========================================================================

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

    # ========================================================================
    # Step 3: Train Autoencoder + Latent Dynamics
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 3: Training Autoencoder + Latent Dynamics")
    print("="*80)

    training_result = train_autoencoder_dynamics(
        x_train, y_train,
        x_val, y_val,
        config,
        verbose=True,
        progress_interval=100
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
    plot_training_curves(
        training_result['train_losses'],
        training_result['val_losses'],
        output_path=f"{dirs['results']}/training_curves.png"
    )

    # ========================================================================
    # Step 4: Compute Latent Bounds and Encode Data
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 4: Computing Latent Space Bounds")
    print("="*80)

    with torch.no_grad():
        z_train = encoder(torch.FloatTensor(x_train).to(device)).cpu().numpy()
        z_val = encoder(torch.FloatTensor(x_val).to(device)).cpu().numpy()
        z_all = encoder(torch.FloatTensor(X).to(device)).cpu().numpy()

    latent_bounds = compute_latent_bounds(z_train, padding_factor=config.latent_bounds_padding)
    print(f"Latent bounds: {latent_bounds.tolist()}")

    # ========================================================================
    # Step 5: Compute 2D Morse Graphs (Latent Space)
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 5: Computing 2D Morse Graphs (Latent Space)")
    print("="*80)

    # Method 1: BoxMapData on E(X_train) and G(E(X_train))
    print("\nMethod 1: BoxMapData on encoded training data...")
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
    print("  Generating large sample for domain restriction...")
    X_large, _, _ = generate_map_trajectory_data(
        config.map_func,
        n_trajectories=min(10000, config.n_trajectories * 2),
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

    # ========================================================================
    # Step 6: Visualize 2D Morse Graphs
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 6: Generating 2D Visualizations")
    print("="*80)

    # BoxMapData Morse graph
    plot_morse_graph_diagram(
        morse_graph_2d_data,
        output_path=f"{dirs['results']}/morse_graph_2d_data.png",
        title="Ives Model - 2D Morse Graph (BoxMapData)"
    )

    # Project equilibrium to latent space
    with torch.no_grad():
        equilibrium_tensor = torch.FloatTensor(EQUILIBRIUM_POINT).unsqueeze(0).to(device)
        equilibrium_latent = encoder(equilibrium_tensor).cpu().numpy()[0]

    plot_latent_space_2d(
        z_train,
        latent_bounds.tolist(),
        morse_graph=morse_graph_2d_data,
        output_path=f"{dirs['results']}/latent_space_data.png",
        title="Ives Model - Latent Space (BoxMapData)",
        equilibria_latent={'Equilibrium': equilibrium_latent}
    )

    # Domain-restricted Morse graph
    plot_morse_graph_diagram(
        morse_graph_2d_restricted,
        output_path=f"{dirs['results']}/morse_graph_2d_restricted.png",
        title="Ives Model - 2D Morse Graph (Domain-Restricted)"
    )

    plot_latent_space_2d(
        z_large,
        latent_bounds.tolist(),
        morse_graph=morse_graph_2d_restricted,
        output_path=f"{dirs['results']}/latent_space_restricted.png",
        title="Ives Model - Latent Space (Domain-Restricted)",
        equilibria_latent={'Equilibrium': equilibrium_latent}
    )

    # ========================================================================
    # Step 7: Save Results Summary
    # ========================================================================

    results = {
        # 3D Morse graph
        'num_morse_sets_3d': result_3d['num_morse_sets'],
        'computation_time_3d': result_3d['computation_time'],

        # 2D Morse graphs
        'num_morse_sets_2d_data': result_2d_data['num_morse_sets'],
        'num_morse_sets_2d_restricted': result_2d_restricted['num_morse_sets'],

        # Training
        'training_time': training_result['training_time'],
        'final_train_loss': training_result['train_losses']['total'][-1],
        'final_val_loss': training_result['val_losses']['total'][-1],

        # Model parameters
        'ives_parameters': {
            'r1': R1, 'r2': R2, 'c': C, 'd': D, 'p': P, 'q': Q,
            'log_offset': LOG_OFFSET
        },
        'equilibrium_point': EQUILIBRIUM_POINT.tolist(),
    }

    save_experiment_metadata(f"{dirs['base']}/metadata.json", config, results)

    print("\n" + "="*80)
    print("IVES MODEL ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults Summary:")
    print(f"  3D Morse Sets:              {results['num_morse_sets_3d']}")
    print(f"  2D Morse Sets (Data):       {results['num_morse_sets_2d_data']}")
    print(f"  2D Morse Sets (Restricted): {results['num_morse_sets_2d_restricted']}")
    print(f"  Training Time:              {results['training_time']:.2f}s")
    print(f"  Final Train Loss:           {results['final_train_loss']:.6f}")
    print(f"  Final Val Loss:             {results['final_val_loss']:.6f}")
    print(f"\nAll results saved to: {base_dir}")
    print(f"\nVisualization files:")
    print(f"  - morse_graph_3d.png: 3D Morse graph diagram")
    print(f"  - morse_sets_3d.png: 3D scatter plot with equilibrium")
    print(f"  - morse_sets_proj_*.png: 2D projections of 3D Morse sets")
    print(f"  - training_curves.png: Training/validation loss curves")
    print(f"  - latent_space_*.png: 2D latent space visualizations")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
