#!/usr/bin/env python3
"""
MorseGraph Example 6: Learned Dynamics - Leslie Map

This example demonstrates learning a low-dimensional representation of a
dynamical system using an autoencoder framework.

We learn:
- E: X → Y (encoder: 3D → 2D)
- D: Y → X (decoder: 2D → 3D)
- G: Y → Y (latent dynamics: 2D → 2D)

Where f(x) is the Leslie map, and we minimize:
  ||D(E(x)) - x||² (reconstruction)
  + ||D(G(E(x))) - f(x)||² (dynamics reconstruction)
  + ||G(E(x)) - E(f(x))||² (dynamics consistency)

This allows us to study the 3D Leslie map dynamics in a simpler 2D latent space.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import math
from scipy.spatial import cKDTree

# MorseGraph imports
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapFunction, BoxMapLearnedLatent
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph
from MorseGraph.plot import plot_morse_graph, plot_morse_sets, plot_morse_sets_3d
from MorseGraph.utils import (
    save_trajectory_data,
    load_trajectory_data,
    compute_latent_bounds,
    filter_boxes_near_data
)


# ML dependencies - check availability
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from MorseGraph.models import Encoder, Decoder, LatentDynamics
    ML_AVAILABLE = True
except ImportError:
    print("PyTorch not available. This example requires PyTorch for ML functionality.")
    ML_AVAILABLE = False

# Set up output directories
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'leslie_model')
figures_dir = os.path.join(output_dir, 'figures')
data_dir = os.path.join(output_dir, "data")
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)


# ============================================================================
# Leslie Map Definition
# ============================================================================
def leslie_map(x):
    """Leslie map dynamics.

    Parameters:
        x: array-like of shape (3,) representing the state [x0, x1, x2]

    Returns:
        Next state as numpy array
    """
    theta_1 = 28.9
    theta_2 = 29.8
    theta_3 = 22.0

    x0_next = (theta_1 * x[0] + theta_2 * x[1] + theta_3 * x[2]) * math.exp(-0.1 * (x[0] + x[1] + x[2]))
    x1_next = 0.7 * x[0]
    x2_next = 0.7 * x[1]

    return np.array([x0_next, x1_next, x2_next])


def generate_leslie_trajectory_data(n_trajectories, n_points, sampling_domain, random_seed=None, skip_initial=0):
    """Generate trajectory data from the Leslie map.

    Args:
        n_trajectories: Number of trajectories to generate
        n_points: Number of transitions per trajectory (will generate n_points + 1 states)
        sampling_domain: Array of shape (2, 3) with [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        random_seed: Random seed for reproducibility
        skip_initial: Number of initial transitions to skip (use tail of trajectory for training)

    Returns:
        x_t: Array of shape (n_trajectories * (n_points - skip_initial), 3) with current states
        x_t_plus_1: Array of shape (n_trajectories * (n_points - skip_initial), 3) with next states
        trajectories: List of n_trajectories trajectories, each of shape (n_points + 1, 3)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    x_t_list = []
    x_t_plus_1_list = []
    trajectories = []

    for _ in range(n_trajectories):
        # Random initial condition in sampling domain
        x_current = np.random.uniform(sampling_domain[0], sampling_domain[1])

        # Generate trajectory
        trajectory = [x_current.copy()]
        for i in range(n_points):
            x_next = leslie_map(x_current)
            # Only add to training data if past skip_initial
            if i >= skip_initial:
                x_t_list.append(x_current.copy())
                x_t_plus_1_list.append(x_next.copy())
            trajectory.append(x_next.copy())
            x_current = x_next

        trajectories.append(np.array(trajectory))

    return np.array(x_t_list), np.array(x_t_plus_1_list), trajectories


# ============================================================================
# Configuration
# ============================================================================
class Config:
    '''Configuration parameters for the learned dynamics example.'''
    # --- Training Control ---
    FORCE_RETRAIN = True
    NUM_EPOCHS = 300
    RANDOM_SEED = 42

    # --- Multiple Runs ---
    NUM_RUNS = 50  # Number of independent runs to perform

    # --- Data Generation (Leslie Map) ---
    N_TRAJECTORIES = 5000  # Number of initial conditions
    N_POINTS = 10  # Number of transitions per trajectory
    SKIP_INITIAL = 8  # Skip first N transitions, use tail of trajectory for training
    DOMAIN_BOUNDS = [[-0.1, -0.1, -0.1], [90.0, 70.0, 70.0]]  # Domain for Leslie map

    # --- Model Architecture ---
    INPUT_DIM = 3
    LATENT_DIM = 2
    HIDDEN_DIM = 128

    # --- Training Parameters ---
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    # Loss weights
    W_RECON = 1.0
    W_DYN_RECON = 1.0
    W_DYN_CONS = 1.0

    # --- MorseGraph Computation ---
    # For Leslie map
    LESLIE_GRID_DIVISIONS = [64, 64, 64]
    LESLIE_EPSILON_BLOAT = 0.1
    # For learned latent system
    LATENT_GRID_DIVISIONS = [256,256]
    LATENT_BOUNDS_PADDING = 1.5
    DATA_RESTRICTION_EPSILON_FACTOR = 1.5

# ============================================================================
# Visualization Functions
# ============================================================================
def create_morsegraph_comparison_figure(
    lorenz_morse_graph, lorenz_grid, lorenz_box_map,
    latent_morse_graph_full, latent_box_map_full, latent_grid_full,
    latent_morse_graph_restricted, latent_box_map_restricted, restricted_active_boxes,
    encoded_train_data,
    output_path
):
    """Create 2x3 comparison figure."""

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Morse Graph Analysis: Leslie Map vs. Learned Latent Dynamics", fontsize=16)

    # Top row: Morse graphs
    ax1 = plt.subplot(2, 3, 1)
    if lorenz_morse_graph is not None:
        plot_morse_graph(lorenz_morse_graph, ax=ax1)
    ax1.set_title('Morse Graph: Leslie Map (3D)')
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    if latent_morse_graph_full is not None:
        plot_morse_graph(latent_morse_graph_full, ax=ax2)
    ax2.set_title('Morse Graph: Latent Dynamics on $R^2$')
    ax2.axis('off')

    ax3 = plt.subplot(2, 3, 3)
    if latent_morse_graph_restricted is not None:
        plot_morse_graph(latent_morse_graph_restricted, ax=ax3)
    ax3.set_title('Morse Graph: Latent Dynamics restricted to E(X)')
    ax3.axis('off')

    # Bottom row: Morse sets
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    if lorenz_morse_graph is not None and lorenz_grid is not None:
        plot_morse_sets_3d(lorenz_grid, lorenz_morse_graph, ax=ax4,
                          box_map=lorenz_box_map, alpha=0.5)
    ax4.set_title('Morse Sets: Leslie Map (3D)')
    ax4.view_init(elev=20, azim=45)

    ax5 = plt.subplot(2, 3, 5)
    if latent_morse_graph_full is not None:
        plot_morse_sets(latent_grid_full, latent_morse_graph_full,
                        box_map=latent_box_map_full, ax=ax5)
    if encoded_train_data is not None:
        ax5.scatter(encoded_train_data[:, 0], encoded_train_data[:, 1], s=1, c='blue', alpha=0.2, zorder=3, rasterized=True)
    ax5.set_title('Morse Sets: in $R^2$')
    ax5.set_aspect('auto', adjustable='box')


    ax6 = plt.subplot(2, 3, 6)
    # Plot inactive boxes in grey for the restricted plot
    if latent_grid_full and restricted_active_boxes is not None:
        all_box_indices = set(range(len(latent_grid_full.get_boxes())))
        active_box_indices = set(restricted_active_boxes)
        inactive_box_indices = all_box_indices - active_box_indices
        
        all_boxes = latent_grid_full.get_boxes()
        for box_idx in inactive_box_indices:
            box = all_boxes[box_idx]
            rect = Rectangle((box[0, 0], box[0, 1]),
                           box[1, 0] - box[0, 0],
                           box[1, 1] - box[0, 1],
                           facecolor='#e0e0e0',
                           edgecolor='none')
            ax6.add_patch(rect)

    if latent_morse_graph_restricted is not None:
        plot_morse_sets(latent_grid_full, latent_morse_graph_restricted,
                        box_map=latent_box_map_restricted, ax=ax6)
    if encoded_train_data is not None:
        ax6.scatter(encoded_train_data[:, 0], encoded_train_data[:, 1], s=1, c='blue', alpha=0.2, zorder=3, rasterized=True)
    ax6.set_title('Morse Sets: Restricted to E(X)')
    ax6.set_aspect('auto', adjustable='box')

    # Enforce same axes for latent plots
    if latent_grid_full is not None:
        xlims = (latent_grid_full.bounds[0, 0], latent_grid_full.bounds[1, 0])
        ylims = (latent_grid_full.bounds[0, 1], latent_grid_full.bounds[1, 1])
        ax5.set_xlim(xlims)
        ax5.set_ylim(ylims)
        ax6.set_xlim(xlims)
        ax6.set_ylim(ylims)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_single_experiment(run_number, base_output_dir, base_data_dir, config_overrides=None):
    """Run a single experiment with the given run number.

    Args:
        run_number: Run number for this experiment
        base_output_dir: Base directory for outputs
        base_data_dir: Base directory for data
        config_overrides: Optional dict of config parameters to override
    """

    # Create run-specific directories
    run_dir = os.path.join(base_output_dir, f"run_{run_number:03d}")
    run_figures_dir = os.path.join(run_dir, "figures")
    run_data_dir = os.path.join(run_dir, "data")
    os.makedirs(run_figures_dir, exist_ok=True)
    os.makedirs(run_data_dir, exist_ok=True)

    print(f"  Run {run_number}/{config_overrides.get('NUM_RUNS', Config.NUM_RUNS) if config_overrides else Config.NUM_RUNS}...", end='', flush=True)

    # --- Configuration with overrides ---
    config_overrides = config_overrides or {}
    force_retrain = config_overrides.get('FORCE_RETRAIN', Config.FORCE_RETRAIN)
    num_epochs = config_overrides.get('NUM_EPOCHS', Config.NUM_EPOCHS)

    # Define model cache paths (in run-specific directory)
    model_cache_dir = os.path.join(run_data_dir, "learned_models")
    os.makedirs(model_cache_dir, exist_ok=True)
    encoder_path = os.path.join(model_cache_dir, f"6_encoder_epochs_{num_epochs}.pt")
    decoder_path = os.path.join(model_cache_dir, f"6_decoder_epochs_{num_epochs}.pt")
    latent_dynamics_path = os.path.join(model_cache_dir, f"6_latent_dynamics_epochs_{num_epochs}.pt")
    # --- End Configuration ---

    # MorseGraph computation parameters
    LESLIE_GRID_DIVISIONS = Config.LESLIE_GRID_DIVISIONS
    LESLIE_EPSILON_BLOAT = Config.LESLIE_EPSILON_BLOAT

    LATENT_GRID_DIVISIONS = Config.LATENT_GRID_DIVISIONS
    LATENT_BOUNDS_PADDING = Config.LATENT_BOUNDS_PADDING

    loss_plot_path = None

    # 1. Load or Generate Training Data (suppress output for runs > 1)
    if run_number == 1:
        print("\n  [Data] Loading/generating training data...")

    domain_bounds = Config.DOMAIN_BOUNDS
    n_trajectories = Config.N_TRAJECTORIES
    n_points = Config.N_POINTS
    skip_initial = Config.SKIP_INITIAL
    random_seed = Config.RANDOM_SEED

    # Check if cached data exists (shared across runs, in base data directory)
    data_cache_path = os.path.join(base_data_dir, "6_trajectory_training_data.npz")

    if os.path.exists(data_cache_path) and not force_retrain:
        x_t, x_t_plus_1, training_trajectories, loaded_metadata = load_trajectory_data(data_cache_path)

        # Verify metadata matches current settings
        if (loaded_metadata.get('n_trajectories') == n_trajectories and
            loaded_metadata.get('n_points') == n_points and
            loaded_metadata.get('skip_initial') == skip_initial and
            loaded_metadata.get('random_seed') == random_seed):
            if run_number == 1:
                print(f"    ✓ Using cached data ({len(x_t)} samples)")
        else:
            if run_number == 1:
                print("    ⚠ Regenerating data (settings changed)...")
            force_retrain = True

    if not os.path.exists(data_cache_path) or force_retrain:
        if run_number == 1:
            print(f"    Generating {n_trajectories} trajectories...")
        x_t, x_t_plus_1, training_trajectories = generate_leslie_trajectory_data(
            n_trajectories=n_trajectories,
            n_points=n_points,
            sampling_domain=np.array(domain_bounds),
            random_seed=random_seed,
            skip_initial=skip_initial
        )
        # Save for future runs
        metadata = {
            'n_trajectories': n_trajectories,
            'n_points': n_points,
            'skip_initial': skip_initial,
            'random_seed': random_seed
        }
        save_trajectory_data(data_cache_path, x_t, x_t_plus_1, training_trajectories, metadata)
        if run_number == 1:
            print(f"    ✓ Generated {len(x_t)} training samples")

    # Visualize training data in 3D (only for first run)
    if run_number == 1:
        fig = plt.figure(figsize=(15, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(x_t[:, 0], x_t[:, 1], x_t[:, 2], s=1, alpha=0.2, c='blue', label='Training points (tail)')
        ax1.set_title(f"Training Data: {n_trajectories} trajectories (skip first {skip_initial})")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.legend()

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(x_t[:, 0], x_t[:, 1], x_t[:, 2], s=1, alpha=0.3, c='blue', label='x(t)')
        ax2.scatter(x_t_plus_1[:, 0], x_t_plus_1[:, 1], x_t_plus_1[:, 2], s=1, alpha=0.3, c='red', label='f(x(t))')
        ax2.set_title(f"Training Pairs: {len(x_t)} samples (trajectory tails)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.legend()

        plt.tight_layout()

        data_plot_path = os.path.join(run_figures_dir, "6_learned_dynamics_data.png")
        plt.savefig(data_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        data_plot_path = None
    
    # 2. Prepare Data for Training
    split_idx = int(0.8 * len(x_t))
    x_train, x_val = x_t[:split_idx], x_t[split_idx:]
    y_train, y_val = x_t_plus_1[:split_idx], x_t_plus_1[split_idx:]

    # Convert to PyTorch tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)
    x_val_tensor = torch.FloatTensor(x_val)
    y_val_tensor = torch.FloatTensor(y_val)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 3. Define Neural Network Models

    # Model architecture parameters (with overrides)
    input_dim = config_overrides.get('INPUT_DIM', Config.INPUT_DIM)
    latent_dim = config_overrides.get('LATENT_DIM', Config.LATENT_DIM)
    hidden_dim = config_overrides.get('HIDDEN_DIM', Config.HIDDEN_DIM)
    num_layers = config_overrides.get('NUM_LAYERS', 3)  # Default to 3
    output_activation = config_overrides.get('OUTPUT_ACTIVATION', None)

    # Create models with architecture parameters
    encoder = Encoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_activation=output_activation
    )
    decoder = Decoder(
        latent_dim=latent_dim,
        output_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_activation=output_activation
    )
    latent_dynamics = LatentDynamics(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_activation=output_activation
    )

    # Move to device (prefer MPS on macOS, then CUDA, then CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    latent_dynamics = latent_dynamics.to(device)

    # Only show model architecture for first run
    if run_number == 1:
        print(f"  [Device] {device}")
        total_params = (sum(p.numel() for p in encoder.parameters()) +
                       sum(p.numel() for p in decoder.parameters()) +
                       sum(p.numel() for p in latent_dynamics.parameters()))
        print(f"  [Models] {num_layers} layers, {hidden_dim} hidden, {total_params:,} params, output_act={output_activation}")

    # 4. Load or Train Models

    models_exist = (os.path.exists(encoder_path) and
                    os.path.exists(decoder_path) and
                    os.path.exists(latent_dynamics_path))

    should_train = not models_exist or force_retrain

    if not should_train:
        print(f"  Found cached models for {num_epochs} epochs, loading from disk...")
        try:
            encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            latent_dynamics.load_state_dict(torch.load(latent_dynamics_path, map_location=device))
            print("  ✓ Models loaded successfully.")
        except Exception as e:
            print(f"  ✗ Error loading models: {e}. Proceeding to retrain...")
            should_train = True

    if should_train:
        if not models_exist:
            print(f"  No cached models found for {num_epochs} epochs, training new models...")
        elif force_retrain:
            print("  `force_retrain` is True, training models...")

        # Optimizer
        parameters = list(encoder.parameters()) + list(decoder.parameters()) + list(latent_dynamics.parameters())
        optimizer = torch.optim.Adam(parameters, lr=Config.LEARNING_RATE)

        # Loss function
        mse_loss = nn.MSELoss()

        # Training loop
        train_losses = {'reconstruction': [], 'dynamics_recon': [], 'dynamics_consistency': [], 'total': []}
        val_losses = {'reconstruction': [], 'dynamics_recon': [], 'dynamics_consistency': [], 'total': []}

        # Loss weights (with overrides)
        w_recon = config_overrides.get('W_RECON', Config.W_RECON)
        w_dyn_recon = config_overrides.get('W_DYN_RECON', Config.W_DYN_RECON)
        w_dyn_cons = config_overrides.get('W_DYN_CONS', Config.W_DYN_CONS)

        print(f"  Training for {num_epochs} epochs...")
        print(f"  Loss weights: reconstruction={w_recon}, dynamics_recon={w_dyn_recon}, dynamics_consistency={w_dyn_cons}")

        for epoch in range(num_epochs):
            # Training phase
            encoder.train()
            decoder.train()
            latent_dynamics.train()

            epoch_loss_recon = 0.0
            epoch_loss_dyn_recon = 0.0
            epoch_loss_dyn_cons = 0.0
            epoch_loss_total = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                # Forward pass
                z_t = encoder(x_batch)                    # E(x_t)
                x_t_recon = decoder(z_t)                  # D(E(x_t))

                z_t_next_pred = latent_dynamics(z_t)      # G(E(x_t))
                x_t_next_pred = decoder(z_t_next_pred)    # D(G(E(x_t)))

                z_t_next_true = encoder(y_batch)          # E(x_{t+1})

                # Compute losses
                loss_recon = mse_loss(x_t_recon, x_batch)                      # ||D(E(x_t)) - x_t||²
                loss_dyn_recon = mse_loss(x_t_next_pred, y_batch)              # ||D(G(E(x_t))) - x_{t+1}||²
                loss_dyn_cons = mse_loss(z_t_next_pred, z_t_next_true)         # ||G(E(x_t)) - E(x_{t+1})||²

                loss_total = w_recon * loss_recon + w_dyn_recon * loss_dyn_recon + w_dyn_cons * loss_dyn_cons

                # Backward pass
                loss_total.backward()
                optimizer.step()

                epoch_loss_recon += loss_recon.item()
                epoch_loss_dyn_recon += loss_dyn_recon.item()
                epoch_loss_dyn_cons += loss_dyn_cons.item()
                epoch_loss_total += loss_total.item()

            # Average training losses
            num_batches = len(train_loader)
            train_losses['reconstruction'].append(epoch_loss_recon / num_batches)
            train_losses['dynamics_recon'].append(epoch_loss_dyn_recon / num_batches)
            train_losses['dynamics_consistency'].append(epoch_loss_dyn_cons / num_batches)
            train_losses['total'].append(epoch_loss_total / num_batches)

            # Validation phase
            encoder.eval()
            decoder.eval()
            latent_dynamics.eval()

            val_loss_recon = 0.0
            val_loss_dyn_recon = 0.0
            val_loss_dyn_cons = 0.0
            val_loss_total = 0.0

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    # Forward pass
                    z_t = encoder(x_batch)
                    x_t_recon = decoder(z_t)

                    z_t_next_pred = latent_dynamics(z_t)
                    x_t_next_pred = decoder(z_t_next_pred)

                    z_t_next_true = encoder(y_batch)

                    # Compute losses
                    loss_recon = mse_loss(x_t_recon, x_batch)
                    loss_dyn_recon = mse_loss(x_t_next_pred, y_batch)
                    loss_dyn_cons = mse_loss(z_t_next_pred, z_t_next_true)

                    loss_total = w_recon * loss_recon + w_dyn_recon * loss_dyn_recon + w_dyn_cons * loss_dyn_cons

                    val_loss_recon += loss_recon.item()
                    val_loss_dyn_recon += loss_dyn_recon.item()
                    val_loss_dyn_cons += loss_dyn_cons.item()
                    val_loss_total += loss_total.item()

            # Average validation losses
            num_batches = len(val_loader)
            val_losses['reconstruction'].append(val_loss_recon / num_batches)
            val_losses['dynamics_recon'].append(val_loss_dyn_recon / num_batches)
            val_losses['dynamics_consistency'].append(val_loss_dyn_cons / num_batches)
            val_losses['total'].append(val_loss_total / num_batches)

            # Print progress only for first run and only every 50 epochs
            if run_number == 1 and (epoch + 1) % 50 == 0:
                print(f"    Epoch [{epoch+1}/{num_epochs}] - Train: {train_losses['total'][-1]:.4f}, Val: {val_losses['total'][-1]:.4f}")

        if run_number == 1:
            print("    ✓ Training completed")

        # Save models
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)
        torch.save(latent_dynamics.state_dict(), latent_dynamics_path)
        if run_number == 1:
            print(f"\n  [Models] Saved to {os.path.basename(model_cache_dir)}/")


        # 5. Plot Training Loss Curves
        if run_number == 1:
            print("\n  [Plotting] Loss curves...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Total loss
        axes[0, 0].plot(train_losses['total'], label='Train', linewidth=2)
        axes[0, 0].plot(val_losses['total'], label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(bottom=0)

        # Reconstruction loss
        axes[0, 1].plot(train_losses['reconstruction'], label='Train', linewidth=2)
        axes[0, 1].plot(val_losses['reconstruction'], label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Reconstruction Loss: ||D(E(x)) - x||²')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(bottom=0)

        # Dynamics reconstruction loss
        axes[1, 0].plot(train_losses['dynamics_recon'], label='Train', linewidth=2)
        axes[1, 0].plot(val_losses['dynamics_recon'], label='Validation', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Dynamics Reconstruction: ||D(G(E(x))) - f(x)||²')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(bottom=0)

        # Dynamics consistency loss
        axes[1, 1].plot(train_losses['dynamics_consistency'], label='Train', linewidth=2)
        axes[1, 1].plot(val_losses['dynamics_consistency'], label='Validation', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Dynamics Consistency: ||G(E(x)) - E(f(x))||²')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(bottom=0)

        plt.tight_layout()

        loss_plot_path = os.path.join(run_figures_dir, "6_learned_dynamics_losses.png")
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    # 6. Model Evaluation
    if run_number == 1:
        print("\n  [Evaluation] Computing errors...")

    encoder.eval()
    decoder.eval()
    latent_dynamics.eval()

    with torch.no_grad():
        # Use validation set for evaluation
        x_val_device = x_val_tensor.to(device)
        y_val_device = y_val_tensor.to(device)

        # Reconstruction error
        z_val = encoder(x_val_device)
        x_val_recon = decoder(z_val)
        recon_error = torch.mean((x_val_device - x_val_recon) ** 2).item()

        # Dynamics prediction error
        z_val_next_pred = latent_dynamics(z_val)
        y_val_pred = decoder(z_val_next_pred)
        dyn_error = torch.mean((y_val_device - y_val_pred) ** 2).item()

        # Dynamics consistency error
        z_val_next_true = encoder(y_val_device)
        cons_error = torch.mean((z_val_next_pred - z_val_next_true) ** 2).item()

        if run_number == 1:
            print(f"    Recon: {recon_error:.6f}, Dyn: {dyn_error:.6f}, Cons: {cons_error:.6f}")

    # 7. Visualize 3D Dynamics and 2D Latent Space Together
    if run_number == 1:
        print("\n  [Plotting] Trajectories...")

    # Generate 3 test orbits with different initial conditions
    test_ics = [
        training_trajectories[0][0],  # First training trajectory IC
        np.array([10.0, 8.0, 6.0]),
        np.array([20.0, 15.0, 10.0])
    ]

    orbit_n_points = 11  # 11 sample points = 10 transitions (x_0 -> x_1 -> ... -> x_10)

    # Create combined figure: 2 rows x 3 columns
    # Top row: 3D discrete sample points
    # Bottom row: 2D latent space showing G(E(x_i)) vs E(f(x_i)) comparison
    fig = plt.figure(figsize=(18, 10))

    # Define color scheme - use colormap for progression
    import matplotlib.cm as cm
    n_transitions = 10  # 10 transitions from x_0 to x_9
    colors = cm.viridis(np.linspace(0, 1, n_transitions))  # Color gradient

    # ========== TOP ROW: 3D Discrete Sample Points ==========
    for orbit_idx, ic in enumerate(test_ics):
        # Generate discrete orbit using Leslie map
        orbit_discrete = [ic.copy()]
        x_current = ic.copy()
        for _ in range(orbit_n_points - 1):
            x_next = leslie_map(x_current)
            orbit_discrete.append(x_next.copy())
            x_current = x_next
        orbit_discrete = np.array(orbit_discrete)

        # Generate more points for smooth visualization (100 points)
        orbit_continuous = [ic.copy()]
        x_current = ic.copy()
        for _ in range(99):
            x_next = leslie_map(x_current)
            orbit_continuous.append(x_next.copy())
            x_current = x_next
        orbit_continuous = np.array(orbit_continuous)

        # Create subplot for this orbit (top row)
        ax = fig.add_subplot(2, 3, orbit_idx + 1, projection='3d')

        # Plot true orbit (blue solid line connecting discrete points)
        ax.plot(orbit_continuous[:, 0], orbit_continuous[:, 1], orbit_continuous[:, 2],
                linewidth=2.0, alpha=0.6, color='C0', label='True orbit', zorder=2)

        # Encode and decode continuous orbit to get D(E(orbit)) reconstruction
        with torch.no_grad():
            orbit_continuous_tensor = torch.FloatTensor(orbit_continuous).to(device)
            orbit_encoded = encoder(orbit_continuous_tensor)
            orbit_reconstructed = decoder(orbit_encoded).cpu().numpy()

        # Plot D(E(orbit)) reconstruction (orange dashed)
        ax.plot(orbit_reconstructed[:, 0], orbit_reconstructed[:, 1], orbit_reconstructed[:, 2],
                linewidth=2.0, alpha=0.7, color='C1', linestyle='--', label='D(E(orbit))', zorder=2)

        # Plot each x_i
        for i in range(n_transitions):
            ax.scatter(orbit_discrete[i, 0], orbit_discrete[i, 1], orbit_discrete[i, 2],
                       s=50, c=[colors[i]], marker='o', edgecolors='none',
                       zorder=5)

        # Add legend entries (using first pair as example)
        ax.scatter([], [], s=50, c=[colors[0]], marker='o', label='x_i')

        ax.set_title(f"3D: Orbit {orbit_idx + 1} - Sample Points", fontsize=11, fontweight='bold')
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("y", fontsize=9)
        ax.set_zlabel("z", fontsize=9)

        # Set consistent axis limits from domain_bounds
        ax.set_xlim(domain_bounds[0][0], domain_bounds[1][0])
        ax.set_ylim(domain_bounds[0][1], domain_bounds[1][1])
        ax.set_zlim(domain_bounds[0][2], domain_bounds[1][2])

        if orbit_idx == 0:
            ax.legend(fontsize=7, loc='upper left', framealpha=0.9)

    # ========== BOTTOM ROW: 2D Latent Space - Emphasize G(E(x_i)) vs E(f(x_i)) ==========

    # Encode the training data to latent space for background
    with torch.no_grad():
        x_train_device = x_train_tensor.to(device)
        z_train = encoder(x_train_device).cpu().numpy()

    # Define color scheme for latent space
    color_background = "#7A7A7A"      # Light gray - training data

    # For each test orbit
    for orbit_idx, ic in enumerate(test_ics):
        # Create subplot for this orbit (bottom row)
        ax = fig.add_subplot(2, 3, orbit_idx + 4)

        # Plot encoded training data as background
        ax.scatter(z_train[:, 0], z_train[:, 1], s=1, alpha=0.15, c=color_background,
                   zorder=0, rasterized=True)

        # Generate the same orbit as before using Leslie map
        orbit_discrete = [ic.copy()]
        x_current = ic.copy()
        for _ in range(orbit_n_points - 1):
            x_next = leslie_map(x_current)
            orbit_discrete.append(x_next.copy())
            x_current = x_next
        orbit_discrete = np.array(orbit_discrete)

        # Generate more points for smooth visualization
        orbit_continuous = [ic.copy()]
        x_current = ic.copy()
        for _ in range(99):
            x_next = leslie_map(x_current)
            orbit_continuous.append(x_next.copy())
            x_current = x_next
        orbit_continuous = np.array(orbit_continuous)

        # Encode continuous orbit to get E(orbit)
        with torch.no_grad():
            orbit_continuous_tensor = torch.FloatTensor(orbit_continuous).to(device)
            orbit_continuous_encoded = encoder(orbit_continuous_tensor).cpu().numpy()

        # Plot continuous E(orbit) (blue solid line)
        ax.plot(orbit_continuous_encoded[:, 0], orbit_continuous_encoded[:, 1],
                linewidth=2.0, alpha=0.6, color='C0', label='E(orbit)', zorder=2)

        # Encode discrete points: E(x_i) and E(f(x_i))
        with torch.no_grad():
            orbit_discrete_tensor = torch.FloatTensor(orbit_discrete).to(device)
            orbit_discrete_encoded = encoder(orbit_discrete_tensor).cpu().numpy()

            # Apply learned dynamics to get predictions: G(E(x_i))
            orbit_discrete_encoded_tensor = torch.FloatTensor(orbit_discrete_encoded).to(device)
            orbit_discrete_dynamics_latent = latent_dynamics(orbit_discrete_encoded_tensor).cpu().numpy()

        # For the 10 transitions (i = 0 to 9):
        # E(f(x_i)) = E(x_{i+1}) are indices 1 to 10
        # G(E(x_i)) are indices 0 to 9 of predictions
        E_fxi = orbit_discrete_encoded[1:]           # True next: E(x_1) to E(x_10)
        G_E_xi = orbit_discrete_dynamics_latent[:-1] # Predictions: G(E(x_0)) to G(E(x_9))

        # Plot each transition with matching colors
        for i in range(n_transitions):
            # Draw arrow from E(x_i) to E(f(x_i)) (ground truth) - colored
            ax.annotate('', xy=(E_fxi[i, 0], E_fxi[i, 1]), xytext=(G_E_xi[i, 0], G_E_xi[i, 1]),
                       arrowprops=dict(arrowstyle='-', color=colors[i], alpha=0.6, lw=1.5),
                       zorder=2)

            # E(f(x_i)) - square with same color
            ax.scatter(E_fxi[i, 0], E_fxi[i, 1],
                       s=50, c=[colors[i]], marker='o', edgecolors='none',
                       zorder=5)

            # G(E(x_i)) - X marker with same color
            ax.scatter(G_E_xi[i, 0], G_E_xi[i, 1],
                       s=60, c=[colors[i]], marker='X', edgecolors='none',
                       zorder=6)

        # Add legend entries (using first pair as example)
        ax.scatter([], [], s=50, c=[colors[0]], marker='o', edgecolors='none',
                   label='E(f(x_i))')
        ax.scatter([], [], s=60, c=[colors[0]], marker='X', edgecolors='none',
                   label='G(E(x_i))')

        ax.set_xlabel('Latent Dimension 1', fontsize=9)
        ax.set_ylabel('Latent Dimension 2', fontsize=9)
        ax.set_title(f'Latent: Orbit {orbit_idx + 1} - Dynamics Consistency', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        if orbit_idx == 0:
            ax.legend(fontsize=7, loc='best', framealpha=0.95)

    # Save combined figure
    plt.tight_layout()

    combined_plot_path = os.path.join(run_figures_dir, "6_learned_dynamics_trajectories.png")
    plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 8. Morse Graph Analysis
    if run_number == 1:
        print("\n  [Morse Graph] Computing...")

    # Step 1: Compute Leslie Map MorseGraph (3D)
    if run_number == 1:
        print("    Leslie map 3D...")
    try:
        leslie_grid = UniformGrid(bounds=np.array(domain_bounds), divisions=LESLIE_GRID_DIVISIONS)
        leslie_dynamics = BoxMapFunction(
            leslie_map, epsilon=LESLIE_EPSILON_BLOAT
        )
        leslie_model = Model(leslie_grid, leslie_dynamics)
        leslie_box_map = leslie_model.compute_box_map()
        leslie_morse_graph = compute_morse_graph(leslie_box_map)
        if run_number == 1:
            print(f"      {len(leslie_morse_graph.nodes())} Morse sets")
    except Exception as e:
        if run_number == 1:
            print(f"      Error: {e}")
        leslie_morse_graph = None
        leslie_box_map = None
        leslie_grid = None

    # Step 2: Compute Latent MorseGraph - Full Grid (2D)
    if run_number == 1:
        print("    Latent full grid 2D...")
    try:
        # Encode training data to find latent bounds
        with torch.no_grad():
            x_train_tensor_device = x_train_tensor.to(device)
            z_train = encoder(x_train_tensor_device).cpu().numpy()

        latent_bounds = compute_latent_bounds(z_train, padding_factor=LATENT_BOUNDS_PADDING)

        latent_grid_full = UniformGrid(bounds=latent_bounds, divisions=LATENT_GRID_DIVISIONS)
        
        latent_dynamics_full = BoxMapLearnedLatent(
            latent_dynamics_model=latent_dynamics,
            decoder_model=decoder,
            encoder_model=encoder,
            device=device,
            epsilon_bloat=latent_grid_full.box_size # Use a bloating factor equal to the cell size
        )

        latent_model_full = Model(latent_grid_full, latent_dynamics_full)
        latent_box_map_full = latent_model_full.compute_box_map()
        latent_morse_graph_full = compute_morse_graph(latent_box_map_full)
        if run_number == 1:
            print(f"      {len(latent_morse_graph_full.nodes())} Morse sets")
    except Exception as e:
        if run_number == 1:
            print(f"      Error: {e}")
        latent_morse_graph_full = None
        latent_box_map_full = None
        latent_grid_full = None

    # Step 3: Compute Latent MorseGraph - Data-Restricted (2D)
    if run_number == 1:
        print("    Latent restricted 2D...")
    active_boxes = None  # Initialize
    try:
        if latent_grid_full is not None:
            # Use Config factor for radius of data restriction
            data_restriction_radius = Config.DATA_RESTRICTION_EPSILON_FACTOR * np.linalg.norm(latent_grid_full.box_size)

            active_boxes = filter_boxes_near_data(
                latent_grid_full,
                z_train,
                data_restriction_radius
            )
            if run_number == 1:
                print(f"      {len(active_boxes)} active boxes")

            latent_box_map_restricted = nx.DiGraph()
            latent_box_map_restricted.add_nodes_from(active_boxes)

            for box_idx in active_boxes:
                box = latent_grid_full.get_boxes()[box_idx]
                image_box = latent_dynamics_full(box)
                target_indices = latent_grid_full.box_to_indices(image_box)
                for target_idx in target_indices:
                    if target_idx in active_boxes:
                        latent_box_map_restricted.add_edge(box_idx, target_idx)

            latent_morse_graph_restricted = compute_morse_graph(latent_box_map_restricted)
            if run_number == 1:
                print(f"      {len(latent_morse_graph_restricted.nodes())} Morse sets")
        else:
            raise RuntimeError("Skipping due to failure in full grid computation.")
    except Exception as e:
        if run_number == 1:
            print(f"      Error: {e}")
        latent_morse_graph_restricted = None
        latent_box_map_restricted = None

    # Step 4: Create 6-panel comparison figure
    if run_number == 1:
        print("\n  [Plotting] Morse graph comparison...")
    comparison_plot_path = os.path.join(run_figures_dir, "6_morsegraph_comparison.png")
    try:
        create_morsegraph_comparison_figure(
            lorenz_morse_graph=leslie_morse_graph,
            lorenz_grid=leslie_grid,
            lorenz_box_map=leslie_box_map,
            latent_morse_graph_full=latent_morse_graph_full,
            latent_box_map_full=latent_box_map_full,
            latent_grid_full=latent_grid_full,
            latent_morse_graph_restricted=latent_morse_graph_restricted,
            latent_box_map_restricted=latent_box_map_restricted,
            restricted_active_boxes=active_boxes,
            encoded_train_data=z_train,
            output_path=comparison_plot_path
        )
    except Exception as e:
        if run_number == 1:
            print(f"      Error: {e}")
        comparison_plot_path = None

    # Print completion for this run
    print(" ✓")

    return {
        'run_number': run_number,
        'encoder_path': encoder_path,
        'decoder_path': decoder_path,
        'latent_dynamics_path': latent_dynamics_path,
        'recon_error': recon_error,
        'dyn_error': dyn_error,
        'cons_error': cons_error,
        'num_leslie_morse_sets': len(leslie_morse_graph.nodes()) if leslie_morse_graph else 0,
        'num_latent_full_morse_sets': len(latent_morse_graph_full.nodes()) if latent_morse_graph_full else 0,
        'num_latent_restricted_morse_sets': len(latent_morse_graph_restricted.nodes()) if latent_morse_graph_restricted else 0,
    }

def main(config_overrides=None, custom_output_dir=None, custom_data_dir=None):
    """Main function to run the learned dynamics experiment.

    Args:
        config_overrides: Optional dict of config parameters to override
        custom_output_dir: Optional custom output directory (defaults to 'leslie_model')
        custom_data_dir: Optional custom data directory (defaults to 'leslie_model/data')
    """
    if not ML_AVAILABLE:
        print("Machine Learning dependencies not available. Please install PyTorch to run this example.")
        return

    # Use custom directories if provided
    exp_output_dir = custom_output_dir or output_dir
    exp_data_dir = custom_data_dir or data_dir

    config_overrides = config_overrides or {}
    num_runs = config_overrides.get('NUM_RUNS', Config.NUM_RUNS)

    print("MorseGraph Example 6: Learned Dynamics - Leslie Map")
    print("===================================================")
    print(f"Running {num_runs} independent experiments")
    if config_overrides:
        print(f"Config overrides: {config_overrides}")

    # Run multiple experiments
    results = []
    for run_num in range(1, num_runs + 1):
        result = run_single_experiment(run_num, exp_output_dir, exp_data_dir, config_overrides)
        results.append(result)

    # Print summary of all runs
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL RUNS")
    print(f"{'='*70}")
    print(f"\n{'Run':<5} {'Recon':<10} {'Dyn':<10} {'Cons':<10} {'Leslie':<8} {'Latent(F)':<10} {'Latent(R)':<10}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['run_number']:<5} {r['recon_error']:<10.6f} {r['dyn_error']:<10.6f} {r['cons_error']:<10.6f} "
              f"{r['num_leslie_morse_sets']:<8} {r['num_latent_full_morse_sets']:<10} {r['num_latent_restricted_morse_sets']:<10}")

    # Compute statistics
    recon_errors = [r['recon_error'] for r in results]
    dyn_errors = [r['dyn_error'] for r in results]
    cons_errors = [r['cons_error'] for r in results]

    print(f"\n{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"{'-'*70}")
    print(f"{'Reconstruction':<20} {np.mean(recon_errors):<12.6f} {np.std(recon_errors):<12.6f} {np.min(recon_errors):<12.6f} {np.max(recon_errors):<12.6f}")
    print(f"{'Dynamics Pred':<20} {np.mean(dyn_errors):<12.6f} {np.std(dyn_errors):<12.6f} {np.min(dyn_errors):<12.6f} {np.max(dyn_errors):<12.6f}")
    print(f"{'Dynamics Cons':<20} {np.mean(cons_errors):<12.6f} {np.std(cons_errors):<12.6f} {np.min(cons_errors):<12.6f} {np.max(cons_errors):<12.6f}")

    print("\nAll runs completed!")
    return results

if __name__ == "__main__":
    main()