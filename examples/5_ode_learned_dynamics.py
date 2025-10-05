#!/usr/bin/env python3
"""
MorseGraph Example 5: Learned Dynamics - Lorenz System

This example demonstrates learning a low-dimensional representation of a
high-dimensional dynamical system using an autoencoder framework.

We learn:
- E: X → Y (encoder: 3D → 2D)
- D: Y → X (decoder: 2D → 3D)
- G: Y → Y (latent dynamics: 2D → 2D)

Where f(x) is the time-one map of the Lorenz system, and we minimize:
  ||D(E(x)) - x||² (reconstruction)
  + ||D(G(E(x))) - f(x)||² (dynamics reconstruction)
  + ||G(E(x)) - E(f(x))||² (dynamics consistency)

This allows us to study the complex 3D Lorenz dynamics in a simpler 2D latent space.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

# MorseGraph imports
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapODE, BoxMapLearnedLatent
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph
from MorseGraph.plot import plot_morse_graph, plot_morse_sets, plot_morse_sets_3d
from MorseGraph.systems import lorenz_ode
from MorseGraph.utils import (
    generate_trajectory_data,
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
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(output_dir, 'figures')
data_dir = os.path.join(output_dir, "data")
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)


# ============================================================================
# Configuration
# ============================================================================
class Config:
    '''Configuration parameters for the learned dynamics example.'''
    # --- Training Control ---
    FORCE_RETRAIN = False
    NUM_EPOCHS = 200
    RANDOM_SEED = 42

    # --- Data Generation (Lorenz System) ---
    N_TRAJECTORIES = 1000
    TIME_SKIP = 6.0  # Time to integrate before sampling starts
    SAMPLING_TIME = 6.0  # Duration of the sampled trajectory
    N_POINTS = 10  # Number of points per trajectory
    DOMAIN_BOUNDS = [[-20, -30, 5], [20, 30, 50]]

    # --- Model Architecture ---
    INPUT_DIM = 3
    LATENT_DIM = 2
    HIDDEN_DIM = 128

    # --- Training Parameters ---
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    # Loss weights
    W_RECON = 1.0
    W_DYN_RECON = 10.0
    W_DYN_CONS = 1.0

    # --- MorseGraph Computation ---
    # For Lorenz system
    LORENZ_GRID_DIVISIONS = [16, 16, 16]
    LORENZ_EPSILON_BLOAT = 0.1
    # For learned latent system
    LATENT_GRID_DIVISIONS = [256, 256]
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
    fig.suptitle("Morse Graph Analysis: Lorenz vs. Learned Latent Dynamics", fontsize=16)

    # Top row: Morse graphs
    ax1 = plt.subplot(2, 3, 1)
    if lorenz_morse_graph is not None:
        plot_morse_graph(lorenz_morse_graph, ax=ax1)
    ax1.set_title('Morse Graph: Lorenz (3D)')
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
    ax4.set_title('Morse Sets: Lorenz (3D)')
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
                           facecolor='#e0e0e0',  # light grey
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

def main():
    if not ML_AVAILABLE:
        print("Machine Learning dependencies not available. Please install PyTorch to run this example.")
        return

    print("MorseGraph Example 5: Learned Dynamics - Lorenz System")
    print("=====================================================")

    # --- Configuration ---
    # Set to True to force retraining even if cached models exist.
    # The number of epochs is included in the filename, so changing it will
    # also trigger retraining.
    force_retrain = Config.FORCE_RETRAIN
    num_epochs = Config.NUM_EPOCHS

    # Define model cache paths
    model_cache_dir = os.path.join(data_dir, "learned_models")
    os.makedirs(model_cache_dir, exist_ok=True)
    encoder_path = os.path.join(model_cache_dir, f"5_encoder_epochs_{num_epochs}.pt")
    decoder_path = os.path.join(model_cache_dir, f"5_decoder_epochs_{num_epochs}.pt")
    latent_dynamics_path = os.path.join(model_cache_dir, f"5_latent_dynamics_epochs_{num_epochs}.pt")
    # --- End Configuration ---

    # MorseGraph computation parameters
    LORENZ_GRID_DIVISIONS = Config.LORENZ_GRID_DIVISIONS
    LORENZ_EPSILON_BLOAT = Config.LORENZ_EPSILON_BLOAT

    LATENT_GRID_DIVISIONS = Config.LATENT_GRID_DIVISIONS
    LATENT_BOUNDS_PADDING = Config.LATENT_BOUNDS_PADDING

    loss_plot_path = None

    # 1. Load or Generate Training Data
    print("\n1. Loading/generating training data from Lorenz trajectories...")

    domain_bounds = Config.DOMAIN_BOUNDS
    n_trajectories = Config.N_TRAJECTORIES
    timeskip = Config.TIME_SKIP
    sampling_time = Config.SAMPLING_TIME
    n_points = Config.N_POINTS
    random_seed = Config.RANDOM_SEED

    # Check if cached data exists
    data_cache_path = os.path.join(data_dir, "5_trajectory_training_data.npz")

    if os.path.exists(data_cache_path) and not force_retrain:
        print(f"  Found cached data at: {data_cache_path}")
        x_t, x_t_plus_1, training_trajectories, loaded_metadata = load_trajectory_data(data_cache_path)

        # Verify metadata matches current settings
        if (loaded_metadata.get('n_trajectories') == n_trajectories and
            loaded_metadata.get('sampling_time') == sampling_time and
            loaded_metadata.get('timeskip') == timeskip and
            loaded_metadata.get('n_points') == n_points and
            loaded_metadata.get('random_seed') == random_seed):
            print("  ✓ Cached data matches current settings, using cached data")
        else:
            print("  ⚠ Cached data has different settings, regenerating...")
            force_retrain = True # Force regeneration

    if not os.path.exists(data_cache_path) or force_retrain:
        if not os.path.exists(data_cache_path):
            print(f"  No cached data found, generating new trajectories...")
        x_t, x_t_plus_1, training_trajectories = generate_trajectory_data(
            ode_func=lorenz_ode,
            ode_params={},
            n_samples=n_trajectories,
            total_time=sampling_time,
            n_points=n_points,
            sampling_domain=np.array(domain_bounds),
            random_seed=random_seed,
            timeskip=timeskip
        )
        # Save for future runs
        metadata = {
            'n_trajectories': n_trajectories,
            'sampling_time': sampling_time,
            'timeskip': timeskip,
            'n_points': n_points,
            'random_seed': random_seed
        }
        save_trajectory_data(data_cache_path, x_t, x_t_plus_1, training_trajectories, metadata)

    print(f"  Dataset ready: {n_trajectories} trajectories")
    print(f"  Points per trajectory: {n_points}")
    print(f"  Total training pairs: {len(x_t)}")
    print(f"  Data shapes: x_t {x_t.shape}, x_t+1 {x_t_plus_1.shape}")

    # Visualize training data in 3D - show trajectory structure
    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    # Plot all training points
    ax1.scatter(x_t[:, 0], x_t[:, 1], x_t[:, 2], s=1, alpha=0.2, c='blue', label='Training Points')
    ax1.set_title(f"Training Data: Points from {n_trajectories} trajectories")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    # Show the pairs structure
    ax2.scatter(x_t[:, 0], x_t[:, 1], x_t[:, 2], s=2, alpha=0.3, c='blue', label='States x(t)')
    ax2.scatter(x_t_plus_1[:, 0], x_t_plus_1[:, 1], x_t_plus_1[:, 2], s=2, alpha=0.3, c='red', label='States x(t+Δt)')
    ax2.set_title(f"Training Pairs: {len(x_t)} samples")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.legend()

    plt.tight_layout()

    # Save training data plot
    data_plot_path = os.path.join(figures_dir, "5_learned_dynamics_data.png")
    plt.savefig(data_plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved training data plot to: {data_plot_path}")
    
    # 2. Prepare Data for Training
    print("\n2. Preparing data for neural network training...")

    # Split into train/validation
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

    print(f"Training set: {len(x_train)} samples")
    print(f"Validation set: {len(x_val)} samples")

    # 3. Define Neural Network Models
    print("\n3. Defining neural network models...")

    # Model architecture parameters
    input_dim = Config.INPUT_DIM
    latent_dim = Config.LATENT_DIM
    hidden_dim = Config.HIDDEN_DIM

    # Create models
    encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    decoder = Decoder(latent_dim=latent_dim, output_dim=input_dim, hidden_dim=hidden_dim)
    latent_dynamics = LatentDynamics(latent_dim=latent_dim, hidden_dim=hidden_dim)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    latent_dynamics = latent_dynamics.to(device)

    # Display model architectures dynamically by introspecting the nn.Modules
    def summarize_model(model, name, indent=2):
        """Print a compact summary of a PyTorch model.

        Shows top-level children, linear layer shapes when available, and
        parameter counts. Keeps the output concise and adapts automatically
        when the architecture changes.
        """
        pad = ' ' * indent
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{pad}{name}:")
        print(f"{pad}  Total params: {total_params:,} (trainable: {trainable_params:,})")

        # Iterate over immediate children to show structure
        for child_name, child in model.named_children():
            child_type = child.__class__.__name__
            # Special-case common layer types to show shapes/args
            if isinstance(child, nn.Linear):
                in_f = getattr(child, 'in_features', '?')
                out_f = getattr(child, 'out_features', '?')
                bias = child.bias is not None
                print(f"{pad}  {child_name}: {child_type}({in_f} → {out_f}, bias={bias})")
            elif isinstance(child, nn.Sequential):
                print(f"{pad}  {child_name}: {child_type} [")
                for i, sub in enumerate(child):
                    sub_type = sub.__class__.__name__
                    if isinstance(sub, nn.Linear):
                        in_f = getattr(sub, 'in_features', '?')
                        out_f = getattr(sub, 'out_features', '?')
                        print(f"{pad}    [{i}] {sub_type}({in_f} → {out_f})")
                    else:
                        # Print simple name for activations and other modules
                        print(f"{pad}    [{i}] {sub_type}")
                print(f"{pad}  ]")
            else:
                # Generic fallback for other module types
                print(f"{pad}  {child_name}: {child_type}")

    print("\n  Model Architectures:")
    print("  " + "="*50)
    summarize_model(encoder, f"Encoder ({input_dim}D → {latent_dim}D)")
    print()
    summarize_model(decoder, f"Decoder ({latent_dim}D → {input_dim}D)")
    print()
    summarize_model(latent_dynamics, f"Latent Dynamics ({latent_dim}D → {latent_dim}D)")
    print()
    total_params = (sum(p.numel() for p in encoder.parameters()) +
                   sum(p.numel() for p in decoder.parameters()) +
                   sum(p.numel() for p in latent_dynamics.parameters()))
    print(f"  Total trainable parameters: {total_params:,}")
    print("  " + "="*50)

    # 4. Load or Train Models
    print("\n4. Loading or training neural network models...")

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

        # Loss weights
        w_recon = Config.W_RECON
        w_dyn_recon = Config.W_DYN_RECON
        w_dyn_cons = Config.W_DYN_CONS

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

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  Train Loss: {train_losses['total'][-1]:.6f} "
                      f"(recon: {train_losses['reconstruction'][-1]:.6f}, "
                      f"dyn_recon: {train_losses['dynamics_recon'][-1]:.6f}, "
                      f"dyn_cons: {train_losses['dynamics_consistency'][-1]:.6f})")
                print(f"  Val Loss:   {val_losses['total'][-1]:.6f} "
                      f"(recon: {val_losses['reconstruction'][-1]:.6f}, "
                      f"dyn_recon: {val_losses['dynamics_recon'][-1]:.6f}, "
                      f"dyn_cons: {val_losses['dynamics_consistency'][-1]:.6f})")

        print("\n  Training completed!")

        # Save models
        print("\n  Saving trained models to disk...")
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)
        torch.save(latent_dynamics.state_dict(), latent_dynamics_path)
        print(f"  - Encoder saved to: {encoder_path}")
        print(f"  - Decoder saved to: {decoder_path}")
        print(f"  - Latent Dynamics saved to: {latent_dynamics_path}")


        # 5. Plot Training Loss Curves
        print("\n5. Plotting training loss curves...")

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

        loss_plot_path = os.path.join(figures_dir, "5_learned_dynamics_losses.png")
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        print(f"  Saved loss curves to: {loss_plot_path}")

    # 6. Model Evaluation
    print("\n6. Evaluating learned model...")

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

        print(f"  Final Validation Errors:")
        print(f"    Reconstruction (MSE):       {recon_error:.6f}")
        print(f"    Dynamics Prediction (MSE):  {dyn_error:.6f}")
        print(f"    Dynamics Consistency (MSE): {cons_error:.6f}")

    # 7. Visualize 3D Dynamics and 2D Latent Space Together
    print("\n7. Visualizing dynamics in 3D and 2D latent space...")

    # Generate 3 test orbits with different initial conditions
    # Orbit 1: Use first training trajectory's initial condition
    # Orbit 2-3: Manually chosen for nice visualization
    test_ics = [
        training_trajectories[0][0],  # First training orbit IC
        np.array([-5.0, -8.0, 25.0]),
        np.array([8.0, 12.0, 30.0])
    ]

    orbit_total_time = 10.0
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
        # Integrate continuous orbit with high accuracy
        sol = solve_ivp(
            lambda t, y: lorenz_ode(t, y),
            [0, orbit_total_time],
            ic,
            dense_output=True,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14
        )

        # Get discrete sample points (11 points as in training)
        t_discrete = np.linspace(0, orbit_total_time, orbit_n_points)
        orbit_discrete = sol.sol(t_discrete).T

        # Get continuous orbit for smooth visualization (1000 points)
        t_continuous = np.linspace(0, orbit_total_time, 1000)
        orbit_continuous = sol.sol(t_continuous).T

        # Create subplot for this orbit (top row)
        ax = fig.add_subplot(2, 3, orbit_idx + 1, projection='3d')

        # Plot continuous true orbit (blue solid)
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

        # Integrate the same orbit as before
        sol = solve_ivp(
            lambda t, y: lorenz_ode(t, y),
            [0, orbit_total_time],
            ic,
            dense_output=True,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14
        )

        # Get discrete sample points
        t_discrete = np.linspace(0, orbit_total_time, orbit_n_points)
        orbit_discrete = sol.sol(t_discrete).T

        # Get continuous orbit for smooth E(orbit) visualization
        t_continuous = np.linspace(0, orbit_total_time, 1000)
        orbit_continuous = sol.sol(t_continuous).T

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

    combined_plot_path = os.path.join(figures_dir, "5_learned_dynamics_trajectories.png")
    plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved combined 3D/2D visualization to: {combined_plot_path}")

    # 8. Morse Graph Analysis
    print("\n8. Performing Morse Graph analysis...")

    # Step 1: Compute Lorenz MorseGraph (3D)
    print("\n  8.1. Computing Morse Graph for Lorenz system...")
    try:
        lorenz_grid = UniformGrid(bounds=np.array(domain_bounds), divisions=LORENZ_GRID_DIVISIONS)
        lorenz_dynamics = BoxMapODE(
            lorenz_ode, tau=1.0, epsilon=LORENZ_EPSILON_BLOAT
        )
        lorenz_model = Model(lorenz_grid, lorenz_dynamics)
        lorenz_box_map = lorenz_model.compute_box_map()
        lorenz_morse_graph = compute_morse_graph(lorenz_box_map)
        print(f"    {len(lorenz_morse_graph.nodes())} Morse sets found.")
    except Exception as e:
        print(f"    ✗ Error computing Lorenz Morse Graph: {e}")
        lorenz_morse_graph = None
        lorenz_box_map = None
        lorenz_grid = None

    # Step 2: Compute Latent MorseGraph - Full Grid (2D)
    print("\n  8.2. Computing Morse Graph for latent dynamics (full grid)...")
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
        print(f"    {len(latent_morse_graph_full.nodes())} Morse sets found.")
    except Exception as e:
        print(f"    ✗ Error computing Latent Full Morse Graph: {e}")
        latent_morse_graph_full = None
        latent_box_map_full = None
        latent_grid_full = None

    # Step 3: Compute Latent MorseGraph - Data-Restricted (2D)
    print("\n  8.3. Computing Morse Graph for latent dynamics restricted to image of E...")
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
            print(f"    Found {len(active_boxes)} boxes near training data out of {len(latent_grid_full.get_boxes())}.")

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
            print(f"    {len(latent_morse_graph_restricted.nodes())} Morse sets found.")
        else:
            raise RuntimeError("Skipping due to failure in full grid computation.")
    except Exception as e:
        print(f"    ✗ Error computing Latent Restricted Morse Graph: {e}")
        latent_morse_graph_restricted = None
        latent_box_map_restricted = None

    # Step 4: Create 6-panel comparison figure
    print("\n  8.4. Generating comparison figure...")
    comparison_plot_path = os.path.join(figures_dir, "5_morsegraph_comparison.png")
    try:
        create_morsegraph_comparison_figure(
            lorenz_morse_graph=lorenz_morse_graph,
            lorenz_grid=lorenz_grid,
            lorenz_box_map=lorenz_box_map,
            latent_morse_graph_full=latent_morse_graph_full,
            latent_box_map_full=latent_box_map_full,
            latent_grid_full=latent_grid_full,
            latent_morse_graph_restricted=latent_morse_graph_restricted,
            latent_box_map_restricted=latent_box_map_restricted,
            restricted_active_boxes=active_boxes,
            encoded_train_data=z_train,
            output_path=comparison_plot_path
        )
        print(f"    ✓ Comparison figure saved to: {comparison_plot_path}")
    except Exception as e:
        print(f"    ✗ Error creating comparison figure: {e}")
        comparison_plot_path = None


    # 9. Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Successfully learned a 2D latent representation of the")
    print(f"3D Lorenz dynamics and performed Morse graph analysis!")
    print(f"\nTraining Configuration:")
    print(f"  Trajectories:         {n_trajectories}")
    print(f"  Points per trajectory: {n_points}")
    print(f"  Total training pairs:  {len(x_t)}")
    print(f"  Training epochs:       {num_epochs}")
    print(f"\nFinal Validation Errors:")
    print(f"  Reconstruction:       {recon_error:.6f}")
    print(f"  Dynamics Prediction:  {dyn_error:.6f}")
    print(f"  Dynamics Consistency: {cons_error:.6f}")
    print(f"\nGenerated Visualizations:")
    print(f"  - {os.path.basename(data_plot_path)}")
    if loss_plot_path:
        print(f"  - {os.path.basename(loss_plot_path)}")
    print(f"  - {os.path.basename(combined_plot_path)}")
    if comparison_plot_path:
        print(f"  - {os.path.basename(comparison_plot_path)}")

    print("\nExample completed!")

if __name__ == "__main__":
    main()