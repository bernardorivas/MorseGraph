#!/usr/bin/env python3
"""
Single-run of Learned Dynamics with cmgdb

It can be used standalone or imported by the multiple runs wrapper.
"""

import os
import json
import numpy as np
from datetime import datetime
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import product
import io
import time
import CMGDB

# Import ML models and utilities
# MorseGraph is a lightweight toy-version of CMGDB.
# BoxMap uses CMGDB, while all else might go through MorseGraph
from MorseGraph.models import Encoder, Decoder, LatentDynamics
from MorseGraph.utils import (
    compute_latent_bounds,
    save_trajectory_data,
    generate_map_trajectory_data,
    count_parameters,
    format_time,
    get_next_run_number,
    count_attractors
)
from MorseGraph.systems import leslie_map_3d

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # --- Run Control ---
    NUM_RUNS = 1

    # --- Data Generation (Leslie Map) ---
    N_TRAJECTORIES = 1000
    N_POINTS = 20
    SKIP_INITIAL = 0
    RANDOM_SEED = None
    DOMAIN_BOUNDS = [[-0.1, -0.1, -0.1], [90.0, 70.0, 70.0]]

    # --- Model Architecture ---
    INPUT_DIM = 3
    LATENT_DIM = 2
    HIDDEN_DIM = 32
    NUM_LAYERS = 3
    OUTPUT_ACTIVATION = None # "tanh" # "sigmoid"

    # --- Training Parameters ---
    NUM_EPOCHS = 1500
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 50
    MIN_DELTA = 1e-5
    W_RECON = 1.0       # DE - I
    W_DYN_RECON = 1.0   # DGE - f
    W_DYN_CONS = 1.0    # GE - fE

    # --- MorseGraph Computation ---
    # For Leslie map (3D)
    LESLIE_SUBDIV_MIN = 36
    LESLIE_SUBDIV_MAX = 42
    # For learned latent system (2D)
    LATENT_SUBDIV_MIN = 20
    LATENT_SUBDIV_MAX = 30
    # Common CMGDB parameters
    SUBDIV_INIT = 0
    SUBDIV_LIMIT = 10000
    PADDING = True
    # Latent space computation
    LATENT_BOUNDS_PADDING = 1.5

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_morse_graph_to_ax(morse_graph, ax, title='', cmap=plt.cm.cool):
    if morse_graph is None:
        ax.text(0.5, 0.5, 'Not computed', ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
        return
    gv_source = CMGDB.PlotMorseGraph(morse_graph, cmap=cmap)
    img_data = gv_source.pipe(format='png')
    img = plt.imread(io.BytesIO(img_data))
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

def plot_morse_sets_3d_cmgdb(morse_graph, ax, lower_bounds, upper_bounds, data=None):
    colors = plt.cm.cool(np.linspace(0, 1, morse_graph.num_vertices()))

    # Calculate appropriate marker size based on box sizes
    all_boxes = []
    for morse_idx in range(morse_graph.num_vertices()):
        boxes = morse_graph.morse_set_boxes(morse_idx)
        if boxes:
            all_boxes.extend(boxes)

    if all_boxes:
        # Compute average box size across all dimensions
        box_sizes = []
        for b in all_boxes:
            dim = len(b) // 2
            size = np.mean([b[d + dim] - b[d] for d in range(dim)])
            box_sizes.append(size)
        avg_box_size = np.mean(box_sizes)

        # Calculate marker size: scale by domain size, with min/max bounds
        domain_size = np.mean([upper_bounds[d] - lower_bounds[d] for d in range(len(lower_bounds))])
        # Marker size in points^2, scaled by box size relative to domain
        marker_size = max(10, min(200, (avg_box_size / domain_size) * 5000))
    else:
        marker_size = 20

    for morse_idx in range(morse_graph.num_vertices()):
        boxes = morse_graph.morse_set_boxes(morse_idx)
        if boxes:
            centers = np.array([[(b[d] + b[d+len(b)//2]) / 2.0 for d in range(len(b)//2)] for b in boxes])
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                      c=[colors[morse_idx]], s=marker_size, alpha=0.4, marker='s')

    # Plot data points if provided
    if data is not None and len(data) > 0:
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='black', s=1, alpha=0.3, label='Data')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])
    ax.set_zlim(lower_bounds[2], upper_bounds[2])
    ax.view_init(elev=30, azim=45)

def plot_data_boxes(ax, data, latent_bounds, subdivision_depth=10):
    """Plot boxes that contain data points as white squares."""
    if data is None or len(data) == 0:
        return

    # Compute box size based on subdivision depth
    lower = np.array(latent_bounds[0])
    upper = np.array(latent_bounds[1])
    n_boxes = 2 ** subdivision_depth
    box_width = (upper - lower) / n_boxes

    # Find unique boxes containing data
    data_boxes = set()
    for point in data:
        # Find which box this point belongs to
        box_idx = np.floor((point - lower) / box_width).astype(int)
        # Clamp to valid range
        box_idx = np.clip(box_idx, 0, n_boxes - 1)
        data_boxes.add(tuple(box_idx))

    if not data_boxes:
        return

    # Convert to centers for scatter plot
    centers = []
    for box_idx in data_boxes:
        box_center = lower + (np.array(box_idx) + 0.5) * box_width
        centers.append(box_center)
    centers = np.array(centers)

    # Calculate marker size: scale by box size relative to domain
    domain_size = np.mean(upper - lower)
    avg_box_size = np.mean(box_width)
    # Marker size in points^2, scaled to roughly fill the box
    marker_size = max(5, min(500, (avg_box_size / domain_size) * 10000))

    # Plot as scatter with square markers
    ax.scatter(centers[:, 0], centers[:, 1],
              s=marker_size, marker='s',
              facecolor='white', edgecolor='none',
              linewidth=0.5, alpha=0.6, zorder=1)

def plot_morse_sets_2d_cmgdb(morse_graph, ax, cmap=plt.cm.cool, data=None):
    if morse_graph is None:
        return
    num_morse_sets = morse_graph.num_vertices()
    if num_morse_sets == 0:
        return
    colors = cmap(np.linspace(0, 1, num_morse_sets))

    for morse_idx in range(num_morse_sets):
        boxes = morse_graph.morse_set_boxes(morse_idx)
        if boxes:
            for b in boxes:
                rect = Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                               facecolor=colors[morse_idx],
                               edgecolor='none')
                ax.add_patch(rect)

    # Plot data points if provided
    if data is not None and len(data) > 0:
        ax.scatter(data[:, 0], data[:, 1], c='black', s=1, alpha=0.3, label='Data')

def plot_latent_morse_sets_restricted_cmgdb(morse_graph, ax, latent_bounds, data=None, subdivision_depth=10, cmap=plt.cm.cool):
    # Plot grey background
    grey_rect = Rectangle((latent_bounds[0][0], latent_bounds[0][1]),
                           latent_bounds[1][0] - latent_bounds[0][0],
                           latent_bounds[1][1] - latent_bounds[0][1],
                           facecolor='#e0e0e0', edgecolor='none')
    ax.add_patch(grey_rect)

    # Plot white boxes containing data
    plot_data_boxes(ax, data, latent_bounds, subdivision_depth=subdivision_depth)

    # Plot morse sets on top
    plot_morse_sets_2d_cmgdb(morse_graph, ax, data=data, cmap=cmap)

    ax.set_xlim(latent_bounds[0][0], latent_bounds[1][0])
    ax.set_ylim(latent_bounds[0][1], latent_bounds[1][1])

def create_comparison_figure(output_path,
                             latent_morse_graph_full,
                             latent_morse_graph_restricted,
                             latent_bounds,
                             latent_data,
                             encoder,
                             device,
                             leslie_barycenters_data,
                             leslie_morse_graph_img_path,
                             domain_bounds,
                             leslie_data):
    fig = plt.figure(figsize=(18, 12))
    latent_cmap = plt.cm.viridis  # New colormap for latent plots

    # --- Top Row ---

    # Top-left: Pre-computed Leslie Morse Graph
    ax1 = fig.add_subplot(2, 3, 1)
    try:
        img = plt.imread(leslie_morse_graph_img_path)
        ax1.imshow(img)
        ax1.set_title('Morse Graph (Pre-computed)')
    except FileNotFoundError:
        ax1.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        ax1.set_title('Morse Graph (Pre-computed)')
    ax1.axis('off')

    # Top-middle: Latent Full Morse Graph
    ax2 = fig.add_subplot(2, 3, 2)
    plot_morse_graph_to_ax(latent_morse_graph_full, ax2, 'Latent dynamics - MG', cmap=latent_cmap)

    # Top-right: Latent Restricted Morse Graph
    ax3 = fig.add_subplot(2, 3, 3)
    plot_morse_graph_to_ax(latent_morse_graph_restricted, ax3, 'BoxMapData on E(X) - MG', cmap=latent_cmap)

    # --- Bottom Row ---

    # Bottom-left: Re-computed Barycenters plot + training data
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.set_title('Scatterplot of barycenters')

    if leslie_barycenters_data is not None:
        num_leslie_morse_sets = len(leslie_barycenters_data.files)
        if num_leslie_morse_sets > 0:
            leslie_colors = plt.cm.cool(np.linspace(0, 1, num_leslie_morse_sets))
            sorted_keys = sorted(leslie_barycenters_data.files, key=lambda k: int(k.split('_')[-1]))

            for key in sorted_keys:
                morse_set_index = int(key.split('_')[-1])
                points_3d = leslie_barycenters_data[key]
                if points_3d.size > 0:
                    color = leslie_colors[morse_set_index]
                    ax4.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=[color], marker='s', s=10)

    if leslie_data is not None and len(leslie_data) > 0:
        ax4.scatter(leslie_data[:, 0], leslie_data[:, 1], leslie_data[:, 2], c='black', s=1, alpha=0.05)

    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    if domain_bounds is not None:
        ax4.set_xlim(domain_bounds[0][0], domain_bounds[1][0])
        ax4.set_ylim(domain_bounds[0][1], domain_bounds[1][1])
        ax4.set_zlim(domain_bounds[0][2], domain_bounds[1][2])
    ax4.view_init(elev=30, azim=45)

    # Function to plot projected barycenters
    def plot_projected_barycenters(ax, leslie_barycenters_data, encoder, device):
        if leslie_barycenters_data is None:
            return

        # Determine number of morse sets to create colormap
        num_leslie_morse_sets = len(leslie_barycenters_data.files)
        if num_leslie_morse_sets == 0:
            return
        leslie_colors = plt.cm.cool(np.linspace(0, 1, num_leslie_morse_sets))

        with torch.no_grad():
            sorted_keys = sorted(leslie_barycenters_data.files, key=lambda k: int(k.split('_')[-1]))
            for key in sorted_keys:
                morse_set_index = int(key.split('_')[-1])
                points_3d = leslie_barycenters_data[key]
                if points_3d.size == 0:
                    continue

                points_3d_tensor = torch.FloatTensor(points_3d).to(device)
                points_2d = encoder(points_3d_tensor).cpu().numpy()

                color = leslie_colors[morse_set_index]
                ax.scatter(points_2d[:, 0], points_2d[:, 1], c=[color], marker='x', s=12, zorder=3, alpha=0.8)

    # Bottom-middle: Latent Full Morse Sets + Projected Barycenters
    ax5 = fig.add_subplot(2, 3, 5)
    plot_morse_sets_2d_cmgdb(latent_morse_graph_full, ax5, data=latent_data, cmap=latent_cmap)
    plot_projected_barycenters(ax5, leslie_barycenters_data, encoder, device)
    ax5.set_title('MorseSets+E(barycenter)+E(data)')
    if latent_bounds is not None:
        ax5.set_xlim(latent_bounds.tolist()[0][0], latent_bounds.tolist()[1][0])
        ax5.set_ylim(latent_bounds.tolist()[0][1], latent_bounds.tolist()[1][1])

    # Bottom-right: Latent Restricted Morse Sets + Projected Barycenters
    ax6 = fig.add_subplot(2, 3, 6)
    plot_latent_morse_sets_restricted_cmgdb(latent_morse_graph_restricted, ax6, latent_bounds.tolist(), data=latent_data, cmap=latent_cmap)
    plot_projected_barycenters(ax6, leslie_barycenters_data, encoder, device)
    ax6.set_title('MorseSets+E(barycenter)+E(data)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# ============================================================================
# cmgdb-specific Morse Graph Computation
# ============================================================================

def compute_leslie_morse_graph_cmgdb(config):
    # Batched version: evaluate all corner points at once
    def leslie_map_batched(points):
        """Batched evaluation of Leslie map on multiple points."""
        return np.array([leslie_map_3d(p) for p in points])

    def F(rect):
        # Get corner points for this rectangle
        dim = len(config['DOMAIN_BOUNDS'][0])
        corners = list(product(*[(rect[d], rect[d+dim]) for d in range(dim)]))
        corners = [list(c) for c in corners]

        # Batch evaluate all corners at once
        corners_next = leslie_map_batched(corners)

        # Compute bounds of image
        padding_size = [(rect[d + dim] - rect[d]) for d in range(dim)] if config.get('PADDING', True) else [0] * dim
        Y_l_bounds = [corners_next[:, d].min() - padding_size[d] for d in range(dim)]
        Y_u_bounds = [corners_next[:, d].max() + padding_size[d] for d in range(dim)]

        return Y_l_bounds + Y_u_bounds

    model = CMGDB.Model(config['LESLIE_SUBDIV_MIN'], config['LESLIE_SUBDIV_MAX'], config['SUBDIV_INIT'], config['SUBDIV_LIMIT'], config['DOMAIN_BOUNDS'][0], config['DOMAIN_BOUNDS'][1], F)
    return CMGDB.ComputeMorseGraph(model)

def compute_latent_morse_graph_cmgdb(latent_dynamics, device, latent_bounds, config):
    # Batched version: evaluate all corner points at once
    def latent_map_func_batched(points):
        """Batched evaluation of latent dynamics on multiple points."""
        with torch.no_grad():
            points_tensor = torch.FloatTensor(points).to(device)
            points_next = latent_dynamics(points_tensor).cpu().numpy()
        return points_next

    def F(rect):
        # Get corner points for this rectangle
        dim = len(latent_bounds[0])
        if config.get('PADDING', True):
            # Use corners mode (default)
            corners = list(product(*[(rect[d], rect[d+dim]) for d in range(dim)]))
            corners = [list(c) for c in corners]
        else:
            # Just use center point if no padding
            corners = [[(rect[d] + rect[d+dim])/2 for d in range(dim)]]

        # Batch evaluate all corners at once
        corners_next = latent_map_func_batched(corners)

        # Compute bounds of image
        padding_size = [(rect[d + dim] - rect[d]) for d in range(dim)] if config.get('PADDING', True) else [0] * dim
        Y_l_bounds = [corners_next[:, d].min() - padding_size[d] for d in range(dim)]
        Y_u_bounds = [corners_next[:, d].max() + padding_size[d] for d in range(dim)]

        return Y_l_bounds + Y_u_bounds

    model = CMGDB.Model(config['LATENT_SUBDIV_MIN'], config['LATENT_SUBDIV_MAX'], config['SUBDIV_INIT'], config['SUBDIV_LIMIT'], latent_bounds[0], latent_bounds[1], F)
    return CMGDB.ComputeMorseGraph(model)

def compute_latent_morse_graph_data_restricted_cmgdb(latent_dynamics, device, z_train, latent_bounds, config):
    with torch.no_grad():
        G_z_train = latent_dynamics(torch.FloatTensor(z_train).to(device)).cpu().numpy()

    box_map_data = CMGDB.BoxMapData(X=z_train, Y=G_z_train, map_empty='outside', lower_bounds=latent_bounds[0], upper_bounds=latent_bounds[1])

    def F_data(rect):
        return box_map_data.compute(rect)

    model = CMGDB.Model(config['LATENT_SUBDIV_MIN'], config['LATENT_SUBDIV_MAX'], config['SUBDIV_INIT'], config['SUBDIV_LIMIT'], latent_bounds[0], latent_bounds[1], F_data)
    return CMGDB.ComputeMorseGraph(model)

# ============================================================================
# Single Experiment Runner
# ============================================================================

def run_single_experiment_cmgdb(run_name, output_dir, config_overrides=None, verbose=True, progress_interval=50):
    """
    Run a single cmgdb experiment with Leslie map learned dynamics.

    Args:
        run_name: Name for this run (can be None for auto-numbering)
        output_dir: Directory to save outputs
        config_overrides: Dictionary of config parameters to override
        verbose: Whether to print detailed progress
        progress_interval: Print progress every N epochs (when verbose=True)

    Returns:
        Dictionary of results including errors and Morse set counts
    """
    # Auto-generate run name if not provided
    if run_name is None:
        run_number = get_next_run_number(output_dir)
        run_name = f"{run_number:03d}"

    run_dir = os.path.join(output_dir, f"run_{run_name}")

    # Create organized directory structure
    run_training_data_dir = os.path.join(run_dir, "training_data")
    run_models_dir = os.path.join(run_dir, "models")
    run_results_dir = os.path.join(run_dir, "results")

    os.makedirs(run_training_data_dir, exist_ok=True)
    os.makedirs(run_models_dir, exist_ok=True)
    os.makedirs(run_results_dir, exist_ok=True)

    config = {k: v for k, v in vars(Config).items() if not k.startswith('__')}
    config.update(config_overrides or {})

    if verbose:
        print(f"\n{'='*80}")
        print(f"Run: {run_name}")
        print(f"Output: {run_dir}")
        print(f"{'='*80}")

    # Data generation
    data_cache_path = os.path.join(run_training_data_dir, "trajectory_data.npz")
    x_t, x_t_plus_1, _ = generate_map_trajectory_data(
        map_func=leslie_map_3d,
        n_trajectories=config['N_TRAJECTORIES'],
        n_points=config['N_POINTS'],
        sampling_domain=np.array(config['DOMAIN_BOUNDS']),
        random_seed=config['RANDOM_SEED'],
        skip_initial=config['SKIP_INITIAL']
    )
    save_trajectory_data(data_cache_path, x_t, x_t_plus_1, [], {})

    # Train/validation split
    split_idx = int(0.8 * len(x_t))
    x_train, x_val = x_t[:split_idx], x_t[split_idx:]
    y_train, y_val = x_t_plus_1[:split_idx], x_t_plus_1[split_idx:]

    # Convert to PyTorch tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)
    x_val_tensor = torch.FloatTensor(x_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # Device setup (prefer MPS on macOS, then CUDA, then CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Model setup with config parameters
    encoder = Encoder(
        config['INPUT_DIM'],
        config['LATENT_DIM'],
        config['HIDDEN_DIM'],
        config.get('NUM_LAYERS', 3),
        output_activation=config.get('OUTPUT_ACTIVATION', None)
    ).to(device)
    decoder = Decoder(
        config['LATENT_DIM'],
        config['INPUT_DIM'],
        config['HIDDEN_DIM'],
        config.get('NUM_LAYERS', 3),
        output_activation=config.get('OUTPUT_ACTIVATION', None)
    ).to(device)
    latent_dynamics = LatentDynamics(
        config['LATENT_DIM'],
        config['HIDDEN_DIM'],
        config.get('NUM_LAYERS', 3),
        output_activation=config.get('OUTPUT_ACTIVATION', None)
    ).to(device)

    # Count parameters
    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)
    dynamics_params = count_parameters(latent_dynamics)
    total_params = encoder_params + decoder_params + dynamics_params

    if verbose:
        print(f"\nModel Architecture:")
        print(f"  Encoder:         {encoder_params:,} parameters")
        print(f"  Decoder:         {decoder_params:,} parameters")
        print(f"  Latent Dynamics: {dynamics_params:,} parameters")
        print(f"  Total:           {total_params:,} parameters")
        print(f"  Training data:   {len(x_train):,} samples")
        print(f"  Device:          {device}")

    # Training setup
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

    parameters = list(encoder.parameters()) + list(decoder.parameters()) + list(latent_dynamics.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config['LEARNING_RATE'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    mse_loss = torch.nn.MSELoss()

    # Loss weights
    w_recon = config.get('W_RECON', 1.0)
    w_dyn_recon = config.get('W_DYN_RECON', 1.0)
    w_dyn_cons = config.get('W_DYN_CONS', 1.0)

    # Loss tracking
    train_losses = {'reconstruction': [], 'dynamics_recon': [], 'dynamics_consistency': [], 'total': []}
    val_losses = {'reconstruction': [], 'dynamics_recon': [], 'dynamics_consistency': [], 'total': []}

    # Training loop
    if verbose:
        print(f"\nTraining {config['NUM_EPOCHS']} epochs (progress shown every {progress_interval} epochs)...")

    training_start_time = time.time()


    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = config.get('EARLY_STOPPING_PATIENCE', 50)
    min_delta = config.get('MIN_DELTA', 1e-5)

    for epoch in range(config['NUM_EPOCHS']):


        # Training phase
        encoder.train()
        decoder.train()
        latent_dynamics.train()

        epoch_loss_recon = 0.0
        epoch_loss_dyn_recon = 0.0
        epoch_loss_dyn_cons = 0.0
        epoch_loss_total = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

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
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

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

        # Calculate time estimate after 10th epoch
        if epoch == 9:
            time_for_10_epochs = time.time() - training_start_time
            avg_epoch_time = time_for_10_epochs / 10
            estimated_total_time = avg_epoch_time * config['NUM_EPOCHS']
            if verbose:
                print(f"  Avg time over first 10 epochs: {format_time(avg_epoch_time)}/epoch | Estimated total: {format_time(estimated_total_time)}")

        # Print progress in verbose mode
        if verbose and (epoch + 1) % progress_interval == 0:
            print(f"  Epoch {epoch + 1}/{config['NUM_EPOCHS']} | "
                  f"DE-I: {train_losses['reconstruction'][-1]:.6f} | "
                  f"DGE-f: {train_losses['dynamics_recon'][-1]:.6f} | "
                  f"GE-fE: {train_losses['dynamics_consistency'][-1]:.6f} | "
                  f"Total: {train_losses['total'][-1]:.6f} | "
                  f"Val Total: {val_losses['total'][-1]:.6f}")

        # Early stopping check
        current_val_loss = val_losses['total'][-1]
        if current_val_loss < best_val_loss - min_delta:
            best_val_loss = current_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Step the scheduler
        scheduler.step(current_val_loss)

        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"  Early stopping triggered at epoch {epoch + 1} due to no improvement in validation loss.")
            break

    # Print final training time
    total_training_time = time.time() - training_start_time
    if verbose:
        print(f"  Training completed in {format_time(total_training_time)}")

    # Save models
    torch.save(encoder.state_dict(), os.path.join(run_models_dir, "encoder.pt"))
    torch.save(decoder.state_dict(), os.path.join(run_models_dir, "decoder.pt"))
    torch.save(latent_dynamics.state_dict(), os.path.join(run_models_dir, "latent_dynamics.pt"))

    # Save training history (losses per epoch)
    final_epoch = epoch + 1
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_epochs': final_epoch,
        'training_time_seconds': total_training_time
    }
    with open(os.path.join(run_results_dir, "training_history.json"), 'w') as f:
        json.dump(training_history, f, indent=2)

    # Plot loss curves
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    def set_dynamic_ylim(ax, train_loss_history, val_loss_history):
        num_epochs = len(train_loss_history)
        start_epoch = int(num_epochs * 0.2) if num_epochs > 10 else 0
        final_losses = train_loss_history[start_epoch:] + val_loss_history[start_epoch:]
        if final_losses:
            upper_limit = max(final_losses) * 1.2
            ax.set_ylim(bottom=0, top=upper_limit + 1e-6)
        else:
            ax.set_ylim(bottom=0)

    # Total loss
    axes[0, 0].plot(train_losses['total'], label='Train', linewidth=2)
    axes[0, 0].plot(val_losses['total'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    set_dynamic_ylim(axes[0, 0], train_losses['total'], val_losses['total'])

    # Reconstruction loss
    axes[0, 1].plot(train_losses['reconstruction'], label='Train', linewidth=2)
    axes[0, 1].plot(val_losses['reconstruction'], label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Reconstruction: ||D(E(x)) - x||²')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    set_dynamic_ylim(axes[0, 1], train_losses['reconstruction'], val_losses['reconstruction'])

    # Dynamics reconstruction loss
    axes[1, 0].plot(train_losses['dynamics_recon'], label='Train', linewidth=2)
    axes[1, 0].plot(val_losses['dynamics_recon'], label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Dynamics Recon: ||D(G(E(x))) - f(x)||²')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    set_dynamic_ylim(axes[1, 0], train_losses['dynamics_recon'], val_losses['dynamics_recon'])

    # Dynamics consistency loss
    axes[1, 1].plot(train_losses['dynamics_consistency'], label='Train', linewidth=2)
    axes[1, 1].plot(val_losses['dynamics_consistency'], label='Validation', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Dynamics Cons: ||G(E(x)) - E(f(x))||²')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    set_dynamic_ylim(axes[1, 1], train_losses['dynamics_consistency'], val_losses['dynamics_consistency'])

    plt.tight_layout()
    loss_plot_path = os.path.join(run_results_dir, "training_losses.png")
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Compute final validation errors
    encoder.eval()
    decoder.eval()
    latent_dynamics.eval()
    with torch.no_grad():
        x_val_device = x_val_tensor.to(device)
        y_val_device = y_val_tensor.to(device)
        z_val = encoder(x_val_device)
        x_val_recon = decoder(z_val)
        recon_error = torch.mean((x_val_device - x_val_recon) ** 2).item()
        z_val_next_pred = latent_dynamics(z_val)
        y_val_pred = decoder(z_val_next_pred)
        dyn_error = torch.mean((y_val_device - y_val_pred) ** 2).item()
        z_val_next_true = encoder(y_val_device)
        cons_error = torch.mean((z_val_next_pred - z_val_next_true) ** 2).item()

    # Morse graph computation
    # We've already computed the 3D Leslie Morse Graph in another file
    compute_leslie = False

    x_full_tensor = torch.FloatTensor(x_t)

    # Leslie 3D computation (optional)
    if compute_leslie:
        if verbose:
            print("  Computing Leslie Morse graph...")
        leslie_morse_graph, _ = compute_leslie_morse_graph_cmgdb(config)
    else:
        leslie_morse_graph = None

    if verbose:
        print("  Computing latent Morse graphs...")
    with torch.no_grad():
        z_train = encoder(x_full_tensor.to(device)).cpu().numpy()
    latent_bounds = compute_latent_bounds(z_train, padding_factor=config['LATENT_BOUNDS_PADDING'])
    latent_morse_graph_full, _ = compute_latent_morse_graph_cmgdb(latent_dynamics, device, latent_bounds.tolist(), config)
    latent_morse_graph_restricted, _ = compute_latent_morse_graph_data_restricted_cmgdb(latent_dynamics, device, z_train, latent_bounds.tolist(), config)

    # Count attractors
    latent_full_attractors = count_attractors(latent_morse_graph_full)
    latent_restricted_attractors = count_attractors(latent_morse_graph_restricted)

    # Plotting Morse graph comparison
    comparison_figure_path = os.path.join(run_results_dir, "morse_graph_comparison.png")

    # Paths to pre-computed Leslie results
    # Assuming the script is run from the project root
    leslie_results_dir = os.path.join('examples', 'leslie_map_3d', 'leslie_map_3d_results')
    leslie_morse_graph_img_path = os.path.join(leslie_results_dir, 'morse_graph.png')
    leslie_barycenters_npz_path = os.path.join(leslie_results_dir, 'barycenters.npz')

    # Load barycenters
    try:
        leslie_barycenters_data = np.load(leslie_barycenters_npz_path, allow_pickle=True)
    except FileNotFoundError:
        if verbose:
            print(f"Warning: Barycenter data not found at {leslie_barycenters_npz_path}")
        leslie_barycenters_data = None

    create_comparison_figure(
        output_path=comparison_figure_path,
        latent_morse_graph_full=latent_morse_graph_full,
        latent_morse_graph_restricted=latent_morse_graph_restricted,
        latent_bounds=latent_bounds,
        latent_data=z_train,
        encoder=encoder,
        device=device,
        leslie_barycenters_data=leslie_barycenters_data,
        leslie_morse_graph_img_path=leslie_morse_graph_img_path,
        domain_bounds=config['DOMAIN_BOUNDS'],
        leslie_data=x_t
    )

    # Save latent space data
    latent_data_path = os.path.join(run_training_data_dir, "latent_data.npz")
    np.savez(latent_data_path, z_train=z_train, latent_bounds=latent_bounds)

    # Compile results
    results = {
        'run_name': run_name,
        'recon_error': recon_error,
        'dyn_error': dyn_error,
        'cons_error': cons_error,
        'num_leslie_morse_sets': leslie_morse_graph.num_vertices() if leslie_morse_graph else 0,
        'num_latent_full_morse_sets': latent_morse_graph_full.num_vertices() if latent_morse_graph_full else 0,
        'num_latent_restricted_morse_sets': latent_morse_graph_restricted.num_vertices() if latent_morse_graph_restricted else 0,
        'latent_full_attractors': latent_full_attractors,
        'latent_restricted_attractors': latent_restricted_attractors,
        'final_train_loss': train_losses['total'][-1],
        'final_val_loss': val_losses['total'][-1],
        'total_training_time': total_training_time,
        'total_parameters': total_params
    }

    # Save comprehensive settings for reproducibility
    settings = {
        'run_info': {
            'run_name': run_name,
            'run_dir': run_dir,
            'timestamp': datetime.now().isoformat()
        },
        'configuration': config,
        'model_architecture': {
            'encoder_params': encoder_params,
            'decoder_params': decoder_params,
            'latent_dynamics_params': dynamics_params,
            'total_params': total_params,
            'input_dim': config['INPUT_DIM'],
            'latent_dim': config['LATENT_DIM'],
            'hidden_dim': config['HIDDEN_DIM'],
            'num_layers': config.get('NUM_LAYERS', 3),
            'output_activation': config.get('OUTPUT_ACTIVATION', None)
        },
        'training_info': {
            'num_epochs': config['NUM_EPOCHS'],
            'batch_size': config['BATCH_SIZE'],
            'learning_rate': config['LEARNING_RATE'],
            'training_time_seconds': total_training_time,
            'device': str(device),
            'loss_weights': {
                'w_recon': w_recon,
                'w_dyn_recon': w_dyn_recon,
                'w_dyn_cons': w_dyn_cons
            }
        },
        'data_info': {
            'n_trajectories': config['N_TRAJECTORIES'],
            'n_points': config['N_POINTS'],
            'skip_initial': config['SKIP_INITIAL'],
            'random_seed': config['RANDOM_SEED'],
            'domain_bounds': config['DOMAIN_BOUNDS'],
            'train_samples': len(x_train),
            'val_samples': len(x_val)
        },
        'results': results
    }

    with open(os.path.join(run_dir, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=2)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Results Summary:")
        print(f"  Reconstruction Error:      {recon_error:.6f}")
        print(f"  Dynamics Error:            {dyn_error:.6f}")
        print(f"  Consistency Error:         {cons_error:.6f}")
        print(f"  Leslie Morse Sets:         {leslie_morse_graph.num_vertices() if leslie_morse_graph else 'N/A'}")
        print(f"  Latent Full Morse Sets:    {latent_morse_graph_full.num_vertices() if latent_morse_graph_full else 0}")
        print(f"  Latent Restricted Morse Sets: {latent_morse_graph_restricted.num_vertices() if latent_morse_graph_restricted else 0}")
        print(f"\nSaved to: {run_dir}")
        print(f"  - Configuration: settings.json")
        print(f"  - Models: models/")
        print(f"  - Training data: training_data/")
        print(f"  - Results: results/")
        print(f"{'='*80}\n")

    return results

# ============================================================================
# Command-Line Interface
# ============================================================================

def parse_args():
    """Parse command-line arguments for single run mode."""
    parser = argparse.ArgumentParser(description='cmgdb Single Run Test')
    parser.add_argument('--name', type=str, default=None,
                        help='Name for this run (default: timestamp)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: examples/learning)')
    parser.add_argument('--progress-interval', type=int, default=50,
                        help='Print progress every N epochs (default: 50)')
    parser.add_argument('--hidden-dim', type=int, default=None,
                        help='Hidden dimension (default: from Config)')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Number of layers (default: from Config)')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='Number of training epochs (default: from Config)')
    return parser.parse_args()

def main():
    """Main entry point for standalone single run."""
    args = parse_args()

    # Set output directory (default to single_runs)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'single_runs')
    os.makedirs(output_dir, exist_ok=True)

    # Set run name (None for auto-numbering, or use provided name)
    run_name = args.name

    # Build config overrides
    config_overrides = {}
    if args.hidden_dim is not None:
        config_overrides['HIDDEN_DIM'] = args.hidden_dim
    if args.num_layers is not None:
        config_overrides['NUM_LAYERS'] = args.num_layers
    if args.num_epochs is not None:
        config_overrides['NUM_EPOCHS'] = args.num_epochs

    if run_name:
        print(f"cmgdb Single Run - {run_name}")
    else:
        next_num = get_next_run_number(output_dir)
        print(f"cmgdb Single Run - Auto-numbering: run_{next_num:03d}")
    print(f"Output directory: {output_dir}")

    # Run experiment
    result = run_single_experiment_cmgdb(
        run_name=run_name,
        output_dir=output_dir,
        config_overrides=config_overrides,
        verbose=True,
        progress_interval=args.progress_interval
    )

    print(f"\nRun completed successfully!")
    run_dir = os.path.join(output_dir, f"run_{result['run_name']}")
    print(f"Results saved to: {run_dir}")

if __name__ == "__main__":
    main()
