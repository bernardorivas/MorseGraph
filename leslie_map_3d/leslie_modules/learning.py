"""
Learning Experiments for Leslie Map 3D

This module handles:
- Autoencoder + latent dynamics training
- Morse graph computation for learned systems
- Data normalization
- Model evaluation and comparison
"""

import os
import json
import numpy as np
from datetime import datetime
import time
import torch
from itertools import product
import CMGDB
import matplotlib.pyplot as plt

from MorseGraph.models import Encoder, Decoder, LatentDynamics
from MorseGraph.utils import (
    compute_latent_bounds,
    save_trajectory_data,
    generate_map_trajectory_data,
    count_parameters,
    format_time,
    get_next_run_number,
    count_attractors,
    compute_similarity_vector,
    format_similarity_report,
    compute_train_val_divergence
)
from MorseGraph.systems import leslie_map_3d

from .config import Config
from .plotting import (
    plot_training_losses,
    plot_latent_transformation_analysis,
    plot_morse_graph_comparison,
    plot_preimage_classification
)
from .ground_truth import load_ground_truth_barycenters


# ============================================================================
# Data Normalization Utilities
# ============================================================================

def needs_normalization(config):
    """
    Determine if normalization is needed based on activation functions.

    Normalization is automatically enabled if any network uses tanh or sigmoid.

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if normalization should be applied
    """
    activations_to_check = [
        config.get('ENCODER_ACTIVATION'),
        config.get('DECODER_ACTIVATION'),
        config.get('LATENT_DYNAMICS_ACTIVATION'),
        config.get('OUTPUT_ACTIVATION')
    ]

    bounded_activations = {'tanh', 'sigmoid'}

    for act in activations_to_check:
        if act in bounded_activations:
            return True

    return False


def compute_normalization_params(data, method='minmax'):
    """
    Compute normalization parameters from training data.

    Args:
        data: Training data array of shape (N, D)
        method: 'minmax' for [-1, 1] scaling, 'minmax_01' for [0, 1] scaling, 'standard' for z-score

    Returns:
        dict: Normalization parameters
    """
    if method == 'minmax':
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0

        return {
            'method': 'minmax',
            'min': data_min.tolist(),
            'max': data_max.tolist(),
            'range': data_range.tolist()
        }
    elif method == 'minmax_01':
        # Normalize to [0, 1] range (for sigmoid decoder)
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0

        return {
            'method': 'minmax_01',
            'min': data_min.tolist(),
            'max': data_max.tolist(),
            'range': data_range.tolist()
        }
    elif method == 'standard':
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std[data_std == 0] = 1.0

        return {
            'method': 'standard',
            'mean': data_mean.tolist(),
            'std': data_std.tolist()
        }
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_data(data, params):
    """Normalize data using precomputed parameters."""
    if params['method'] == 'minmax':
        data_min = np.array(params['min'])
        data_range = np.array(params['range'])
        return 2.0 * (data - data_min) / data_range - 1.0
    elif params['method'] == 'minmax_01':
        # Normalize to [0, 1] range
        data_min = np.array(params['min'])
        data_range = np.array(params['range'])
        return (data - data_min) / data_range
    elif params['method'] == 'standard':
        data_mean = np.array(params['mean'])
        data_std = np.array(params['std'])
        return (data - data_mean) / data_std
    else:
        raise ValueError(f"Unknown normalization method: {params['method']}")


def denormalize_data(data, params):
    """Denormalize data back to original scale."""
    if params['method'] == 'minmax':
        data_min = np.array(params['min'])
        data_range = np.array(params['range'])
        return (data + 1.0) * data_range / 2.0 + data_min
    elif params['method'] == 'minmax_01':
        # Denormalize from [0, 1] range
        data_min = np.array(params['min'])
        data_range = np.array(params['range'])
        return data * data_range + data_min
    elif params['method'] == 'standard':
        data_mean = np.array(params['mean'])
        data_std = np.array(params['std'])
        return data * data_std + data_mean
    else:
        raise ValueError(f"Unknown normalization method: {params['method']}")


# ============================================================================
# Morse Graph Computation
# ============================================================================

def compute_leslie_morse_graph_cmgdb(config):
    """Compute Morse graph for the Leslie map using CMGDB."""
    def leslie_map_batched(points):
        return np.array([leslie_map_3d(p) for p in points])

    def F(rect):
        dim = len(config['DOMAIN_BOUNDS'][0])
        corners = list(product(*[(rect[d], rect[d+dim]) for d in range(dim)]))
        corners = [list(c) for c in corners]

        corners_next = leslie_map_batched(corners)

        padding_size = [(rect[d + dim] - rect[d]) for d in range(dim)] if config.get('PADDING', True) else [0] * dim
        Y_l_bounds = [corners_next[:, d].min() - padding_size[d] for d in range(dim)]
        Y_u_bounds = [corners_next[:, d].max() + padding_size[d] for d in range(dim)]

        return Y_l_bounds + Y_u_bounds

    model = CMGDB.Model(
        config['SUBDIV_MIN'],
        config['SUBDIV_MAX'],
        config['SUBDIV_INIT'],
        config['SUBDIV_LIMIT'],
        config['DOMAIN_BOUNDS'][0],
        config['DOMAIN_BOUNDS'][1],
        F
    )
    return CMGDB.ComputeMorseGraph(model)


def compute_latent_morse_graph_cmgdb(latent_dynamics, device, latent_bounds, config):
    """Compute Morse graph for learned latent dynamics."""
    def latent_map_func_batched(points):
        with torch.no_grad():
            points_tensor = torch.FloatTensor(points).to(device)
            points_next = latent_dynamics(points_tensor).cpu().numpy()
        return points_next

    def F(rect):
        dim = len(latent_bounds[0])
        if config.get('LATENT_PADDING', True):
            corners = list(product(*[(rect[d], rect[d+dim]) for d in range(dim)]))
            corners = [list(c) for c in corners]
        else:
            corners = [[(rect[d] + rect[d+dim])/2 for d in range(dim)]]

        corners_next = latent_map_func_batched(corners)

        padding_size = [(rect[d + dim] - rect[d]) for d in range(dim)] if config.get('LATENT_PADDING', True) else [0] * dim
        Y_l_bounds = [corners_next[:, d].min() - padding_size[d] for d in range(dim)]
        Y_u_bounds = [corners_next[:, d].max() + padding_size[d] for d in range(dim)]

        return Y_l_bounds + Y_u_bounds

    model = CMGDB.Model(
        config['LATENT_SUBDIV_MIN'],
        config['LATENT_SUBDIV_MAX'],
        config['LATENT_SUBDIV_INIT'],
        config['LATENT_SUBDIV_LIMIT'],
        latent_bounds[0],
        latent_bounds[1],
        F
    )
    return CMGDB.ComputeMorseGraph(model)


def compute_latent_morse_graph_data_restricted_cmgdb(latent_dynamics, device, z_data, latent_bounds, config):
    """Compute Morse graph using data-restricted BoxMap."""
    with torch.no_grad():
        G_z = latent_dynamics(torch.FloatTensor(z_data).to(device)).cpu().numpy()

    box_map_data = CMGDB.BoxMapData(
        X=z_data,
        Y=G_z,
        map_empty='outside',
        lower_bounds=latent_bounds[0],
        upper_bounds=latent_bounds[1]
    )

    def F_data(rect):
        return box_map_data.compute(rect)

    model = CMGDB.Model(
        config['LATENT_SUBDIV_MIN'],
        config['LATENT_SUBDIV_MAX'],
        config['LATENT_SUBDIV_INIT'],
        config['LATENT_SUBDIV_LIMIT'],
        latent_bounds[0],
        latent_bounds[1],
        F_data
    )
    return CMGDB.ComputeMorseGraph(model)


# ============================================================================
# Main Learning Experiment Runner
# ============================================================================

def run_learning_experiment(
        run_name=None,
        output_dir=None,
        config_overrides=None,
        verbose=True,
        progress_interval=50):
    """
    Run a single learning experiment with Leslie map.

    Args:
        run_name: Name for this run (can be None for auto-numbering)
        output_dir: Directory to save outputs (if None, uses Config default)
        config_overrides: Dictionary of config parameters to override
        verbose: Whether to print detailed progress
        progress_interval: Print progress every N epochs (when verbose=True)

    Returns:
        Dictionary of results including errors and Morse set counts
    """
    # Import plot_morse_graph for individual graph saves
    from .plotting import plot_morse_graph
    
    # Set output directory
    if output_dir is None:
        output_dir = Config.get_single_runs_dir()
    os.makedirs(output_dir, exist_ok=True)

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

    # Build configuration
    config = {k: v for k, v in vars(Config).items() if not k.startswith('__') and not callable(v)}
    config.update(config_overrides or {})

    if verbose:
        print(f"\n{'='*80}")
        print(f"Run: {run_name}")
        print(f"Output: {run_dir}")
        print(f"{'='*80}")

    # Define paths to pre-computed ground truth files
    gt_dir = Config.get_ground_truth_dir()
    ground_truth_graph_img_path = os.path.join(gt_dir, "morse_graph.png")

    # Load precomputed ground truth barycenters
    ground_truth_barycenters = load_ground_truth_barycenters()
    ground_truth_morse_graph = None  # The C++ object is not loaded, only its image

    if not os.path.exists(ground_truth_graph_img_path) and verbose:
        print("\n⚠️  WARNING: Ground truth Morse graph IMAGE not found!")
        print(f"   Expected at: {ground_truth_graph_img_path}")
        print("   Run 1_compute_ground_truth.py first for complete visualizations.\n")

    if ground_truth_barycenters is None and verbose:
        print("\n⚠️  WARNING: Ground truth barycenters not found!")
        print("   Run 1_compute_ground_truth.py first for complete visualizations.\n")

    # Data generation
    data_cache_path = os.path.join(run_training_data_dir, "trajectory_data.npz")
    x_t, x_t_plus_1, trajectories = generate_map_trajectory_data(
        map_func=leslie_map_3d,
        n_trajectories=config['N_TRAJECTORIES'],
        n_points=config['N_POINTS'],
        sampling_domain=np.array(config['DOMAIN_BOUNDS']),
        random_seed=config['RANDOM_SEED'],
        skip_initial=config['SKIP_INITIAL']
    )
    save_trajectory_data(data_cache_path, x_t, x_t_plus_1, [], {})

    # Train/validation split
    def trajectories_to_pairs(traj_list):
        X_list, Y_list = [], []
        for traj in traj_list:
            for i in range(len(traj) - 1):
                X_list.append(traj[i])
                Y_list.append(traj[i + 1])
        return np.array(X_list), np.array(Y_list)

    n_train_trajectories = int(0.8 * len(trajectories))
    trajectories_train = trajectories[:n_train_trajectories]
    trajectories_val = trajectories[n_train_trajectories:]

    x_train, y_train = trajectories_to_pairs(trajectories_train)
    x_val, y_val = trajectories_to_pairs(trajectories_val)

    if verbose:
        print(f"\nTrain/Validation Split:")
        print(f"  Train: {len(trajectories_train)} trajectories → {len(x_train)} pairs")
        print(f"  Val:   {len(trajectories_val)} trajectories → {len(x_val)} pairs")

    # Data normalization
    use_normalization = needs_normalization(config)
    if use_normalization:
        # Choose normalization method based on decoder activation
        # Sigmoid decoder needs [0, 1] range, tanh/other needs [-1, 1]
        decoder_activation = config.get('DECODER_ACTIVATION') or config.get('OUTPUT_ACTIVATION')
        if decoder_activation == 'sigmoid':
            norm_method = 'minmax_01'
            norm_range_str = '[0, 1]'
        else:
            norm_method = 'minmax'
            norm_range_str = '[-1, 1]'

        if verbose:
            print(f"\nData Normalization:")
            print(f"  Detected bounded activations (tanh/sigmoid)")
            print(f"  Decoder activation: {decoder_activation}")
            print(f"  Normalizing data to {norm_range_str} range")

        norm_params = compute_normalization_params(np.vstack([x_train, y_train]), method=norm_method)
        x_train = normalize_data(x_train, norm_params)
        y_train = normalize_data(y_train, norm_params)
        x_val = normalize_data(x_val, norm_params)
        y_val = normalize_data(y_val, norm_params)
        x_t = normalize_data(x_t, norm_params)
        x_t_plus_1 = normalize_data(x_t_plus_1, norm_params)

        if verbose:
            print(f"  Data range after normalization: [{x_train.min():.3f}, {x_train.max():.3f}]")
    else:
        norm_params = None
        if verbose:
            print(f"\nNo bounded activations detected - skipping normalization")

    # Convert to PyTorch tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)
    x_val_tensor = torch.FloatTensor(x_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Model setup
    encoder = Encoder(
        config['INPUT_DIM'],
        config['LATENT_DIM'],
        config['HIDDEN_DIM'],
        config.get('NUM_LAYERS', 3),
        output_activation=config.get('ENCODER_ACTIVATION') or config.get('OUTPUT_ACTIVATION', None)
    ).to(device)
    decoder = Decoder(
        config['LATENT_DIM'],
        config['INPUT_DIM'],
        config['HIDDEN_DIM'],
        config.get('NUM_LAYERS', 3),
        output_activation=config.get('DECODER_ACTIVATION') or config.get('OUTPUT_ACTIVATION', None)
    ).to(device)
    latent_dynamics = LatentDynamics(
        config['LATENT_DIM'],
        config['HIDDEN_DIM'],
        config.get('NUM_LAYERS', 3),
        output_activation=config.get('LATENT_DYNAMICS_ACTIVATION') or config.get('OUTPUT_ACTIVATION', None)
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

            z_t = encoder(x_batch)
            x_t_recon = decoder(z_t)
            z_t_next_pred = latent_dynamics(z_t)
            x_t_next_pred = decoder(z_t_next_pred)
            z_t_next_true = encoder(y_batch)

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

                z_t = encoder(x_batch)
                x_t_recon = decoder(z_t)
                z_t_next_pred = latent_dynamics(z_t)
                x_t_next_pred = decoder(z_t_next_pred)
                z_t_next_true = encoder(y_batch)

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

        # Print progress
        if verbose and (epoch + 1) % progress_interval == 0:
            print(f"  Epoch {epoch + 1}/{config['NUM_EPOCHS']}")
            print(f"    Train: Recon={train_losses['reconstruction'][-1]:.6f} | "
                  f"DynRecon={train_losses['dynamics_recon'][-1]:.6f} | "
                  f"DynCons={train_losses['dynamics_consistency'][-1]:.6f} | "
                  f"Total={train_losses['total'][-1]:.6f}")
            print(f"    Val:   Recon={val_losses['reconstruction'][-1]:.6f} | "
                  f"DynRecon={val_losses['dynamics_recon'][-1]:.6f} | "
                  f"DynCons={val_losses['dynamics_consistency'][-1]:.6f} | "
                  f"Total={val_losses['total'][-1]:.6f}")

        # Early stopping check
        current_val_loss = val_losses['total'][-1]
        if current_val_loss < best_val_loss - min_delta:
            best_val_loss = current_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(current_val_loss)

        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"  Early stopping triggered at epoch {epoch + 1}")
            break

    total_training_time = time.time() - training_start_time
    if verbose:
        print(f"  Training completed in {format_time(total_training_time)}")

    # Save models
    torch.save(encoder.state_dict(), os.path.join(run_models_dir, "encoder.pt"))
    torch.save(decoder.state_dict(), os.path.join(run_models_dir, "decoder.pt"))
    torch.save(latent_dynamics.state_dict(), os.path.join(run_models_dir, "latent_dynamics.pt"))

    # Save training history
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
    plot_training_losses(train_losses, val_losses, os.path.join(run_results_dir, "training_losses.png"))

    if verbose:
        print("  Generating visualization figures...")

    # Compute Morse graphs
    if verbose:
        print("  Computing latent Morse graphs...")

    x_full_tensor = torch.FloatTensor(x_t)
    with torch.no_grad():
        z_train = encoder(x_full_tensor.to(device)).cpu().numpy()
        z_val = encoder(x_val_tensor.to(device)).cpu().numpy()

    latent_bounds = compute_latent_bounds(z_train, padding_factor=config['LATENT_BOUNDS_PADDING'])
    latent_morse_graph_full, _ = compute_latent_morse_graph_cmgdb(latent_dynamics, device, latent_bounds.tolist(), config)
    latent_morse_graph_restricted, _ = compute_latent_morse_graph_data_restricted_cmgdb(latent_dynamics, device, z_train, latent_bounds.tolist(), config)
    latent_morse_graph_val, _ = compute_latent_morse_graph_data_restricted_cmgdb(latent_dynamics, device, z_val, latent_bounds.tolist(), config)

    # Save individual Morse graphs
    if verbose:
        print("  Saving individual Morse graphs...")
    plot_morse_graph(
        latent_morse_graph_full,
        os.path.join(run_results_dir, "morse_graph_latent_full.png"),
        title="Latent Dynamics - Full Morse Graph",
        cmap=plt.cm.viridis
    )
    plot_morse_graph(
        latent_morse_graph_restricted,
        os.path.join(run_results_dir, "morse_graph_latent_train.png"),
        title="BoxMapData on E(X_train) - Morse Graph",
        cmap=plt.cm.viridis
    )
    plot_morse_graph(
        latent_morse_graph_val,
        os.path.join(run_results_dir, "morse_graph_latent_val.png"),
        title="BoxMapData on E(X_val) - Morse Graph",
        cmap=plt.cm.viridis
    )

    # Save latent space data
    latent_data_path = os.path.join(run_training_data_dir, "latent_data.npz")
    np.savez(latent_data_path, z_train=z_train, latent_bounds=latent_bounds)

    # Generate comprehensive visualizations
    if verbose:
        print("  Generating Figure 1: Latent transformation analysis...")
    plot_latent_transformation_analysis(
        x_t, encoder, decoder, device, latent_bounds.tolist(),
        os.path.join(run_results_dir, "latent_transformation_analysis.png")
    )

    if verbose:
        print("  Generating Figure 2: Morse graph comparison (5 variants)...")
    plot_morse_graph_comparison(
        ground_truth_graph_img_path,
        ground_truth_barycenters,
        latent_morse_graph_full,
        latent_morse_graph_restricted,
        latent_morse_graph_val,
        encoder,
        device,
        x_t,
        z_train,
        z_val,
        latent_bounds.tolist(),
        config['DOMAIN_BOUNDS'],
        run_results_dir
    )

    if verbose:
        print("  Generating Figure 3: Preimage classification...")
    plot_preimage_classification(
        latent_morse_graph_full,
        latent_morse_graph_restricted,
        latent_morse_graph_val,
        encoder,
        decoder,
        device,
        x_t,
        z_train,
        z_val,
        latent_bounds.tolist(),
        config['DOMAIN_BOUNDS'],
        os.path.join(run_results_dir, "preimage_classification.png")
    )

    # Compile results
    results = {
        'run_name': run_name,
        'num_latent_full_morse_sets': latent_morse_graph_full.num_vertices() if latent_morse_graph_full else 0,
        'num_latent_restricted_morse_sets': latent_morse_graph_restricted.num_vertices() if latent_morse_graph_restricted else 0,
        'num_latent_val_morse_sets': latent_morse_graph_val.num_vertices() if latent_morse_graph_val else 0,
        'final_train_loss': train_losses['total'][-1],
        'final_val_loss': val_losses['total'][-1],
        'total_training_time': total_training_time,
        'total_parameters': total_params,
    }

    # Save settings
    settings = {
        'run_info': {
            'run_name': run_name,
            'run_dir': run_dir,
            'timestamp': datetime.now().isoformat()
        },
        'configuration': {k: v for k, v in config.items() if not callable(v)},
        'results': results
    }

    with open(os.path.join(run_dir, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=2)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Results Summary:")
        print(f"  Latent Full Morse Sets:    {latent_morse_graph_full.num_vertices() if latent_morse_graph_full else 0}")
        print(f"  Latent Train Morse Sets:   {latent_morse_graph_restricted.num_vertices() if latent_morse_graph_restricted else 0}")
        print(f"  Latent Val Morse Sets:     {latent_morse_graph_val.num_vertices() if latent_morse_graph_val else 0}")
        print(f"\nSaved to: {run_dir}")
        print(f"{'='*80}\n")

    return results
