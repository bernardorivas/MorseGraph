#!/usr/bin/env python3
"""
Ives Ecological Model - MorseGraph analysis through CMGDB/MORALS framework

This example studies the Ives midge-algae-detritus ecological model
using the Morse Graphs. The model operates in log10 scale
to handle many orders of magnitude in population abundances.

Model: Ives et al. (2008) - "High-amplitude fluctuations and alternative
       dynamical states of midges in Lake Myvatn", Nature 452: 84-87

The system models nonlinear interaction in the Lake Myvatn of
- Midge larvae (M)
- Algae (A)
- Detritus (D)

Usage:
    python 7_ives_model.py

Output:
    All results saved to examples/ives_model_output/
"""

import numpy as np
import os
import sys
import argparse
from functools import partial

# Set matplotlib to non-interactive backend (prevents plots from showing)
import matplotlib
matplotlib.use('Agg')
import torch

# Add parent directory to path for MorseGraph imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.utils import (
    generate_map_trajectory_data,
    setup_experiment_dirs,
    save_experiment_metadata,
    compute_latent_bounds,
    get_next_run_number,
    load_or_compute_3d_morse_graph,
    compute_trajectory_hash,
    compute_training_hash,
    compute_cmgdb_2d_hash,
    load_or_train_autoencoder,
    load_or_compute_2d_morse_graphs,
    load_or_generate_trajectory_data
)
from MorseGraph.config import load_experiment_config, save_config_to_yaml
from MorseGraph.core import (
    compute_morse_graph_2d_data,
    compute_morse_graph_2d_restricted
)
from MorseGraph.training import train_autoencoder_dynamics
from MorseGraph.plot import (
    plot_morse_graph_diagram,  # Used for 2D Morse graphs
    plot_training_curves,
    plot_latent_space_flexible,
    plot_encoder_decoder_roundtrip,
    plot_trajectory_analysis,
    plot_morse_graph_comparison,
    plot_2x2_morse_comparison,
    plot_preimage_classification,
    compute_encoded_barycenters
)
# Note: 3D visualizations (plot_morse_sets_3d_scatter, 3D projections) are now
# generated in load_or_compute_3d_morse_graph() and cached in cmgdb_3d/{hash}/results/
from MorseGraph.systems import ives_model_log


# ============================================================================
# Ives Model Parameters & Configuration
# ============================================================================
# All parameters are loaded from YAML config files in configs/
# See configs/README.md for documentation on:
#   - configs/ives_default.yaml  (standard settings)
#   - configs/ives_high_res.yaml (publication quality)
#   - configs/ives_fast.yaml     (quick testing)
#
# Model parameters, CMGDB settings, and training hyperparameters are all
# defined in the config files for easy experimentation and reproducibility.
# ============================================================================



# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Run complete Ives model analysis pipeline."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Ives Ecological Model - Morse Graph Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python 7_ives_model.py                                     # Run with default config
        python 7_ives_model.py --config configs/ives_fast.yaml     # Quick test run
        python 7_ives_model.py --config configs/ives_high_res.yaml # High-resolution
        python 7_ives_model.py --force-recompute-3d                # Force 3D recompute
        python 7_ives_model.py --force-regenerate-data             # Force trajectory regeneration
        python 7_ives_model.py --force-retrain                     # Force autoencoder retrain
        python 7_ives_model.py --force-recompute-2d                # Force 2D recompute
        python 7_ives_model.py --force-all                         # Force all computations
        python 7_ives_model.py --help                              # Show this help
                """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='examples/configs/ives_default.yaml',
        help='Path to YAML configuration file (default: examples/configs/ives_default.yaml)'
    )
    parser.add_argument(
        '--force-recompute-3d',
        action='store_true',
        help='Force recomputation of 3D Morse graph (ignore cache)'
    )
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retraining of autoencoder models (ignore cached training)'
    )
    parser.add_argument(
        '--force-recompute-2d',
        action='store_true',
        help='Force recomputation of 2D Morse graphs (ignore cache)'
    )
    parser.add_argument(
        '--force-regenerate-data',
        action='store_true',
        help='Force regeneration of trajectory data (ignore cache)'
    )
    parser.add_argument(
        '--force-all',
        action='store_true',
        help='Force all computations (equivalent to --force-recompute-3d --force-regenerate-data --force-retrain --force-recompute-2d)'
    )
    # Legacy support
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        dest='force_recompute_legacy',
        help=argparse.SUPPRESS  # Hidden - legacy support for --force-recompute
    )
    args = parser.parse_args()

    # Handle --force-all and legacy --force-recompute
    if args.force_all:
        args.force_recompute_3d = True
        args.force_regenerate_data = True
        args.force_retrain = True
        args.force_recompute_2d = True

    # Legacy support: --force-recompute maps to --force-recompute-3d
    if hasattr(args, 'force_recompute_legacy') and args.force_recompute_legacy:
        args.force_recompute_3d = True

    # Load configuration from YAML first
    print("\n" + "="*80)
    config = load_experiment_config(args.config, verbose=True)
    print("="*80)

    # Extract dynamics parameters from config - fail fast if missing
    if not hasattr(config, '_yaml_config') or 'dynamics' not in config._yaml_config:
        raise ValueError(
            "Config must have 'dynamics' section. Please check your YAML configuration file.\n"
            "Required fields: r1, r2, c, d, p, q, log_offset"
        )

    dyn_params = config._yaml_config['dynamics']

    # Extract required dynamics parameters (no defaults - must be explicit)
    required_params = ['r1', 'r2', 'c', 'd', 'p', 'q', 'log_offset']
    missing = [p for p in required_params if p not in dyn_params]
    if missing:
        raise ValueError(
            f"Missing required dynamics parameters in config: {missing}\n"
            f"Please add these to the 'dynamics' section of your YAML config."
        )

    R1 = dyn_params['r1']
    R2 = dyn_params['r2']
    C = dyn_params['c']
    D = dyn_params['d']
    P = dyn_params['p']
    Q = dyn_params['q']
    LOG_OFFSET = dyn_params['log_offset']

    # Optional: equilibrium point (for visualization)
    if 'domain' in config._yaml_config and 'equilibrium' in config._yaml_config['domain']:
        EQUILIBRIUM_POINT = np.array(config._yaml_config['domain']['equilibrium'])
    else:
        EQUILIBRIUM_POINT = None
        print("  Note: No equilibrium point provided in config (optional)")

    # Optional: period-12 orbit (for visualization)
    if 'domain' in config._yaml_config and 'period_12_orbit' in config._yaml_config['domain']:
        PERIOD_12_ORBIT = np.array(config._yaml_config['domain']['period_12_orbit'])
    else:
        PERIOD_12_ORBIT = None
        print("  Note: No period-12 orbit provided in config (optional)")

    # Print header with loaded parameters
    print("\nIVES ECOLOGICAL MODEL - MORSE GRAPH ANALYSIS")
    print("Model: Midge-Algae-Detritus dynamics (Ives et al. 2008)")
    print(f"Parameters: R1={R1:.3f}, R2={R2:.3f}, C={C:.2e}, D={D:.4f}, P={P:.5f}, Q={Q:.3f}")
    print(f"Domain (log): {config.domain_bounds[0]} to {config.domain_bounds[1]}")

    if EQUILIBRIUM_POINT is not None:
        print(f"Equilibrium (log): [{EQUILIBRIUM_POINT[0]:.4f}, {EQUILIBRIUM_POINT[1]:.4f}, {EQUILIBRIUM_POINT[2]:.4f}]")
    if PERIOD_12_ORBIT is not None:
        print(f"Period-12 orbit: {len(PERIOD_12_ORBIT)} points loaded from config")

    # Print force flags status
    if args.force_recompute_3d:
        print("\nForce recompute 3D enabled - will ignore 3D CMGDB cache")
    if args.force_regenerate_data:
        print("Force regenerate data enabled - will ignore trajectory data cache")
    if args.force_retrain:
        print("Force retrain enabled - will ignore training cache")
    if args.force_recompute_2d:
        print("Force recompute 2D enabled - will ignore 2D CMGDB cache")

    # Create map function with loaded parameters
    ives_map = partial(
        ives_model_log,
        r1=R1, r2=R2, c=C, d=D, p=P, q=Q, offset=LOG_OFFSET
    )
    config.set_map_func(ives_map)

    # Setup directories with run_XXX structure
    base_output_dir = os.path.join(os.path.dirname(__file__), "ives_model_output")
    run_number = get_next_run_number(base_output_dir)
    run_dir = os.path.join(base_output_dir, f"run_{run_number:03d}")
    dirs = setup_experiment_dirs(run_dir)

    # Save config to run directory for reproducibility
    config_save_path = os.path.join(run_dir, "config.yaml")
    save_config_to_yaml(config, config_save_path)

    print(f"\nRun Directory: {run_dir}")
    print(f"Run Number: {run_number}")
    print(f"Config saved to: {config_save_path}")

    # ========================================================================
    # Step 1: Load or Compute 3D Morse Graph (Ground Truth)
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 1: Loading/Computing 3D Morse Graph (Log Scale)")
    print("="*80)

    result_3d, was_cached_3d = load_or_compute_3d_morse_graph(
        config.map_func,
        config.domain_bounds,
        subdiv_min=config.subdiv_min,
        subdiv_max=config.subdiv_max,
        subdiv_init=config.subdiv_init,
        subdiv_limit=config.subdiv_limit,
        padding=config.padding,
        base_dir=base_output_dir,
        force_recompute=args.force_recompute_3d,
        verbose=True,
        equilibria={'Equilibrium': EQUILIBRIUM_POINT} if EQUILIBRIUM_POINT is not None else None,
        periodic_orbits={'Period-12': PERIOD_12_ORBIT} if PERIOD_12_ORBIT is not None else None,
        labels={'x': 'log(Midge)', 'y': 'log(Algae)', 'z': 'log(Detritus)'}
    )

    # Store hash for dependent computations
    cmgdb_3d_hash = result_3d['param_hash']

    morse_graph_3d = result_3d['morse_graph']
    barycenters_3d = result_3d['barycenters']

    if was_cached_3d:
        print(f"\n✓ Using cached 3D Morse graph (hash: {cmgdb_3d_hash})")
    else:
        print(f"\n✓ Computed and saved 3D Morse graph (hash: {cmgdb_3d_hash})")

    # Note: 3D visualizations are already saved in cmgdb_3d/{hash}/results/:
    #   - morse_graph_3d.png (Morse graph diagram)
    #   - morse_sets_3d.png (3D scatter with barycenters)
    #   - morse_sets_proj_01.png, morse_sets_proj_02.png, morse_sets_proj_12.png (2D projections)
    results_3d_dir = os.path.join(result_3d['cache_path'], 'results')
    print(f"  3D visualizations available in: {results_3d_dir}/")

    # ========================================================================
    # Step 2: Load or Generate Training Data
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 2: Loading/Generating Training Data")
    print("="*80)

    # Compute trajectory hash based on config and 3D hash
    trajectory_hash = compute_trajectory_hash(config, cmgdb_3d_hash)
    print(f"Trajectory hash: {trajectory_hash}")

    # Load or generate trajectory data
    trajectory_result, was_cached_traj = load_or_generate_trajectory_data(
        config=config,
        trajectory_hash=trajectory_hash,
        map_func=config.map_func,
        domain_bounds=np.array(config.domain_bounds),
        output_dir=base_output_dir,
        force_regenerate=args.force_regenerate_data
    )

    X = trajectory_result['X']
    Y = trajectory_result['Y']
    trajectories = trajectory_result['trajectories']

    if was_cached_traj:
        print(f"\n✓ Using cached trajectory data (hash: {trajectory_hash})")
    else:
        print(f"\n✓ Generated and cached trajectory data (hash: {trajectory_hash})")

    # Train/val split
    split = int(0.8 * len(X))
    x_train, y_train = X[:split], Y[:split]
    x_val, y_val = X[split:], Y[split:]

    print(f"Train: {len(x_train)} samples")
    print(f"Val:   {len(x_val)} samples")

    # ========================================================================
    # Step 3: Load or Train Autoencoder + Latent Dynamics
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 3: Loading/Training Autoencoder + Latent Dynamics")
    print("="*80)

    # Compute training hash based on config and 3D hash
    training_hash = compute_training_hash(config, cmgdb_3d_hash)
    print(f"Training hash: {training_hash}")

    # Prepare training data
    training_data = {
        'X_train': x_train,
        'Xnext_train': y_train,
        'X_val': x_val,
        'Xnext_val': y_val
    }

    # Load or train models
    training_result, was_cached_training = load_or_train_autoencoder(
        config=config,
        training_hash=training_hash,
        training_data=training_data,
        map_func=config.map_func,
        output_dir=base_output_dir,
        force_retrain=args.force_retrain
    )

    encoder = training_result['encoder']
    decoder = training_result['decoder']
    latent_dynamics = training_result['latent_dynamics']

    # Get device from encoder
    device = next(encoder.parameters()).device

    if was_cached_training:
        print(f"\n✓ Using cached training (hash: {training_hash})")
    else:
        print(f"\n✓ Training completed and cached (hash: {training_hash})")

    # Copy models to run directory for reference
    torch.save(encoder.state_dict(), f"{dirs['models']}/encoder.pt")
    torch.save(decoder.state_dict(), f"{dirs['models']}/decoder.pt")
    torch.save(latent_dynamics.state_dict(), f"{dirs['models']}/latent_dynamics.pt")

    # Plot training curves
    losses = training_result['training_losses']
    plot_training_curves(
        losses['train'],
        losses['val'],
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

    # Get latent bounds from training result (or compute if not cached)
    if 'latent_bounds' in training_result:
        latent_bounds = training_result['latent_bounds']
    else:
        latent_bounds = compute_latent_bounds(z_train, padding_factor=config.latent_bounds_padding)

    print(f"Latent bounds: {latent_bounds.tolist()}")

    # Encode 3D barycenters to latent space for visualization
    print("\nEncoding 3D barycenters to latent space...")
    barycenters_latent = compute_encoded_barycenters(barycenters_3d, encoder, device)
    total_barycenters = sum(len(barys) for barys in barycenters_latent.values())
    print(f"  Encoded {total_barycenters} barycenters from {len(barycenters_latent)} Morse sets")

    # ========================================================================
    # Step 4a: Encoder/Decoder Round-Trip Analysis
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 4a: Encoder/Decoder Round-Trip Analysis")
    print("="*80)

    plot_encoder_decoder_roundtrip(
        encoder,
        decoder,
        device,
        original_bounds=config.domain_bounds,
        latent_bounds=latent_bounds.tolist(),
        n_grid_points=config.n_grid_points,
        output_path=f"{dirs['results']}/encoder_decoder_roundtrip.png",
        title_prefix="Ives Model - ",
        labels={'x': 'log(Midge)', 'y': 'log(Algae)', 'z': 'log(Detritus)'}
    )
    print("  ✓ Saved encoder/decoder round-trip analysis")

    # ========================================================================
    # Step 4b: Generate and Visualize Trajectory Simulations
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 4b: Trajectory Simulation Analysis")
    print("="*80)

    # Generate 5 trajectories from random initial conditions
    n_test_trajectories = 5
    n_trajectory_steps = 100
    print(f"  Generating {n_test_trajectories} test trajectories ({n_trajectory_steps} steps each)...")

    np.random.seed(42)  # For reproducibility
    test_trajectories_3d = []
    test_trajectories_latent = []

    for i in range(n_test_trajectories):
        # Random initial condition in domain
        x0 = np.array([
            np.random.uniform(config.domain_bounds[0][0], config.domain_bounds[1][0]),
            np.random.uniform(config.domain_bounds[0][1], config.domain_bounds[1][1]),
            np.random.uniform(config.domain_bounds[0][2], config.domain_bounds[1][2])
        ])

        # Simulate in original space
        traj_3d = [x0]
        for _ in range(n_trajectory_steps - 1):
            traj_3d.append(config.map_func(traj_3d[-1]))
        traj_3d = np.array(traj_3d)

        # Encode to latent space and simulate there
        with torch.no_grad():
            z0 = encoder(torch.FloatTensor(x0).unsqueeze(0).to(device)).cpu().numpy()[0]
            traj_latent = [z0]
            for _ in range(n_trajectory_steps - 1):
                z_next = latent_dynamics(torch.FloatTensor(traj_latent[-1]).unsqueeze(0).to(device)).cpu().numpy()[0]
                traj_latent.append(z_next)
            traj_latent = np.array(traj_latent)

        test_trajectories_3d.append(traj_3d)
        test_trajectories_latent.append(traj_latent)

    print(f"  Generated {len(test_trajectories_3d)} trajectories")

    plot_trajectory_analysis(
        test_trajectories_3d,
        test_trajectories_latent,
        output_path=f"{dirs['results']}/trajectory_analysis.png",
        title_prefix="Ives Model - ",
        labels={'x': 'log(Midge)', 'y': 'log(Algae)', 'z': 'log(Detritus)'}
    )
    print("  ✓ Saved trajectory analysis visualization")

    # ========================================================================
    # Step 5: Load or Compute Learned Latent Dynamics (2D Morse Graphs)
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 5: Loading/Computing Learned Latent Dynamics (2D Morse Graphs)")
    print("="*80)

    # Compute 2D hash based on config and training hash
    cmgdb_2d_hash = compute_cmgdb_2d_hash(config, training_hash)
    print(f"2D CMGDB hash: {cmgdb_2d_hash}")

    # Load or compute 2D Morse graphs
    morse_2d_result, was_cached_2d = load_or_compute_2d_morse_graphs(
        config=config,
        cmgdb_2d_hash=cmgdb_2d_hash,
        encoder=encoder,
        decoder=decoder,
        latent_dynamics=latent_dynamics,
        latent_bounds=latent_bounds,
        map_func=config.map_func,
        output_dir=base_output_dir,
        force_recompute=args.force_recompute_2d
    )

    morse_graph_2d = morse_2d_result['morse_graph']
    barycenters_2d = morse_2d_result['barycenters']

    if was_cached_2d:
        print(f"\n✓ Using cached 2D Morse graph (hash: {cmgdb_2d_hash})")
    else:
        print(f"\n✓ Computed and cached 2D Morse graph (hash: {cmgdb_2d_hash})")

    # ========================================================================
    # Step 5a: Generate Large Grid for Comparisons and Preimage Analysis
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 5a: Generating Large Grid for Analysis")
    print("="*80)

    # Generate dense uniform grid in original space for encoding
    original_grid_subdiv = config.original_grid_subdiv
    n_per_dim = 2 ** (original_grid_subdiv // 3)
    print(f"  Grid: {n_per_dim} points per dimension ({n_per_dim**3} total)")

    # Create meshgrid for 3D space
    grid_1d = [np.linspace(config.domain_bounds[0][i], config.domain_bounds[1][i], n_per_dim)
               for i in range(3)]
    mesh = np.meshgrid(*grid_1d, indexing='ij')
    X_large_grid = np.stack([m.flatten() for m in mesh], axis=1)
    print(f"  Generated {len(X_large_grid)} grid points in original space")

    # Encode grid to latent space
    with torch.no_grad():
        z_large_grid = encoder(torch.FloatTensor(X_large_grid).to(device)).cpu().numpy()
    print(f"  Encoded to latent space: {z_large_grid.shape}")

    # ========================================================================
    # Step 6: Visualize Learned Latent Dynamics (2D)
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 6: Generating 2D Visualizations")
    print("="*80)

    # Padded Morse graph
    plot_morse_graph_diagram(
        morse_graph_2d,
        output_path=f"{dirs['results']}/morse_graph_2d.png",
        title="Ives Model - Learned Latent Dynamics (2D) - Padded"
    )

    # Project equilibrium and period-12 orbit to latent space
    with torch.no_grad():
        # Encode equilibrium if available
        if EQUILIBRIUM_POINT is not None:
            equilibrium_tensor = torch.FloatTensor(EQUILIBRIUM_POINT).unsqueeze(0).to(device)
            equilibrium_latent = encoder(equilibrium_tensor).cpu().numpy()[0]
        else:
            equilibrium_latent = None

        # Encode period-12 orbit if available
        if PERIOD_12_ORBIT is not None:
            period12_tensor = torch.FloatTensor(PERIOD_12_ORBIT).to(device)
            period12_latent = encoder(period12_tensor).cpu().numpy()
        else:
            period12_latent = None

    # ========================================================================
    # Step 6.1: Enhanced Latent Space Visualizations (Padded Method)
    # ========================================================================

    print("\n  Generating enhanced latent space visualizations (Padded method)...")

    # 1. Morse sets only
    plot_latent_space_flexible(
        morse_graph=morse_graph_2d,
        latent_bounds=latent_bounds.tolist(),
        output_path=f"{dirs['results']}/latent_morse_sets_only.png",
        title="Latent Space - Morse Sets Only",
        show_morse_sets=True
    )

    # 2. E(Barycenters) only
    plot_latent_space_flexible(
        barycenters_latent=barycenters_latent,
        latent_bounds=latent_bounds.tolist(),
        output_path=f"{dirs['results']}/latent_barycenters_only.png",
        title="Latent Space - E(3D Barycenters)",
        show_barycenters=True,
        barycenter_size=6,
        cmap_barycenters='cool'  # E(3D barycenters) use cool colormap
    )

    # 3. Equilibrium only
    plot_latent_space_flexible(
        equilibrium_latent=equilibrium_latent,
        latent_bounds=latent_bounds.tolist(),
        output_path=f"{dirs['results']}/latent_equilibrium_only.png",
        title="Latent Space - Equilibrium",
        show_equilibrium=True
    )

    # 4. Data + Morse sets
    plot_latent_space_flexible(
        z_data=z_train,
        morse_graph=morse_graph_2d,
        latent_bounds=latent_bounds.tolist(),
        output_path=f"{dirs['results']}/latent_data_morse_sets.png",
        title="Latent Space - Data + Morse Sets",
        show_data=True,
        show_morse_sets=True
    )

    # 5. Data + E(Barycenters) + Equilibrium
    plot_latent_space_flexible(
        z_data=z_train,
        barycenters_latent=barycenters_latent,
        equilibrium_latent=equilibrium_latent,
        latent_bounds=latent_bounds.tolist(),
        output_path=f"{dirs['results']}/latent_data_barycenters.png",
        title="Latent Space - Data + E(Barycenters) + Equilibrium",
        show_data=True,
        show_barycenters=True,
        show_equilibrium=True,
        barycenter_size=6,
        cmap_barycenters='cool'  # E(3D barycenters) use cool colormap
    )

    # 6. Morse sets + E(Barycenters) + Equilibrium
    plot_latent_space_flexible(
        morse_graph=morse_graph_2d,
        barycenters_latent=barycenters_latent,
        equilibrium_latent=equilibrium_latent,
        latent_bounds=latent_bounds.tolist(),
        output_path=f"{dirs['results']}/latent_morse_barycenters.png",
        title="Latent Space - Morse Sets + E(Barycenters) + Equilibrium",
        show_morse_sets=True,
        show_barycenters=True,
        show_equilibrium=True,
        barycenter_size=6,
        cmap_morse_sets='viridis',    # 2D Morse sets use viridis
        cmap_barycenters='cool'        # E(3D barycenters) use cool colormap
    )

    # 7-9. E(Period-12 orbit) visualizations
    if period12_latent is not None:
        # Just orbit
        plot_latent_space_flexible(
            period12_latent=period12_latent,
            latent_bounds=latent_bounds.tolist(),
            output_path=f"{dirs['results']}/latent_period12_only.png",
            title="Latent Space - E(Period-12 Orbit)",
            show_period12=True
        )

        # Data + orbit
        plot_latent_space_flexible(
            z_data=z_train,
            period12_latent=period12_latent,
            latent_bounds=latent_bounds.tolist(),
            output_path=f"{dirs['results']}/latent_data_period12.png",
            title="Latent Space - Data + E(Period-12 Orbit)",
            show_data=True,
            show_period12=True
        )

        # Morse sets + orbit
        plot_latent_space_flexible(
            morse_graph=morse_graph_2d,
            period12_latent=period12_latent,
            latent_bounds=latent_bounds.tolist(),
            output_path=f"{dirs['results']}/latent_morse_period12.png",
            title="Latent Space - Morse Sets + E(Period-12 Orbit)",
            show_morse_sets=True,
            show_period12=True
        )

    print("  ✓ Saved 9 enhanced latent space visualizations")

    # ========================================================================
    # Step 6a: Morse Graph Comparison
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 6a: Morse Graph Comparison")
    print("="*80)

    # Create 2x2 clean comparison
    print("\n  Creating 2x2 comparison visualization...")
    plot_2x2_morse_comparison(
        morse_graph_3d,
        morse_graph_2d,
        config.domain_bounds,
        latent_bounds.tolist(),
        encoder,
        device,
        z_train,
        output_path=f"{dirs['results']}/morse_2x2_comparison.png",
        title_prefix="Ives - ",
        equilibria={'Equilibrium': EQUILIBRIUM_POINT} if EQUILIBRIUM_POINT is not None else None,
        periodic_orbits={'Period-12': PERIOD_12_ORBIT} if PERIOD_12_ORBIT is not None else None,
        equilibria_latent={'Equilibrium': equilibrium_latent},
        labels={'x': 'log(Midge)', 'y': 'log(Algae)', 'z': 'log(Detritus)'}
    )

    print(f"  3D Morse Sets: {morse_graph_3d.num_vertices()}")
    print(f"  2D Morse Sets: {morse_graph_2d.num_vertices()}")
    print("  ✓ Saved 2x2 comparison")

    # ========================================================================
    # Step 6b: Preimage Classification Analysis
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 6b: Preimage Classification Analysis")
    print("="*80)

    # Analyze preimages
    print("  Analyzing preimages...")
    preimages = plot_preimage_classification(
        morse_graph_2d,
        encoder,
        decoder,
        device,
        X_large_grid,
        latent_bounds.tolist(),
        config.domain_bounds,
        subdiv_max=config.latent_subdiv_max,
        output_path=f"{dirs['results']}/preimage_classification.png",
        title_prefix="Ives Model - ",
        method_name="BoxMapData",
        labels={'x': 'log(Midge)', 'y': 'log(Algae)', 'z': 'log(Detritus)'},
        max_points_per_set=2000
    )
    print("  ✓ Saved preimage analysis")

    # ========================================================================
    # Step 7: Save Results Summary
    # ========================================================================

    results = {
        # Run info
        'run_number': run_number,
        'run_directory': run_dir,

        # 3D Morse graph
        'num_morse_sets_3d': result_3d['num_morse_sets'],
        'computation_time_3d': result_3d['computation_time'],
        'was_3d_cached': was_cached_3d,

        # 2D Morse graph
        'num_morse_sets_2d': morse_graph_2d.num_vertices(),

        # Training
        'training_time': training_result.get('training_time', 0.0),
        'final_train_loss': losses['train']['total'][-1],
        'final_val_loss': losses['val']['total'][-1],

        # Analysis
        'n_test_trajectories': n_test_trajectories,
        'n_trajectory_steps': n_trajectory_steps,

        # Model parameters
        'ives_parameters': {
            'r1': R1, 'r2': R2, 'c': C, 'd': D, 'p': P, 'q': Q,
            'log_offset': LOG_OFFSET
        },
        'equilibrium_point': EQUILIBRIUM_POINT.tolist() if EQUILIBRIUM_POINT is not None else None,
    }

    save_experiment_metadata(f"{dirs['base']}/metadata.json", config, results)

    print("\n" + "="*80)
    print("IVES MODEL ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults Summary:")
    print(f"  Run Number:                 {run_number}")
    print(f"  3D Morse Sets:              {results['num_morse_sets_3d']} {'(cached)' if was_cached_3d else '(computed)'}")
    print(f"  2D Morse Sets:              {results['num_morse_sets_2d']}")
    training_time_str = f"{results['training_time']:.2f}s" if results['training_time'] > 0 else "(cached)"
    print(f"  Training Time:              {training_time_str}")
    print(f"  Final Train Loss:           {results['final_train_loss']:.6f}")
    print(f"  Final Val Loss:             {results['final_val_loss']:.6f}")
    param_hash_short = result_3d['param_hash'][:8]
    print(f"\nOutput Structure:")
    print(f"  {base_output_dir}/")
    print(f"  ├── cmgdb_3d/                 # Cached 3D Morse graphs (organized by parameter hash)")
    print(f"  │   ├── {param_hash_short}.../         # This parameter configuration")
    print(f"  │   │   ├── morse_graph_data.pkl")
    print(f"  │   │   ├── barycenters.npz")
    print(f"  │   │   ├── metadata.json")
    print(f"  │   │   └── results/          # 3D visualizations for this config")
    print(f"  │   └── <other_hashes>.../    # Other parameter configurations")
    print(f"  └── run_{run_number:03d}/               # This run's results")
    print(f"      ├── config.yaml")
    print(f"      ├── models/")
    print(f"      ├── training_data/")
    print(f"      └── results/              # 2D and analysis visualizations")
    print(f"\n3D Visualizations (in {results_3d_dir}/):")
    print(f"  - morse_graph_3d.png: 3D Morse graph diagram")
    print(f"  - morse_sets_3d.png: 3D scatter with barycenters (size ∝ box volume)")
    print(f"  - morse_sets_proj_01.png: Projection onto dims 0-1")
    print(f"  - morse_sets_proj_02.png: Projection onto dims 0-2")
    print(f"  - morse_sets_proj_12.png: Projection onto dims 1-2")
    print(f"\nRun-Specific Visualizations (in {run_dir}/results/):")
    print(f"  - training_curves.png: Training/validation loss curves")
    print(f"\n  Quality Analysis:")
    print(f"    - encoder_decoder_roundtrip.png: E/D transformation quality (2x3 grid)")
    print(f"    - trajectory_analysis.png: Trajectory simulations (3D + latent)")
    print(f"\n  Latent Space (2D) - Enhanced Visualizations:")
    print(f"    Basic views:")
    print(f"      - latent_morse_sets_only.png: Just morse sets")
    print(f"      - latent_barycenters_only.png: Just E(3D barycenters)")
    print(f"      - latent_equilibrium_only.png: Just equilibrium")
    print(f"    Combined views:")
    print(f"      - latent_data_morse_sets.png: Data + morse sets")
    print(f"      - latent_data_barycenters.png: Data + E(barycenters) + equilibrium")
    print(f"      - latent_morse_barycenters.png: Morse sets + E(barycenters) + equilibrium")
    print(f"    Period-12 orbit views:")
    print(f"      - latent_period12_only.png: Just E(period-12 orbit)")
    print(f"      - latent_data_period12.png: Data + E(period-12 orbit)")
    print(f"      - latent_morse_period12.png: Morse sets + E(period-12 orbit)")
    print(f"\n  Morse Graphs:")
    print(f"    - morse_graph_2d.png: 2D Morse graph")
    print(f"\n  Comprehensive Comparisons:")
    print(f"    - morse_2x2_comparison.png: Clean 2x2 comparison (3D vs 2D)")
    print(f"    - preimage_classification.png: Preimage analysis")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
