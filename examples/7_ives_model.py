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
    load_or_compute_3d_morse_graph
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
# generated in load_or_compute_3d_morse_graph() and cached in cmgdb_3d/results/
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
        python 7_ives_model.py --force-recompute                   # Force 3D recompute
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
        '--force-recompute', '--force-recompute-3d',
        action='store_true',
        dest='force_recompute',
        help='Force recomputation of 3D Morse graph (ignore cache)'
    )
    args = parser.parse_args()

    # Load configuration from YAML first
    print("\n" + "="*80)
    config = load_experiment_config(args.config, verbose=True)
    print("="*80)

    # Extract dynamics parameters from config
    if hasattr(config, '_yaml_config') and 'dynamics' in config._yaml_config:
        dyn_params = config._yaml_config['dynamics']
        R1 = dyn_params.get('r1', 3.873)
        R2 = dyn_params.get('r2', 11.746)
        C = dyn_params.get('c', 3.67e-07)
        D = dyn_params.get('d', 0.5517)
        P = dyn_params.get('p', 0.06659)
        Q = dyn_params.get('q', 0.9026)
        LOG_OFFSET = dyn_params.get('log_offset', 0.001)

        # Get equilibrium and period-12 orbit if available
        if 'domain' in config._yaml_config and 'equilibrium' in config._yaml_config['domain']:
            EQUILIBRIUM_POINT = np.array(config._yaml_config['domain']['equilibrium'])
        else:
            EQUILIBRIUM_POINT = np.array([0.792107, 0.209010, 0.376449])
        
        # Read period-12 orbit from config
        if 'domain' in config._yaml_config and 'period_12_orbit' in config._yaml_config['domain']:
            PERIOD_12_ORBIT = np.array(config._yaml_config['domain']['period_12_orbit'])
        else:
            PERIOD_12_ORBIT = None
    else:
        # Fallback to defaults
        R1, R2, C, D, P, Q, LOG_OFFSET = 3.873, 11.746, 3.67e-07, 0.5517, 0.06659, 0.9026, 0.001
        EQUILIBRIUM_POINT = np.array([0.792107, 0.209010, 0.376449])
        PERIOD_12_ORBIT = None

    # Print header with loaded parameters
    print("\nIVES ECOLOGICAL MODEL - MORSE GRAPH ANALYSIS")
    print("Model: Midge-Algae-Detritus dynamics (Ives et al. 2008)")
    print(f"Parameters: R1={R1:.3f}, R2={R2:.3f}, C={C:.2e}, D={D:.4f}, P={P:.5f}, Q={Q:.3f}")
    print(f"Domain (log): {config.domain_bounds[0]} to {config.domain_bounds[1]}")
    print(f"Equilibrium (log): [{EQUILIBRIUM_POINT[0]:.4f}, {EQUILIBRIUM_POINT[1]:.4f}, {EQUILIBRIUM_POINT[2]:.4f}]")
    if PERIOD_12_ORBIT is not None:
        print(f"Period-12 orbit: {len(PERIOD_12_ORBIT)} points loaded from config")

    if args.force_recompute:
        print("\nForce recompute enabled - will ignore 3D CMGDB cache")

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

    result_3d, was_cached = load_or_compute_3d_morse_graph(
        config.map_func,
        config.domain_bounds,
        subdiv_min=config.subdiv_min,
        subdiv_max=config.subdiv_max,
        subdiv_init=config.subdiv_init,
        subdiv_limit=config.subdiv_limit,
        padding=config.padding,
        base_dir=base_output_dir,
        force_recompute=args.force_recompute,
        verbose=True,
        equilibria={'Equilibrium': EQUILIBRIUM_POINT},
        periodic_orbits={'Period-12': PERIOD_12_ORBIT} if PERIOD_12_ORBIT is not None else None,
        labels={'x': 'log(Midge)', 'y': 'log(Algae)', 'z': 'log(Detritus)'}
    )

    morse_graph_3d = result_3d['morse_graph']
    barycenters_3d = result_3d['barycenters']

    if was_cached:
        print("\n  ✓ Using cached 3D Morse graph from cmgdb_3d/")
    else:
        print("\n  ✓ Computed and saved 3D Morse graph to cmgdb_3d/")

    # Note: 3D visualizations are already saved in cmgdb_3d/results/:
    #   - morse_graph_3d.png (Morse graph diagram)
    #   - morse_sets_3d.png (3D scatter with barycenters)
    #   - morse_sets_proj_01.png, morse_sets_proj_02.png, morse_sets_proj_12.png (2D projections)
    print(f"  3D visualizations available in: {base_output_dir}/cmgdb_3d/results/")

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
    # Step 5: Compute Learned Latent Dynamics (2D Morse Graphs)
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 5: Computing Learned Latent Dynamics (2D Morse Graphs)")
    print("="*80)

    # Generate dense uniform grid in original 3D space (MORALS approach)
    print("\nGenerating dense uniform grid in original space...")

    # Use subdivision to create uniform grid (adjustable for memory/resolution trade-off)
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

    # Method 1: Domain-restricted with padding (MORALS default)
    print("\nMethod 1: Domain-restricted + Padding...")
    result_2d_padded = compute_morse_graph_2d_restricted(
        latent_dynamics,
        device,
        z_large_grid,
        latent_bounds.tolist(),
        subdiv_min=config.latent_subdiv_min,
        subdiv_max=config.latent_subdiv_max,
        subdiv_init=config.latent_subdiv_init,
        subdiv_limit=config.latent_subdiv_limit,
        include_neighbors=True,
        padding=True,
        verbose=True
    )

    morse_graph_2d_padded = result_2d_padded['morse_graph']

    # Method 2: Domain-restricted without padding
    print("\nMethod 2: Domain-restricted + No Padding...")
    result_2d_unpadded = compute_morse_graph_2d_restricted(
        latent_dynamics,
        device,
        z_large_grid,
        latent_bounds.tolist(),
        subdiv_min=config.latent_subdiv_min,
        subdiv_max=config.latent_subdiv_max,
        subdiv_init=config.latent_subdiv_init,
        subdiv_limit=config.latent_subdiv_limit,
        include_neighbors=True,
        padding=False,
        verbose=True
    )

    morse_graph_2d_unpadded = result_2d_unpadded['morse_graph']

    # ========================================================================
    # Step 6: Visualize Learned Latent Dynamics (2D)
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 6: Generating 2D Visualizations")
    print("="*80)

    # Padded Morse graph
    plot_morse_graph_diagram(
        morse_graph_2d_padded,
        output_path=f"{dirs['results']}/morse_graph_2d_padded.png",
        title="Ives Model - Learned Latent Dynamics (2D) - Padded"
    )

    # Project equilibrium and period-12 orbit to latent space
    with torch.no_grad():
        equilibrium_tensor = torch.FloatTensor(EQUILIBRIUM_POINT).unsqueeze(0).to(device)
        equilibrium_latent = encoder(equilibrium_tensor).cpu().numpy()[0]

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
        morse_graph=morse_graph_2d_padded,
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
        barycenter_size=6
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
        morse_graph=morse_graph_2d_padded,
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
        barycenter_size=6
    )

    # 6. Morse sets + E(Barycenters) + Equilibrium
    plot_latent_space_flexible(
        morse_graph=morse_graph_2d_padded,
        barycenters_latent=barycenters_latent,
        equilibrium_latent=equilibrium_latent,
        latent_bounds=latent_bounds.tolist(),
        output_path=f"{dirs['results']}/latent_morse_barycenters.png",
        title="Latent Space - Morse Sets + E(Barycenters) + Equilibrium",
        show_morse_sets=True,
        show_barycenters=True,
        show_equilibrium=True,
        barycenter_size=6
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
            morse_graph=morse_graph_2d_padded,
            period12_latent=period12_latent,
            latent_bounds=latent_bounds.tolist(),
            output_path=f"{dirs['results']}/latent_morse_period12.png",
            title="Latent Space - Morse Sets + E(Period-12 Orbit)",
            show_morse_sets=True,
            show_period12=True
        )

    print("  ✓ Saved 9 enhanced latent space visualizations")

    # Unpadded Morse graph
    plot_morse_graph_diagram(
        morse_graph_2d_unpadded,
        output_path=f"{dirs['results']}/morse_graph_2d_unpadded.png",
        title="Ives Model - Learned Latent Dynamics (2D) - Unpadded"
    )

    # ========================================================================
    # Step 6a: Comprehensive Morse Graph Comparison
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 6a: Comprehensive Morse Graph Comparison")
    print("="*80)

    comparison_stats = plot_morse_graph_comparison(
        morse_graph_3d,
        morse_graph_2d_padded,
        morse_graph_2d_unpadded,
        barycenters_3d,
        encoder,
        device,
        z_train,
        z_large_grid,
        latent_bounds.tolist(),
        config.domain_bounds,
        output_path=f"{dirs['results']}/morse_graph_comparison.png",
        title_prefix="Ives Model - ",
        equilibria={'Equilibrium': EQUILIBRIUM_POINT},
        equilibria_latent={'Equilibrium': equilibrium_latent},
        labels={'x': 'log(Midge)', 'y': 'log(Algae)', 'z': 'log(Detritus)'}
    )

    print(f"  3D Morse Sets: {comparison_stats['num_morse_sets_3d']} ({comparison_stats['num_edges_3d']} edges)")
    print(f"  2D Morse Sets (Padded): {comparison_stats['num_morse_sets_2d_data']} ({comparison_stats['num_edges_2d_data']} edges)")
    print(f"  2D Morse Sets (Unpadded): {comparison_stats['num_morse_sets_2d_restricted']} ({comparison_stats['num_edges_2d_restricted']} edges)")
    print("  ✓ Saved comprehensive comparison")

    # Create 2x2 clean comparison (Padded method)
    print("\n  Creating 2x2 comparison visualization (Padded method)...")
    plot_2x2_morse_comparison(
        morse_graph_3d,
        morse_graph_2d_padded,
        config.domain_bounds,
        latent_bounds.tolist(),
        encoder,
        device,
        z_train,
        output_path=f"{dirs['results']}/morse_2x2_comparison_padded.png",
        title_prefix="Ives - ",
        equilibria={'Equilibrium': EQUILIBRIUM_POINT},
        periodic_orbits={'Period-12': PERIOD_12_ORBIT} if PERIOD_12_ORBIT is not None else None,
        equilibria_latent={'Equilibrium': equilibrium_latent},
        labels={'x': 'log(Midge)', 'y': 'log(Algae)', 'z': 'log(Detritus)'}
    )
    print("  ✓ Saved 2x2 comparison (Padded method)")

    # Create 2x2 clean comparison (Unpadded method)
    print("  Creating 2x2 comparison visualization (Unpadded method)...")
    plot_2x2_morse_comparison(
        morse_graph_3d,
        morse_graph_2d_unpadded,
        config.domain_bounds,
        latent_bounds.tolist(),
        encoder,
        device,
        z_large_grid,
        output_path=f"{dirs['results']}/morse_2x2_comparison_unpadded.png",
        title_prefix="Ives - ",
        equilibria={'Equilibrium': EQUILIBRIUM_POINT},
        periodic_orbits={'Period-12': PERIOD_12_ORBIT} if PERIOD_12_ORBIT is not None else None,
        equilibria_latent={'Equilibrium': equilibrium_latent},
        labels={'x': 'log(Midge)', 'y': 'log(Algae)', 'z': 'log(Detritus)'}
    )
    print("  ✓ Saved 2x2 comparison (Unpadded method)")

    # ========================================================================
    # Step 6b: Preimage Classification Analysis
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 6b: Preimage Classification Analysis")
    print("="*80)

    # Analyze preimages for Padded method
    print("  Analyzing preimages for Padded method...")
    preimages_padded = plot_preimage_classification(
        morse_graph_2d_padded,
        encoder,
        decoder,
        device,
        X_large_grid,
        latent_bounds.tolist(),
        config.domain_bounds,
        subdiv_max=config.latent_subdiv_max,
        output_path=f"{dirs['results']}/preimage_classification_padded.png",
        title_prefix="Ives Model - ",
        method_name="Padded",
        labels={'x': 'log(Midge)', 'y': 'log(Algae)', 'z': 'log(Detritus)'},
        max_points_per_set=2000
    )
    print("  ✓ Saved preimage analysis (Padded method)")

    # Analyze preimages for Unpadded method
    print("  Analyzing preimages for Unpadded method...")
    preimages_unpadded = plot_preimage_classification(
        morse_graph_2d_unpadded,
        encoder,
        decoder,
        device,
        X_large_grid,
        latent_bounds.tolist(),
        config.domain_bounds,
        subdiv_max=config.latent_subdiv_max,
        output_path=f"{dirs['results']}/preimage_classification_unpadded.png",
        title_prefix="Ives Model - ",
        method_name="Unpadded",
        labels={'x': 'log(Midge)', 'y': 'log(Algae)', 'z': 'log(Detritus)'},
        max_points_per_set=2000
    )
    print("  ✓ Saved preimage analysis (Unpadded method)")

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
        'was_3d_cached': was_cached,

        # 2D Morse graphs
        'num_morse_sets_2d_padded': result_2d_padded['num_morse_sets'],
        'num_morse_sets_2d_unpadded': result_2d_unpadded['num_morse_sets'],

        # Training
        'training_time': training_result['training_time'],
        'final_train_loss': training_result['train_losses']['total'][-1],
        'final_val_loss': training_result['val_losses']['total'][-1],

        # Analysis
        'n_test_trajectories': n_test_trajectories,
        'n_trajectory_steps': n_trajectory_steps,

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
    print(f"  Run Number:                 {run_number}")
    print(f"  3D Morse Sets:              {results['num_morse_sets_3d']} {'(cached)' if was_cached else '(computed)'}")
    print(f"  2D Morse Sets (Padded):     {results['num_morse_sets_2d_padded']}")
    print(f"  2D Morse Sets (Unpadded):   {results['num_morse_sets_2d_unpadded']}")
    print(f"  Training Time:              {results['training_time']:.2f}s")
    print(f"  Final Train Loss:           {results['final_train_loss']:.6f}")
    print(f"  Final Val Loss:             {results['final_val_loss']:.6f}")
    print(f"\nOutput Structure:")
    print(f"  {base_output_dir}/")
    print(f"  ├── cmgdb_3d/                 # Cached 3D Morse graph (reusable)")
    print(f"  │   └── results/              # 3D visualizations (shared)")
    print(f"  └── run_{run_number:03d}/               # This run's results")
    print(f"      ├── config.yaml")
    print(f"      ├── models/")
    print(f"      ├── training_data/")
    print(f"      └── results/              # 2D and analysis visualizations")
    print(f"\n3D Visualizations (in cmgdb_3d/results/, shared across runs):")
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
    print(f"    - morse_graph_2d_data.png: 2D Morse graph (Data)")
    print(f"    - morse_graph_2d_restricted.png: 2D Morse graph (Restricted)")
    print(f"\n  Comprehensive Comparisons:")
    print(f"    - morse_graph_comparison.png: Side-by-side 3D vs 2D comparison (2x3 grid)")
    print(f"    - morse_2x2_comparison_data.png: Clean 2x2 comparison (Data method)")
    print(f"    - morse_2x2_comparison_restricted.png: Clean 2x2 comparison (Restricted method)")
    print(f"    - preimage_classification_data.png: Preimage analysis - Data method")
    print(f"    - preimage_classification_restricted.png: Preimage analysis - Restricted method")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
