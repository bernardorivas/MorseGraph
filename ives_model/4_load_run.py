#!/usr/bin/env python3
"""
Load and Regenerate Visualizations for Existing Run

This script loads a previously trained model and regenerates all visualization
figures. Useful when training succeeded but visualization failed, or when you
want to regenerate figures with different settings.

Usage:
    python 4_load_run.py experiments/learning/single/run_001
    python 4_load_run.py experiments/learning/sweeps/sweep_001/baseline/run_01
    python 4_load_run.py experiments/learning/single/run_004 --force
"""

import argparse
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from ives_modules import Config
from ives_modules.plotting import (
    plot_training_losses,
    plot_morse_graph,
    plot_latent_transformation_analysis,
    plot_morse_graph_comparison,
    plot_preimage_classification
)
from ives_modules.learning import (
    compute_latent_bounds,
    compute_latent_morse_graph_cmgdb,
    compute_latent_morse_graph_domain_restricted,
    estimate_large_sample_size,
    generate_map_trajectory_data
)
from ives_modules.ground_truth import load_ground_truth_barycenters
from MorseGraph.models import Encoder, Decoder, LatentDynamics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Load and regenerate visualizations for existing run',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('run_path', type=str,
                       help='Path to run directory (e.g., experiments/learning/single/run_001)')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing visualizations')
    parser.add_argument('--skip-morse', action='store_true',
                       help='Skip Morse graph computation (faster, only regenerate existing data)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')

    return parser.parse_args()


def load_run_and_regenerate_plots(run_path, force=False, skip_morse=False, verbose=True):
    """
    Load a trained run and regenerate all visualizations.

    Args:
        run_path: Path to run directory
        force: If True, overwrite existing visualizations
        skip_morse: If True, skip Morse graph recomputation
        verbose: If True, print progress messages
    """
    # Validate run path
    if not os.path.exists(run_path):
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    # Define expected directories
    run_training_data_dir = os.path.join(run_path, "training_data")
    run_models_dir = os.path.join(run_path, "models")
    run_results_dir = os.path.join(run_path, "results")

    if not os.path.exists(run_models_dir):
        raise FileNotFoundError(f"Models directory not found: {run_models_dir}")

    os.makedirs(run_results_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Loading run from: {run_path}")
        print(f"{'='*80}")

    # Load settings (or use defaults if not found)
    settings_path = os.path.join(run_path, "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        config = settings['configuration']
        run_name = settings['run_info']['run_name']
        if verbose:
            print(f"Run name: {run_name}")
            print(f"Configuration loaded from settings.json")
    else:
        # Fall back to default config
        config = {k: v for k, v in vars(Config).items() if not k.startswith('__') and not callable(v)}
        run_name = os.path.basename(run_path)
        if verbose:
            print(f"⚠️  settings.json not found - using default configuration")
            print(f"Run name: {run_name}")

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"Device: {device}")

    # Load models
    if verbose:
        print("\nLoading trained models...")

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

    # Load model weights
    encoder.load_state_dict(torch.load(os.path.join(run_models_dir, "encoder.pt"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(run_models_dir, "decoder.pt"), map_location=device))
    latent_dynamics.load_state_dict(torch.load(os.path.join(run_models_dir, "latent_dynamics.pt"), map_location=device))

    encoder.eval()
    decoder.eval()
    latent_dynamics.eval()

    if verbose:
        print("  ✓ Models loaded successfully")

    # Load training data
    if verbose:
        print("\nLoading training data...")

    trajectory_data_path = os.path.join(run_training_data_dir, "trajectory_data.npz")
    if not os.path.exists(trajectory_data_path):
        raise FileNotFoundError(f"Training data not found: {trajectory_data_path}")

    data = np.load(trajectory_data_path)
    x_t = data['x_t']

    # Reconstruct train/val split (80/20)
    n_total = len(x_t)
    n_train = int(0.8 * n_total)

    x_train = x_t[:n_train]
    x_val = x_t[n_train:]

    if verbose:
        print(f"  Total samples: {len(x_t)}")
        print(f"  Train samples: {len(x_train)}")
        print(f"  Val samples:   {len(x_val)}")

    # Load training history for loss plots
    training_history_path = os.path.join(run_results_dir, "training_history.json")
    if os.path.exists(training_history_path):
        with open(training_history_path, 'r') as f:
            training_history = json.load(f)
        train_losses = training_history['train_losses']
        val_losses = training_history['val_losses']

        if verbose:
            print(f"  Training history loaded ({training_history['total_epochs']} epochs)")
    else:
        if verbose:
            print("  ⚠️  Training history not found - loss plots will be skipped")
        train_losses = None
        val_losses = None

    # Load or compute latent embeddings
    if verbose:
        print("\nComputing latent embeddings...")

    x_train_tensor = torch.FloatTensor(x_train).to(device)
    x_val_tensor = torch.FloatTensor(x_val).to(device)
    x_full_tensor = torch.FloatTensor(x_t).to(device)

    with torch.no_grad():
        z_train = encoder(x_train_tensor).cpu().numpy()
        z_val = encoder(x_val_tensor).cpu().numpy()

    latent_bounds = compute_latent_bounds(z_train, padding_factor=config['LATENT_BOUNDS_PADDING'])

    if verbose:
        print(f"  Latent bounds: {latent_bounds.tolist()}")

    # Load ground truth data
    gt_dir = Config.get_ground_truth_dir()
    ground_truth_graph_img_path = os.path.join(gt_dir, "morse_graph.png")
    ground_truth_barycenters = load_ground_truth_barycenters()

    if not os.path.exists(ground_truth_graph_img_path) and verbose:
        print("\n⚠️  WARNING: Ground truth Morse graph image not found!")
        print(f"   Expected at: {ground_truth_graph_img_path}")

    # Compute or load Morse graphs
    if skip_morse:
        if verbose:
            print("\nSkipping Morse graph computation (--skip-morse flag)")
        latent_morse_graph_full = None
        latent_morse_graph_train = None
        latent_morse_graph_val = None
        latent_morse_graph_large = None
    else:
        if verbose:
            print("\nComputing latent Morse graphs...")
            print("  This may take several minutes...")

        # Variant 1: Full latent dynamics (unrestricted)
        latent_morse_graph_full, _ = compute_latent_morse_graph_cmgdb(
            latent_dynamics, device, latent_bounds.tolist(), config
        )
        if verbose:
            print(f"  ✓ Full latent MG: {latent_morse_graph_full.num_vertices()} Morse sets")

        # Variant 2: Train-restricted (domain-restricted to E(X_train), no neighbors)
        latent_morse_graph_train, _ = compute_latent_morse_graph_domain_restricted(
            latent_dynamics, device, z_train, latent_bounds.tolist(), config, include_neighbors=False
        )
        if verbose:
            print(f"  ✓ Train-restricted MG: {latent_morse_graph_train.num_vertices()} Morse sets")

        # Variant 3: Val-restricted (domain-restricted to E(X_val), no neighbors)
        latent_morse_graph_val, _ = compute_latent_morse_graph_domain_restricted(
            latent_dynamics, device, z_val, latent_bounds.tolist(), config, include_neighbors=False
        )
        if verbose:
            print(f"  ✓ Val-restricted MG: {latent_morse_graph_val.num_vertices()} Morse sets")

        # Variant 4: Large sample with neighbors
        large_sample_size = estimate_large_sample_size(latent_bounds.tolist(), config, target_points_per_box=2)
        if verbose:
            print(f"  Generating large sample ({large_sample_size} points)...")

        x_large_sample = generate_map_trajectory_data(
            n_trajectories=large_sample_size // config['N_POINTS'],
            n_points=config['N_POINTS'],
            skip_initial=config['SKIP_INITIAL'],
            random_seed=None,  # Different sample each time
            config=config
        )

        with torch.no_grad():
            z_large = encoder(torch.FloatTensor(x_large_sample).to(device)).cpu().numpy()

        latent_morse_graph_large, _ = compute_latent_morse_graph_domain_restricted(
            latent_dynamics, device, z_large, latent_bounds.tolist(), config, include_neighbors=True
        )
        if verbose:
            print(f"  ✓ Large (+neighbors) MG: {latent_morse_graph_large.num_vertices()} Morse sets")

    # Generate visualizations
    if verbose:
        print("\nRegenerating visualizations...")

    # 1. Training loss curves
    if train_losses is not None:
        loss_plot_path = os.path.join(run_results_dir, "training_losses.png")
        if force or not os.path.exists(loss_plot_path):
            plot_training_losses(train_losses, val_losses, loss_plot_path)
            if verbose:
                print(f"  ✓ Training losses plot")
        else:
            if verbose:
                print(f"  ⊘ Skipping training losses (already exists, use --force to overwrite)")

    # 2. Individual Morse graph plots
    if not skip_morse:
        morse_graph_paths = [
            ("morse_graph_latent_full.png", latent_morse_graph_full, "Latent Dynamics - Full Morse Graph"),
            ("morse_graph_latent_train.png", latent_morse_graph_train, "Train-restricted Morse Graph"),
            ("morse_graph_latent_val.png", latent_morse_graph_val, "Val-restricted Morse Graph"),
            ("morse_graph_latent_large.png", latent_morse_graph_large, "Large (+neighbors) Morse Graph")
        ]

        for filename, mg, title in morse_graph_paths:
            plot_path = os.path.join(run_results_dir, filename)
            if force or not os.path.exists(plot_path):
                if mg is not None:
                    plot_morse_graph(mg, plot_path, title=title, cmap=plt.cm.viridis)
                    if verbose:
                        print(f"  ✓ {filename}")
            else:
                if verbose:
                    print(f"  ⊘ Skipping {filename} (already exists)")

    # 3. Latent transformation analysis
    transform_plot_path = os.path.join(run_results_dir, "latent_transformation_analysis.png")
    if force or not os.path.exists(transform_plot_path):
        plot_latent_transformation_analysis(
            x_t, encoder, decoder, device, latent_bounds.tolist(), transform_plot_path
        )
        if verbose:
            print(f"  ✓ Latent transformation analysis")
    else:
        if verbose:
            print(f"  ⊘ Skipping latent transformation analysis (already exists)")

    # 4. Morse graph comparison (5 variants)
    if not skip_morse:
        comparison_variants = ['', '_grey', '_clean', '_minimal', '_no_overlay']
        all_exist = all(
            os.path.exists(os.path.join(run_results_dir, f"morse_graph_comparison{v}.png"))
            for v in comparison_variants
        )

        if force or not all_exist:
            plot_morse_graph_comparison(
                ground_truth_graph_img_path,
                ground_truth_barycenters,
                latent_morse_graph_full,
                latent_morse_graph_train,
                latent_morse_graph_val,
                latent_morse_graph_large,
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
                print(f"  ✓ Morse graph comparison (5 variants)")
        else:
            if verbose:
                print(f"  ⊘ Skipping Morse graph comparison (already exists)")

    # 5. Preimage classification
    if not skip_morse:
        preimage_plot_path = os.path.join(run_results_dir, "preimage_classification.png")
        if force or not os.path.exists(preimage_plot_path):
            plot_preimage_classification(
                latent_morse_graph_full,
                latent_morse_graph_train,
                latent_morse_graph_val,
                latent_morse_graph_large,
                encoder,
                decoder,
                device,
                x_t,
                z_train,
                z_val,
                latent_bounds.tolist(),
                config['DOMAIN_BOUNDS'],
                preimage_plot_path
            )
            if verbose:
                print(f"  ✓ Preimage classification")
        else:
            if verbose:
                print(f"  ⊘ Skipping preimage classification (already exists)")

    if verbose:
        print(f"\n{'='*80}")
        print(f"Visualization regeneration complete!")
        print(f"Output directory: {run_results_dir}")
        print(f"{'='*80}\n")


def main():
    """Main entry point."""
    args = parse_args()

    try:
        load_run_and_regenerate_plots(
            args.run_path,
            force=args.force,
            skip_morse=args.skip_morse,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
