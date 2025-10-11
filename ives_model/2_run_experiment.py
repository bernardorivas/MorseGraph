#!/usr/bin/env python3
"""
Run Single Learning Experiment for Ives Ecological Model

This script runs a single autoencoder + latent dynamics learning experiment
for the Ives midge-algae-detritus ecological model.

Usage:
    python 2_run_experiment.py                           # Use defaults
    python 2_run_experiment.py --name exp_001            # Named run
    python 2_run_experiment.py --epochs 2000 --hidden-dim 64  # Custom hyperparameters
    python 2_run_experiment.py --output-dir ./my_experiments  # Custom output
"""

import argparse
from ives_modules import Config, run_learning_experiment


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run single learning experiment for Ives Ecological Model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--name', type=str, default=None,
                       help='Name for this run (default: auto-numbered)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: experiments/learning/single/)')

    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=None,
                       help=f'Hidden dimension (default: {Config.HIDDEN_DIM})')
    parser.add_argument('--num-layers', type=int, default=None,
                       help=f'Number of layers (default: {Config.NUM_LAYERS})')
    parser.add_argument('--activation', type=str, default=None,
                       choices=['None', 'tanh', 'sigmoid'],
                       help=f'Output activation (default: {Config.OUTPUT_ACTIVATION})')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help=f'Number of epochs (default: {Config.NUM_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                       help=f'Batch size (default: {Config.BATCH_SIZE})')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help=f'Learning rate (default: {Config.LEARNING_RATE})')

    # Data generation
    parser.add_argument('--n-trajectories', type=int, default=None,
                       help=f'Number of trajectories (default: {Config.N_TRAJECTORIES})')
    parser.add_argument('--random-seed', type=int, default=None,
                       help=f'Random seed (default: {Config.RANDOM_SEED})')

    # Progress
    parser.add_argument('--progress-interval', type=int, default=50,
                       help='Print progress every N epochs (default: 50)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build config overrides
    config_overrides = {}

    if args.hidden_dim is not None:
        config_overrides['HIDDEN_DIM'] = args.hidden_dim
    if args.num_layers is not None:
        config_overrides['NUM_LAYERS'] = args.num_layers
    if args.activation is not None:
        config_overrides['OUTPUT_ACTIVATION'] = None if args.activation == 'None' else args.activation

    if args.epochs is not None:
        config_overrides['NUM_EPOCHS'] = args.epochs
    if args.batch_size is not None:
        config_overrides['BATCH_SIZE'] = args.batch_size
    if args.learning_rate is not None:
        config_overrides['LEARNING_RATE'] = args.learning_rate

    if args.n_trajectories is not None:
        config_overrides['N_TRAJECTORIES'] = args.n_trajectories
    if args.random_seed is not None:
        config_overrides['RANDOM_SEED'] = args.random_seed

    # Run experiment
    results = run_learning_experiment(
        run_name=args.name,
        output_dir=args.output_dir,
        config_overrides=config_overrides,
        verbose=not args.quiet,
        progress_interval=args.progress_interval
    )

    if not args.quiet:
        print("\nExperiment completed successfully!")
        print(f"Run name: {results['run_name']}")


if __name__ == "__main__":
    main()
