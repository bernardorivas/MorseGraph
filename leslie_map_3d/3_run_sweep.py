#!/usr/bin/env python3
"""
Run Parameter Sweep for Leslie Map 3D Learning Experiments

This script runs multiple learning experiments with different hyperparameter configurations
to explore the effect of various settings on learned Morse graphs.

Usage:
    python 3_run_sweep.py                                # Run default experiment sweep
    python 3_run_sweep.py --sweep-name my_sweep          # Named sweep
    python 3_run_sweep.py --experiments 1,2,3            # Run specific experiments
"""

import argparse
import os
import json
import numpy as np
from datetime import datetime
from leslie_modules import Config, run_learning_experiment


# Define experimental configurations
EXPERIMENTS = {
    'baseline': {
        'description': 'Baseline configuration',
        'config': {}
    },
    'morals_full': {
        'description': 'Full MORALS method: shallow (1 layer), large batch (1024), heavy recon weight (500:1:1)',
        'config': {
            'ENCODER_ACTIVATION': 'tanh',
            'DECODER_ACTIVATION': 'sigmoid',
            'LATENT_DYNAMICS_ACTIVATION': 'tanh',
            'NUM_LAYERS': 1,
            'BATCH_SIZE': 1024,
            'W_RECON': 500.0,
            'W_DYN_RECON': 1.0,
            'W_DYN_CONS': 1.0
        }
    },
    'morals_shallow': {
        'description': 'MORALS activations + shallow network (1 layer only)',
        'config': {
            'ENCODER_ACTIVATION': 'tanh',
            'DECODER_ACTIVATION': 'sigmoid',
            'LATENT_DYNAMICS_ACTIVATION': 'tanh',
            'NUM_LAYERS': 1
        }
    },
    'morals_weights': {
        'description': 'MORALS activations + loss weights (500:1:1)',
        'config': {
            'ENCODER_ACTIVATION': 'tanh',
            'DECODER_ACTIVATION': 'sigmoid',
            'LATENT_DYNAMICS_ACTIVATION': 'tanh',
            'W_RECON': 500.0,
            'W_DYN_RECON': 1.0,
            'W_DYN_CONS': 1.0
        }
    },
    'morals_batch': {
        'description': 'MORALS activations + large batch size (1024)',
        'config': {
            'ENCODER_ACTIVATION': 'tanh',
            'DECODER_ACTIVATION': 'sigmoid',
            'LATENT_DYNAMICS_ACTIVATION': 'tanh',
            'BATCH_SIZE': 1024
        }
    },
    'morals': {
        'description': 'MORALS activations only (original config for backward compatibility)',
        'config': {
            'ENCODER_ACTIVATION': 'tanh',
            'DECODER_ACTIVATION': 'sigmoid',
            'LATENT_DYNAMICS_ACTIVATION': 'tanh'
        }
    },
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run parameter sweep for Leslie Map 3D learning',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--sweep-name', type=str, default=None,
                       help='Name for this sweep (default: auto-numbered)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Base output directory (default: experiments/learning/sweeps/)')
    parser.add_argument('--experiments', type=str, default=None,
                       help='Comma-separated list of experiment names to run (default: all)')
    parser.add_argument('--num-runs', type=int, default=1,
                       help='Number of runs per experiment (default: 1)')
    parser.add_argument('--progress-interval', type=int, default=100,
                       help='Print progress every N epochs (default: 100)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed progress messages')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set up output directory
    if args.output_dir is None:
        base_dir = Config.get_sweeps_dir()
    else:
        base_dir = args.output_dir

    os.makedirs(base_dir, exist_ok=True)

    # Auto-generate sweep name if needed
    if args.sweep_name is None:
        existing_sweeps = [d for d in os.listdir(base_dir) if d.startswith('sweep_')]
        sweep_num = len(existing_sweeps) + 1
        sweep_name = f"sweep_{sweep_num:03d}"
    else:
        sweep_name = args.sweep_name

    sweep_dir = os.path.join(base_dir, sweep_name)
    os.makedirs(sweep_dir, exist_ok=True)

    # Determine which experiments to run
    if args.experiments is None:
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        experiments_to_run = [e.strip() for e in args.experiments.split(',')]

    print("="*80)
    print(f"Parameter Sweep: {sweep_name}")
    print("="*80)
    print(f"Output directory: {sweep_dir}")
    print(f"Experiments to run: {', '.join(experiments_to_run)}")
    print(f"Runs per experiment: {args.num_runs}")
    print()

    # Run experiments
    all_results = []
    total_experiments = len(experiments_to_run) * args.num_runs
    current_experiment = 0

    for exp_name in experiments_to_run:
        if exp_name not in EXPERIMENTS:
            print(f"WARNING: Unknown experiment '{exp_name}', skipping...")
            continue

        exp_config = EXPERIMENTS[exp_name]
        print(f"\n{'='*80}")
        print(f"Experiment: {exp_name}")
        print(f"Description: {exp_config['description']}")
        print(f"{'='*80}")

        for run_num in range(args.num_runs):
            current_experiment += 1
            run_name = f"{exp_name}_{run_num+1:02d}"

            print(f"\n[{current_experiment}/{total_experiments}] Running {run_name}...")

            exp_output_dir = os.path.join(sweep_dir, exp_name)

            results = run_learning_experiment(
                run_name=f"{run_num+1:02d}",
                output_dir=exp_output_dir,
                config_overrides=exp_config['config'],
                verbose=not args.quiet,
                progress_interval=args.progress_interval
            )

            # Add experiment metadata
            results['experiment_name'] = exp_name
            results['experiment_description'] = exp_config['description']
            results['run_number'] = run_num + 1
            all_results.append(results)

    # Save sweep summary
    print(f"\n{'='*80}")
    print("Sweep Summary")
    print(f"{'='*80}")

    summary = {
        'sweep_name': sweep_name,
        'sweep_dir': sweep_dir,
        'timestamp': datetime.now().isoformat(),
        'total_experiments': current_experiment,
        'experiments_run': experiments_to_run,
        'num_runs_per_experiment': args.num_runs,
        'results': all_results
    }

    summary_path = os.path.join(sweep_dir, 'sweep_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Sweep completed!")
    print(f"  Total experiments: {current_experiment}")
    print(f"  Summary saved to: {summary_path}")
    print(f"{'='*80}\n")

    # Print quick statistics
    print("\nQuick Statistics:")
    print(f"{'Experiment':<25} {'Avg Train Loss':<15} {'Avg Val Loss':<15} {'Avg Morse Sets':<15}")
    print("-" * 70)

    for exp_name in experiments_to_run:
        exp_results = [r for r in all_results if r['experiment_name'] == exp_name]
        if exp_results:
            avg_train_loss = np.mean([r['final_train_loss'] for r in exp_results])
            avg_val_loss = np.mean([r['final_val_loss'] for r in exp_results])
            avg_morse_sets = np.mean([r['num_latent_full_morse_sets'] for r in exp_results])

            print(f"{exp_name:<25} {avg_train_loss:<15.6f} {avg_val_loss:<15.6f} {avg_morse_sets:<15.1f}")

    print()


if __name__ == "__main__":
    main()
