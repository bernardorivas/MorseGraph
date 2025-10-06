#!/usr/bin/env python3
"""
CMGDB Multiple Runs Test: Orchestrator for Multiple Experiment Runs

This orchestrator runs multiple experiments using the cmgdb_single_run module.
It handles aggregating results and generating summary statistics.
"""

import os
import json
import numpy as np
from datetime import datetime
import argparse
import sys
import importlib.util

# Import single run functionality
spec = importlib.util.spec_from_file_location("cmgdb_single_run",
                                              os.path.join(os.path.dirname(__file__), "cmgdb_single_run.py"))
cmgdb_single_run = importlib.util.module_from_spec(spec)
sys.modules["cmgdb_single_run"] = cmgdb_single_run
spec.loader.exec_module(cmgdb_single_run)

# Import from the loaded module
Config = cmgdb_single_run.Config
run_single_experiment_cmgdb = cmgdb_single_run.run_single_experiment_cmgdb

# ============================================================================
# Experiment Configurations
# ============================================================================

EXPERIMENTS = [
    {
        "name": "no_activation",
        "description": "10 runs with no output activation, balanced weights",
        "config": {
            "OUTPUT_ACTIVATION": None,
            "NUM_RUNS": 30,
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
        }
    },
    {
        "name": "tanh_activation",
        "description": "10 runs with tanh output activation and reduced reconstruction weight",
        "config": {
            "OUTPUT_ACTIVATION": "tanh",
            "NUM_RUNS": 10,
            "W_RECON": 0.1,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
        }
    },
    {
        "name": "sigmoid_activation",
        "description": "10 runs with sigmoid output activation and reduced reconstruction weight",
        "config": {
            "OUTPUT_ACTIVATION": "sigmoid",
            "NUM_RUNS": 10,
            "W_RECON": 0.1,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
        }
    }
]

#
# EXPERIMENTS = [
# # --- Baseline ---
#     {
#         "name": "baseline",
#         "description": "Baseline: 3 layers, 128 hidden, no output activation, balanced weights",
#         "config": {
#             "HIDDEN_DIM": 128,
#             "NUM_LAYERS": 3,
#             "OUTPUT_ACTIVATION": None,
#             "W_RECON": 1.0,
#             "W_DYN_RECON": 1.0,
#             "W_DYN_CONS": 1.0,
#             "NUM_RUNS": 5,
#         }
#     },
#     {
#         "name": "tanh_output",
#         "description": "Tanh output activation on encoder/decoder/dynamics",
#         "config": {
#             "HIDDEN_DIM": 128,
#             "NUM_LAYERS": 3,
#             "OUTPUT_ACTIVATION": "tanh",
#             "W_RECON": 1.0,
#             "W_DYN_RECON": 1.0,
#             "W_DYN_CONS": 1.0,
#             "NUM_RUNS": 5,
#         }
#     },
#     # ... more experiments ...
# ]
# """

# Base output directory - save in tests/learning
base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multiple_run')
os.makedirs(base_output_dir, exist_ok=True)

# ============================================================================
# Command-Line Argument Parsing
# ============================================================================

def parse_args():
    """Parse command-line arguments for test mode and experiment selection."""
    parser = argparse.ArgumentParser(description='CMGDB Multiple Runs Test with optional test mode')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode: single run, skip 3D Leslie computation, show progress')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Name of specific experiment to run (defaults to first experiment in test mode)')
    parser.add_argument('--progress-interval', type=int, default=50,
                        help='Print progress every N epochs (default: 50)')
    return parser.parse_args()

# ============================================================================
# Main Orchestrator
# ============================================================================

def run_all_experiments(args=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Handle test mode
    test_mode = args.test if args else False
    progress_interval = args.progress_interval if args else 50
    selected_experiment = args.experiment if args else None

    if test_mode:
        print(f"CMGDB Test Mode - Start time: {timestamp}")
        print("Test mode: Running single run, skipping 3D Leslie computation")
    else:
        print(f"CMGDB Multiple Runs Test - Start time: {timestamp}")

    all_experiment_summaries = []
    all_attractor_counts = {}

    # Filter experiments based on test mode and selection
    experiments_to_run = EXPERIMENTS
    if test_mode:
        if selected_experiment:
            # Find the specified experiment
            experiments_to_run = [exp for exp in EXPERIMENTS if exp['name'] == selected_experiment]
            if not experiments_to_run:
                print(f"Error: Experiment '{selected_experiment}' not found")
                print(f"Available experiments: {[exp['name'] for exp in EXPERIMENTS]}")
                return
        else:
            # Just run the first experiment
            experiments_to_run = [EXPERIMENTS[0]]

    for exp in experiments_to_run:
        print(f"\n{'='*80}")
        print(f"Running Experiment: {exp['name']}")
        print(f"Description: {exp['description']}")
        print(f"{'='*80}")

        results = []
        exp_attractor_counts = {}
        # In test mode, force NUM_RUNS to 1
        if test_mode:
            num_runs = 1
        else:
            num_runs = exp['config'].get('NUM_RUNS', Config.NUM_RUNS)

        for run_num in range(1, num_runs + 1):
            run_name = f"{exp['name']}_{run_num:03d}"
            exp_specific_output_dir = os.path.join(base_output_dir, exp['name'])

            # Run single experiment with minimal output unless in test mode
            print(f"  Run {run_num}/{num_runs}: {run_name}...", end='', flush=True)

            # Create a config for this specific run with a unique seed
            run_config_overrides = exp['config'].copy()
            run_config_overrides['RANDOM_SEED'] = run_num

            result = run_single_experiment_cmgdb(
                run_name,
                exp_specific_output_dir,
                config_overrides=run_config_overrides,
                verbose=test_mode,
                progress_interval=progress_interval
            )
            if not test_mode:
                print(" ✓")
            results.append(result)
            exp_attractor_counts[result['run_name']] = {
                'latent_full_attractors': result.get('latent_full_attractors', 'N/A'),
                'latent_restricted_attractors': result.get('latent_restricted_attractors', 'N/A')
            }

        # Save raw results for this experiment
        exp_summary_file = os.path.join(base_output_dir, f"experiment_{exp['name']}_results.json")
        with open(exp_summary_file, 'w') as f:
            json.dump({'experiment_name': exp['name'], 'results': results}, f, indent=2)
        print(f"\nSaved full results to {exp_summary_file}")

        all_attractor_counts[exp['name']] = exp_attractor_counts

        # --- Compute and Print Statistics for this Experiment ---
        recon_errors = [r['recon_error'] for r in results]
        dyn_errors = [r['dyn_error'] for r in results]
        cons_errors = [r['cons_error'] for r in results]
        val_losses = [r['final_val_loss'] for r in results]

        # Calculate stats
        stats = {
            'name': exp['name'],
            'recon_error': {'mean': np.mean(recon_errors), 'std': np.std(recon_errors)},
            'dyn_error': {'mean': np.mean(dyn_errors), 'std': np.std(dyn_errors)},
            'cons_error': {'mean': np.mean(cons_errors), 'std': np.std(cons_errors)},
            'final_val_loss': {'mean': np.mean(val_losses), 'std': np.std(val_losses)},
        }
        all_experiment_summaries.append(stats)

        print(f"\n--- Summary for Experiment: {exp['name']} ({num_runs} runs) ---")
        print(f"{'Metric':<20} {'Mean':<15} {'Std Dev':<15}")
        print("-" * 50)
        print(f"{'Reconstruction Err':<20} {stats['recon_error']['mean']:<15.6f} {stats['recon_error']['std']:<15.6f}")
        print(f"{'Dynamics Err':<20} {stats['dyn_error']['mean']:<15.6f} {stats['dyn_error']['std']:<15.6f}")
        print(f"{'Consistency Err':<20} {stats['cons_error']['mean']:<15.6f} {stats['cons_error']['std']:<15.6f}")
        print(f"{'Final Val Loss':<20} {stats['final_val_loss']['mean']:<15.6f} {stats['final_val_loss']['std']:<15.6f}")
        print("-" * 50)

    # --- Final Summary Table ---
    print(f"\n\n{'='*100}")
    print("Overall Experiment Comparison")
    print(f"{'='*100}")
    header = f"{'Experiment':<20} | {'Recon Err (μ ± σ)':<22} | {'Dyn Err (μ ± σ)':<20} | {'Cons Err (μ ± σ)':<20} | {'Val Loss (μ ± σ)':<20}"
    print(header)
    print("-" * len(header))

    for summary in all_experiment_summaries:
        recon_str = f"{summary['recon_error']['mean']:.4f} ± {summary['recon_error']['std']:.4f}"
        dyn_str = f"{summary['dyn_error']['mean']:.4f} ± {summary['dyn_error']['std']:.4f}"
        cons_str = f"{summary['cons_error']['mean']:.4f} ± {summary['cons_error']['std']:.4f}"
        loss_str = f"{summary['final_val_loss']['mean']:.4f} ± {summary['final_val_loss']['std']:.4f}"
        print(f"{summary['name']:<20} | {recon_str:<22} | {dyn_str:<20} | {cons_str:<20} | {loss_str:<20}")

    print("-" * len(header))

    # Save summary to a file
    summary_path = os.path.join(base_output_dir, "all_experiments_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_experiment_summaries, f, indent=2)
    print(f"\nSaved overall summary to {summary_path}")

    # Save attractor counts
    attractors_path = os.path.join(base_output_dir, "experiment_number_of_attractors.json")
    with open(attractors_path, 'w') as f:
        json.dump(all_attractor_counts, f, indent=2)
    print(f"Saved attractor counts to {attractors_path}")

    print(f"\nAll experiments completed. End time: {datetime.now().strftime('%Y%m%d_%H%M%S')}")

if __name__ == "__main__":
    args = parse_args()
    run_all_experiments(args)
