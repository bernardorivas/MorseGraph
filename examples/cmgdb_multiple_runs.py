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
        "name": "baseline",
        "description": "Baseline: no activation, balanced weights",
        "config": {
            "OUTPUT_ACTIVATION": None,
            "NUM_RUNS": 30,
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
        }
    },
    {
        "name": "morals_baseline",
        "description": "MORALS paper config: E=tanh, G=sigmoid, D=tanh",
        "config": {
            "ENCODER_ACTIVATION": "tanh",
            "LATENT_DYNAMICS_ACTIVATION": "sigmoid",
            "DECODER_ACTIVATION": "tanh",
            "NUM_RUNS": 30,
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
        }
    },
    {
        "name": "tanh_all",
        "description": "Tanh activation on all networks",
        "config": {
            "OUTPUT_ACTIVATION": "tanh",
            "NUM_RUNS": 10,
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
        }
    },
    {
        "name": "sigmoid_all",
        "description": "Sigmoid activation on all networks",
        "config": {
            "OUTPUT_ACTIVATION": "sigmoid",
            "NUM_RUNS": 10,
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
        }
    },
    # Loss weight ratio experiments
    {
        "name": "recon_low",
        "description": "Low reconstruction weight (W_RECON=0.1)",
        "config": {
            "NUM_RUNS": 10,
            "W_RECON": 0.1,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
        }
    },
    {
        "name": "recon_high",
        "description": "High reconstruction weight (W_RECON=10.0)",
        "config": {
            "NUM_RUNS": 10,
            "W_RECON": 10.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
        }
    },
    {
        "name": "cons_low",
        "description": "Low consistency weight (W_DYN_CONS=0.1) - test for collapse",
        "config": {
            "NUM_RUNS": 10,
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 0.1,
        }
    },
    {
        "name": "cons_high",
        "description": "High consistency weight (W_DYN_CONS=10.0)",
        "config": {
            "NUM_RUNS": 10,
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 10.0,
        }
    },
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
# Visualization Functions
# ============================================================================

def plot_error_topology_correlations(all_experiment_summaries, experiments_results, output_dir):
    """
    Create scatter plots showing correlation between training errors and topological metrics.

    This visualization helps identify which aspects of topological accuracy are most
    correlated with the consistency loss, directly testing whether low consistency
    error corresponds to better topological fidelity.

    Args:
        all_experiment_summaries: List of summary dicts with stats for each experiment
        experiments_results: Dict mapping experiment names to lists of individual run results
        output_dir: Directory to save correlation plots
    """
    # Collect all runs across all experiments
    all_cons_errors = []
    all_node_diffs = []
    all_attr_diffs = []
    all_conn_diffs = []
    all_train_val_divs = []
    experiment_labels = []

    for exp_name, results in experiments_results.items():
        for result in results:
            all_cons_errors.append(result['cons_error'])
            all_node_diffs.append(result['topology_similarity_full']['node_diff'])
            all_attr_diffs.append(result['topology_similarity_full']['attractor_diff'])
            all_conn_diffs.append(result['topology_similarity_full']['connection_diff'])
            all_train_val_divs.append(result['train_val_divergence']['node_divergence'])
            experiment_labels.append(exp_name)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Consistency Error vs Topological Metrics Correlation', fontsize=16, y=0.995)

    # Plot 1: Consistency Error vs Node Diff
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(all_cons_errors, all_node_diffs, alpha=0.6, s=50)
    ax1.set_xlabel('Consistency Error (GE - fE)', fontsize=11)
    ax1.set_ylabel('Node Difference vs Ground Truth', fontsize=11)
    ax1.set_title('Node Topology Correlation')
    ax1.grid(True, alpha=0.3)
    # Add correlation coefficient
    corr1 = np.corrcoef(all_cons_errors, all_node_diffs)[0, 1]
    ax1.text(0.05, 0.95, f'ρ = {corr1:.3f}', transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Consistency Error vs Attractor Diff
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(all_cons_errors, all_attr_diffs, alpha=0.6, s=50, c='orange')
    ax2.set_xlabel('Consistency Error (GE - fE)', fontsize=11)
    ax2.set_ylabel('Attractor Difference vs Ground Truth', fontsize=11)
    ax2.set_title('Attractor Topology Correlation')
    ax2.grid(True, alpha=0.3)
    corr2 = np.corrcoef(all_cons_errors, all_attr_diffs)[0, 1]
    ax2.text(0.05, 0.95, f'ρ = {corr2:.3f}', transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Consistency Error vs Connection Diff
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(all_cons_errors, all_conn_diffs, alpha=0.6, s=50, c='green')
    ax3.set_xlabel('Consistency Error (GE - fE)', fontsize=11)
    ax3.set_ylabel('Connection Difference vs Ground Truth', fontsize=11)
    ax3.set_title('Edge Topology Correlation')
    ax3.grid(True, alpha=0.3)
    corr3 = np.corrcoef(all_cons_errors, all_conn_diffs)[0, 1]
    ax3.text(0.05, 0.95, f'ρ = {corr3:.3f}', transform=ax3.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Consistency Error vs Train-Val Divergence
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(all_cons_errors, all_train_val_divs, alpha=0.6, s=50, c='red')
    ax4.set_xlabel('Consistency Error (GE - fE)', fontsize=11)
    ax4.set_ylabel('Train-Val Node Divergence', fontsize=11)
    ax4.set_title('Overfitting Correlation')
    ax4.grid(True, alpha=0.3)
    corr4 = np.corrcoef(all_cons_errors, all_train_val_divs)[0, 1]
    ax4.text(0.05, 0.95, f'ρ = {corr4:.3f}', transform=ax4.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    correlation_plot_path = os.path.join(output_dir, "error_topology_correlations.png")
    plt.savefig(correlation_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved correlation plots to {correlation_plot_path}")

    # Print correlation summary
    print(f"\n{'='*60}")
    print("Error-Topology Correlation Summary")
    print(f"{'='*60}")
    print(f"  Consistency Error vs Node Diff:        ρ = {corr1:.3f}")
    print(f"  Consistency Error vs Attractor Diff:   ρ = {corr2:.3f}")
    print(f"  Consistency Error vs Connection Diff:  ρ = {corr3:.3f}")
    print(f"  Consistency Error vs Train-Val Div:    ρ = {corr4:.3f}")
    print(f"{'='*60}")

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
    all_experiments_results = {}  # For correlation plots

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
        all_experiments_results[exp['name']] = results  # Store for correlation plots

        # --- Compute and Print Statistics for this Experiment ---
        recon_errors = [r['recon_error'] for r in results]
        dyn_errors = [r['dyn_error'] for r in results]
        cons_errors = [r['cons_error'] for r in results]
        val_losses = [r['final_val_loss'] for r in results]

        # Topological similarity statistics
        topo_full_node_diff = [r['topology_similarity_full']['node_diff'] for r in results]
        topo_full_attr_diff = [r['topology_similarity_full']['attractor_diff'] for r in results]
        topo_full_conn_diff = [r['topology_similarity_full']['connection_diff'] for r in results]
        topo_train_node_diff = [r['topology_similarity_train']['node_diff'] for r in results]
        topo_val_node_diff = [r['topology_similarity_val']['node_diff'] for r in results]

        # Train-val divergence statistics
        train_val_node_div = [r['train_val_divergence']['node_divergence'] for r in results]
        train_val_edge_div = [r['train_val_divergence']['edge_divergence'] for r in results]

        # Calculate stats
        stats = {
            'name': exp['name'],
            'recon_error': {'mean': np.mean(recon_errors), 'std': np.std(recon_errors)},
            'dyn_error': {'mean': np.mean(dyn_errors), 'std': np.std(dyn_errors)},
            'cons_error': {'mean': np.mean(cons_errors), 'std': np.std(cons_errors)},
            'final_val_loss': {'mean': np.mean(val_losses), 'std': np.std(val_losses)},
            'topology': {
                'full_node_diff': {'mean': np.mean(topo_full_node_diff), 'std': np.std(topo_full_node_diff)},
                'full_attractor_diff': {'mean': np.mean(topo_full_attr_diff), 'std': np.std(topo_full_attr_diff)},
                'full_connection_diff': {'mean': np.mean(topo_full_conn_diff), 'std': np.std(topo_full_conn_diff)},
                'train_node_diff': {'mean': np.mean(topo_train_node_diff), 'std': np.std(topo_train_node_diff)},
                'val_node_diff': {'mean': np.mean(topo_val_node_diff), 'std': np.std(topo_val_node_diff)},
                'train_val_node_div': {'mean': np.mean(train_val_node_div), 'std': np.std(train_val_node_div)},
                'train_val_edge_div': {'mean': np.mean(train_val_edge_div), 'std': np.std(train_val_edge_div)},
            }
        }
        all_experiment_summaries.append(stats)

        print(f"\n--- Summary for Experiment: {exp['name']} ({num_runs} runs) ---")
        print(f"{'Metric':<30} {'Mean':<15} {'Std Dev':<15}")
        print("-" * 60)
        print(f"{'Reconstruction Err':<30} {stats['recon_error']['mean']:<15.6f} {stats['recon_error']['std']:<15.6f}")
        print(f"{'Dynamics Err':<30} {stats['dyn_error']['mean']:<15.6f} {stats['dyn_error']['std']:<15.6f}")
        print(f"{'Consistency Err':<30} {stats['cons_error']['mean']:<15.6f} {stats['cons_error']['std']:<15.6f}")
        print(f"{'Final Val Loss':<30} {stats['final_val_loss']['mean']:<15.6f} {stats['final_val_loss']['std']:<15.6f}")
        print(f"\n{'Topological Metrics:':<30}")
        print(f"{'  Full Node Diff':<30} {stats['topology']['full_node_diff']['mean']:<15.4f} {stats['topology']['full_node_diff']['std']:<15.4f}")
        print(f"{'  Full Attractor Diff':<30} {stats['topology']['full_attractor_diff']['mean']:<15.4f} {stats['topology']['full_attractor_diff']['std']:<15.4f}")
        print(f"{'  Full Connection Diff':<30} {stats['topology']['full_connection_diff']['mean']:<15.4f} {stats['topology']['full_connection_diff']['std']:<15.4f}")
        print(f"{'  Train Node Diff':<30} {stats['topology']['train_node_diff']['mean']:<15.4f} {stats['topology']['train_node_diff']['std']:<15.4f}")
        print(f"{'  Val Node Diff':<30} {stats['topology']['val_node_diff']['mean']:<15.4f} {stats['topology']['val_node_diff']['std']:<15.4f}")
        print(f"{'  Train-Val Node Divergence':<30} {stats['topology']['train_val_node_div']['mean']:<15.4f} {stats['topology']['train_val_node_div']['std']:<15.4f}")
        print(f"{'  Train-Val Edge Divergence':<30} {stats['topology']['train_val_edge_div']['mean']:<15.4f} {stats['topology']['train_val_edge_div']['std']:<15.4f}")
        print("-" * 60)

    # --- Final Summary Table ---
    print(f"\n\n{'='*140}")
    print("Overall Experiment Comparison - Training Errors")
    print(f"{'='*140}")
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

    # Topological metrics table
    print(f"\n\n{'='*140}")
    print("Overall Experiment Comparison - Topological Metrics")
    print(f"{'='*140}")
    topo_header = f"{'Experiment':<20} | {'Node Diff (μ ± σ)':<22} | {'Attr Diff (μ ± σ)':<22} | {'Conn Diff (μ ± σ)':<22} | {'Train-Val Div (μ ± σ)':<25}"
    print(topo_header)
    print("-" * len(topo_header))

    for summary in all_experiment_summaries:
        node_str = f"{summary['topology']['full_node_diff']['mean']:.2f} ± {summary['topology']['full_node_diff']['std']:.2f}"
        attr_str = f"{summary['topology']['full_attractor_diff']['mean']:.2f} ± {summary['topology']['full_attractor_diff']['std']:.2f}"
        conn_str = f"{summary['topology']['full_connection_diff']['mean']:.2f} ± {summary['topology']['full_connection_diff']['std']:.2f}"
        div_str = f"{summary['topology']['train_val_node_div']['mean']:.2f} ± {summary['topology']['train_val_node_div']['std']:.2f}"
        print(f"{summary['name']:<20} | {node_str:<22} | {attr_str:<22} | {conn_str:<22} | {div_str:<25}")

    print("-" * len(topo_header))

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

    # Generate correlation plots (skip in test mode or if only 1 experiment)
    if not test_mode and len(all_experiments_results) > 0:
        total_runs = sum(len(results) for results in all_experiments_results.values())
        if total_runs > 1:  # Need at least 2 runs for correlation
            print(f"\n{'='*60}")
            print("Generating Error-Topology Correlation Plots...")
            print(f"{'='*60}")
            plot_error_topology_correlations(all_experiment_summaries, all_experiments_results, base_output_dir)

    print(f"\nAll experiments completed. End time: {datetime.now().strftime('%Y%m%d_%H%M%S')}")

if __name__ == "__main__":
    args = parse_args()
    run_all_experiments(args)
