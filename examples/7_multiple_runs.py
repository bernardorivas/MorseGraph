#!/usr/bin/env python3
"""
MorseGraph Example 7: Multiple Experiment Runner - Architecture & Hyperparameter Study

This script orchestrates multiple runs of the Leslie map learned dynamics experiment
(example 6) with different model architectures and hyperparameters to systematically
compare their performance.

Experiments test variations in:
- Output activation functions (None, Tanh, Sigmoid)
- Network depth (2, 3, 4 layers)
- Network width (64, 128, 256 hidden units)
- Loss weight combinations for the three loss terms
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import importlib.util
import sys

# Import example 6's main function
# We need to import it as a module to call its main() function
example_6_path = os.path.join(os.path.dirname(__file__), '6_map_learned_dynamics.py')
spec = importlib.util.spec_from_file_location("example_6", example_6_path)
example_6 = importlib.util.module_from_spec(spec)
sys.modules['example_6'] = example_6
spec.loader.exec_module(example_6)

# Base output directory
base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'leslie_experiments')
os.makedirs(base_output_dir, exist_ok=True)

# ============================================================================
# Experiment Grid
# ============================================================================

EXPERIMENTS = [
    # --- Baseline ---
    {
        "name": "baseline",
        "description": "Baseline: 3 layers, 128 hidden, no output activation, balanced weights",
        "config": {
            "HIDDEN_DIM": 128,
            "NUM_LAYERS": 3,
            "OUTPUT_ACTIVATION": None,
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
            "NUM_RUNS": 50,  # 50 runs per experiment
        }
    },

    # --- Output Activation Variations ---
    {
        "name": "tanh_output",
        "description": "Tanh output activation on encoder/decoder/dynamics",
        "config": {
            "HIDDEN_DIM": 128,
            "NUM_LAYERS": 3,
            "OUTPUT_ACTIVATION": "tanh",
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
            "NUM_RUNS": 50,
        }
    },
    {
        "name": "sigmoid_output",
        "description": "Sigmoid output activation on encoder/decoder/dynamics",
        "config": {
            "HIDDEN_DIM": 128,
            "NUM_LAYERS": 3,
            "OUTPUT_ACTIVATION": "sigmoid",
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
            "NUM_RUNS": 50,
        }
    },

    # --- Network Depth Variations ---
    {
        "name": "shallow_2layer",
        "description": "Shallow network: 2 layers",
        "config": {
            "HIDDEN_DIM": 128,
            "NUM_LAYERS": 2,
            "OUTPUT_ACTIVATION": None,
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
            "NUM_RUNS": 50,
        }
    },
    {
        "name": "deep_4layer",
        "description": "Deep network: 4 layers",
        "config": {
            "HIDDEN_DIM": 128,
            "NUM_LAYERS": 4,
            "OUTPUT_ACTIVATION": None,
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
            "NUM_RUNS": 50,
        }
    },

    # --- Network Width Variations ---
    {
        "name": "narrow_64",
        "description": "Narrow network: 64 hidden units",
        "config": {
            "HIDDEN_DIM": 64,
            "NUM_LAYERS": 3,
            "OUTPUT_ACTIVATION": None,
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
            "NUM_RUNS": 50,
        }
    },
    {
        "name": "wide_256",
        "description": "Wide network: 256 hidden units",
        "config": {
            "HIDDEN_DIM": 256,
            "NUM_LAYERS": 3,
            "OUTPUT_ACTIVATION": None,
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
            "NUM_RUNS": 50,
        }
    },

    # --- Loss Weight Variations ---
    {
        "name": "recon_heavy",
        "description": "Reconstruction-heavy: (100, 1, 1)",
        "config": {
            "HIDDEN_DIM": 128,
            "NUM_LAYERS": 3,
            "OUTPUT_ACTIVATION": None,
            "W_RECON": 100.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 1.0,
            "NUM_RUNS": 50,
        }
    },
    {
        "name": "dyn_recon_heavy",
        "description": "Dynamics reconstruction-heavy: (1, 100, 1)",
        "config": {
            "HIDDEN_DIM": 128,
            "NUM_LAYERS": 3,
            "OUTPUT_ACTIVATION": None,
            "W_RECON": 1.0,
            "W_DYN_RECON": 100.0,
            "W_DYN_CONS": 1.0,
            "NUM_RUNS": 50,
        }
    },
    {
        "name": "dyn_cons_heavy",
        "description": "Dynamics consistency-heavy: (1, 1, 100)",
        "config": {
            "HIDDEN_DIM": 128,
            "NUM_LAYERS": 3,
            "OUTPUT_ACTIVATION": None,
            "W_RECON": 1.0,
            "W_DYN_RECON": 1.0,
            "W_DYN_CONS": 100.0,
            "NUM_RUNS": 50,
        }
    },
    {
        "name": "balanced_10",
        "description": "Balanced emphasis on all terms: (10, 10, 1)",
        "config": {
            "HIDDEN_DIM": 128,
            "NUM_LAYERS": 3,
            "OUTPUT_ACTIVATION": None,
            "W_RECON": 10.0,
            "W_DYN_RECON": 10.0,
            "W_DYN_CONS": 1.0,
            "NUM_RUNS": 50,
        }
    },
]

# ============================================================================
# Main Orchestrator
# ============================================================================

def run_all_experiments():
    """Run all experiments in the grid."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("="*80)
    print("MorseGraph Example 7: Multiple Experiment Runner")
    print("="*80)
    print(f"Start time: {timestamp}")
    print(f"Number of experiments: {len(EXPERIMENTS)}")
    print(f"Total runs: {sum(exp['config']['NUM_RUNS'] for exp in EXPERIMENTS)}")
    print("="*80)

    all_experiment_results = []

    for exp_idx, experiment in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {exp_idx}/{len(EXPERIMENTS)}: {experiment['name']}")
        print(f"{'='*80}")

        # Create experiment-specific directory
        exp_output_dir = os.path.join(base_output_dir, f"experiment_{experiment['name']}")
        exp_data_dir = os.path.join(base_output_dir, "data")  # Shared data directory
        os.makedirs(exp_output_dir, exist_ok=True)
        os.makedirs(exp_data_dir, exist_ok=True)

        # Run experiment
        try:
            results = example_6.main(
                config_overrides=experiment['config'],
                custom_output_dir=exp_output_dir,
                custom_data_dir=exp_data_dir
            )

            # Store results with experiment metadata
            experiment_summary = {
                'experiment_name': experiment['name'],
                'description': experiment['description'],
                'config': experiment['config'],
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            all_experiment_results.append(experiment_summary)

            # Save individual experiment results
            results_file = os.path.join(exp_output_dir, "experiment_results.json")
            with open(results_file, 'w') as f:
                json.dump(experiment_summary, f, indent=2)

            # Print summary stats for this experiment
            recon_errors = [r['recon_error'] for r in results]
            dyn_errors = [r['dyn_error'] for r in results]
            cons_errors = [r['cons_error'] for r in results]
            print(f"\nResults: Recon={np.mean(recon_errors):.4f}±{np.std(recon_errors):.4f}, "
                  f"Dyn={np.mean(dyn_errors):.4f}±{np.std(dyn_errors):.4f}, "
                  f"Cons={np.mean(cons_errors):.4f}±{np.std(cons_errors):.4f}")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Save aggregated results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    summary_file = os.path.join(base_output_dir, "all_experiments_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_experiment_results, f, indent=2)
    print(f"✓ {summary_file}")

    # Generate comparison statistics and visualizations
    generate_comparison_report(all_experiment_results)

    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"End time: {end_timestamp}")

def generate_comparison_report(all_experiment_results):
    """Generate comparison statistics and visualizations across all experiments."""

    print("\nGenerating comparison visualizations...")

    # Extract statistics for each experiment
    experiment_stats = []
    for exp_result in all_experiment_results:
        results = exp_result['results']
        recon_errors = [r['recon_error'] for r in results]
        dyn_errors = [r['dyn_error'] for r in results]
        cons_errors = [r['cons_error'] for r in results]

        stats = {
            'name': exp_result['experiment_name'],
            'description': exp_result['description'],
            'config': exp_result['config'],
            'recon_mean': np.mean(recon_errors),
            'recon_std': np.std(recon_errors),
            'recon_min': np.min(recon_errors),
            'recon_max': np.max(recon_errors),
            'dyn_mean': np.mean(dyn_errors),
            'dyn_std': np.std(dyn_errors),
            'dyn_min': np.min(dyn_errors),
            'dyn_max': np.max(dyn_errors),
            'cons_mean': np.mean(cons_errors),
            'cons_std': np.std(cons_errors),
            'cons_min': np.min(cons_errors),
            'cons_max': np.max(cons_errors),
        }
        experiment_stats.append(stats)

    # Create comparison visualizations
    create_comparison_plots(all_experiment_results, experiment_stats)

def create_comparison_plots(all_experiment_results, experiment_stats):
    """Create comparison visualizations across experiments."""

    # 1. Box plots of errors across experiments
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Error Distribution Across Experiments', fontsize=16)

    exp_names = [exp['experiment_name'] for exp in all_experiment_results]

    # Reconstruction errors
    recon_data = [[r['recon_error'] for r in exp['results']] for exp in all_experiment_results]
    bp1 = axes[0].boxplot(recon_data, labels=exp_names, patch_artist=True)
    axes[0].set_title('Reconstruction Error')
    axes[0].set_ylabel('MSE')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)

    # Dynamics prediction errors
    dyn_data = [[r['dyn_error'] for r in exp['results']] for exp in all_experiment_results]
    bp2 = axes[1].boxplot(dyn_data, labels=exp_names, patch_artist=True)
    axes[1].set_title('Dynamics Prediction Error')
    axes[1].set_ylabel('MSE')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

    # Dynamics consistency errors
    cons_data = [[r['cons_error'] for r in exp['results']] for exp in all_experiment_results]
    bp3 = axes[2].boxplot(cons_data, labels=exp_names, patch_artist=True)
    axes[2].set_title('Dynamics Consistency Error')
    axes[2].set_ylabel('MSE')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(base_output_dir, 'comparison_boxplots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ {plot_path}")

    # 2. Bar plot comparison of means
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Mean Error Comparison Across Experiments', fontsize=16)

    x = np.arange(len(experiment_stats))
    width = 0.6

    recon_means = [s['recon_mean'] for s in experiment_stats]
    recon_stds = [s['recon_std'] for s in experiment_stats]
    axes[0].bar(x, recon_means, width, yerr=recon_stds, capsize=5)
    axes[0].set_title('Reconstruction Error')
    axes[0].set_ylabel('Mean MSE ± Std')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')

    dyn_means = [s['dyn_mean'] for s in experiment_stats]
    dyn_stds = [s['dyn_std'] for s in experiment_stats]
    axes[1].bar(x, dyn_means, width, yerr=dyn_stds, capsize=5)
    axes[1].set_title('Dynamics Prediction Error')
    axes[1].set_ylabel('Mean MSE ± Std')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')

    cons_means = [s['cons_mean'] for s in experiment_stats]
    cons_stds = [s['cons_std'] for s in experiment_stats]
    axes[2].bar(x, cons_means, width, yerr=cons_stds, capsize=5)
    axes[2].set_title('Dynamics Consistency Error')
    axes[2].set_ylabel('Mean MSE ± Std')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(base_output_dir, 'comparison_barplots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ {plot_path}")

    # 3. Best/worst summary
    best_recon = min(experiment_stats, key=lambda s: s['recon_mean'])
    best_dyn = min(experiment_stats, key=lambda s: s['dyn_mean'])
    best_cons = min(experiment_stats, key=lambda s: s['cons_mean'])

    summary_text = f"""BEST PERFORMERS
===============
Reconstruction: {best_recon['name']} ({best_recon['recon_mean']:.6f}±{best_recon['recon_std']:.6f})
Dynamics Pred:  {best_dyn['name']} ({best_dyn['dyn_mean']:.6f}±{best_dyn['dyn_std']:.6f})
Consistency:    {best_cons['name']} ({best_cons['cons_mean']:.6f}±{best_cons['cons_std']:.6f})
"""

    summary_file = os.path.join(base_output_dir, 'best_performers.txt')
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    print(summary_text)
    print(f"✓ {summary_file}")

if __name__ == "__main__":
    run_all_experiments()
