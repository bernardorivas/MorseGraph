#!/usr/bin/env python3
"""
Parameter Sweep: 3D Morse Graph Minimal Nodes Analysis

Systematically test CMGDB parameter combinations to find which ones produce
exactly 2 minimal nodes (sources without incoming edges) in the Morse graph.

Results are saved to a CSV file with periodic checkpointing.

Usage:
    python tests/parameter_sweep_morse_minimal_nodes.py
    python tests/parameter_sweep_morse_minimal_nodes.py --save-freq 5
    python tests/parameter_sweep_morse_minimal_nodes.py --output results.csv
"""

import numpy as np
import os
import sys
import csv
import argparse
import time
from functools import partial
from datetime import datetime
from itertools import product

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.core import compute_morse_graph_3d
from MorseGraph.systems import ives_model_log


def analyze_morse_graph_structure(morse_graph):
    """Count sources (minimal nodes) and sinks (maximal nodes) from CMGDB MorseGraph."""
    vertices = list(range(morse_graph.num_vertices()))
    edges = morse_graph.edges()

    # Build adjacency structure to find sources/sinks
    has_incoming = set()
    has_outgoing = set()

    for edge_idx in range(len(edges)):
        u, v = edges[edge_idx]
        has_outgoing.add(u)
        has_incoming.add(v)

    sources = [v for v in vertices if v not in has_incoming]
    sinks = [v for v in vertices if v not in has_outgoing]

    return {
        'num_nodes': len(vertices),
        'num_edges': len(edges),
        'num_sources': len(sources),
        'num_sinks': len(sinks),
        'sources': sources,
        'sinks': sinks,
    }


def generate_parameter_combinations():
    """Generate all parameter combinations for the sweep."""
    padding_values = [False, True]
    subdiv_min_values = [30, 33, 36]
    subdiv_max_offsets = [0, 3, 6]  # Offset from subdiv_min
    subdiv_init_values = [0, 5, 10, 15]
    subdiv_limit = 100000  # Fixed

    combinations = []
    for padding, subdiv_min, offset, subdiv_init in product(
        padding_values, subdiv_min_values, subdiv_max_offsets, subdiv_init_values
    ):
        subdiv_max = subdiv_min + offset
        combinations.append({
            'padding': padding,
            'subdiv_min': subdiv_min,
            'subdiv_max': subdiv_max,
            'subdiv_init': subdiv_init,
            'subdiv_limit': subdiv_limit,
        })

    return combinations


def test_configuration(config, ives_map, domain_bounds, verbose=False):
    """Test a single parameter configuration and return results."""
    if verbose:
        print(f"\nTesting: padding={config['padding']}, "
              f"subdiv_min={config['subdiv_min']}, subdiv_max={config['subdiv_max']}, "
              f"subdiv_init={config['subdiv_init']}")

    start_time = time.time()
    result = {
        **config,  # Include all parameters
        'num_morse_sets': None,
        'num_edges': None,
        'num_sources': None,
        'num_sinks': None,
        'has_two_minimal_nodes': False,
        'computation_time': None,
        'error': '',
    }

    try:
        morse_result = compute_morse_graph_3d(
            ives_map,
            domain_bounds,
            subdiv_min=config['subdiv_min'],
            subdiv_max=config['subdiv_max'],
            subdiv_init=config['subdiv_init'],
            subdiv_limit=config['subdiv_limit'],
            padding=config['padding'],
            cache_dir=None,  # Don't cache
            use_cache=False,
            verbose=False  # Suppress CMGDB output for cleaner logs
        )

        morse_graph = morse_result['morse_graph']
        analysis = analyze_morse_graph_structure(morse_graph)

        result['num_morse_sets'] = analysis['num_nodes']
        result['num_edges'] = analysis['num_edges']
        result['num_sources'] = analysis['num_sources']
        result['num_sinks'] = analysis['num_sinks']
        result['has_two_minimal_nodes'] = (analysis['num_sources'] == 2)
        result['computation_time'] = time.time() - start_time

        if verbose:
            print(f"  → Morse sets: {result['num_morse_sets']}, "
                  f"Sources: {result['num_sources']}, "
                  f"Sinks: {result['num_sinks']}, "
                  f"Time: {result['computation_time']:.2f}s")
            if result['has_two_minimal_nodes']:
                print(f"  ⭐ FOUND: 2 minimal nodes!")

    except Exception as e:
        result['error'] = str(e)
        result['computation_time'] = time.time() - start_time
        if verbose:
            print(f"  ✗ ERROR: {e}")

    return result


def save_results_to_csv(results, output_file):
    """Save results to CSV file."""
    if not results:
        return

    # Define CSV columns
    fieldnames = [
        'padding', 'subdiv_min', 'subdiv_max', 'subdiv_init', 'subdiv_limit',
        'num_morse_sets', 'num_edges', 'num_sources', 'num_sinks',
        'has_two_minimal_nodes', 'computation_time', 'error'
    ]

    # Write or append to CSV
    file_exists = os.path.exists(output_file)
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(
        description='Parameter sweep for 3D Morse Graph minimal nodes analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/parameter_sweep_morse_minimal_nodes.py
  python tests/parameter_sweep_morse_minimal_nodes.py --save-freq 5
  python tests/parameter_sweep_morse_minimal_nodes.py --output my_results.csv
        """
    )

    parser.add_argument('--output', '-o', type=str,
                       default='tests/morse_parameter_sweep_results.csv',
                       help='Output CSV file (default: tests/morse_parameter_sweep_results.csv)')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save results to CSV every N iterations (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed progress for each configuration')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output file (default: append)')

    args = parser.parse_args()

    # Handle overwrite
    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)
        print(f"Removed existing file: {args.output}")

    # Ives model parameters (matching configs/ives_default.yaml)
    R1 = 3.873
    R2 = 11.746
    C = 3.67e-07
    D = 0.5517
    P = 0.06659
    Q = 0.9026
    LOG_OFFSET = 0.001
    DOMAIN_BOUNDS = [[-5, -20, -8], [10, 20, 8]]

    # Create map function
    ives_map = partial(
        ives_model_log,
        r1=R1, r2=R2, c=C, d=D, p=P, q=Q, offset=LOG_OFFSET
    )

    # Generate parameter combinations
    combinations = generate_parameter_combinations()
    total_combinations = len(combinations)

    # Print header
    print("\n" + "="*80)
    print("PARAMETER SWEEP: 3D MORSE GRAPH MINIMAL NODES ANALYSIS")
    print("="*80)
    print(f"\nIves Model: Midge-Algae-Detritus dynamics")
    print(f"Domain (log): {DOMAIN_BOUNDS[0]} to {DOMAIN_BOUNDS[1]}")
    print(f"\nParameter Space:")
    print(f"  padding: [False, True]")
    print(f"  subdiv_min: [30, 33, 36]")
    print(f"  subdiv_max: subdiv_min + [0, 3, 6]")
    print(f"  subdiv_init: [0, 5, 10, 15]")
    print(f"  subdiv_limit: 100000 (fixed)")
    print(f"\nTotal configurations: {total_combinations}")
    print(f"Output file: {args.output}")
    print(f"Save frequency: every {args.save_freq} iterations")
    print("="*80)

    # Run parameter sweep
    results_buffer = []
    successful_configs = []
    start_time_total = time.time()

    for i, config in enumerate(combinations, 1):
        print(f"\n[{i}/{total_combinations}] ", end='', flush=True)

        result = test_configuration(config, ives_map, DOMAIN_BOUNDS, verbose=args.verbose)
        results_buffer.append(result)

        # Track successful configurations
        if result['has_two_minimal_nodes']:
            successful_configs.append(config)

        # Periodic save
        if i % args.save_freq == 0 or i == total_combinations:
            save_results_to_csv(results_buffer, args.output)
            print(f"  [Saved {len(results_buffer)} results to {args.output}]")
            results_buffer = []

        # Progress update
        elapsed = time.time() - start_time_total
        avg_time_per_config = elapsed / i
        remaining = (total_combinations - i) * avg_time_per_config
        print(f"  Progress: {i}/{total_combinations} ({100*i/total_combinations:.1f}%), "
              f"Elapsed: {elapsed:.0f}s, ETA: {remaining:.0f}s")

    # Final summary
    total_time = time.time() - start_time_total
    print("\n" + "="*80)
    print("PARAMETER SWEEP COMPLETE")
    print("="*80)
    print(f"\nTotal configurations tested: {total_combinations}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"Average time per configuration: {total_time/total_combinations:.2f}s")
    print(f"\nResults saved to: {args.output}")

    # Show successful configurations
    if successful_configs:
        print(f"\n⭐ Found {len(successful_configs)} configuration(s) with exactly 2 minimal nodes:")
        for config in successful_configs:
            print(f"  - padding={config['padding']}, "
                  f"subdiv_min={config['subdiv_min']}, "
                  f"subdiv_max={config['subdiv_max']}, "
                  f"subdiv_init={config['subdiv_init']}")
    else:
        print(f"\nNo configurations with exactly 2 minimal nodes found.")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
