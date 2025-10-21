#!/usr/bin/env python3
"""
Compute 3D Morse Graph for Ives Model (Standalone)

This script ONLY computes the 3D Morse graph without any learning pipeline.
Useful for:
- Exploring different subdivision levels
- Finding ground truth Morse structure
- Generating high-resolution reference graphs

The results are cached in cmgdb_3d/ for reuse by other scripts.

Usage:
    python compute_ives_3d_morse_graph.py --subdiv 33
    python compute_ives_3d_morse_graph.py --subdiv-min 30 --subdiv-max 42
    python compute_ives_3d_morse_graph.py --list-cache  # View cached results
"""

import numpy as np
import os
import sys
import argparse
from functools import partial
import json
from datetime import datetime

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.utils import load_or_compute_3d_morse_graph
from MorseGraph.systems import ives_model_log


def list_cache(base_dir):
    """List all cached 3D Morse graphs with readable names."""
    cmgdb_3d_dir = os.path.join(base_dir, 'cmgdb_3d')

    if not os.path.exists(cmgdb_3d_dir):
        print(f"No cache directory found at: {cmgdb_3d_dir}")
        return

    print("\n" + "="*80)
    print("CACHED 3D MORSE GRAPHS")
    print("="*80)

    # Get all subdirectories (hash-based cache folders)
    cache_folders = []
    for item in os.listdir(cmgdb_3d_dir):
        item_path = os.path.join(cmgdb_3d_dir, item)
        if os.path.isdir(item_path) and item != 'results':
            metadata_path = os.path.join(item_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    metadata['hash'] = item
                    metadata['path'] = item_path
                    cache_folders.append(metadata)

    if not cache_folders:
        print("No cached results found.")
        return

    # Sort by cached_at timestamp (most recent first)
    cache_folders.sort(key=lambda x: x.get('cached_at', ''), reverse=True)

    for i, meta in enumerate(cache_folders, 1):
        subdiv_min = meta.get('subdiv_min', '?')
        subdiv_max = meta.get('subdiv_max', '?')
        subdiv_init = meta.get('subdiv_init', '?')
        padding = meta.get('padding', '?')
        num_sets = meta.get('num_morse_sets', '?')
        comp_time = meta.get('computation_time', 0.0)
        cached_at = meta.get('cached_at', 'unknown')
        param_hash = meta.get('hash', meta.get('param_hash', 'unknown'))

        # Determine if uniform grid
        if subdiv_min == subdiv_max:
            grid_type = f"UNIFORM (subdiv={subdiv_min})"
        else:
            grid_type = f"ADAPTIVE (min={subdiv_min}, max={subdiv_max}, init={subdiv_init})"

        print(f"\n{i}. {grid_type}")
        print(f"   Morse Sets: {num_sets}")
        print(f"   Padding: {padding}")
        print(f"   Compute Time: {comp_time:.2f}s")
        print(f"   Cached: {cached_at}")
        print(f"   Hash: {param_hash}")
        print(f"   Path: {meta['path']}")

    print("\n" + "="*80)
    print(f"Total cached results: {len(cache_folders)}")
    print(f"Cache directory: {cmgdb_3d_dir}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compute 3D Morse Graph for Ives Ecological Model (Standalone)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Uniform grid (subdiv_min = subdiv_max)
  python compute_ives_3d_morse_graph.py --subdiv 33
  python compute_ives_3d_morse_graph.py --subdiv 36  # Fine resolution

  # Adaptive grid
  python compute_ives_3d_morse_graph.py --subdiv-min 30 --subdiv-max 42

  # Force recompute (ignore cache)
  python compute_ives_3d_morse_graph.py --subdiv 33 --force

  # List cached results
  python compute_ives_3d_morse_graph.py --list-cache
        """
    )

    # Subdivision parameters
    parser.add_argument('--subdiv', type=int, help='Subdivision depth for uniform grid (sets both min and max)')
    parser.add_argument('--subdiv-min', type=int, default=30, help='Minimum subdivision depth (default: 30)')
    parser.add_argument('--subdiv-max', type=int, default=42, help='Maximum subdivision depth (default: 42)')
    parser.add_argument('--subdiv-init', type=int, default=0, help='Initial subdivision depth (default: 0)')
    parser.add_argument('--subdiv-limit', type=int, default=100000, help='Maximum number of boxes (default: 100000)')
    parser.add_argument('--padding', action='store_true', help='Enable boundary padding (default: disabled)')
    parser.add_argument('--no-padding', action='store_false', dest='padding', help='Disable boundary padding')
    parser.set_defaults(padding=False)

    # Output directory
    parser.add_argument('--output-dir', type=str, default='examples/ives_model_output',
                       help='Output directory (default: examples/ives_model_output)')

    # Cache control
    parser.add_argument('--force', '--force-recompute', action='store_true',
                       help='Force recomputation (ignore cache)')
    parser.add_argument('--list-cache', action='store_true', help='List all cached 3D Morse graphs')

    # Verbosity
    parser.add_argument('--quiet', action='store_true', help='Suppress progress messages')

    args = parser.parse_args()

    # If listing cache, do that and exit
    if args.list_cache:
        list_cache(args.output_dir)
        return

    # Handle uniform grid shorthand
    if args.subdiv is not None:
        subdiv_min = args.subdiv
        subdiv_max = args.subdiv
        grid_type_str = f"UNIFORM (subdiv={args.subdiv})"
    else:
        subdiv_min = args.subdiv_min
        subdiv_max = args.subdiv_max
        if subdiv_min == subdiv_max:
            grid_type_str = f"UNIFORM (subdiv={subdiv_min})"
        else:
            grid_type_str = f"ADAPTIVE (min={subdiv_min}, max={subdiv_max})"

    verbose = not args.quiet

    # Ives model parameters (matching configs/ives_default.yaml)
    R1 = 3.873
    R2 = 11.746
    C = 3.67e-07
    D = 0.5517
    P = 0.06659
    Q = 0.9026
    LOG_OFFSET = 0.001
    DOMAIN_BOUNDS = [[-5, -20, -8], [10, 20, 8]]
    EQUILIBRIUM_POINT = np.array([0.792107, 0.209010, 0.376449])

    # Print header
    if verbose:
        print("\n" + "="*80)
        print("IVES 3D MORSE GRAPH COMPUTATION (STANDALONE)")
        print("="*80)
        print(f"\nModel: Midge-Algae-Detritus dynamics (Ives et al. 2008)")
        print(f"Parameters: R1={R1:.3f}, R2={R2:.3f}, C={C:.2e}, D={D:.4f}, P={P:.5f}, Q={Q:.3f}")
        print(f"Domain (log): {DOMAIN_BOUNDS[0]} to {DOMAIN_BOUNDS[1]}")
        print(f"Equilibrium (log): [{EQUILIBRIUM_POINT[0]:.4f}, {EQUILIBRIUM_POINT[1]:.4f}, {EQUILIBRIUM_POINT[2]:.4f}]")
        print(f"\nGrid Configuration:")
        print(f"  Type: {grid_type_str}")
        print(f"  subdiv_min: {subdiv_min}")
        print(f"  subdiv_max: {subdiv_max}")
        print(f"  subdiv_init: {args.subdiv_init}")
        print(f"  subdiv_limit: {args.subdiv_limit:,}")
        print(f"  padding: {args.padding}")

        if args.force:
            print(f"\n⚠ Force recompute enabled - will ignore cache")

        print(f"\nOutput directory: {args.output_dir}")

    # Create map function
    ives_map = partial(
        ives_model_log,
        r1=R1, r2=R2, c=C, d=D, p=P, q=Q, offset=LOG_OFFSET
    )

    # Compute or load 3D Morse graph
    if verbose:
        print("\n" + "="*80)
        print("COMPUTING 3D MORSE GRAPH")
        print("="*80)

    result_3d, was_cached = load_or_compute_3d_morse_graph(
        ives_map,
        DOMAIN_BOUNDS,
        subdiv_min=subdiv_min,
        subdiv_max=subdiv_max,
        subdiv_init=args.subdiv_init,
        subdiv_limit=args.subdiv_limit,
        padding=args.padding,
        base_dir=args.output_dir,
        force_recompute=args.force,
        verbose=verbose,
        equilibria={'Equilibrium': EQUILIBRIUM_POINT},
        labels={'x': 'log(Midge)', 'y': 'log(Algae)', 'z': 'log(Detritus)'}
    )

    morse_graph_3d = result_3d['morse_graph']
    barycenters_3d = result_3d['barycenters']
    num_morse_sets = result_3d['num_morse_sets']
    computation_time = result_3d['computation_time']

    # Print summary
    if verbose:
        print("\n" + "="*80)
        print("COMPUTATION COMPLETE")
        print("="*80)
        status_str = "LOADED FROM CACHE" if was_cached else "COMPUTED"
        print(f"\nStatus: {status_str}")
        print(f"Morse Sets: {num_morse_sets}")
        print(f"Number of Nodes: {morse_graph_3d.number_of_nodes()}")
        print(f"Number of Edges: {morse_graph_3d.number_of_edges()}")
        if not was_cached:
            print(f"Computation Time: {computation_time:.2f}s")

        print(f"\nCache Directory: {result_3d['cache_path']}")
        print(f"Visualizations: {result_3d['cache_path']}/results/")

        # Analyze Morse graph structure
        import networkx as nx

        # Find sources (minimal elements) and sinks (maximal elements)
        sources = [n for n in morse_graph_3d.nodes() if morse_graph_3d.in_degree(n) == 0]
        sinks = [n for n in morse_graph_3d.nodes() if morse_graph_3d.out_degree(n) == 0]

        print(f"\nMorse Graph Structure:")
        print(f"  Sources (minimal elements): {len(sources)}")
        print(f"  Sinks (maximal elements): {len(sinks)}")

        if len(sources) <= 5:
            print(f"  Source nodes: {sources}")
        if len(sinks) <= 5:
            print(f"  Sink nodes: {sinks}")

        # Check if acyclic
        is_dag = nx.is_directed_acyclic_graph(morse_graph_3d)
        print(f"  Is DAG (acyclic): {is_dag}")

        if not is_dag:
            # Find cycles
            try:
                cycles = list(nx.simple_cycles(morse_graph_3d))
                print(f"  Number of cycles: {len(cycles)}")
                if len(cycles) <= 3:
                    for i, cycle in enumerate(cycles, 1):
                        print(f"    Cycle {i}: {cycle}")
            except:
                pass

        print("="*80 + "\n")


if __name__ == "__main__":
    main()
