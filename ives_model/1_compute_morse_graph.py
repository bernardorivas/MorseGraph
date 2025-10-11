#!/usr/bin/env python3
"""
Compute Morse Graph for Ives Ecological Model

This script computes the Morse graph for the Ives model in 3D
using CMGDB and saves the results for later use in learning experiments and plotting.

Usage:
    python 1_compute_morse_graph.py                     # Use defaults from Config
    python 1_compute_morse_graph.py --r1 4.0 --r2 12.0  # Custom parameters
    python 1_compute_morse_graph.py --subdiv-min 30 --subdiv-max 40  # Custom subdivisions
    python 1_compute_morse_graph.py --output-dir ./my_results  # Custom output directory
"""

import argparse
from ives_modules import Config, compute_morse_graph


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute 3D Morse graph for Ives Ecological Model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--r1', type=float, default=Config.R1,
                       help=f'Midge reproduction rate (default: {Config.R1})')
    parser.add_argument('--r2', type=float, default=Config.R2,
                       help=f'Algae growth rate (default: {Config.R2})')
    parser.add_argument('--c', type=float, default=Config.C,
                       help=f'Constant input (default: {Config.C:.2e})')
    parser.add_argument('--d', type=float, default=Config.D,
                       help=f'Detritus decay rate (default: {Config.D})')
    parser.add_argument('--p', type=float, default=Config.P,
                       help=f'Relative palatability of detritus (default: {Config.P})')
    parser.add_argument('--q', type=float, default=Config.Q,
                       help=f'Exponent in midge consumption (default: {Config.Q})')
    parser.add_argument('--log-offset', type=float, default=Config.LOG_OFFSET,
                       help=f'Offset before log transform (default: {Config.LOG_OFFSET})')
    parser.add_argument('--subdiv-min', type=int, default=Config.SUBDIV_MIN,
                       help=f'Minimum subdivision depth (default: {Config.SUBDIV_MIN})')
    parser.add_argument('--subdiv-max', type=int, default=Config.SUBDIV_MAX,
                       help=f'Maximum subdivision depth (default: {Config.SUBDIV_MAX})')
    parser.add_argument('--subdiv-init', type=int, default=Config.SUBDIV_INIT,
                       help=f'Initial subdivision depth (default: {Config.SUBDIV_INIT})')
    parser.add_argument('--subdiv-limit', type=int, default=Config.SUBDIV_LIMIT,
                       help=f'Subdivision limit (default: {Config.SUBDIV_LIMIT})')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: experiments/morse_graph/)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("="*80)
    print("3D Morse Graph Computation (Ives Model)")
    print("="*80)
    print(f"Parameters:")
    print(f"  r1={args.r1:.3f}, r2={args.r2:.3f}, c={args.c:.2e}")
    print(f"  d={args.d:.4f}, p={args.p:.5f}, q={args.q:.3f}")
    print(f"  log_offset={args.log_offset}")
    print(f"  subdivisions: min={args.subdiv_min}, max={args.subdiv_max}, init={args.subdiv_init}")
    print()

    # Compute morse graph
    results = compute_morse_graph(
        r1=args.r1,
        r2=args.r2,
        c=args.c,
        d=args.d,
        p=args.p,
        q=args.q,
        log_offset=args.log_offset,
        subdiv_min=args.subdiv_min,
        subdiv_max=args.subdiv_max,
        subdiv_init=args.subdiv_init,
        subdiv_limit=args.subdiv_limit,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )

    print("="*80)
    print("Computation Complete!")
    print(f"  Morse sets: {results['num_morse_sets']}")
    print(f"  Computation time: {results['computation_time']:.2f}s")
    print(f"  Results saved to: {results['output_dir']}")
    print("="*80)


if __name__ == "__main__":
    main()
