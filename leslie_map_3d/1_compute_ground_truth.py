#!/usr/bin/env python3
"""
Compute Ground Truth Morse Graph for Leslie Map 3D

This script computes the ground truth Morse graph for the Leslie map
using CMGDB and saves the results for later use in learning experiments.

Usage:
    python 1_compute_ground_truth.py                    # Use defaults from Config
    python 1_compute_ground_truth.py --theta 30.0       # Custom theta value
    python 1_compute_ground_truth.py --subdiv-min 30 --subdiv-max 36  # Custom subdivisions
    python 1_compute_ground_truth.py --output-dir ./my_results  # Custom output directory
"""

import argparse
from leslie_modules import Config, compute_ground_truth_morse_graph


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute ground truth Morse graph for Leslie Map 3D',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--theta', type=float, default=None,
                       help='Common theta value for all age classes (overrides individual theta values)')
    parser.add_argument('--theta-1', type=float, default=Config.THETA_1,
                       help=f'Fertility for age class 0 (default: {Config.THETA_1})')
    parser.add_argument('--theta-2', type=float, default=Config.THETA_2,
                       help=f'Fertility for age class 1 (default: {Config.THETA_2})')
    parser.add_argument('--theta-3', type=float, default=Config.THETA_3,
                       help=f'Fertility for age class 2 (default: {Config.THETA_3})')
    parser.add_argument('--survival-1', type=float, default=Config.SURVIVAL_1,
                       help=f'Survival rate 0→1 (default: {Config.SURVIVAL_1})')
    parser.add_argument('--survival-2', type=float, default=Config.SURVIVAL_2,
                       help=f'Survival rate 1→2 (default: {Config.SURVIVAL_2})')
    parser.add_argument('--subdiv-min', type=int, default=Config.SUBDIV_MIN,
                       help=f'Minimum subdivision depth (default: {Config.SUBDIV_MIN})')
    parser.add_argument('--subdiv-max', type=int, default=Config.SUBDIV_MAX,
                       help=f'Maximum subdivision depth (default: {Config.SUBDIV_MAX})')
    parser.add_argument('--subdiv-init', type=int, default=Config.SUBDIV_INIT,
                       help=f'Initial subdivision depth (default: {Config.SUBDIV_INIT})')
    parser.add_argument('--subdiv-limit', type=int, default=Config.SUBDIV_LIMIT,
                       help=f'Subdivision limit (default: {Config.SUBDIV_LIMIT})')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: experiments/ground_truth/)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle theta override
    if args.theta is not None:
        theta_1 = theta_2 = theta_3 = args.theta
    else:
        theta_1 = args.theta_1
        theta_2 = args.theta_2
        theta_3 = args.theta_3

    print("="*80)
    print("Ground Truth Morse Graph Computation")
    print("="*80)
    print(f"Parameters:")
    print(f"  theta = ({theta_1}, {theta_2}, {theta_3})")
    print(f"  survival = ({args.survival_1}, {args.survival_2})")
    print(f"  subdivisions: min={args.subdiv_min}, max={args.subdiv_max}, init={args.subdiv_init}")
    print()

    # Compute ground truth
    results = compute_ground_truth_morse_graph(
        theta_1=theta_1,
        theta_2=theta_2,
        theta_3=theta_3,
        survival_1=args.survival_1,
        survival_2=args.survival_2,
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
