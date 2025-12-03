#!/usr/bin/env python3
"""
Leslie Population Model (3D) - MorseGraph analysis through CMGDB/MORALS framework

This example studies the three-dimensional Leslie population model,
a structured population dynamics model with three age classes.

Model: Leslie, P.H., "On the use of matrices in certain population mathematics",
       Biometrika 33(3): 183-212 (1945)

The system models population dynamics with:
- x0: Age class 0 (juveniles)
- x1: Age class 1 (young adults)
- x2: Age class 2 (mature adults)

The map is defined by:
    x0_{n+1} = (theta_1*x0_n + theta_2*x1_n + theta_3*x2_n) * exp(-0.1*(x0_n + x1_n + x2_n))
    x1_{n+1} = survival_1 * x0_n
    x2_{n+1} = survival_2 * x1_n

Usage:
    python 8_leslie_map_3d.py

Output:
    All results saved to examples/leslie_map_3d_output/
"""

import os
import sys
import argparse

# Add parent directory to path for MorseGraph imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.pipeline import MorseGraphPipeline

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Run complete Leslie model analysis pipeline."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Leslie Population Model (3D) - Morse Graph Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python 8_leslie_map_3d.py                                     # Run with default config
        python 8_leslie_map_3d.py --config configs/leslie_morals.yaml # Alternative config
        python 8_leslie_map_3d.py --config configs/leslie_default.yaml  # Explicit default
        python 8_leslie_map_3d.py --force-recompute-3d                # Force 3D recompute
        python 8_leslie_map_3d.py --force-regenerate-data             # Force trajectory regeneration
        python 8_leslie_map_3d.py --force-retrain                     # Force autoencoder retrain
        python 8_leslie_map_3d.py --force-recompute-2d                # Force 2D recompute
        python 8_leslie_map_3d.py --force-all                         # Force all computations
        python 8_leslie_map_3d.py --help                              # Show this help
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/leslie_default.yaml',
        help='Path to YAML configuration file (default: configs/leslie_default.yaml)'
    )
    parser.add_argument(
        '--force-recompute-3d',
        action='store_true',
        help='Force recomputation of 3D Morse graph (ignore cache)'
    )
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retraining of autoencoder models (ignore cached training)'
    )
    parser.add_argument(
        '--force-recompute-2d',
        action='store_true',
        help='Force recomputation of 2D Morse graphs (ignore cache)'
    )
    parser.add_argument(
        '--force-regenerate-data',
        action='store_true',
        help='Force regeneration of trajectory data (ignore cache)'
    )
    parser.add_argument(
        '--force-all',
        action='store_true',
        help='Force all computations (equivalent to --force-recompute-3d --force-regenerate-data --force-retrain --force-recompute-2d)'
    )
    parser.add_argument(
        '--latent-morse-method',
        type=str,
        default=None,
        help='Specify 2D Morse graph computation method: data, restricted, latent_enclosure (overrides config)'
    )
    args = parser.parse_args()

    # Resolve config path relative to script location if needed
    if not os.path.isabs(args.config) and not os.path.exists(args.config):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, args.config)
        if os.path.exists(config_path):
            args.config = config_path

    # Handle --force-all
    if args.force_all:
        args.force_recompute_3d = True
        args.force_regenerate_data = True
        args.force_retrain = True
        args.force_recompute_2d = True

    # Initialize and run the pipeline
    output_dir = os.path.join(os.path.dirname(__file__), "leslie_map_3d_output")
    pipeline = MorseGraphPipeline(config_path=args.config, output_dir=output_dir)
    pipeline.run(
        force_recompute_3d=args.force_recompute_3d, 
        force_regenerate_data=args.force_regenerate_data,
        force_retrain=args.force_retrain,
        force_recompute_2d=args.force_recompute_2d,
        latent_morse_method=args.latent_morse_method
    )

if __name__ == "__main__":
    main()