#!/usr/bin/env python3
"""
Ives Ecological Model - MorseGraph analysis through CMGDB/MORALS framework

This example studies the Ives midge-algae-detritus ecological model
using the Morse Graphs. The model operates in log10 scale
to handle many orders of magnitude in population abundances.

Model: Ives et al. (2008) - "High-amplitude fluctuations and alternative
       dynamical states of midges in Lake Myvatn", Nature 452: 84-87

Usage:
    python 7_ives_model.py

Output:
    All results saved to examples/ives_model_output/
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
    """Run complete Ives model analysis pipeline."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Ives Ecological Model - Morse Graph Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python 7_ives_model.py                                     # Run with default config
        python 7_ives_model.py --config configs/ives_morals.yaml   # Alternative config
        python 7_ives_model.py --config configs/ives_default.yaml  # Explicit default
        python 7_ives_model.py --force-recompute-3d                # Force 3D recompute
        python 7_ives_model.py --force-regenerate-data             # Force trajectory regeneration
        python 7_ives_model.py --force-retrain                     # Force autoencoder retrain
        python 7_ives_model.py --force-recompute-2d                # Force 2D recompute
        python 7_ives_model.py --force-all                         # Force all computations
        python 7_ives_model.py --help                              # Show this help
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ives_default.yaml',
        help='Path to YAML configuration file (default: configs/ives_default.yaml)'
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
    output_dir = os.path.join(os.path.dirname(__file__), "ives_model_output")
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