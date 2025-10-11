"""
Leslie Map 3D Example - Modular Components

This package contains reusable components for Leslie map 3D experiments including:
- Ground truth Morse graph computation
- Learning experiments (autoencoder + latent dynamics)
- Plotting and visualization utilities
- Shared configuration
"""

from .config import Config
from .ground_truth import (
    compute_ground_truth_morse_graph,
    load_ground_truth_barycenters,
    load_ground_truth_morse_graph,
    extract_graph_structure
)
from .learning import run_learning_experiment
from . import plotting

__all__ = [
    'Config',
    'compute_ground_truth_morse_graph',
    'load_ground_truth_barycenters',
    'load_ground_truth_morse_graph',
    'extract_graph_structure',
    'run_learning_experiment',
    'plotting'
]
