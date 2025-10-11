"""
Ives Ecological Model - Modular Components

This package contains reusable components for Ives model experiments including:
- Ground truth Morse graph computation
- Learning experiments (autoencoder + latent dynamics)
- Plotting and visualization utilities
- Shared configuration
"""

from .config import Config
from .morse_graph import (
    compute_morse_graph,
    load_morse_sets_barycenters,
    load_morse_graph,
    extract_graph_structure
)
from .learning import run_learning_experiment
from . import plotting

__all__ = [
    'Config',
    'compute_morse_graph',
    'load_morse_sets_barycenters',
    'load_morse_graph',
    'extract_graph_structure',
    'run_learning_experiment',
    'plotting'
]
