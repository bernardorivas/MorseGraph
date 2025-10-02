"""
MorseGraph: Computational topology for dynamical systems.

This library provides tools for computing and analyzing Morse graphs of
dynamical systems through discrete abstractions on grids.
"""

__version__ = "0.1.0"

# Core components
from .grids import AbstractGrid, UniformGrid, AdaptiveGrid
from .dynamics import (
    Dynamics,
    BoxMapFunction,
    BoxMapData,
    BoxMapODE
)
from .core import Model
from .analysis import (
    compute_morse_graph,
    compute_all_morse_set_basins,
    compute_basins_of_attraction,
    iterative_morse_computation,
    diameter_criterion,
    analyze_refinement_convergence
)
from .plot import (
    plot_morse_sets,
    plot_basins_of_attraction,
    plot_morse_graph,
    plot_data_coverage,
    plot_data_points_overlay
)

# Optional ML components (conditional import)
try:
    from .dynamics import BoxMapLearnedLatent
    from .models import Encoder, Decoder, LatentDynamics
    from .training import Training
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

# Utility modules
from . import systems
from . import utils

__all__ = [
    # Core
    'AbstractGrid',
    'UniformGrid',
    'AdaptiveGrid',
    'Dynamics',
    'BoxMapFunction',
    'BoxMapData',
    'BoxMapODE',
    'Model',
    # Analysis
    'compute_morse_graph',
    'compute_all_morse_set_basins',
    'compute_basins_of_attraction',
    'iterative_morse_computation',
    'diameter_criterion',
    'analyze_refinement_convergence',
    # Plotting
    'plot_morse_sets',
    'plot_basins_of_attraction',
    'plot_morse_graph',
    'plot_data_coverage',
    'plot_data_points_overlay',
    # Modules
    'systems',
    'utils',
]

# Add ML components to __all__ if available
if _ML_AVAILABLE:
    __all__.extend([
        'BoxMapLearnedLatent',
        'Encoder',
        'Decoder',
        'LatentDynamics',
        'Training',
    ])
