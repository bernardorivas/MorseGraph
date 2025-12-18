"""
Plotting functions for MorseGraph package.

This module provides visualization functions for Morse graphs, basins of attraction,
and related structures. The implementation has been refactored into focused submodules
for better maintainability.

All public functions are re-exported here for backward compatibility.
"""

# Re-export all plotting functions from focused modules
from .plotting.utils import (
    get_num_morse_sets,
    get_morse_set_color,
    get_morse_set_colors,
    create_latent_grid,
    compute_encoded_barycenters,
)

from .plotting.morse_sets import (
    plot_morse_sets,
    plot_morse_sets_3d,
    plot_morse_sets_3d_scatter,
    plot_morse_sets_3d_projections,
    plot_morse_sets_3d_with_trajectories,
    plot_morse_sets_3d_projections_with_trajectories,
)

from .plotting.basins import (
    plot_basins_of_attraction,
    plot_data_coverage,
    plot_data_points_overlay,
)

from .plotting.graphs import (
    plot_morse_graph,
    morse_graph_to_graphviz_string,
    plot_morse_graph_diagram,
)

from .plotting.comparison import (
    plot_morse_graph_comparison,
    plot_2x2_morse_comparison,
    plot_preimage_classification,
)

from .plotting.training import (
    plot_training_curves,
    plot_encoder_decoder_roundtrip,
)

from .plotting.trajectories import (
    plot_trajectory_analysis,
)

from .plotting.latent import (
    plot_latent_space_2d,
    classify_points_to_morse_sets,
    plot_data_boxes,
)

from .plotting.attractor import (
    plot_attractor_barycenter_comparison,
)

# Re-export for backward compatibility
__all__ = [
    # Utils
    'get_num_morse_sets',
    'get_morse_set_color',
    'get_morse_set_colors',
    'create_latent_grid',
    'compute_encoded_barycenters',
    # Morse sets
    'plot_morse_sets',
    'plot_morse_sets_3d',
    'plot_morse_sets_3d_scatter',
    'plot_morse_sets_3d_projections',
    'plot_morse_sets_3d_with_trajectories',
    'plot_morse_sets_3d_projections_with_trajectories',
    # Basins
    'plot_basins_of_attraction',
    'plot_data_coverage',
    'plot_data_points_overlay',
    # Graphs
    'plot_morse_graph',
    'morse_graph_to_graphviz_string',
    'plot_morse_graph_diagram',
    # Comparison
    'plot_morse_graph_comparison',
    'plot_2x2_morse_comparison',
    'plot_preimage_classification',
    # Training
    'plot_training_curves',
    'plot_encoder_decoder_roundtrip',
    # Trajectories
    'plot_trajectory_analysis',
    # Latent
    'plot_latent_space_2d',
    'classify_points_to_morse_sets',
    'plot_data_boxes',
    # Attractor
    'plot_attractor_barycenter_comparison',
]
