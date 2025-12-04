"""
Plotting modules for MorseGraph package.

This package is split into focused modules:
- utils: Plotting utilities (colors, helpers)
- morse_sets: Morse sets visualization
- basins: Basins of attraction visualization
- graphs: Morse graph diagrams
- comparison: Comparison plots
- training: Training visualization
- trajectories: Trajectory visualization
- latent: Latent space visualization

All public functions are re-exported here for backward compatibility.
"""

# Re-export from utils
from .utils import (
    get_num_morse_sets,
    get_morse_set_color,
    get_morse_set_colors,
    create_latent_grid,
    compute_encoded_barycenters,
    _box_to_cuboid_faces,
    _get_graph_statistics,
)

# Re-export from morse_sets
from .morse_sets import (
    plot_morse_sets,
    plot_morse_sets_3d,
    plot_morse_sets_3d_scatter,
    plot_morse_sets_3d_projections,
    plot_morse_sets_3d_with_trajectories,
    plot_morse_sets_3d_projections_with_trajectories,
)

# Re-export from basins
from .basins import (
    plot_basins_of_attraction,
    plot_data_coverage,
    plot_data_points_overlay,
)

# Re-export from graphs
from .graphs import (
    plot_morse_graph,
    morse_graph_to_graphviz_string,
    plot_morse_graph_diagram,
)

# Re-export from comparison
from .comparison import (
    plot_morse_graph_comparison,
    plot_2x2_morse_comparison,
    plot_preimage_classification,
)

# Re-export from training
from .training import (
    plot_training_curves,
    plot_encoder_decoder_roundtrip,
)

# Re-export from trajectories
from .trajectories import (
    plot_trajectory_analysis,
)

# Re-export from latent
from .latent import (
    plot_latent_space_2d,
    classify_points_to_morse_sets,
    plot_data_boxes,
)

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
]
