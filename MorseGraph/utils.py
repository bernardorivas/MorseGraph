"""
Utility functions for data generation, management, and spatial operations.

This module provides helper functions for working with trajectory data,
latent space operations, and spatial filtering of grid boxes.

NOTE: This module has been refactored. All utility functions have been
moved to focused submodules in MorseGraph.utils.*. This file maintains 
backward compatibility by re-exporting from the new modules.
"""

# Re-export functions from new modules for backward compatibility
from .utils.trajectory import (
    generate_trajectory_data,
    generate_map_trajectory_data,
)
from .utils.io import (
    _save_trajectory_data_file,
    _load_trajectory_data_file,
    get_next_run_number,
)
from .utils.spatial import (
    filter_boxes_near_data,
)
from .utils.latent import (
    compute_latent_bounds,
    compute_latent_bounds_from_data,
)
from .utils.training import (
    count_parameters,
    format_time,
    save_models,
    load_models,
    save_training_history,
    load_training_history,
)

# Re-export analysis functions from new module for backward compatibility
from .utils.analysis import (
    count_attractors,
    extract_edge_set,
    compute_similarity_vector,
    format_similarity_report,
    compute_train_val_divergence,
)


# Re-export config classes and functions from new module for backward compatibility
from .utils.config import (
    ExperimentConfig,
    setup_experiment_dirs,
    save_experiment_metadata,
)

# Re-export caching functions from new module for backward compatibility
from .utils.caching import (
    get_cache_paths,
    compute_parameter_hash,
    compute_trajectory_hash,
    compute_training_hash,
    compute_cmgdb_2d_hash,
    compute_cmgdb_3d_hash,
    compute_trajectory_data_hash,
    get_cache_path,
    CachedMorseGraph,
    load_or_train_autoencoder,
    load_or_generate_trajectory_data,
    save_trajectory_data,
    load_trajectory_data,
    load_or_compute_2d_morse_graphs,
    save_morse_graph_cache,
    load_morse_graph_cache,
    save_morse_graph_data,
    load_morse_graph_data,
    load_or_compute_3d_morse_graph,
)

# =============================================================================
# Pipeline Utilities
# =============================================================================
# (All utility functions have been moved to focused submodules)
# This file now serves as a backward-compatibility re-export layer
