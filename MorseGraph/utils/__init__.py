"""
Utility modules for MorseGraph package.

This package is split into focused modules:
- trajectory: Trajectory data generation
- io: Data I/O operations
- latent: Latent space utilities
- training: ML training utilities
- spatial: Spatial filtering
- analysis: Morse graph analysis utilities
- config: Experiment configuration
- caching: CMGDB caching utilities
- pipeline: Pipeline utilities

All public functions are re-exported here for backward compatibility.
"""

# Re-export all public functions for backward compatibility
from .trajectory import (
    generate_trajectory_data,
    generate_map_trajectory_data,
    generate_random_trajectories_3d,
)
from .io import (
    _save_trajectory_data_file,
    _load_trajectory_data_file,
    get_next_run_number,
)
from .spatial import (
    filter_boxes_near_data,
)
from .latent import (
    compute_latent_bounds,
    compute_latent_bounds_from_data,
    generate_3d_grid_for_encoding,
)
from .training import (
    count_parameters,
    format_time,
    save_models,
    load_models,
    save_training_history,
    load_training_history,
)
from .analysis import (
    count_attractors,
    extract_edge_set,
    compute_similarity_vector,
    format_similarity_report,
    compute_train_val_divergence,
)
from .config import (
    ExperimentConfig,
    setup_experiment_dirs,
    save_experiment_metadata,
)

from .caching import (
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

__all__ = [
    'generate_trajectory_data',
    'generate_map_trajectory_data',
    'generate_random_trajectories_3d',
    '_save_trajectory_data_file',
    '_load_trajectory_data_file',
    'get_next_run_number',
    'filter_boxes_near_data',
    'compute_latent_bounds',
    'compute_latent_bounds_from_data',
    'generate_3d_grid_for_encoding',
    'count_parameters',
    'format_time',
    'save_models',
    'load_models',
    'save_training_history',
    'load_training_history',
    'count_attractors',
    'extract_edge_set',
    'compute_similarity_vector',
    'format_similarity_report',
    'compute_train_val_divergence',
    'ExperimentConfig',
    'setup_experiment_dirs',
    'save_experiment_metadata',
    'get_cache_paths',
    'compute_parameter_hash',
    'compute_trajectory_hash',
    'compute_training_hash',
    'compute_cmgdb_2d_hash',
    'compute_cmgdb_3d_hash',
    'compute_trajectory_data_hash',
    'get_cache_path',
    'CachedMorseGraph',
    'load_or_train_autoencoder',
    'load_or_generate_trajectory_data',
    'save_trajectory_data',
    'load_trajectory_data',
    'load_or_compute_2d_morse_graphs',
    'save_morse_graph_cache',
    'load_morse_graph_cache',
    'save_morse_graph_data',
    'load_morse_graph_data',
    'load_or_compute_3d_morse_graph',
]

