"""
Experiment configuration and management utilities.

This module provides the ExperimentConfig class and related functions for
managing experiment parameters and directory structures.
"""

import os
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime


class ExperimentConfig:
    """
    Base configuration class for 3D map experiments with learning pipeline.

    This class centralizes all parameters for:
    - Domain specification
    - CMGDB parameters for 3D Morse graph computation
    - Data generation settings
    - Neural network architecture
    - Training parameters
    - Latent space parameters

    Can be subclassed for specific dynamical systems.

    Example:
        >>> config = ExperimentConfig(
        ...     domain_bounds=[[-10, -10, -10], [10, 10, 10]],
        ...     subdiv_min=30,
        ...     subdiv_max=42
        ... )
        >>> config.set_map_func(my_map_function)
    """

    def __init__(
        self,
        # System info
        system_type: str = 'map',
        dynamics_name: str = 'unknown',

        # Domain specification
        domain_bounds: List[List[float]] = None,

        # CMGDB parameters for 3D
        subdiv_min: int = 30,
        subdiv_max: int = 42,
        subdiv_init: int = 0,
        subdiv_limit: int = 10000,
        padding: bool = True,

        # Data generation
        n_trajectories: int = 5000,
        n_points: int = 20,
        skip_initial: int = 0,
        random_seed: Optional[int] = 42,

        # Model architecture - Simple mode (shared)
        input_dim: int = 3,
        latent_dim: int = 2,
        hidden_dim: int = 32,
        num_layers: int = 3,
        output_activation: Optional[str] = None,
        encoder_activation: Optional[str] = None,
        decoder_activation: Optional[str] = None,
        latent_dynamics_activation: Optional[str] = None,

        # Model architecture - Advanced mode (component-specific)
        encoder_hidden_dim: Optional[int] = None,
        encoder_num_layers: Optional[int] = None,
        decoder_hidden_dim: Optional[int] = None,
        decoder_num_layers: Optional[int] = None,
        latent_dynamics_hidden_dim: Optional[int] = None,
        latent_dynamics_num_layers: Optional[int] = None,

        # Training parameters
        num_epochs: int = 1500,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 50,
        min_delta: float = 1e-5,

        # Loss weights
        w_recon: float = 1.0,
        w_dyn_recon: float = 1.0,
        w_dyn_cons: float = 1.0,

        # Latent space parameters
        latent_subdiv_min: int = 20,
        latent_subdiv_max: int = 28,
        latent_subdiv_init: int = 0,
        latent_subdiv_limit: int = 10000,
        latent_padding: bool = True,
        latent_bounds_padding: float = 1.01,
        original_grid_subdiv: int = 15,
        latent_morse_graph_method: Optional[str] = 'data',

        # Large sample for domain-restricted computation
        large_sample_size: Optional[int] = None,
        target_points_per_box: int = 2,

        # Visualization parameters
        n_grid_points: int = 20,
    ):
        """
        Initialize experiment configuration.
        """
        # System info
        self.system_type = system_type
        self.dynamics_name = dynamics_name

        # Domain
        self.domain_bounds = domain_bounds or [[-10, -10, -10], [10, 10, 10]]

        # 3D CMGDB
        self.subdiv_min = subdiv_min
        self.subdiv_max = subdiv_max
        self.subdiv_init = subdiv_init
        self.subdiv_limit = subdiv_limit
        self.padding = padding

        # Data generation
        self.n_trajectories = n_trajectories
        self.n_points = n_points
        self.skip_initial = skip_initial
        self.random_seed = random_seed

        # Architecture - Simple mode (shared)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_activation = output_activation
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.latent_dynamics_activation = latent_dynamics_activation

        # Architecture - Advanced mode (component-specific)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_num_layers = decoder_num_layers
        self.latent_dynamics_hidden_dim = latent_dynamics_hidden_dim
        self.latent_dynamics_num_layers = latent_dynamics_num_layers

        # Training
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta

        # Loss weights
        self.w_recon = w_recon
        self.w_dyn_recon = w_dyn_recon
        self.w_dyn_cons = w_dyn_cons

        # Latent space
        self.latent_subdiv_min = latent_subdiv_min
        self.latent_subdiv_max = latent_subdiv_max
        self.latent_subdiv_init = latent_subdiv_init
        self.latent_subdiv_limit = latent_subdiv_limit
        self.latent_padding = latent_padding
        self.latent_bounds_padding = latent_bounds_padding
        self.original_grid_subdiv = original_grid_subdiv
        self.latent_morse_graph_method = latent_morse_graph_method

        # Large sample
        self.large_sample_size = large_sample_size
        self.target_points_per_box = target_points_per_box

        # Visualization
        self.n_grid_points = n_grid_points

        # Map function (to be set)
        self.map_func = None

    def set_map_func(self, map_func: Callable):
        """Set the map function f: R^3 -> R^3."""
        self.map_func = map_func

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {
            'system_type': self.system_type,
            'dynamics_name': self.dynamics_name,
            'domain_bounds': self.domain_bounds,
            'subdiv_min': self.subdiv_min,
            'subdiv_max': self.subdiv_max,
            'subdiv_init': self.subdiv_init,
            'subdiv_limit': self.subdiv_limit,
            'padding': self.padding,
            'n_trajectories': self.n_trajectories,
            'n_points': self.n_points,
            'skip_initial': self.skip_initial,
            'random_seed': self.random_seed,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_activation': self.output_activation,
            'encoder_activation': self.encoder_activation,
            'decoder_activation': self.decoder_activation,
            'latent_dynamics_activation': self.latent_dynamics_activation,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'early_stopping_patience': self.early_stopping_patience,
            'min_delta': self.min_delta,
            'w_recon': self.w_recon,
            'w_dyn_recon': self.w_dyn_recon,
            'w_dyn_cons': self.w_dyn_cons,
            'latent_subdiv_min': self.latent_subdiv_min,
            'latent_subdiv_max': self.latent_subdiv_max,
            'latent_subdiv_init': self.latent_subdiv_init,
            'latent_subdiv_limit': self.latent_subdiv_limit,
            'latent_padding': self.latent_padding,
            'latent_bounds_padding': self.latent_bounds_padding,
            'original_grid_subdiv': self.original_grid_subdiv,
            'latent_morse_graph_method': self.latent_morse_graph_method,
            'large_sample_size': self.large_sample_size,
            'target_points_per_box': self.target_points_per_box,
            'n_grid_points': self.n_grid_points,
        }
        
        # Include component-specific architecture parameters if set
        if self.encoder_hidden_dim is not None:
            config_dict['encoder_hidden_dim'] = self.encoder_hidden_dim
        if self.encoder_num_layers is not None:
            config_dict['encoder_num_layers'] = self.encoder_num_layers
        if self.decoder_hidden_dim is not None:
            config_dict['decoder_hidden_dim'] = self.decoder_hidden_dim
        if self.decoder_num_layers is not None:
            config_dict['decoder_num_layers'] = self.decoder_num_layers
        if self.latent_dynamics_hidden_dim is not None:
            config_dict['latent_dynamics_hidden_dim'] = self.latent_dynamics_hidden_dim
        if self.latent_dynamics_num_layers is not None:
            config_dict['latent_dynamics_num_layers'] = self.latent_dynamics_num_layers
        
        return config_dict


def setup_experiment_dirs(base_dir: str) -> Dict[str, str]:
    """
    Create organized directory structure for experiment outputs.

    Args:
        base_dir: Base directory for this experiment

    Returns:
        Dictionary with paths to subdirectories:
            - 'base': Base directory
            - 'training_data': Training data directory
            - 'models': Model checkpoints directory
            - 'results': Results and visualizations directory

    Example:
        >>> dirs = setup_experiment_dirs('/path/to/experiment')
        >>> save_trajectory_data(os.path.join(dirs['training_data'], 'data.npz'), X, Y, trajs, {})
    """
    dirs = {
        'base': base_dir,
        'training_data': os.path.join(base_dir, 'training_data'),
        'models': os.path.join(base_dir, 'models'),
        'results': os.path.join(base_dir, 'results'),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def save_experiment_metadata(
    filepath: str,
    config: ExperimentConfig,
    results: Dict[str, Any]
) -> None:
    """
    Save experiment configuration and results to JSON.

    Args:
        filepath: Path to save JSON file
        config: ExperimentConfig instance
        results: Dictionary of experiment results

    Example:
        >>> save_experiment_metadata(
        ...     'experiment/metadata.json',
        ...     config,
        ...     {'num_morse_sets_3d': 4, 'final_train_loss': 0.0012}
        ... )
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'configuration': config.to_dict(),
        'results': results,
    }

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved experiment metadata to: {filepath}")


__all__ = [
    'ExperimentConfig',
    'setup_experiment_dirs',
    'save_experiment_metadata',
]

