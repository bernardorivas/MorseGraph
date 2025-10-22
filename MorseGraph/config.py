"""
Configuration management for MorseGraph experiments.

This module provides utilities for loading, validating, and managing
experiment configurations from YAML files. Supports hierarchical configs,
validation, and integration with the ExperimentConfig class.
"""

import os
import yaml
from typing import Dict, Any, Optional
from functools import partial


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed

    Example:
        >>> config_dict = load_yaml_config('configs/ives_default.yaml')
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML config {config_path}: {e}")

    if config_dict is None:
        raise ValueError(f"Empty config file: {config_path}")

    return config_dict


def validate_config(config_dict: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.

    Checks for required fields and valid value ranges.

    Args:
        config_dict: Configuration dictionary to validate

    Raises:
        ValueError: If validation fails

    Example:
        >>> validate_config(config_dict)
    """
    required_sections = ['domain', 'cmgdb_3d']

    # Check required sections exist
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Missing required config section: '{section}'")

    # Validate domain bounds
    if 'bounds' in config_dict['domain']:
        bounds = config_dict['domain']['bounds']
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("domain.bounds must be a list of [lower, upper] bounds")
        if not isinstance(bounds[0], list) or not isinstance(bounds[1], list):
            raise ValueError("domain.bounds must be [[x_min, y_min, z_min], [x_max, y_max, z_max]]")
        if len(bounds[0]) != len(bounds[1]):
            raise ValueError("domain.bounds lower and upper must have same dimension")

    # Validate CMGDB parameters
    cmgdb_3d = config_dict['cmgdb_3d']
    if 'subdiv_min' in cmgdb_3d and 'subdiv_max' in cmgdb_3d:
        if cmgdb_3d['subdiv_min'] > cmgdb_3d['subdiv_max']:
            raise ValueError("cmgdb_3d.subdiv_min must be <= subdiv_max")

    if 'cmgdb_2d' in config_dict:
        cmgdb_2d = config_dict['cmgdb_2d']
        if 'subdiv_min' in cmgdb_2d and 'subdiv_max' in cmgdb_2d:
            if cmgdb_2d['subdiv_min'] > cmgdb_2d['subdiv_max']:
                raise ValueError("cmgdb_2d.subdiv_min must be <= subdiv_max")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Override config takes precedence. Handles nested dictionaries recursively.

    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to override with

    Returns:
        Merged configuration dictionary

    Example:
        >>> base = {'cmgdb_3d': {'subdiv_min': 30}}
        >>> override = {'cmgdb_3d': {'subdiv_max': 42}}
        >>> merged = merge_configs(base, override)
        >>> # Result: {'cmgdb_3d': {'subdiv_min': 30, 'subdiv_max': 42}}
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


def load_experiment_config(config_path: str, system_type: str = 'map',
                          base_config_path: Optional[str] = None,
                          verbose: bool = True) -> 'ExperimentConfig':
    """
    Load ExperimentConfig from YAML file.

    This is the main entry point for loading configurations. It handles:
    - Loading YAML file
    - Optional inheritance from base config
    - Validation
    - Conversion to ExperimentConfig object

    Args:
        config_path: Path to YAML configuration file
        system_type: Type of dynamical system ('map' or 'ode')
        base_config_path: Optional path to base config to inherit from
        verbose: Whether to print loading messages

    Returns:
        ExperimentConfig object initialized from YAML

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails

    Example:
        >>> from MorseGraph.config import load_experiment_config
        >>> config = load_experiment_config('configs/ives_default.yaml')
        >>> print(f"Latent dim: {config.latent_dim}")
    """
    from MorseGraph.utils import ExperimentConfig

    if verbose:
        print(f"Loading config from: {config_path}")

    # Load main config
    config_dict = load_yaml_config(config_path)

    # Optionally inherit from base config
    if base_config_path is not None:
        if verbose:
            print(f"  Inheriting from: {base_config_path}")
        base_dict = load_yaml_config(base_config_path)
        config_dict = merge_configs(base_dict, config_dict)

    # Validate configuration
    validate_config(config_dict)

    # Extract parameters for ExperimentConfig
    params = {}

    # Domain parameters
    if 'domain' in config_dict:
        if 'bounds' in config_dict['domain']:
            params['domain_bounds'] = config_dict['domain']['bounds']

    # 3D CMGDB parameters
    if 'cmgdb_3d' in config_dict:
        cmgdb_3d = config_dict['cmgdb_3d']
        for key in ['subdiv_min', 'subdiv_max', 'subdiv_init', 'subdiv_limit', 'padding']:
            if key in cmgdb_3d:
                params[key] = cmgdb_3d[key]

    # 2D CMGDB parameters
    if 'cmgdb_2d' in config_dict:
        cmgdb_2d = config_dict['cmgdb_2d']
        mapping = {
            'subdiv_min': 'latent_subdiv_min',
            'subdiv_max': 'latent_subdiv_max',
            'subdiv_init': 'latent_subdiv_init',
            'subdiv_limit': 'latent_subdiv_limit',
            'padding': 'latent_padding',
            'bounds_padding': 'latent_bounds_padding',
            'original_grid_subdiv': 'original_grid_subdiv',
            'neighbor_radius_factor': 'neighbor_radius_factor'
        }
        for yaml_key, param_key in mapping.items():
            if yaml_key in cmgdb_2d:
                params[param_key] = cmgdb_2d[yaml_key]

    # Training parameters
    if 'training' in config_dict:
        training = config_dict['training']
        for key in ['n_trajectories', 'n_points', 'skip_initial', 'random_seed',
                    'num_epochs', 'batch_size', 'learning_rate',
                    'early_stopping_patience', 'min_delta']:
            if key in training:
                params[key] = training[key]

    # Model architecture parameters
    if 'model' in config_dict:
        model = config_dict['model']

        # Always load these
        for key in ['input_dim', 'latent_dim']:
            if key in model:
                params[key] = model[key]

        # Check if using advanced mode (component-specific) or simple mode (shared)
        using_advanced_mode = any(k in model for k in ['encoder_hidden_dim', 'decoder_hidden_dim', 'latent_dynamics_hidden_dim'])

        if using_advanced_mode:
            # Advanced mode: Component-specific architecture
            for key in ['encoder_hidden_dim', 'encoder_num_layers', 'encoder_activation',
                       'decoder_hidden_dim', 'decoder_num_layers', 'decoder_activation',
                       'latent_dynamics_hidden_dim', 'latent_dynamics_num_layers', 'latent_dynamics_activation']:
                if key in model:
                    params[key] = model[key]
        else:
            # Simple mode (backward compatible): Shared architecture
            for key in ['hidden_dim', 'num_layers', 'output_activation',
                       'encoder_activation', 'decoder_activation', 'latent_dynamics_activation']:
                if key in model:
                    params[key] = model[key]

    # Loss weights
    if 'loss_weights' in config_dict:
        loss = config_dict['loss_weights']
        for key in ['w_recon', 'w_dyn_recon', 'w_dyn_cons']:
            if key in loss:
                params[key] = loss[key]

    # Visualization parameters
    if 'visualization' in config_dict:
        viz = config_dict['visualization']
        if 'n_grid_points' in viz:
            params['n_grid_points'] = viz['n_grid_points']

    # Create ExperimentConfig
    config = ExperimentConfig(**params)

    # Store original config dict for reference
    config._yaml_config = config_dict

    if verbose:
        print(f"  ✓ Config loaded successfully")
        if 'system' in config_dict and 'name' in config_dict['system']:
            print(f"  System: {config_dict['system']['name']}")

    return config


def save_config_to_yaml(config: 'ExperimentConfig', output_path: str) -> None:
    """
    Save ExperimentConfig to YAML file.

    Useful for saving the exact configuration used in an experiment.

    Args:
        config: ExperimentConfig object to save
        output_path: Path where YAML file will be saved

    Example:
        >>> save_config_to_yaml(config, 'run_001/config.yaml')
    """
    # Build config dictionary from ExperimentConfig
    config_dict = {
        'domain': {
            'bounds': config.domain_bounds
        },
        'cmgdb_3d': {
            'subdiv_min': config.subdiv_min,
            'subdiv_max': config.subdiv_max,
            'subdiv_init': config.subdiv_init,
            'subdiv_limit': config.subdiv_limit,
            'padding': config.padding
        },
        'cmgdb_2d': {
            'subdiv_min': config.latent_subdiv_min,
            'subdiv_max': config.latent_subdiv_max,
            'subdiv_init': config.latent_subdiv_init,
            'subdiv_limit': config.latent_subdiv_limit,
            'padding': config.latent_padding,
            'bounds_padding': config.latent_bounds_padding
        },
        'training': {
            'n_trajectories': config.n_trajectories,
            'n_points': config.n_points,
            'skip_initial': config.skip_initial,
            'random_seed': config.random_seed,
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'early_stopping_patience': config.early_stopping_patience,
            'min_delta': config.min_delta
        },
        'model': {
            'input_dim': config.input_dim,
            'latent_dim': config.latent_dim,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'output_activation': config.output_activation
        },
        'loss_weights': {
            'w_recon': config.w_recon,
            'w_dyn_recon': config.w_dyn_recon,
            'w_dyn_cons': config.w_dyn_cons
        },
        'visualization': {
            'n_grid_points': config.n_grid_points
        }
    }

    # Include original YAML config if available
    if hasattr(config, '_yaml_config'):
        # Preserve system info and dynamics params
        if 'system' in config._yaml_config:
            config_dict['system'] = config._yaml_config['system']
        if 'dynamics' in config._yaml_config:
            config_dict['dynamics'] = config._yaml_config['dynamics']

    # Save to YAML
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
