"""
CMGDB caching utilities.

This module provides functions for caching Morse graph computations, including
hash computation, cache path management, and save/load operations for various
types of cached data (Morse graphs, trajectory data, autoencoder models).
"""

import os
import numpy as np
from typing import Callable, Dict, Tuple, List, Optional, Any, Union

# Import from other utils modules
from .latent import compute_latent_bounds
from .trajectory import generate_map_trajectory_data
from .io import _load_trajectory_data_file


def _raise_not_implemented(method_name: str):
    """Helper to raise NotImplementedError for unsupported methods."""
    raise NotImplementedError(f"Method '{method_name}' is not yet implemented")


# =============================================================================
# Hash Computation Functions
# =============================================================================

def get_cache_paths(base_dir: str, cache_type: str, hash_key: str) -> Dict[str, str]:
    """
    Build standard cache directory structure for Morse graph data.
    
    Args:
        base_dir: Base output directory
        cache_type: Type of cache ('cmgdb_2d', 'cmgdb_3d', etc.)
        hash_key: Hash key identifying the computation
    
    Returns:
        Dictionary with paths:
        - 'cache_dir': Main cache directory
        - 'metadata': Path to metadata.json
        - 'morse_graph': Path to morse_graph.pkl
        - 'barycenters': Path to barycenters.npz
    """
    cache_dir = os.path.join(base_dir, cache_type, hash_key)
    return {
        'cache_dir': cache_dir,
        'metadata': os.path.join(cache_dir, 'metadata.json'),
        'morse_graph': os.path.join(cache_dir, 'morse_graph.pkl'),
        'barycenters': os.path.join(cache_dir, 'barycenters.npz'),
    }


def compute_parameter_hash(
    func,
    domain_bounds,
    subdiv_min,
    subdiv_max,
    subdiv_init,
    subdiv_limit,
    padding,
    extra_params=None
) -> str:
    """
    Compute unique hash for CMGDB computation parameters.

    This hash is used as a cache key to avoid recomputing Morse graphs
    with identical parameters. The hash includes:
    - Function source code (to detect if map changed)
    - Domain bounds
    - All CMGDB subdivision parameters
    - Optional extra parameters (for 2D computations with learned models)

    Args:
        func: Map function (will hash its source code)
        domain_bounds: Domain boundaries
        subdiv_min, subdiv_max, subdiv_init, subdiv_limit: CMGDB parameters
        padding: Whether padding is used
        extra_params: Optional dict of additional parameters to include in hash

    Returns:
        SHA256 hash string (first 16 characters for readability)

    Example:
        >>> hash_key = compute_parameter_hash(
        ...     my_map_func,
        ...     [[-5,-5,-5], [5,5,5]],
        ...     subdiv_min=30,
        ...     subdiv_max=42,
        ...     subdiv_init=0,
        ...     subdiv_limit=10000,
        ...     padding=True
        ... )
    """
    import hashlib
    import json
    import inspect
    import functools

    # Create parameter dictionary
    params = {
        'domain_bounds': domain_bounds,
        'subdiv_min': subdiv_min,
        'subdiv_max': subdiv_max,
        'subdiv_init': subdiv_init,
        'subdiv_limit': subdiv_limit,
        'padding': padding,
    }

    if extra_params is not None:
        params['extra_params'] = extra_params

    # Try to get function source code for hashing
    # If not available (e.g., built-in or lambda), use function name
    try:
        func_source = inspect.getsource(func)
    except (OSError, TypeError):
        # Fallback for partial functions or other callables
        if isinstance(func, functools.partial):
            # Access the original function from the partial object
            func_source = f"{func.func.__module__}.{func.func.__name__}"

            # CRITICAL: Include bound parameters in hash
            # This ensures changes to dynamics parameters trigger recomputation
            if func.keywords:
                # Sort keywords for consistent hashing
                sorted_kwargs = sorted(func.keywords.items())
                func_source += f"_kwargs:{sorted_kwargs}"
            if func.args:
                func_source += f"_args:{func.args}"
        else:
            # Fallback for other cases (e.g., built-in functions)
            func_source = f"{func.__module__}.{func.__name__}"

    params['function_source'] = func_source

    # Create sorted JSON string for consistent hashing
    params_str = json.dumps(params, sort_keys=True)

    # Compute SHA256 hash
    hash_obj = hashlib.sha256(params_str.encode('utf-8'))
    hash_full = hash_obj.hexdigest()

    # Return first 16 characters for readability
    return hash_full[:16]


def compute_trajectory_hash(config, cmgdb_3d_hash: str) -> str:
    """
    Compute hash for trajectory data generation configuration.

    This hash depends on the 3D CMGDB hash (which includes the map function
    and domain) plus all trajectory generation parameters to enable caching
    of generated trajectory data.

    Args:
        config: Experiment configuration object
        cmgdb_3d_hash: Hash of 3D CMGDB computation (includes map and domain)

    Returns:
        SHA256 hash string (first 16 characters)
    """
    import hashlib
    import json

    params = {
        # Dependency on 3D computation (includes map function and domain)
        '3d_hash': cmgdb_3d_hash,

        # Trajectory generation parameters
        'n_trajectories': config.n_trajectories,
        'n_points': config.n_points,
        'skip_initial': config.skip_initial,
        'random_seed': config.random_seed,
    }

    # Create sorted JSON string for consistent hashing
    params_str = json.dumps(params, sort_keys=True)

    # Compute SHA256 hash
    hash_obj = hashlib.sha256(params_str.encode('utf-8'))
    hash_full = hash_obj.hexdigest()

    return hash_full[:16]


def compute_training_hash(
    config_traj: Dict[str, Any],
    input_dim: int,
    latent_dim: int,
    hidden_dim: int,
    num_layers: int,
    w_recon: float,
    w_dyn_recon: float,
    w_dyn_cons: float,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    early_stopping_patience: int,
    min_delta: float,
    encoder_activation: Optional[str],
    decoder_activation: Optional[str],
    latent_dynamics_activation: Optional[str]
) -> str:
    """
    Compute hash for training configuration.
    """
    import hashlib
    import json

    # Serialize the trajectory config to get a consistent hash base
    # We can use the hash of the config if it has one, or just the config itself
    # Assuming config_traj is a dict
    config_str = json.dumps(config_traj, sort_keys=True)
    traj_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:16]

    params = {
        'traj_config_hash': traj_hash,
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'w_recon': w_recon,
        'w_dyn_recon': w_dyn_recon,
        'w_dyn_cons': w_dyn_cons,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'early_stopping_patience': early_stopping_patience,
        'min_delta': min_delta,
        'encoder_activation': str(encoder_activation),
        'decoder_activation': str(decoder_activation),
        'latent_dynamics_activation': str(latent_dynamics_activation)
    }

    # Create sorted JSON string for consistent hashing
    params_str = json.dumps(params, sort_keys=True)

    # Compute SHA256 hash
    hash_obj = hashlib.sha256(params_str.encode('utf-8'))
    hash_full = hash_obj.hexdigest()

    return hash_full[:16]


def compute_cmgdb_2d_hash(
    config_train: Dict[str, Any],
    method: str,
    subdiv_min: int,
    subdiv_max: int,
    subdiv_init: int,
    subdiv_limit: int,
    padding: bool,
    original_grid_subdiv: int,
    latent_bounds: List[List[float]]
) -> str:
    """
    Compute hash for 2D CMGDB configuration.
    """
    import hashlib
    import json

    # Serialize the training config
    config_str = json.dumps(config_train, sort_keys=True)
    train_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:16]

    params = {
        'training_config_hash': train_hash,
        'method': method,
        'subdiv_min': subdiv_min,
        'subdiv_max': subdiv_max,
        'subdiv_init': subdiv_init,
        'subdiv_limit': subdiv_limit,
        'padding': padding,
        'original_grid_subdiv': original_grid_subdiv,
        'latent_bounds': latent_bounds
    }

    # Create sorted JSON string for consistent hashing
    params_str = json.dumps(params, sort_keys=True)

    # Compute SHA256 hash
    hash_obj = hashlib.sha256(params_str.encode('utf-8'))
    hash_full = hash_obj.hexdigest()

    return hash_full[:16]


def compute_cmgdb_3d_hash(
    dynamics_name: str,
    domain_bounds: List[List[float]],
    subdiv_min: int,
    subdiv_max: int,
    subdiv_init: int,
    subdiv_limit: int,
    padding: bool,
    system_parameters: Dict[str, Any]
) -> str:
    """Compute unique hash for 3D CMGDB computation."""
    import hashlib
    import json
    
    params = {
        'dynamics_name': dynamics_name,
        'domain_bounds': domain_bounds,
        'subdiv_min': subdiv_min,
        'subdiv_max': subdiv_max,
        'subdiv_init': subdiv_init,
        'subdiv_limit': subdiv_limit,
        'padding': padding,
        'system_parameters': system_parameters
    }
    
    params_str = json.dumps(params, sort_keys=True)
    hash_obj = hashlib.sha256(params_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


def compute_trajectory_data_hash(
    config_3d: Dict[str, Any],
    n_trajectories: int,
    n_points: int,
    skip_initial: int,
    random_seed: int
) -> str:
    """Compute unique hash for trajectory data."""
    import hashlib
    import json
    
    # Hash of the config used for 3D computation (which includes dynamics info)
    # We use this as a base to ensure trajectories correspond to the same system
    config_str = json.dumps(config_3d, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:16]
    
    params = {
        'base_config_hash': config_hash,
        'n_trajectories': n_trajectories,
        'n_points': n_points,
        'skip_initial': skip_initial,
        'random_seed': random_seed
    }
    
    params_str = json.dumps(params, sort_keys=True)
    hash_obj = hashlib.sha256(params_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


# =============================================================================
# Cache Path Management
# =============================================================================

def get_cache_path(cache_dir: str, param_hash: str) -> Dict[str, str]:
    """
    Get cache file paths for a given parameter hash.

    Args:
        cache_dir: Base cache directory
        param_hash: Parameter hash from compute_parameter_hash()

    Returns:
        Dictionary with paths:
            - 'dir': Cache subdirectory for this hash
            - 'morse_graph': Path to morse_graph_data (native CMGDB format)
            - 'barycenters': Path to barycenters.npz
            - 'metadata': Path to metadata.json

    Example:
        >>> paths = get_cache_path('/path/to/cache', 'a1b2c3d4e5f6g7h8')
        >>> print(paths['morse_graph'])
        /path/to/cache/a1b2c3d4e5f6g7h8/morse_graph_data
    """
    cache_subdir = os.path.join(cache_dir, param_hash)

    return {
        'dir': cache_subdir,
        'morse_graph': os.path.join(cache_subdir, 'morse_graph_data'),
        'barycenters': os.path.join(cache_subdir, 'barycenters.npz'),
        'metadata': os.path.join(cache_subdir, 'metadata.json'),
    }


# =============================================================================
# CachedMorseGraph Class
# =============================================================================

class CachedMorseGraph:
    """
    Lightweight wrapper for cached Morse graph data that mimics CMGDB.MorseGraph interface.

    Uses NetworkX DiGraph internally for easy graph manipulation while providing
    CMGDB-compatible interface. This avoids dealing with C++ chomp DirectedGraph
    structure and allows full Python-based graph analysis.
    """
    def __init__(self, graph=None):
        """
        Args:
            graph: NetworkX DiGraph with morse_set_boxes stored as node attributes.
                   If None, creates empty graph.
        """
        import networkx as nx

        if graph is None:
            self._graph = nx.DiGraph()
        else:
            self._graph = graph

    @property
    def graph(self):
        """Access underlying NetworkX DiGraph for advanced analysis."""
        return self._graph

    def num_vertices(self) -> int:
        """Return number of vertices in the Morse graph."""
        return self._graph.number_of_nodes()

    def vertices(self) -> list:
        """Return list of vertex IDs."""
        return list(self._graph.nodes())

    def adjacencies(self, vertex: int) -> list:
        """Return list of vertices that the given vertex has edges to."""
        return list(self._graph.successors(vertex))

    def morse_set_boxes(self, vertex: int) -> list:
        """Return list of boxes for the given Morse set."""
        if vertex not in self._graph.nodes:
            return []
        return self._graph.nodes[vertex].get('morse_set_boxes', [])

    def edges(self) -> list:
        """Return list of edges as (source, target) tuples."""
        return list(self._graph.edges())


# =============================================================================
# Autoencoder Caching
# =============================================================================

def _prepare_autoencoder_cache_paths(output_dir: str, training_hash: str) -> Dict[str, str]:
    """Prepare all cache paths for autoencoder training."""
    cache_dir = os.path.join(output_dir, 'training', training_hash)
    models_dir = os.path.join(cache_dir, 'models')
    return {
        'cache_dir': cache_dir,
        'models_dir': models_dir,
        'metadata': os.path.join(cache_dir, 'metadata.json'),
        'losses': os.path.join(cache_dir, 'training_losses.pkl'),
        'bounds': os.path.join(cache_dir, 'latent_bounds.npz'),
        'encoder': os.path.join(models_dir, 'encoder.pt'),
        'decoder': os.path.join(models_dir, 'decoder.pt'),
        'latent_dynamics': os.path.join(models_dir, 'latent_dynamics.pt'),
    }


def _check_autoencoder_cache_valid(paths: Dict[str, str]) -> bool:
    """Check if all required cache files exist."""
    return all(os.path.exists(paths[key]) for key in ['encoder', 'decoder', 'latent_dynamics', 
                                                       'metadata', 'losses', 'bounds'])


def _load_cached_autoencoder(config, paths: Dict[str, str]) -> Dict:
    """Load autoencoder models and metadata from cache."""
    import pickle
    import torch
    import json
    from MorseGraph.models import Encoder, Decoder, LatentDynamics

    with open(paths['metadata'], 'r') as f:
        metadata = json.load(f)

    encoder = Encoder(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        output_activation=config.encoder_activation
    )
    encoder.load_state_dict(torch.load(paths['encoder']))
    encoder.eval()

    decoder = Decoder(
        latent_dim=config.latent_dim,
        output_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        output_activation=config.decoder_activation
    )
    decoder.load_state_dict(torch.load(paths['decoder']))
    decoder.eval()

    latent_dynamics = LatentDynamics(
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        output_activation=config.latent_dynamics_activation
    )
    latent_dynamics.load_state_dict(torch.load(paths['latent_dynamics']))
    latent_dynamics.eval()

    with open(paths['losses'], 'rb') as f:
        training_losses = pickle.load(f)

    latent_bounds_data = np.load(paths['bounds'])
    latent_bounds = latent_bounds_data['bounds']

    return {
        'encoder': encoder,
        'decoder': decoder,
        'latent_dynamics': latent_dynamics,
        'training_losses': training_losses,
        'latent_bounds': latent_bounds,
        'config': config,
        'metadata': metadata
    }


def _save_trained_autoencoder(training_result: Dict, training_data: Dict, 
                              config, training_hash: str, paths: Dict[str, str]) -> None:
    """Save trained autoencoder models and metadata to cache."""
    import pickle
    import torch
    import json
    from datetime import datetime

    os.makedirs(paths['models_dir'], exist_ok=True)

    torch.save(training_result['encoder'].state_dict(), paths['encoder'])
    torch.save(training_result['decoder'].state_dict(), paths['decoder'])
    torch.save(training_result['latent_dynamics'].state_dict(), paths['latent_dynamics'])

    device = training_result['device']
    encoder = training_result['encoder']
    with torch.no_grad():
        z_train = encoder(torch.FloatTensor(training_data['X_train']).to(device)).cpu().numpy()

    latent_bounds = compute_latent_bounds(z_train, padding_factor=config.latent_bounds_padding)

    training_losses = {
        'train': training_result['train_losses'],
        'val': training_result['val_losses']
    }
    with open(paths['losses'], 'wb') as f:
        pickle.dump(training_losses, f)

    np.savez(paths['bounds'], bounds=latent_bounds)

    metadata = {
        'training_hash': training_hash,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'input_dim': config.input_dim,
            'latent_dim': config.latent_dim,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'n_trajectories': config.n_trajectories,
            'n_points': config.n_points,
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
        }
    }

    with open(paths['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Training results cached to {paths['cache_dir']}")


def load_or_train_autoencoder(
    config,
    training_hash: str,
    training_data: Dict,
    map_func: Callable,
    output_dir: str = 'examples/ives_model_output',
    force_retrain: bool = False
) -> Tuple[Dict, bool]:
    """
    Load cached autoencoder models or train new ones if cache doesn't exist.

    Args:
        config: Configuration object with training parameters
        training_hash: Hash identifying this training configuration
        training_data: Dictionary with 'X_train', 'Xnext_train', 'X_val', 'Xnext_val' arrays
        map_func: Original map function for validation
        output_dir: Base output directory
        force_retrain: If True, ignore cache and retrain

    Returns:
        Tuple of (training_result, was_cached):
            - training_result: Dict with keys:
                - 'encoder': Trained encoder model
                - 'decoder': Trained decoder model
                - 'latent_dynamics': Trained latent dynamics model
                - 'training_losses': Dict of loss curves
                - 'latent_bounds': Bounds of latent space
                - 'config': Configuration used
            - was_cached: True if loaded from cache, False if newly trained
    """
    paths = _prepare_autoencoder_cache_paths(output_dir, training_hash)

    if _check_autoencoder_cache_valid(paths) and not force_retrain:
        print(f"Loading cached training results from {paths['cache_dir']}")
        training_result = _load_cached_autoencoder(config, paths)
        return training_result, True

    print(f"Training new autoencoder (hash: {training_hash})")
    from MorseGraph.training import train_autoencoder_dynamics

    training_result = train_autoencoder_dynamics(
        x_train=training_data['X_train'],
        y_train=training_data['Xnext_train'],
        x_val=training_data['X_val'],
        y_val=training_data['Xnext_val'],
        config=config,
        verbose=True,
        progress_interval=100
    )

    _save_trained_autoencoder(training_result, training_data, config, training_hash, paths)

    import torch
    device = training_result['device']
    encoder = training_result['encoder']
    with torch.no_grad():
        z_train = encoder(torch.FloatTensor(training_data['X_train']).to(device)).cpu().numpy()

    latent_bounds = compute_latent_bounds(z_train, padding_factor=config.latent_bounds_padding)
    training_losses = {
        'train': training_result['train_losses'],
        'val': training_result['val_losses']
    }

    training_result['latent_bounds'] = latent_bounds
    training_result['training_losses'] = training_losses

    return training_result, False


# =============================================================================
# Trajectory Data Caching
# =============================================================================

def load_or_generate_trajectory_data(
    config,
    trajectory_hash: str,
    map_func: Callable,
    domain_bounds: np.ndarray,
    output_dir: str = 'examples/ives_model_output',
    force_regenerate: bool = False
) -> Tuple[Dict, bool]:
    """
    Load cached trajectory data or generate new data if cache doesn't exist.

    Args:
        config: Configuration object with trajectory generation parameters
        trajectory_hash: Hash identifying this trajectory configuration
        map_func: Map function for trajectory generation
        domain_bounds: Domain bounds for sampling initial conditions
        output_dir: Base output directory
        force_regenerate: If True, ignore cache and regenerate data

    Returns:
        Tuple of (trajectory_result, was_cached):
            - trajectory_result: Dict with keys:
                - 'X': Current states array
                - 'Y': Next states array
                - 'trajectories': List of trajectory arrays
                - 'config': Configuration used
            - was_cached: True if loaded from cache, False if newly generated
    """
    import json
    from datetime import datetime

    # Define cache directory
    cache_dir = os.path.join(output_dir, 'trajectory_data', trajectory_hash)
    metadata_path = os.path.join(cache_dir, 'metadata.json')
    X_path = os.path.join(cache_dir, 'X.npz')
    Y_path = os.path.join(cache_dir, 'Y.npz')
    trajectories_path = os.path.join(cache_dir, 'trajectories.npz')

    # Check if cache exists and is valid
    cache_valid = (
        os.path.exists(X_path) and
        os.path.exists(Y_path) and
        os.path.exists(trajectories_path) and
        os.path.exists(metadata_path)
    )

    if cache_valid and not force_regenerate:
        print(f"Loading cached trajectory data from {cache_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load trajectory data
        X_data = np.load(X_path)
        X = X_data['X']

        Y_data = np.load(Y_path)
        Y = Y_data['Y']

        trajectories_data = np.load(trajectories_path, allow_pickle=True)
        trajectories = list(trajectories_data['trajectories'])

        trajectory_result = {
            'X': X,
            'Y': Y,
            'trajectories': trajectories,
            'config': config,
            'metadata': metadata
        }

        return trajectory_result, True

    else:
        # Generate new trajectory data
        print(f"Generating trajectory data (hash: {trajectory_hash})")

        X, Y, trajectories = generate_map_trajectory_data(
            map_func,
            config.n_trajectories,
            config.n_points,
            domain_bounds,
            random_seed=config.random_seed,
            skip_initial=config.skip_initial
        )

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Save trajectory data
        np.savez(X_path, X=X)
        np.savez(Y_path, Y=Y)
        np.savez(trajectories_path, trajectories=trajectories)

        # Save metadata
        metadata = {
            'trajectory_hash': trajectory_hash,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'n_trajectories': config.n_trajectories,
                'n_points': config.n_points,
                'skip_initial': config.skip_initial,
                'random_seed': config.random_seed,
            },
            'data_shapes': {
                'X': X.shape,
                'Y': Y.shape,
                'n_trajectories': len(trajectories),
            }
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Trajectory data cached to {cache_dir}")

        trajectory_result = {
            'X': X,
            'Y': Y,
            'trajectories': trajectories,
            'config': config,
            'metadata': metadata
        }

        return trajectory_result, False


def save_trajectory_data(directory: str, data: Dict[str, Any]) -> None:
    """Save trajectory data dict to directory."""
    import json
    
    os.makedirs(directory, exist_ok=True)
    
    np.savez_compressed(os.path.join(directory, 'data.npz'), X=data['X'], Y=data['Y'])
    
    if 'config' in data:
        with open(os.path.join(directory, 'config.json'), 'w') as f:
            json.dump(data['config'], f, indent=2)


def load_trajectory_data(directory: str) -> Optional[Dict[str, Any]]:
    """
    Load trajectory data from directory (preferred) or file (legacy).
    Returns dict with keys 'X', 'Y', 'config'.
    """
    import json
    
    # If passed a file path, use legacy loader
    if os.path.isfile(directory):
        try:
            X, Y, _, meta = _load_trajectory_data_file(directory)
            return {'X': X, 'Y': Y, 'config': meta}
        except Exception:
            return None

    # Directory loading
    if not os.path.exists(os.path.join(directory, 'data.npz')):
        return None
        
    try:
        data = np.load(os.path.join(directory, 'data.npz'))
        X = data['X']
        Y = data['Y']
        
        config = None
        if os.path.exists(os.path.join(directory, 'config.json')):
            with open(os.path.join(directory, 'config.json'), 'r') as f:
                config = json.load(f)
                
        return {'X': X, 'Y': Y, 'config': config}
    except Exception as e:
        print(f"Error loading trajectory data from {directory}: {e}")
        return None


# =============================================================================
# 2D Morse Graph Caching
# =============================================================================

def _load_cached_2d_morse_graph(cache_paths: Dict[str, str], config) -> Tuple[Dict, bool]:
    """Load cached 2D Morse graph from disk."""
    import pickle
    import json
    
    with open(cache_paths['metadata'], 'r') as f:
        metadata = json.load(f)

    with open(cache_paths['morse_graph'], 'rb') as f:
        morse_graph_nx = pickle.load(f)
    morse_graph = CachedMorseGraph(morse_graph_nx)

    barycenters = {}
    barycenters_keys = metadata.get('barycenters_keys', [])
    if not barycenters_keys:
        # Fallback: extract from metadata keys that match pattern
        barycenters_keys = [k for k in metadata.keys() if k.startswith('morse_set_')]
    
    for key in barycenters_keys:
        barycenters[int(key.split('_')[-1])] = np.array(metadata[key])

    return {
        'morse_graph': morse_graph,
        'barycenters': barycenters,
        'config': config,
        'metadata': metadata,
        'method': metadata['method']
    }, True


def _compute_2d_morse_graph_data_method(config, encoder, decoder, latent_dynamics,
                                       latent_bounds, device):
    """Compute 2D Morse graph using 'data' method."""
    from MorseGraph.core import compute_morse_graph_2d_data
    import torch
    
    print("\nGenerating dense uniform grid in original space...")
    original_grid_subdiv = config.original_grid_subdiv
    input_dim = config.input_dim
    n_per_dim = 2 ** (original_grid_subdiv // input_dim)
    print(f"  Grid: {n_per_dim} points per dimension ({n_per_dim**input_dim} total)")

    domain_bounds = config.domain_bounds
    grid_1d = [np.linspace(domain_bounds[0][i], domain_bounds[1][i], n_per_dim)
               for i in range(input_dim)]
    
    mesh = np.meshgrid(*grid_1d, indexing='ij')
    X_large_grid = np.stack([m.flatten() for m in mesh], axis=1)
    print(f"  Generated {len(X_large_grid)} grid points in original space")

    with torch.no_grad():
        z_large_grid = encoder(torch.FloatTensor(X_large_grid).to(device)).cpu().numpy()
    print(f"  Encoded to latent space: {z_large_grid.shape}")

    print(f"\nComputing Morse graph using BoxMapData (padding={config.latent_padding})...")
    result_2d = compute_morse_graph_2d_data(
        latent_dynamics, device, z_large_grid, latent_bounds.tolist(),
        subdiv_min=config.latent_subdiv_min, subdiv_max=config.latent_subdiv_max,
        subdiv_init=config.latent_subdiv_init, subdiv_limit=config.latent_subdiv_limit,
        padding=config.latent_padding,
        cache_dir=None,
        use_cache=False,
        verbose=True
    )
    
    morse_graph_cmgdb = result_2d['morse_graph']
    barycenters = {}
    for i in range(morse_graph_cmgdb.num_vertices()):
        boxes = morse_graph_cmgdb.morse_set_boxes(i)
        barycenters[i] = [np.array([(box[j] + box[j + 2]) / 2.0 for j in range(2)]) 
                          for box in boxes] if boxes else []
    
    return morse_graph_cmgdb, barycenters


def _compute_2d_morse_graph_restricted_method(config, encoder, latent_dynamics,
                                              latent_bounds, device):
    """Compute 2D Morse graph using 'restricted' method."""
    from MorseGraph.core import _run_cmgdb_compute
    from MorseGraph.dynamics import BoxMapLearnedLatent
    from MorseGraph.grids import UniformGrid
    import torch
    
    print(f"\nComputing Morse graph using restricted method (padding={config.latent_padding})...")
    
    z_train = encoder(torch.FloatTensor(config.trajectory_data['X']).to(device)).detach().cpu().numpy()
    latent_dim = z_train.shape[1]
    
    dims = [2**config.latent_subdiv_max] * latent_dim
    temp_grid = UniformGrid(np.array([latent_bounds[0], latent_bounds[1]]), dims)
    
    cell_size = (np.array(latent_bounds[1]) - np.array(latent_bounds[0])) / np.array(dims)
    indices_vec = np.floor((z_train - np.array(latent_bounds[0])) / cell_size).astype(int)
    indices_vec = np.clip(indices_vec, 0, np.array(dims) - 1)
    flat_indices = np.ravel_multi_index(indices_vec.T, dims)
    active_set = set(flat_indices)
    
    active_array = np.array(list(active_set))
    dilated_array = temp_grid.dilate_indices(active_array, radius=1)
    allowed_indices = set(dilated_array)
    
    print(f"  Restricted domain: {len(active_set)} data boxes -> {len(allowed_indices)} allowed boxes")
    
    pad_val = 1e-6 if config.latent_padding else 0.0
    dynamics = BoxMapLearnedLatent(
        latent_dynamics, device, padding=pad_val, allowed_indices=allowed_indices
    )
    
    morse_graph_cmgdb, morse_sets, barycenters_dict, map_graph = _run_cmgdb_compute(
        dynamics,
        [latent_bounds[0], latent_bounds[1]],
        config.latent_subdiv_min,
        config.latent_subdiv_max,
        config.latent_subdiv_init,
        config.latent_subdiv_limit,
        verbose=True
    )
    
    barycenters = {}
    for i in range(morse_graph_cmgdb.num_vertices()):
        boxes = morse_graph_cmgdb.morse_set_boxes(i)
        barycenters[i] = [np.array([(box[j] + box[j + 2]) / 2.0 for j in range(2)]) 
                          for box in boxes] if boxes else []
    
    return morse_graph_cmgdb, barycenters


def _convert_cmgdb_to_networkx_for_cache(morse_graph_cmgdb):
    """Convert CMGDB MorseGraph to NetworkX for caching."""
    import networkx as nx
    
    nx_graph = nx.DiGraph()
    for v in range(morse_graph_cmgdb.num_vertices()):
        boxes = morse_graph_cmgdb.morse_set_boxes(v)
        nx_graph.add_node(v, morse_set_boxes=[list(b) for b in boxes] if boxes else [])
    for v in range(morse_graph_cmgdb.num_vertices()):
        for target in morse_graph_cmgdb.adjacencies(v):
            nx_graph.add_edge(v, target)
    
    return nx_graph


def _save_2d_morse_graph_cache(cache_paths: Dict[str, str], nx_graph, barycenters,
                               config, cmgdb_2d_hash, method):
    """Save 2D Morse graph to cache."""
    import pickle
    import json
    from datetime import datetime
    
    os.makedirs(cache_paths['cache_dir'], exist_ok=True)
    with open(cache_paths['morse_graph'], 'wb') as f:
        pickle.dump(nx_graph, f)
    
    barycenters_serializable = {f'morse_set_{k}': [arr.tolist() for arr in v] 
                                for k, v in barycenters.items()}
    
    metadata = {
        'cmgdb_2d_hash': cmgdb_2d_hash,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'subdiv_min': config.latent_subdiv_min,
            'subdiv_max': config.latent_subdiv_max,
            'subdiv_init': config.latent_subdiv_init,
            'subdiv_limit': config.latent_subdiv_limit,
            'padding': config.latent_padding,
            'bounds_padding': config.latent_bounds_padding,
            'original_grid_subdiv': config.original_grid_subdiv,
            'method': method
        },
        'num_morse_sets': nx_graph.number_of_nodes(),
        'num_edges': nx_graph.number_of_edges(),
        **barycenters_serializable
    }

    with open(cache_paths['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"2D Morse graph cached to {cache_paths['cache_dir']}")
    return metadata


def load_or_compute_2d_morse_graphs(
    config,
    cmgdb_2d_hash: str,
    encoder,
    decoder,
    latent_dynamics,
    latent_bounds: np.ndarray,
    output_dir: str = 'examples/ives_model_output',
    force_recompute: bool = False
) -> Tuple[Dict, bool]:
    """
    Load cached 2D Morse graph or compute new one if cache doesn't exist.

    Computes the 2D Morse graph in latent space using the method specified in config.

    Args:
        config: Configuration object with 2D CMGDB parameters
        cmgdb_2d_hash: Hash identifying this 2D CMGDB configuration
        encoder: Trained encoder model
        decoder: Trained decoder model
        latent_dynamics: Trained latent dynamics model
        latent_bounds: Bounds of latent space [2, 2] array
        output_dir: Base output directory
        force_recompute: If True, ignore cache and recompute

    Returns:
        Tuple of (morse_2d_result, was_cached):
            - morse_2d_result: Dict with keys:
                - 'morse_graph': 2D Morse graph
                - 'barycenters': Barycenters of Morse sets
                - 'config': Configuration used
                - 'metadata': Metadata dict
                - 'method': The method used for computation
            - was_cached: True if loaded from cache, False if newly computed
    """
    import pickle
    import json
    from datetime import datetime

    cache_paths = get_cache_paths(output_dir, 'cmgdb_2d', cmgdb_2d_hash)
    
    cache_valid = (
        os.path.exists(cache_paths['morse_graph']) and
        os.path.exists(cache_paths['barycenters']) and
        os.path.exists(cache_paths['metadata'])
    )

    if cache_valid and not force_recompute:
        print(f"Loading cached 2D Morse graph from {cache_paths['cache_dir']}")
        return _load_cached_2d_morse_graph(cache_paths, config)

    method = config.latent_morse_graph_method
    print(f"Computing 2D Morse graph (hash: {cmgdb_2d_hash}) using method: {method}")

    import torch
    device = next(encoder.parameters()).device

    _COMPUTATION_METHODS = {
        'data': lambda: _compute_2d_morse_graph_data_method(
            config, encoder, decoder, latent_dynamics, latent_bounds, device
        ),
        'restricted': lambda: _compute_2d_morse_graph_restricted_method(
            config, encoder, latent_dynamics, latent_bounds, device
        ),
        'latent_enclosure': lambda: _raise_not_implemented('latent_enclosure'),
    }

    if method not in _COMPUTATION_METHODS:
        raise ValueError(f"Unknown 2D Morse graph computation method: {method}")

    morse_graph_cmgdb, barycenters = _COMPUTATION_METHODS[method]()

    nx_graph = _convert_cmgdb_to_networkx_for_cache(morse_graph_cmgdb)
    metadata = _save_2d_morse_graph_cache(
        cache_paths, nx_graph, barycenters, config, cmgdb_2d_hash, method
    )

    return {
        'morse_graph': CachedMorseGraph(nx_graph),
        'barycenters': barycenters,
        'config': config,
        'metadata': metadata,
        'method': method
    }, False


# =============================================================================
# 3D Morse Graph Caching
# =============================================================================

def save_morse_graph_cache(
    morse_graph,
    map_graph,
    barycenters: Dict[int, list],
    metadata: Dict[str, Any],
    cache_dir: str,
    param_hash: str,
    verbose: bool = True
) -> None:
    """
    Save CMGDB Morse graph computation to cache.

    Saves three files to enable future loading:
    1. morse_graph_data.pkl: Serialized graph structure and boxes
    2. barycenters.npz: NumPy archive of barycenter coordinates
    3. metadata.json: Parameters and computation info

    Note: map_graph is not cached as it's not needed for visualization/analysis.

    Args:
        morse_graph: CMGDB MorseGraph object
        map_graph: CMGDB MapGraph object (not cached, kept for API compatibility)
        barycenters: Dict mapping Morse set index to list of barycenter arrays
        metadata: Dictionary of parameters and computation info
        cache_dir: Base cache directory
        param_hash: Parameter hash (from compute_parameter_hash)
        verbose: Whether to print progress messages

    Example:
        >>> save_morse_graph_cache(
        ...     morse_graph, map_graph, barycenters,
        ...     {'subdiv_min': 30, 'computation_time': 120.5},
        ...     cache_dir='/path/to/cache',
        ...     param_hash='a1b2c3d4e5f6g7h8'
        ... )
    """
    import json
    import pickle
    from datetime import datetime

    try:
        import CMGDB
    except ImportError:
        if verbose:
            print("  WARNING: CMGDB not available, cannot save cache")
        return

    # Get cache paths
    paths = get_cache_path(cache_dir, param_hash)

    # Create cache directory
    os.makedirs(paths['dir'], exist_ok=True)

    # Save barycenters
    barycenters_to_save = {
        f'morse_set_{k}': np.array(v) if v else np.array([])
        for k, v in barycenters.items()
    }
    if barycenters_to_save:
        np.savez(paths['barycenters'], **barycenters_to_save)

    # Add timestamp to metadata
    metadata_with_timestamp = metadata.copy()
    metadata_with_timestamp['cached_at'] = datetime.now().isoformat()
    metadata_with_timestamp['param_hash'] = param_hash

    # Save metadata
    with open(paths['metadata'], 'w') as f:
        json.dump(metadata_with_timestamp, f, indent=2)

    # Build NetworkX DiGraph from CMGDB MorseGraph
    import networkx as nx

    graph = nx.DiGraph()
    num_verts = morse_graph.num_vertices()

    # Add nodes with morse_set_boxes as attributes
    for v in range(num_verts):
        boxes = morse_graph.morse_set_boxes(v)
        # Convert boxes to list of lists for proper serialization
        boxes_serializable = [list(box) for box in boxes] if boxes else []
        graph.add_node(v, morse_set_boxes=boxes_serializable)

    # Add edges
    for v in range(num_verts):
        adjacent = morse_graph.adjacencies(v)
        for target in adjacent:
            graph.add_edge(v, target)

    # Save NetworkX graph using pickle
    with open(paths['morse_graph'], 'wb') as f:
        pickle.dump(graph, f)

    if verbose:
        print(f"  Cached Morse graph to: {paths['dir']}")


def load_morse_graph_cache(
    cache_dir: str,
    param_hash: str,
    verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Load CMGDB Morse graph from cache if it exists.

    Args:
        cache_dir: Base cache directory
        param_hash: Parameter hash to look for
        verbose: Whether to print progress messages

    Returns:
        Dictionary with loaded data if cache exists, None otherwise:
            - 'morse_graph': CMGDB MorseGraph object (or CachedMorseGraph wrapper)
            - 'barycenters': Dict mapping Morse set index to list of barycenters
            - 'metadata': Cached metadata dict
            - 'num_morse_sets': Number of Morse sets
        Returns None if cache not found or load fails.

    Example:
        >>> cached = load_morse_graph_cache('/path/to/cache', 'a1b2c3d4e5f6g7h8')
        >>> if cached is not None:
        ...     morse_graph = cached['morse_graph']
        ...     print(f"Loaded {cached['num_morse_sets']} Morse sets from cache")
    """
    import json
    import pickle

    # Get cache paths
    paths = get_cache_path(cache_dir, param_hash)

    # Check if cache exists
    if not os.path.exists(paths['morse_graph']):
        return None

    try:
        # Load NetworkX graph from pickle
        with open(paths['morse_graph'], 'rb') as f:
            nx_graph = pickle.load(f)

        # Wrap in CachedMorseGraph for CMGDB-compatible interface
        morse_graph = CachedMorseGraph(nx_graph)

        # Load barycenters
        barycenters = {}
        if os.path.exists(paths['barycenters']):
            barycenters_data = np.load(paths['barycenters'], allow_pickle=True)
            for key in barycenters_data.files:
                morse_set_idx = int(key.split('_')[-1])
                barys_array = barycenters_data[key]
                if barys_array.size > 0:
                    # Convert back to list of arrays
                    if barys_array.ndim == 1:
                        barycenters[morse_set_idx] = [barys_array]
                    else:
                        barycenters[morse_set_idx] = [barys_array[i] for i in range(len(barys_array))]
                else:
                    barycenters[morse_set_idx] = []

        # Load metadata
        metadata = {}
        if os.path.exists(paths['metadata']):
            with open(paths['metadata'], 'r') as f:
                metadata = json.load(f)

        if verbose:
            print(f"  Loaded Morse graph from cache: {paths['dir']}")

        return {
            'morse_graph': morse_graph,
            'barycenters': barycenters,
            'metadata': metadata,
            'num_morse_sets': morse_graph.num_vertices(),
        }

    except Exception as e:
        if verbose:
            error_type = type(e).__name__
            error_msg = str(e)

            # Provide helpful error messages based on error type
            if "pickle" in error_msg.lower() or "unpickling" in error_msg.lower():
                print(f"  WARNING: Cache file corrupted or incompatible format")
                print(f"           Error: {error_type}: {error_msg}")
                print(f"           Cache location: {paths['dir']}")
                print(f"           Suggestion: Delete cache directory to force recomputation")
            elif isinstance(e, FileNotFoundError):
                print(f"  WARNING: Cache files incomplete (missing {e.filename})")
            else:
                print(f"  WARNING: Failed to load cache: {error_type}: {error_msg}")
                print(f"           Cache location: {paths['dir']}")

        return None


def save_morse_graph_data(directory: str, data: Dict[str, Any]) -> None:
    """Save Morse graph data (graph, sets, barycenters, config) to directory."""
    import pickle
    import json
    import networkx as nx
    
    os.makedirs(directory, exist_ok=True)
    
    morse_graph = data['morse_graph']
    
    # Save graph (convert to NetworkX if CMGDB object)
    with open(os.path.join(directory, 'morse_graph.pkl'), 'wb') as f:
        if hasattr(morse_graph, 'num_vertices') and not isinstance(morse_graph, CachedMorseGraph):
            # Convert CMGDB to NetworkX
            nx_graph = nx.DiGraph()
            for v in range(morse_graph.num_vertices()):
                # Store box info if available
                # CMGDB might expose morse_set_boxes
                if hasattr(morse_graph, 'morse_set_boxes'):
                    try:
                        boxes = morse_graph.morse_set_boxes(v)
                        # boxes is likely a list of list/array
                        nx_graph.add_node(v, morse_set_boxes=[list(b) for b in boxes] if boxes else [])
                    except Exception:
                        nx_graph.add_node(v)
                else:
                    nx_graph.add_node(v)
                    
            for v in range(morse_graph.num_vertices()):
                for target in morse_graph.adjacencies(v):
                    nx_graph.add_edge(v, target)
            pickle.dump(nx_graph, f)
        elif isinstance(morse_graph, CachedMorseGraph):
             pickle.dump(morse_graph.graph, f)
        else:
            # Assume already NetworkX or picklable
            pickle.dump(morse_graph, f)
        
    if 'morse_sets' in data and data['morse_sets'] is not None:
        with open(os.path.join(directory, 'morse_sets.pkl'), 'wb') as f:
            pickle.dump(data['morse_sets'], f)
            
    # Save barycenters (JSON friendly if possible, or npz)
    barycenters_serializable = {}
    if 'morse_set_barycenters' in data and data['morse_set_barycenters'] is not None:
        for k, v in data['morse_set_barycenters'].items():
            barycenters_serializable[str(k)] = [arr.tolist() for arr in v]
    
    with open(os.path.join(directory, 'barycenters.json'), 'w') as f:
        json.dump(barycenters_serializable, f, indent=2)
        
    # Save config
    if 'config' in data:
        with open(os.path.join(directory, 'config.json'), 'w') as f:
            json.dump(data['config'], f, indent=2)
            
    # Save metadata/method if present
    if 'method' in data:
        with open(os.path.join(directory, 'metadata.json'), 'w') as f:
            json.dump({'method': data['method']}, f, indent=2)


def load_morse_graph_data(directory: str) -> Optional[Dict[str, Any]]:
    """Load Morse graph data from directory."""
    import pickle
    import json
    import networkx as nx
    
    if not os.path.exists(os.path.join(directory, 'morse_graph.pkl')):
        return None
        
    try:
        with open(os.path.join(directory, 'morse_graph.pkl'), 'rb') as f:
            morse_graph_obj = pickle.load(f)
            
        # Wrap in CachedMorseGraph if it's NetworkX
        if isinstance(morse_graph_obj, nx.DiGraph):
            morse_graph = CachedMorseGraph(morse_graph_obj)
        else:
            morse_graph = morse_graph_obj
            
        morse_sets = None
        if os.path.exists(os.path.join(directory, 'morse_sets.pkl')):
            with open(os.path.join(directory, 'morse_sets.pkl'), 'rb') as f:
                morse_sets = pickle.load(f)
                
        morse_set_barycenters = {}
        if os.path.exists(os.path.join(directory, 'barycenters.json')):
            with open(os.path.join(directory, 'barycenters.json'), 'r') as f:
                bary_json = json.load(f)
                for k, v in bary_json.items():
                    morse_set_barycenters[int(k)] = [np.array(arr) for arr in v]
                    
        config = None
        if os.path.exists(os.path.join(directory, 'config.json')):
            with open(os.path.join(directory, 'config.json'), 'r') as f:
                config = json.load(f)
                
        method = None
        if os.path.exists(os.path.join(directory, 'metadata.json')):
            with open(os.path.join(directory, 'metadata.json'), 'r') as f:
                meta = json.load(f)
                method = meta.get('method')
                
        return {
            'morse_graph': morse_graph,
            'morse_sets': morse_sets,
            'morse_set_barycenters': morse_set_barycenters,
            'config': config,
            'method': method
        }
    except Exception as e:
        print(f"Error loading cache from {directory}: {e}")
        return None


def _load_cached_3d_morse_graph(cmgdb_3d_dir: str, param_hash: str, cache_subdir: str,
                                results_dir: str, verbose: bool) -> Optional[Dict[str, Any]]:
    """Load cached 3D Morse graph from disk."""
    cached = load_morse_graph_cache(cmgdb_3d_dir, param_hash, verbose=verbose)
    
    if cached is None:
        return None
    
    return {
        'morse_graph': cached['morse_graph'],
        'barycenters': cached['barycenters'],
        'num_morse_sets': cached['num_morse_sets'],
        'computation_time': cached['metadata'].get('computation_time', 0.0),
        'cached': True,
        'cache_path': cache_subdir,
        'param_hash': param_hash,
        'results_dir': results_dir
    }


def _prepare_3d_cache_paths(base_dir: str, param_hash: str) -> Dict[str, str]:
    """Prepare cache directory paths for 3D Morse graph."""
    cmgdb_3d_dir = os.path.join(base_dir, 'cmgdb_3d')
    cache_paths = get_cache_path(cmgdb_3d_dir, param_hash)
    cache_subdir = cache_paths['dir']
    results_dir = os.path.join(cache_subdir, 'results')
    
    return {
        'cmgdb_3d_dir': cmgdb_3d_dir,
        'cache_subdir': cache_subdir,
        'results_dir': results_dir
    }


def _compute_3d_morse_graph(map_func: Callable, domain_bounds: List,
                           subdiv_min: int, subdiv_max: int, subdiv_init: int,
                           subdiv_limit: int, padding: bool, verbose: bool) -> Dict[str, Any]:
    """Compute 3D Morse graph from scratch."""
    from MorseGraph.core import compute_morse_graph_3d
    
    return compute_morse_graph_3d(
        map_func,
        domain_bounds,
        subdiv_min=subdiv_min,
        subdiv_max=subdiv_max,
        subdiv_init=subdiv_init,
        subdiv_limit=subdiv_limit,
        padding=padding,
        cache_dir=None,
        use_cache=False,
        verbose=verbose
    )


def _generate_3d_visualizations(morse_graph_3d, domain_bounds, results_dir: str,
                               equilibria: Optional[Dict[str, np.ndarray]],
                               periodic_orbits: Optional[Dict[str, np.ndarray]],
                               labels: Optional[Dict[str, str]], verbose: bool):
    """Generate and save 3D Morse graph visualizations."""
    from MorseGraph.plot import plot_morse_graph_diagram, plot_morse_sets_3d_scatter
    
    if verbose:
        print(f"\n  Generating 3D visualizations...")
    
    plot_morse_graph_diagram(
        morse_graph_3d,
        output_path=f"{results_dir}/morse_graph_3d.png",
        title="3D Morse Graph Diagram"
    )
    if verbose:
        print(f"    ✓ Saved morse_graph_3d.png")
    
    plot_morse_sets_3d_scatter(
        morse_graph_3d,
        domain_bounds,
        output_path=f"{results_dir}/morse_sets_3d.png",
        title="3D Morse Sets (Barycenters)",
        equilibria=equilibria,
        periodic_orbits=periodic_orbits,
        labels=labels
    )
    if verbose:
        print(f"    ✓ Saved morse_sets_3d.png")
    
    _generate_projection_plots(morse_graph_3d, results_dir, labels, verbose)


def _generate_projection_plots(morse_graph_3d, results_dir: str,
                               labels: Optional[Dict[str, str]], verbose: bool):
    """Generate 2D projection plots using CMGDB.PlotMorseSets."""
    try:
        import CMGDB
        from matplotlib import cm
        
        if labels is None:
            labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}
        
        projections = [
            ([0, 1], f"{labels['x']}-{labels['y']}"),
            ([0, 2], f"{labels['x']}-{labels['z']}"),
            ([1, 2], f"{labels['y']}-{labels['z']}")
        ]
        
        for proj_dims, proj_name in projections:
            proj_filename = f"morse_sets_proj_{''.join(map(str, proj_dims))}.png"
            output_file = f"{results_dir}/{proj_filename}"
            
            xlabel_key = ['x', 'y', 'z'][proj_dims[0]]
            ylabel_key = ['x', 'y', 'z'][proj_dims[1]]
            
            CMGDB.PlotMorseSets(
                morse_graph_3d,
                proj_dims=proj_dims,
                cmap=cm.cool,
                fig_w=8,
                fig_h=8,
                xlabel=labels[xlabel_key],
                ylabel=labels[ylabel_key],
                fig_fname=output_file,
                dpi=150
            )
            
            if verbose:
                print(f"    ✓ Saved {proj_filename}")
    
    except ImportError:
        if verbose:
            print(f"    ⚠ CMGDB not available for projection plots")
    except Exception as e:
        if verbose:
            print(f"    ⚠ Could not generate projection plots: {e}")


def _save_3d_morse_graph_cache(cmgdb_3d_dir: str, cache_subdir: str, results_dir: str,
                               morse_graph_3d, barycenters_3d, map_graph,
                               param_hash: str, domain_bounds: List,
                               subdiv_min: int, subdiv_max: int, subdiv_init: int,
                               subdiv_limit: int, padding: bool, computation_time: float,
                               verbose: bool):
    """Save 3D Morse graph to cache."""
    metadata = {
        'subdiv_min': subdiv_min,
        'subdiv_max': subdiv_max,
        'subdiv_init': subdiv_init,
        'subdiv_limit': subdiv_limit,
        'padding': padding,
        'domain_bounds': domain_bounds,
        'computation_time': computation_time,
    }
    
    save_morse_graph_cache(
        morse_graph_3d,
        map_graph,
        barycenters_3d,
        metadata,
        cache_dir=cmgdb_3d_dir,
        param_hash=param_hash,
        verbose=verbose
    )
    
    os.makedirs(results_dir, exist_ok=True)


def load_or_compute_3d_morse_graph(
    map_func: Callable,
    domain_bounds: List,
    subdiv_min: int,
    subdiv_max: int,
    subdiv_init: int,
    subdiv_limit: int,
    padding: bool,
    base_dir: str,
    force_recompute: bool = False,
    verbose: bool = True,
    equilibria: Optional[Dict[str, np.ndarray]] = None,
    periodic_orbits: Optional[Dict[str, np.ndarray]] = None,
    labels: Optional[Dict[str, str]] = None
) -> Tuple[Dict[str, Any], bool]:
    """
    Load 3D Morse graph from cmgdb_3d/{hash}/ cache or compute if not found.

    This function implements a reusable caching strategy for expensive 3D CMGDB
    computations. Results are organized by parameter hash, allowing multiple
    parameter configurations to coexist without overwriting each other.

    Directory structure:
        cmgdb_3d/
          ├── {hash1}/
          │   ├── morse_graph_data.pkl
          │   ├── barycenters.npz
          │   ├── metadata.json
          │   └── results/
          │       ├── morse_graph_3d.png
          │       ├── morse_sets_3d.png
          │       └── morse_sets_proj_*.png
          └── {hash2}/
              └── ...

    When computing, also generates and saves standard visualizations:
    - Morse graph diagram
    - 3D scatter plot of barycenters (marker size proportional to box volume)
    - 2D projection plots for dims [0,1], [0,2], [1,2]

    Args:
        map_func: Map function for dynamics
        domain_bounds: Domain bounds for computation
        subdiv_min: Minimum subdivision depth
        subdiv_max: Maximum subdivision depth
        subdiv_init: Initial subdivision depth
        subdiv_limit: Maximum number of boxes
        padding: Whether to use padding
        base_dir: Base experiment directory (e.g., 'ives_model_output/')
        force_recompute: If True, ignore cache and recompute
        verbose: Whether to print progress messages
        equilibria: Optional dict of equilibrium points for plotting
        periodic_orbits: Optional dict of periodic orbits for plotting (each is Nx3 array)
        labels: Optional dict with 'x', 'y', 'z' keys for axis labels

    Returns:
        Tuple of (result_dict, was_cached) where:
            - result_dict: Same format as compute_morse_graph_3d() output
                          plus 'param_hash' and 'cache_path'
            - was_cached: True if loaded from cache, False if computed

    Example:
        >>> result_3d, was_cached = load_or_compute_3d_morse_graph(
        ...     ives_map, domain_bounds, 15, 18, 0, 50000, True,
        ...     base_dir='examples/ives_model_output',
        ...     equilibria={'Eq': equilibrium_point},
        ...     periodic_orbits={'Period-12': period_12_orbit},
        ...     labels={'x': 'log(M)', 'y': 'log(A)', 'z': 'log(D)'}
        ... )
        >>> print(f"Hash: {result_3d['param_hash']}")
        >>> print(f"Cached: {was_cached}")
    """
    import time
    
    param_hash = compute_parameter_hash(
        map_func, domain_bounds, subdiv_min, subdiv_max,
        subdiv_init, subdiv_limit, padding
    )

    paths = _prepare_3d_cache_paths(base_dir, param_hash)
    cmgdb_3d_dir = paths['cmgdb_3d_dir']
    cache_subdir = paths['cache_subdir']
    results_dir = paths['results_dir']

    if not force_recompute and os.path.exists(cache_subdir):
        if verbose:
            print(f"\nFound cache for parameters (hash: {param_hash[:8]}...), attempting to load...")
        
        cached_result = _load_cached_3d_morse_graph(
            cmgdb_3d_dir, param_hash, cache_subdir, results_dir, verbose
        )
        
        if cached_result is not None:
            if verbose:
                print(f"  ✓ Successfully loaded 3D Morse graph from cache")
                print(f"    Morse sets: {cached_result['num_morse_sets']}")
                print(f"    Cache directory: {cache_subdir}")
                print(f"    Visualizations: {results_dir}/")
            return cached_result, True
        
        if verbose:
            print(f"  ✗ Cache load failed, will recompute")

    if verbose:
        print(f"\nComputing 3D Morse graph (hash: {param_hash[:8]}...)...")
        print(f"  Will cache to: {cache_subdir}")

    start_time = time.time()
    result_3d = _compute_3d_morse_graph(
        map_func, domain_bounds, subdiv_min, subdiv_max,
        subdiv_init, subdiv_limit, padding, verbose
    )
    computation_time = time.time() - start_time

    morse_graph_3d = result_3d['morse_graph']
    barycenters_3d = result_3d['barycenters']
    map_graph = result_3d.get('map_graph')

    _save_3d_morse_graph_cache(
        cmgdb_3d_dir, cache_subdir, results_dir,
        morse_graph_3d, barycenters_3d, map_graph,
        param_hash, domain_bounds, subdiv_min, subdiv_max,
        subdiv_init, subdiv_limit, padding, computation_time, verbose
    )

    _generate_3d_visualizations(
        morse_graph_3d, domain_bounds, results_dir,
        equilibria, periodic_orbits, labels, verbose
    )

    if verbose:
        print(f"\n  ✓ Saved 3D Morse graph and visualizations to: {cache_subdir}")

    result_3d['cached'] = False
    result_3d['cache_path'] = cache_subdir
    result_3d['param_hash'] = param_hash

    return result_3d, False

