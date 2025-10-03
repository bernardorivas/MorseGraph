"""
Utility functions for data generation, management, and spatial operations.

This module provides helper functions for working with trajectory data,
latent space operations, and spatial filtering of grid boxes.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from typing import Callable, Dict, Tuple, List, Optional, Any


# =============================================================================
# Trajectory Data Generation
# =============================================================================

def generate_trajectory_data(
    ode_func: Callable,
    ode_params: Dict[str, Any],
    n_samples: int,
    total_time: float,
    n_points: int,
    sampling_domain: np.ndarray,
    random_seed: Optional[int] = 42,
    timeskip: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Generate trajectory data from an ODE by sampling random initial conditions.

    This function creates training data for dynamics learning by:
    1. Sampling random initial conditions from a domain
    2. Integrating each trajectory forward in time (with optional timeskip)
    3. Extracting discrete time points along each trajectory
    4. Creating (X, Y) pairs where Y = f(X) after time dt

    Args:
        ode_func: ODE function with signature f(t, y, **params)
        ode_params: Dictionary of parameters to pass to ode_func
        n_samples: Number of random initial conditions to generate
        total_time: Total integration time per trajectory (after timeskip)
        n_points: Number of discrete points to extract per trajectory
        sampling_domain: Domain bounds of shape (2, D) for [[mins], [maxs]]
        random_seed: Random seed for reproducibility (default: 42)
        timeskip: Time to skip before sampling starts (default: 0.0).
                   If > 0, trajectories are first integrated from 0 to timeskip
                   (to reach attractor), then sampled from timeskip to timeskip+total_time.

    Returns:
        X: Array of shape (n_samples * (n_points-1), D) - current states
        Y: Array of shape (n_samples * (n_points-1), D) - next states
        trajectories: List of n_samples trajectory arrays, each of shape (n_points, D)

    Example:
        >>> from MorseGraph.utils import generate_trajectory_data
        >>> from MorseGraph.systems import van_der_pol_ode
        >>> domain = np.array([[-4, -4], [4, 4]])
        >>> # Sample from t=0 to t=10
        >>> X, Y, trajs = generate_trajectory_data(
        ...     van_der_pol_ode, {'mu': 1.0}, 100, 10.0, 11, domain
        ... )
        >>> # Sample from t=5 to t=10 (5s timeskip, then 5s sampling)
        >>> X, Y, trajs = generate_trajectory_data(
        ...     van_der_pol_ode, {'mu': 1.0}, 100, 5.0, 6, domain, timeskip=5.0
        ... )
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dim = sampling_domain.shape[1]

    # Sample random initial conditions
    initial_conditions = np.random.uniform(
        sampling_domain[0],
        sampling_domain[1],
        (n_samples, dim)
    )

    X = []
    Y = []
    trajectories = []

    if timeskip > 0:
        print(f"  Generating {n_samples} trajectories with {n_points} points each...")
        print(f"  Timeskip period: t=0 to t={timeskip}, then sampling t={timeskip} to t={timeskip + total_time}")
    else:
        print(f"  Generating {n_samples} trajectories with {n_points} points each...")

    for i, ic in enumerate(initial_conditions):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{n_samples}")

        # If timeskip > 0, do timeskip integration first
        if timeskip > 0:
            # Timeskip: integrate from 0 to timeskip
            timeskip_sol = solve_ivp(
                lambda t, y: ode_func(t, y, **ode_params),
                [0, timeskip],
                ic,
                method='DOP853',
                rtol=1e-12,
                atol=1e-14
            )
            # Use final state of timeskip as new initial condition
            ic_after_timeskip = timeskip_sol.y[:, -1]
        else:
            ic_after_timeskip = ic

        # Integrate ODE with high accuracy
        sol = solve_ivp(
            lambda t, y: ode_func(t, y, **ode_params),
            [timeskip, timeskip + total_time],
            ic_after_timeskip,
            dense_output=True,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14
        )

        # Extract n_points uniformly spaced along trajectory
        times = np.linspace(timeskip, timeskip + total_time, n_points)
        trajectory = sol.sol(times).T  # Shape: (n_points, dim)

        trajectories.append(trajectory)

        # Create (X, Y) pairs from the trajectory
        X.append(trajectory[:-1])
        Y.append(trajectory[1:])

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    return X, Y, trajectories


# =============================================================================
# Trajectory Data I/O
# =============================================================================

def save_trajectory_data(
    filepath: str,
    x_t: np.ndarray,
    x_t_plus_1: np.ndarray,
    trajectories: List[np.ndarray],
    metadata: Dict[str, Any]
) -> None:
    """
    Save trajectory data and metadata to disk in compressed NPZ format.

    Args:
        filepath: Path to save the .npz file
        x_t: Current states array
        x_t_plus_1: Next states array
        trajectories: List of trajectory arrays
        metadata: Dictionary of metadata to save

    Example:
        >>> save_trajectory_data(
        ...     'data.npz', X, Y, trajs,
        ...     {'n_trajectories': 100, 'total_time': 10.0}
        ... )
    """
    save_dict = {
        'x_t': x_t,
        'x_t_plus_1': x_t_plus_1,
        'n_trajectories': len(trajectories),
        **metadata
    }

    # Save trajectories as separate arrays
    for i, traj in enumerate(trajectories):
        save_dict[f'trajectory_{i}'] = traj

    np.savez_compressed(filepath, **save_dict)
    print(f"  Saved training data to: {filepath}")


def load_trajectory_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Dict[str, Any]]:
    """
    Load trajectory data and metadata from disk.

    Args:
        filepath: Path to the .npz file

    Returns:
        x_t: Current states array
        x_t_plus_1: Next states array
        trajectories: List of trajectory arrays
        metadata: Dictionary of metadata

    Example:
        >>> X, Y, trajs, meta = load_trajectory_data('data.npz')
    """
    data = np.load(filepath)

    x_t = data['x_t']
    x_t_plus_1 = data['x_t_plus_1']
    n_trajectories = int(data['n_trajectories'])

    # Reconstruct trajectories list
    trajectories = []
    for i in range(n_trajectories):
        trajectories.append(data[f'trajectory_{i}'])

    # Extract metadata
    metadata = {
        'n_trajectories': n_trajectories,
        'n_points': int(data['n_points']),
        'random_seed': int(data['random_seed'])
    }
    if 'total_time' in data:
        metadata['total_time'] = float(data['total_time'])
    if 'sampling_time' in data:
        metadata['sampling_time'] = float(data['sampling_time'])
    if 'timeskip' in data:
        metadata['timeskip'] = float(data['timeskip'])


    print(f"  Loaded training data from: {filepath}")
    print(f"  Metadata: {metadata}")

    return x_t, x_t_plus_1, trajectories, metadata


# =============================================================================
# Latent Space Utilities
# =============================================================================

def compute_latent_bounds(
    encoded_data: np.ndarray,
    padding_factor: float = 1.2
) -> np.ndarray:
    """
    Compute bounding box for latent space with padding.

    Takes encoded data points and computes a bounding box that contains
    all points, then expands it by a padding factor to ensure coverage.

    Args:
        encoded_data: Array of shape (N, latent_dim)
        padding_factor: Factor to expand bounds by (default: 1.2 for 20% padding)

    Returns:
        bounds: Array of shape (2, latent_dim) for [[xmin, ymin, ...], [xmax, ymax, ...]]

    Example:
        >>> encoded = encoder(data)  # Shape: (1000, 2)
        >>> bounds = compute_latent_bounds(encoded, padding_factor=1.5)
        >>> # bounds might be [[-3.5, -2.1], [3.5, 2.1]]
    """
    mins = encoded_data.min(axis=0)
    maxs = encoded_data.max(axis=0)

    center = (mins + maxs) / 2
    range_vec = (maxs - mins) * padding_factor / 2

    return np.array([center - range_vec, center + range_vec])


# =============================================================================
# Spatial Filtering
# =============================================================================

def filter_boxes_near_data(
    grid,
    data_points: np.ndarray,
    epsilon_radius: float
) -> List[int]:
    """
    Return list of box indices that contain or are epsilon-close to data points.

    Uses cKDTree for efficient spatial queries to identify which grid boxes
    are within epsilon distance of any data point. Useful for restricting
    computations to regions with data coverage.

    Args:
        grid: Grid object (UniformGrid or AdaptiveGrid)
        data_points: Array of shape (N, D) with data point coordinates
        epsilon_radius: Maximum distance for a box to be considered "near" data

    Returns:
        List of box indices that are within epsilon_radius of any data point

    Example:
        >>> from MorseGraph.grids import UniformGrid
        >>> grid = UniformGrid(bounds=np.array([[-5, -5], [5, 5]]), divisions=[100, 100])
        >>> data = np.random.randn(1000, 2)
        >>> active_boxes = filter_boxes_near_data(grid, data, epsilon_radius=0.5)
        >>> print(f"Active boxes: {len(active_boxes)} out of {len(grid.get_boxes())}")
    """
    tree = cKDTree(data_points)

    # Get all box centers
    all_boxes = grid.get_boxes()
    box_centers = (all_boxes[:, 0, :] + all_boxes[:, 1, :]) / 2.0

    # Query the tree for all box centers at once
    distances, _ = tree.query(box_centers, k=1)

    # Find active boxes where distance is within the radius
    active_indices = np.where(distances <= epsilon_radius)[0]

    return active_indices.tolist()
