"""
Trajectory data generation utilities.

This module provides functions for generating trajectory data from ODEs and maps,
which are used for training dynamics learning models.
"""

import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
from typing import Callable, Dict, Tuple, List, Optional, Any

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def generate_trajectory_data(
    ode_func: Callable,
    ode_params: Dict[str, Any],
    n_samples: int,
    total_time: float,
    n_points: int,
    sampling_domain: np.ndarray,
    random_seed: Optional[int] = 42,
    timeskip: float = 0.0,
    n_jobs: int = -1
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
        n_jobs: Number of parallel jobs. -1 uses all CPUs (default: -1)

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

    if timeskip > 0:
        print(f"  Generating {n_samples} trajectories with {n_points} points each...")
        print(f"  Timeskip period: t=0 to t={timeskip}, then sampling t={timeskip} to t={timeskip + total_time}")
    else:
        print(f"  Generating {n_samples} trajectories with {n_points} points each...")

    def integrate_trajectory(ic):
        """Integrate a single trajectory from initial condition."""
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

        # Create (X, Y) pairs from the trajectory
        X_traj = trajectory[:-1]
        Y_traj = trajectory[1:]

        return X_traj, Y_traj, trajectory

    # Parallel computation with progress bar
    if HAS_TQDM:
        results = Parallel(n_jobs=n_jobs)(
            delayed(integrate_trajectory)(ic)
            for ic in tqdm(initial_conditions, desc="  Progress", ncols=80)
        )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(integrate_trajectory)(ic) for ic in initial_conditions
        )
        print(f"  Completed {n_samples} trajectories")

    # Unpack results
    X = [r[0] for r in results]
    Y = [r[1] for r in results]
    trajectories = [r[2] for r in results]

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    return X.astype(np.float32), Y.astype(np.float32), trajectories


def generate_map_trajectory_data(
    map_func: Callable,
    n_trajectories: int,
    n_points: int,
    sampling_domain: np.ndarray,
    random_seed: Optional[int] = 42,
    skip_initial: int = 0,
    n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Generate trajectory data from a discrete map by iterating from random initial conditions.

    This function creates training data for dynamics learning by:
    1. Sampling random initial conditions from a domain
    2. Iterating the map forward for n_points steps
    3. Optionally skipping initial iterations (to reach attractor)
    4. Creating (X, Y) pairs where Y = map(X)

    Args:
        map_func: Map function with signature f(x) -> x_next, where x is a state vector
        n_trajectories: Number of random initial conditions to generate
        n_points: Number of iterations per trajectory (including skip_initial)
        sampling_domain: Domain bounds of shape (2, D) for [[mins], [maxs]]
        random_seed: Random seed for reproducibility (default: 42)
        skip_initial: Number of initial iterations to skip before collecting data (default: 0).
                      If > 0, the map is iterated skip_initial times before data collection starts,
                      useful for reaching an attractor.
        n_jobs: Number of parallel jobs. -1 uses all CPUs (default: -1)

    Returns:
        X: Array of shape (n_trajectories * (n_points - skip_initial - 1), D) - current states
        Y: Array of shape (n_trajectories * (n_points - skip_initial - 1), D) - next states
        trajectories: List of n_trajectories trajectory arrays, each of shape (n_points - skip_initial, D)

    Example:
        >>> from MorseGraph.utils import generate_map_trajectory_data
        >>> from MorseGraph.systems import henon_map
        >>> domain = np.array([[-2, -2], [2, 2]])
        >>> # Generate 100 trajectories with 50 points each
        >>> X, Y, trajs = generate_map_trajectory_data(
        ...     henon_map, 100, 50, domain
        ... )
        >>> # Skip first 10 iterations to reach attractor
        >>> X, Y, trajs = generate_map_trajectory_data(
        ...     henon_map, 100, 50, domain, skip_initial=10
        ... )
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dim = sampling_domain.shape[1]

    # Sample random initial conditions
    initial_conditions = np.random.uniform(
        sampling_domain[0],
        sampling_domain[1],
        (n_trajectories, dim)
    )

    if skip_initial > 0:
        print(f"  Generating {n_trajectories} map trajectories with {n_points} iterations each...")
        print(f"  Skipping first {skip_initial} iterations, collecting {n_points - skip_initial} points")
    else:
        print(f"  Generating {n_trajectories} map trajectories with {n_points} iterations each...")

    def iterate_map(ic):
        """Iterate a single map trajectory from initial condition."""
        x_current = ic.copy()
        trajectory_points = []
        X_traj = []
        Y_traj = []

        for step in range(n_points):
            x_next = map_func(x_current)

            # Collect data after skip_initial iterations
            if step >= skip_initial:
                trajectory_points.append(x_current.copy())
                # Create (X, Y) pairs, except for the last point
                if step < n_points - 1:
                    X_traj.append(x_current.copy())
                    Y_traj.append(x_next.copy())

            x_current = x_next

        return X_traj, Y_traj, np.array(trajectory_points)

    # Parallel computation with progress bar
    if HAS_TQDM:
        results = Parallel(n_jobs=n_jobs)(
            delayed(iterate_map)(ic)
            for ic in tqdm(initial_conditions, desc="  Progress", ncols=80)
        )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(iterate_map)(ic) for ic in initial_conditions
        )
        print(f"  Completed {n_trajectories} trajectories")

    # Unpack results
    X = []
    Y = []
    trajectories = []
    for X_traj, Y_traj, trajectory in results:
        X.extend(X_traj)
        Y.extend(Y_traj)
        trajectories.append(trajectory)

    X = np.concatenate([x[np.newaxis, :] for x in X], axis=0) if X else np.array([])
    Y = np.concatenate([y[np.newaxis, :] for y in Y], axis=0) if Y else np.array([])

    return X.astype(np.float32), Y.astype(np.float32), trajectories


def generate_random_trajectories_3d(
    dynamics_func: Callable,
    domain_bounds: np.ndarray,
    n_trajectories: int,
    n_points: int,
    skip_initial: int = 0,
    random_seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random 3D trajectories using a map function.
    
    This is a convenience wrapper around generate_map_trajectory_data for 3D systems.
    
    Args:
        dynamics_func: Map function f(x) -> x_next
        domain_bounds: Domain bounds of shape (2, 3) for [[mins], [maxs]]
        n_trajectories: Number of trajectories to generate
        n_points: Number of points per trajectory
        skip_initial: Number of initial iterations to skip
        random_seed: Random seed for reproducibility
        
    Returns:
        X: Array of shape (N, 3) - current states
        Y: Array of shape (N, 3) - next states
    """
    X, Y, _ = generate_map_trajectory_data(
        dynamics_func,
        n_trajectories,
        n_points,
        domain_bounds,
        random_seed=random_seed,
        skip_initial=skip_initial
    )
    return X, Y


__all__ = [
    'generate_trajectory_data',
    'generate_map_trajectory_data',
    'generate_random_trajectories_3d',
]

