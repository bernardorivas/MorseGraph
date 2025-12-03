"""
Utility functions for data generation, management, and spatial operations.

This module provides helper functions for working with trajectory data,
latent space operations, and spatial filtering of grid boxes.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from typing import Callable, Dict, Tuple, List, Optional, Any, Union, Set

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


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


# =============================================================================
# Trajectory Data I/O
# =============================================================================

def _save_trajectory_data_file(
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
        >>> _save_trajectory_data_file(
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


def _load_trajectory_data_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Dict[str, Any]]:
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
        >>> X, Y, trajs, meta = _load_trajectory_data_file('data.npz')
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


def get_next_run_number(base_dir: str) -> int:
    """
    Find the next available run number in a directory containing run_XXX folders.

    Scans the base directory for existing folders named 'run_001', 'run_002', etc.,
    and returns the next available number in the sequence.

    Args:
        base_dir: Directory to scan for existing run folders

    Returns:
        Next available run number (starting from 1)

    Example:
        >>> # Directory contains run_001, run_002, run_005
        >>> get_next_run_number('/path/to/experiments')
        6
        >>> # Empty directory
        >>> get_next_run_number('/path/to/new_experiments')
        1
    """
    if not os.path.exists(base_dir):
        return 1

    existing_runs = [
        d for d in os.listdir(base_dir)
        if d.startswith('run_') and os.path.isdir(os.path.join(base_dir, d))
    ]
    if not existing_runs:
        return 1

    # Extract run numbers
    run_numbers = []
    for run_dir in existing_runs:
        try:
            num = int(run_dir.split('_')[1])
            run_numbers.append(num)
        except (IndexError, ValueError):
            continue

    return max(run_numbers) + 1 if run_numbers else 1


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
# ML Training Utilities
# =============================================================================

def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model instance

    Returns:
        Total number of trainable parameters

    Example:
        >>> from MorseGraph.models import Encoder
        >>> encoder = Encoder(input_dim=3, latent_dim=2, hidden_dim=64, num_layers=3)
        >>> num_params = count_parameters(encoder)
        >>> print(f"Encoder has {num_params:,} parameters")
    """
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except AttributeError:
        raise TypeError("Model must be a PyTorch model with .parameters() method")


def format_time(seconds: float) -> str:
    """
    Format time duration in seconds as a human-readable string.

    Args:
        seconds: Time duration in seconds

    Returns:
        Formatted string (e.g., "1.5s", "2m 30s", "1h 15m")

    Example:
        >>> format_time(45.3)
        '45.3s'
        >>> format_time(150)
        '2m 30s'
        >>> format_time(3900)
        '1h 5m'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


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


# =============================================================================
# Morse Graph Analysis
# =============================================================================

def count_attractors(morse_graph) -> int:
    """
    Counts the number of attractors (nodes with no out-edges) in a Morse graph.

    An attractor in a Morse graph corresponds to a minimal node in the graph,
    which is a vertex with no outgoing edges. This function iterates through
    all vertices in the Morse graph and counts how many have an out-degree of 0.

    Args:
        morse_graph: A Morse graph object from cmgdb. It is expected to have
                     `vertices()` and `adjacencies(v)` methods.

    Returns:
        The number of attractors in the Morse graph. Returns 0 if the graph
        is None or has no vertices.

    Example:
        >>> morse_graph = cmgdb.ComputeMorseGraph(model)
        >>> num_attractors = count_attractors(morse_graph)
        >>> print(f"Found {num_attractors} attractors.")
    """
    if morse_graph is None:
        return 0
    
    vertices = morse_graph.vertices()
    if not vertices:
        return 0

    attractor_count = 0
    for v in vertices:
        if not morse_graph.adjacencies(v):
            attractor_count += 1
    return attractor_count

def extract_edge_set(morse_graph) -> Set[Tuple[int, int]]:
    """
    Extract the set of directed edges from a CMGDB Morse graph.

    Args:
        morse_graph: A CMGDB Morse graph object with methods:
                     - num_vertices() -> int
                     - adjacencies(vertex_id) -> list of adjacent vertex IDs

    Returns:
        Set of (source, target) tuples representing directed edges

    Example:
        >>> edges = extract_edge_set(morse_graph)
        >>> print(edges)
        {(3, 2), (2, 0), (2, 1)}
    """
    if morse_graph is None:
        return set()

    edges = set()
    num_vertices = morse_graph.num_vertices()

    for v in range(num_vertices):
        # Get all vertices that v has edges to
        adjacent_vertices = morse_graph.adjacencies(v)
        for target in adjacent_vertices:
            edges.add((v, target))

    return edges


def count_attractors(morse_graph) -> int:
    """
    Count the number of attractors (sinks) in a Morse graph.

    An attractor is a node with no outgoing edges (out-degree = 0).

    Args:
        morse_graph: A CMGDB Morse graph object

    Returns:
        Number of attractor nodes

    Example:
        >>> num_attractors = count_attractors(morse_graph)
        >>> print(f"Found {num_attractors} attractors")
    """
    if morse_graph is None:
        return 0

    num_attractors = 0
    for v in range(morse_graph.num_vertices()):
        if len(morse_graph.adjacencies(v)) == 0:
            num_attractors += 1

    return num_attractors


def compute_similarity_vector(
    morse_graph,
    ground_truth: Dict[str, Union[int, Set[Tuple[int, int]]]]
) -> Dict[str, Union[int, float]]:
    """
    Compute a multi-dimensional topological similarity vector comparing
    a learned Morse graph to a ground truth structure.

    This function provides a quantitative alternative to visual comparison,
    measuring similarity across multiple topological features:
    - Number of attractors (sinks)
    - Total number of nodes (Morse sets)
    - Edge connectivity structure

    Args:
        morse_graph: A CMGDB Morse graph object (can be None)
        ground_truth: Dictionary containing:
            - 'num_nodes': Expected number of nodes (int)
            - 'edges': Expected edge set as {(source, target), ...} (set)
            - 'num_attractors': Expected number of attractors (int)

    Returns:
        Dictionary with keys:
            - 'attractor_diff': |attractors_learned - attractors_true|
            - 'node_diff': |nodes_learned - nodes_true|
            - 'connection_diff': Symmetric difference count |E_learned Δ E_true|
            - 'attractors_learned': Number of attractors in learned graph
            - 'attractors_true': Number of attractors in ground truth
            - 'nodes_learned': Number of nodes in learned graph
            - 'nodes_true': Number of nodes in ground truth
            - 'edges_learned': Set of edges in learned graph
            - 'edges_true': Set of edges in ground truth
            - 'missing_edges': Edges in ground truth but not learned
            - 'extra_edges': Edges learned but not in ground truth

    Example:
        >>> ground_truth = {
        ...     'num_nodes': 4,
        ...     'edges': {(3, 2), (2, 0), (2, 1)},
        ...     'num_attractors': 2
        ... }
        >>> similarity = compute_similarity_vector(morse_graph, ground_truth)
        >>> print(f"Node difference: {similarity['node_diff']}")
        >>> print(f"Connection difference: {similarity['connection_diff']}")
    """
    # Extract learned graph properties
    if morse_graph is None:
        nodes_learned = 0
        attractors_learned = 0
        edges_learned = set()
    else:
        nodes_learned = morse_graph.num_vertices()
        attractors_learned = count_attractors(morse_graph)
        edges_learned = extract_edge_set(morse_graph)

    # Extract ground truth properties
    nodes_true = ground_truth.get('num_nodes', 0)
    attractors_true = ground_truth.get('num_attractors', 0)
    edges_true = ground_truth.get('edges', set())

    # Compute differences
    node_diff = abs(nodes_learned - nodes_true)
    attractor_diff = abs(attractors_learned - attractors_true)

    # Edge set symmetric difference
    missing_edges = edges_true - edges_learned  # In ground truth but not learned
    extra_edges = edges_learned - edges_true     # Learned but not in ground truth
    connection_diff = len(missing_edges) + len(extra_edges)

    return {
        'attractor_diff': attractor_diff,
        'node_diff': node_diff,
        'connection_diff': connection_diff,
        'attractors_learned': attractors_learned,
        'attractors_true': attractors_true,
        'nodes_learned': nodes_learned,
        'nodes_true': nodes_true,
        'edges_learned': edges_learned,
        'edges_true': edges_true,
        'missing_edges': missing_edges,
        'extra_edges': extra_edges,
    }


def format_similarity_report(similarity: Dict[str, Union[int, float]],
                             title: str = "Topological Similarity") -> str:
    """
    Create a human-readable summary of topological similarity metrics.

    Args:
        similarity: Output from compute_similarity_vector()
        title: Optional title for the report

    Returns:
        Formatted string report

    Example:
        >>> report = format_similarity_report(similarity, title="Full Latent Graph")
        >>> print(report)
    """
    lines = [f"\n{title}:"]
    lines.append(f"  Nodes:      {similarity['nodes_learned']} vs {similarity['nodes_true']} " +
                 f"(diff: {similarity['node_diff']})")
    lines.append(f"  Attractors: {similarity['attractors_learned']} vs {similarity['attractors_true']} " +
                 f"(diff: {similarity['attractor_diff']})")
    lines.append(f"  Edges:      {len(similarity['edges_learned'])} vs {len(similarity['edges_true'])} " +
                 f"(diff: {similarity['connection_diff']})")

    if similarity['missing_edges']:
        lines.append(f"    Missing edges: {sorted(similarity['missing_edges'])}")
    if similarity['extra_edges']:
        lines.append(f"    Extra edges:   {sorted(similarity['extra_edges'])}")

    # Perfect match indicator
    if (similarity['node_diff'] == 0 and
        similarity['attractor_diff'] == 0 and
        similarity['connection_diff'] == 0):
        lines.append("  ✓ PERFECT MATCH!")

    return '\n'.join(lines)


def compute_train_val_divergence(similarity_train: Dict, similarity_val: Dict) -> Dict[str, int]:
    """
    Compute divergence between training and validation Morse graphs.

    This helps detect overfitting: if the training and validation graphs
    differ significantly, the model may not generalize well.

    Args:
        similarity_train: Similarity vector for training data graph
        similarity_val: Similarity vector for validation data graph

    Returns:
        Dictionary with:
            - 'node_divergence': |nodes_train - nodes_val|
            - 'attractor_divergence': |attractors_train - attractors_val|
            - 'edge_divergence': Symmetric difference of edge sets

    Example:
        >>> divergence = compute_train_val_divergence(sim_train, sim_val)
        >>> if divergence['node_divergence'] > 1:
        ...     print("WARNING: Train and validation graphs differ substantially")
    """
    node_div = abs(similarity_train['nodes_learned'] - similarity_val['nodes_learned'])
    attractor_div = abs(similarity_train['attractors_learned'] - similarity_val['attractors_learned'])

    edges_train = similarity_train['edges_learned']
    edges_val = similarity_val['edges_learned']
    edge_div = len((edges_train - edges_val) | (edges_val - edges_train))

    return {
        'node_divergence': node_div,
        'attractor_divergence': attractor_div,
        'edge_divergence': edge_div,
    }


# =============================================================================
# Experiment Configuration and Management
# =============================================================================

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
    import json
    from datetime import datetime

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'configuration': config.to_dict(),
        'results': results,
    }

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved experiment metadata to: {filepath}")


# =============================================================================
# CMGDB Caching Utilities
# =============================================================================

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
    import os
    import pickle
    import torch
    import json
    from datetime import datetime

    # Define cache directory
    cache_dir = os.path.join(output_dir, 'training', training_hash)
    models_dir = os.path.join(cache_dir, 'models')
    metadata_path = os.path.join(cache_dir, 'metadata.json')
    losses_path = os.path.join(cache_dir, 'training_losses.pkl')
    bounds_path = os.path.join(cache_dir, 'latent_bounds.npz')

    encoder_path = os.path.join(models_dir, 'encoder.pt')
    decoder_path = os.path.join(models_dir, 'decoder.pt')
    latent_dynamics_path = os.path.join(models_dir, 'latent_dynamics.pt')

    # Check if cache exists and is valid
    cache_valid = (
        os.path.exists(encoder_path) and
        os.path.exists(decoder_path) and
        os.path.exists(latent_dynamics_path) and
        os.path.exists(metadata_path) and
        os.path.exists(losses_path) and
        os.path.exists(bounds_path)
    )

    if cache_valid and not force_retrain:
        print(f"Loading cached training results from {cache_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load models
        from MorseGraph.models import Encoder, Decoder, LatentDynamics

        encoder = Encoder(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            output_activation=config.encoder_activation
        )
        encoder.load_state_dict(torch.load(encoder_path))
        encoder.eval()

        decoder = Decoder(
            latent_dim=config.latent_dim,
            output_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            output_activation=config.decoder_activation
        )
        decoder.load_state_dict(torch.load(decoder_path))
        decoder.eval()

        latent_dynamics = LatentDynamics(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            output_activation=config.latent_dynamics_activation
        )
        latent_dynamics.load_state_dict(torch.load(latent_dynamics_path))
        latent_dynamics.eval()

        # Load training losses
        with open(losses_path, 'rb') as f:
            training_losses = pickle.load(f)

        # Load latent bounds
        latent_bounds_data = np.load(bounds_path)
        latent_bounds = latent_bounds_data['bounds']

        training_result = {
            'encoder': encoder,
            'decoder': decoder,
            'latent_dynamics': latent_dynamics,
            'training_losses': training_losses,
            'latent_bounds': latent_bounds,
            'config': config,
            'metadata': metadata
        }

        return training_result, True

    else:
        # Train new models
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

        # Create cache directory
        os.makedirs(models_dir, exist_ok=True)

        # Save models
        torch.save(training_result['encoder'].state_dict(), encoder_path)
        torch.save(training_result['decoder'].state_dict(), decoder_path)
        torch.save(training_result['latent_dynamics'].state_dict(), latent_dynamics_path)

        # Compute latent bounds from training data
        device = training_result['device']
        encoder = training_result['encoder']
        with torch.no_grad():
            z_train = encoder(torch.FloatTensor(training_data['X_train']).to(device)).cpu().numpy()

        latent_bounds = compute_latent_bounds(z_train, padding_factor=config.latent_bounds_padding)

        # Save training losses (combining train and val)
        training_losses = {
            'train': training_result['train_losses'],
            'val': training_result['val_losses']
        }
        with open(losses_path, 'wb') as f:
            pickle.dump(training_losses, f)

        # Save latent bounds
        np.savez(bounds_path, bounds=latent_bounds)

        # Save metadata
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

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Training results cached to {cache_dir}")

        # Add latent_bounds and training_losses to result
        training_result['latent_bounds'] = latent_bounds
        training_result['training_losses'] = training_losses

        return training_result, False


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
    import os
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
    import os
    import pickle
    import json
    from datetime import datetime

    # Define cache directory
    cache_dir = os.path.join(output_dir, 'cmgdb_2d', cmgdb_2d_hash)
    metadata_path = os.path.join(cache_dir, 'metadata.json')
    morse_graph_path = os.path.join(cache_dir, 'morse_graph.pkl')
    barycenters_path = os.path.join(cache_dir, 'barycenters.npz')

    # Check if cache exists and is valid
    cache_valid = (
        os.path.exists(morse_graph_path) and
        os.path.exists(barycenters_path) and
        os.path.exists(metadata_path)
    )

    if cache_valid and not force_recompute:
        print(f"Loading cached 2D Morse graph from {cache_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load Morse graph
        with open(morse_graph_path, 'rb') as f:
            morse_graph_nx = pickle.load(f)
        morse_graph = CachedMorseGraph(morse_graph_nx)

        # Load barycenters - reconstruct dict from npz format
        barycenters = {}
        for key in metadata['barycenters_keys']: # Use keys from metadata
            barycenters[int(key.split('_')[-1])] = np.array(metadata[key])

        morse_2d_result = {
            'morse_graph': morse_graph,
            'barycenters': barycenters,
            'config': config,
            'metadata': metadata,
            'method': metadata['method']
        }

        return morse_2d_result, True

    else:
        # Compute new 2D Morse graph
        method = config.latent_morse_graph_method
        print(f"Computing 2D Morse graph (hash: {cmgdb_2d_hash}) using method: {method}")

        from MorseGraph.core import compute_morse_graph_2d_data, compute_morse_graph_2d_restricted, compute_morse_graph_2d_latent_enclosure
        import torch
        import networkx as nx
        import pickle

        # Get device from encoder
        device = next(encoder.parameters()).device
        
        morse_graph_cmgdb = None
        barycenters = {}
        
        if method == 'data':
            # Generate dense uniform grid in original space for encoding
            print("\nGenerating dense uniform grid in original space...")
            original_grid_subdiv = config.original_grid_subdiv
            input_dim = config.input_dim
            n_per_dim = 2 ** (original_grid_subdiv // input_dim) # Adjust N for higher dimensions
            print(f"  Grid: {n_per_dim} points per dimension ({n_per_dim**input_dim} total)")

            # Create meshgrid for 3D space
            domain_bounds = config.domain_bounds
            grid_1d = [np.linspace(domain_bounds[0][i], domain_bounds[1][i], n_per_dim)
                       for i in range(input_dim)]
            
            # Dynamically create meshgrid based on input_dim
            if input_dim == 3:
                mesh = np.meshgrid(*grid_1d, indexing='ij')
            elif input_dim == 2:
                mesh = np.meshgrid(*grid_1d, indexing='ij')
            elif input_dim == 1:
                mesh = np.meshgrid(*grid_1d, indexing='ij')
            else:
                # Handle cases for other input dimensions as needed
                mesh = np.meshgrid(*grid_1d, indexing='ij') # Default to ij for consistency


            X_large_grid = np.stack([m.flatten() for m in mesh], axis=1)
            print(f"  Generated {len(X_large_grid)} grid points in original space")

            # Encode grid to latent space
            with torch.no_grad():
                z_large_grid = encoder(torch.FloatTensor(X_large_grid).to(device)).cpu().numpy()
            print(f"  Encoded to latent space: {z_large_grid.shape}")

            # Compute Morse graph using BoxMapData
            print(f"\nComputing Morse graph using BoxMapData (padding={config.latent_padding})...")
            result_2d = compute_morse_graph_2d_data(
                latent_dynamics, device, z_large_grid, latent_bounds.tolist(),
                subdiv_min=config.latent_subdiv_min, subdiv_max=config.latent_subdiv_max,
                subdiv_init=config.latent_subdiv_init, subdiv_limit=config.latent_subdiv_limit,
                padding=config.latent_padding,
                cache_dir=None,  # Don't use internal cache
                use_cache=False,
                verbose=True
            )
            morse_graph_cmgdb = result_2d['morse_graph']
            # Extract barycenters
            for i in range(morse_graph_cmgdb.num_vertices()):
                boxes = morse_graph_cmgdb.morse_set_boxes(i)
                barycenters[i] = [np.array([(box[j] + box[j + 2]) / 2.0 for j in range(2)]) for box in boxes] if boxes else []


        elif method == 'restricted':
            print(f"\nComputing Morse graph using restricted method (padding={config.latent_padding})...")
            # For 'restricted' method, we need the raw z_data from training
            # Assuming training_data from earlier stages is available or passed.
            # For now, let's just use the latent data from a previous stage as z_data.
            # In a full pipeline, this z_data might come from the training result.
            z_data_for_restricted = encoder(torch.FloatTensor(config.trajectory_data['X']).to(device)).detach().cpu().numpy()

            result_2d = compute_morse_graph_2d_restricted(
                latent_dynamics, device, z_data_for_restricted, latent_bounds.tolist(),
                subdiv_min=config.latent_subdiv_min, subdiv_max=config.latent_subdiv_max,
                subdiv_init=config.latent_subdiv_init, subdiv_limit=config.latent_subdiv_limit,
                include_neighbors=True, # This should be configurable
                padding=config.latent_padding,
                cache_dir=None,
                use_cache=False,
                verbose=True
            )
            morse_graph_cmgdb = result_2d['morse_graph']
            # Extract barycenters
            for i in range(morse_graph_cmgdb.num_vertices()):
                boxes = morse_graph_cmgdb.morse_set_boxes(i)
                barycenters[i] = [np.array([(box[j] + box[j + 2]) / 2.0 for j in range(2)]) for box in boxes] if boxes else []


        elif method == 'latent_enclosure':
            print(f"\nComputing Morse graph using latent enclosure method (padding={config.latent_padding})...")
            result_2d = compute_morse_graph_2d_latent_enclosure(
                latent_dynamics, device, latent_bounds.tolist(),
                subdiv_min=config.latent_subdiv_min, subdiv_max=config.latent_subdiv_max,
                subdiv_init=config.latent_subdiv_init, subdiv_limit=config.latent_subdiv_limit,
                padding=config.latent_padding,
                cache_dir=None,
                use_cache=False,
                verbose=True
            )
            morse_graph_cmgdb = result_2d['morse_graph']
            # Extract barycenters
            for i in range(morse_graph_cmgdb.num_vertices()):
                boxes = morse_graph_cmgdb.morse_set_boxes(i)
                barycenters[i] = [np.array([(box[j] + box[j + 2]) / 2.0 for j in range(2)]) for box in boxes] if boxes else []
                
        else:
            raise ValueError(f"Unknown 2D Morse graph computation method: {method}")

        # Convert to NetworkX for caching
        nx_graph = nx.DiGraph()
        for v in range(morse_graph_cmgdb.num_vertices()):
            boxes = morse_graph_cmgdb.morse_set_boxes(v)
            nx_graph.add_node(v, morse_set_boxes=[list(b) for b in boxes] if boxes else [])
        for v in range(morse_graph_cmgdb.num_vertices()):
            for target in morse_graph_cmgdb.adjacencies(v):
                nx_graph.add_edge(v, target)

        # Save to cache
        os.makedirs(cache_dir, exist_ok=True)
        with open(morse_graph_path, 'wb') as f:
            pickle.dump(nx_graph, f)
        
        # Save barycenters (convert to list of lists for JSON serializability in metadata)
        barycenters_serializable = {f'morse_set_{k}': [arr.tolist() for arr in v] for k, v in barycenters.items()}
        
        # Save metadata
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
                'method': method # Save the method used
            },
            'num_morse_sets': nx_graph.number_of_nodes(),
            'num_edges': nx_graph.number_of_edges(),
            **barycenters_serializable # Include barycenters as part of metadata
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"2D Morse graph cached to {cache_dir}")

        morse_2d_result = {
            'morse_graph': CachedMorseGraph(nx_graph),
            'barycenters': barycenters,
            'config': config,
            'metadata': metadata,
            'method': method
        }

        return morse_2d_result, False


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
# Pipeline Utilities (New)
# =============================================================================

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


def save_morse_graph_data(directory: str, data: Dict[str, Any]) -> None:
    """Save Morse graph data (graph, sets, barycenters, config) to directory."""
    import pickle
    import json
    import os
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
    import os
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


def save_trajectory_data(directory: str, data: Dict[str, Any]) -> None:
    """Save trajectory data dict to directory."""
    import os
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
    import os
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


def save_models(directory: str, encoder, decoder, latent_dynamics, config: Optional[Dict] = None) -> None:
    """Save PyTorch models and optional config."""
    import torch
    import os
    import json
    
    os.makedirs(directory, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(directory, 'encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join(directory, 'decoder.pt'))
    torch.save(latent_dynamics.state_dict(), os.path.join(directory, 'latent_dynamics.pt'))
    
    if config is not None:
        # Construct model args from config for reconstruction
        model_config = {
            'encoder_args': {
                'input_dim': config.get('input_dim'),
                'latent_dim': config.get('latent_dim'),
                'hidden_dim': config.get('hidden_dim'),
                'num_layers': config.get('num_layers'),
                'output_activation': config.get('encoder_activation')
            },
            'decoder_args': {
                'latent_dim': config.get('latent_dim'),
                'output_dim': config.get('input_dim'),
                'hidden_dim': config.get('hidden_dim'),
                'num_layers': config.get('num_layers'),
                'output_activation': config.get('decoder_activation')
            },
            'dynamics_args': {
                'latent_dim': config.get('latent_dim'),
                'hidden_dim': config.get('hidden_dim'),
                'num_layers': config.get('num_layers'),
                'output_activation': config.get('latent_dynamics_activation')
            }
        }
        # Handle advanced mode if present in config (omitted for brevity, assuming simple mode for now or keys match)
        # Ideally, ExperimentConfig.to_dict() preserves all.
        
        with open(os.path.join(directory, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)


def load_models(directory: str) -> Tuple[Any, Any, Any]:
    """Load PyTorch models. Returns (None, None, None) if not found."""
    import json
    import torch
    import os
    from MorseGraph.models import Encoder, Decoder, LatentDynamics
    
    if not (os.path.exists(os.path.join(directory, 'encoder.pt')) and
            os.path.exists(os.path.join(directory, 'decoder.pt')) and
            os.path.exists(os.path.join(directory, 'latent_dynamics.pt')) and
            os.path.exists(os.path.join(directory, 'training_history.json'))): # Check config/history to reconstruct
        return None, None, None
        
    # Load config to reconstruct models
    # Assuming config is in history or separate file. 
    # pipeline.py saves history separately.
    # Models need dims.
    # We should save model config.
    
    # Try to load config from a file if saved, otherwise rely on user to know?
    # pipeline.py passes config to train, but load_models needs to know dimensions to instantiate.
    
    # Let's assume a 'model_config.json' is saved or we extract from training history
    
    try:
        # Look for training history which contains config params usually
        with open(os.path.join(directory, 'training_history.json'), 'r') as f:
            history_data = json.load(f)
            
        # Hack: assume pipeline saves config inside history or we saved it separately?
        # pipeline.py: save_training_history(training_cache_dir, history)
        # It doesn't seem to save model config explicitly in save_models.
        # But training_hash includes dimensions.
        
        # Ideally we save a 'model_config.json'.
        # For now, let's check if we can load without it? No.
        # We need input_dim, latent_dim etc.
        
        # Let's look if 'config.json' exists (saved by pipeline maybe?)
        # pipeline doesn't call save_config there.
        
        # NOTE: This is a weakness. I'll make save_training_history include config if possible,
        # or pipeline should save it.
        # In pipeline.py: save_models, then save_training_history.
        
        # Let's assume for now we can't load without config.
        # But wait, pipeline.py calls load_models(training_cache_dir).
        # It doesn't pass config.
        
        # I will implement save_models to also save a 'model_config.json' if passed? 
        # Or save_models in utils doesn't take config.
        
        # Correct approach: save_models should take config or dimensions.
        # But signature in pipeline is `save_models(training_cache_dir, encoder, decoder, latent_dynamics)`.
        # So I should extract dims from models themselves!
        
        encoder_state = torch.load(os.path.join(directory, 'encoder.pt'))
        # Infer dims from state dict shapes
        # input_dim: weight of first layer
        # latent_dim: weight of last layer (mu)
        
        # This is brittle.
        
        # Alternative: pipeline.py could save config.
        
        # For now, I will define load_models to return None if it can't instantiate.
        # But how to instantiate?
        
        # Check if 'model_config.json' exists.
        if os.path.exists(os.path.join(directory, 'model_config.json')):
             with open(os.path.join(directory, 'model_config.json'), 'r') as f:
                mc = json.load(f)
        else:
            # Fallback: try to infer or fail
            # Since I am writing this, I can enforce save_models to save config if I could change pipeline.
            # But I don't want to change pipeline call signature if possible.
            # pipeline: `save_models(training_cache_dir, encoder, decoder, latent_dynamics)`
            
            # I will inspect the models to get attributes if they are stored.
            # PyTorch models don't store init args by default.
            
            # I will update pipeline.py to save model config!
            # Or `save_models` in utils can extract it if I modify models to store it.
            
            return None, None, None

        encoder = Encoder(**mc['encoder_args'])
        decoder = Decoder(**mc['decoder_args'])
        latent_dynamics = LatentDynamics(**mc['dynamics_args'])
        
        encoder.load_state_dict(encoder_state)
        decoder.load_state_dict(torch.load(os.path.join(directory, 'decoder.pt')))
        latent_dynamics.load_state_dict(torch.load(os.path.join(directory, 'latent_dynamics.pt')))
        
        return encoder, decoder, latent_dynamics
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None


def save_training_history(directory: str, history: Dict) -> None:
    """Save training history."""
    import json
    import os
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


def load_training_history(directory: str) -> Optional[Dict]:
    """Load training history."""
    import json
    import os
    path = os.path.join(directory, 'training_history.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def compute_latent_bounds_from_data(z_data: np.ndarray, padding: float = 1.1) -> List[List[float]]:
    """Compute padded bounds for latent data."""
    mins = z_data.min(axis=0)
    maxs = z_data.max(axis=0)
    
    ranges = maxs - mins
    centers = (maxs + mins) / 2
    
    padded_ranges = ranges * padding
    
    padded_mins = centers - padded_ranges / 2
    padded_maxs = centers + padded_ranges / 2
    
    return [padded_mins.tolist(), padded_maxs.tolist()]


def generate_3d_grid_for_encoding(bounds: List[List[float]], subdiv: int, input_dim: int) -> np.ndarray:
    """Generate a dense grid in original space."""
    # Approx points per dim
    n_per_dim = int(2**(subdiv / input_dim))
    
    grid_1d = [np.linspace(bounds[0][i], bounds[1][i], n_per_dim) for i in range(input_dim)]
    mesh = np.meshgrid(*grid_1d, indexing='ij')
    
    X_grid = np.stack([m.flatten() for m in mesh], axis=1)
    return X_grid


def generate_random_trajectories_3d(
    dynamics_func: Callable,
    domain_bounds: np.ndarray,
    n_trajectories: int,
    n_points: int,
    skip_initial: int,
    random_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random trajectories for 3D map.
    Returns X, Y arrays.
    """
    np.random.seed(random_seed)
    
    # Sample initial conditions
    ics = np.random.uniform(domain_bounds[0], domain_bounds[1], (n_trajectories, domain_bounds.shape[1]))
    
    X = []
    Y = []
    
    for ic in ics:
        curr = ic
        # Skip
        for _ in range(skip_initial):
            curr = dynamics_func(curr)
            
        # Collect
        for _ in range(n_points):
            next_val = dynamics_func(curr)
            X.append(curr)
            Y.append(next_val)
            curr = next_val
            
    return np.array(X), np.array(Y)


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
    from MorseGraph.core import compute_morse_graph_3d
    from MorseGraph.plot import plot_morse_graph_diagram, plot_morse_sets_3d_scatter
    import json
    from datetime import datetime

    # Compute hash of current parameters
    param_hash = compute_parameter_hash(
        map_func,
        domain_bounds,
        subdiv_min,
        subdiv_max,
        subdiv_init,
        subdiv_limit,
        padding
    )

    cmgdb_3d_dir = os.path.join(base_dir, 'cmgdb_3d')

    # Get paths for this specific parameter configuration
    cache_paths = get_cache_path(cmgdb_3d_dir, param_hash)
    cache_subdir = cache_paths['dir']
    results_dir = os.path.join(cache_subdir, 'results')

    # Try to load from cache if it exists and not forcing recompute
    if not force_recompute and os.path.exists(cache_subdir):
        if verbose:
            print(f"\nFound cache for parameters (hash: {param_hash[:8]}...), attempting to load...")

        # Try to load cached data
        cached = load_morse_graph_cache(cmgdb_3d_dir, param_hash, verbose=verbose)

        if cached is not None:
            # Successfully loaded from cache
            result_3d = {
                'morse_graph': cached['morse_graph'],
                'barycenters': cached['barycenters'],
                'num_morse_sets': cached['num_morse_sets'],
                'computation_time': cached['metadata'].get('computation_time', 0.0),
                'cached': True,
                'cache_path': cache_subdir,
                'param_hash': param_hash
            }
            if verbose:
                print(f"  ✓ Successfully loaded 3D Morse graph from cache")
                print(f"    Morse sets: {result_3d['num_morse_sets']}")
                print(f"    Cache directory: {cache_subdir}")
                print(f"    Visualizations: {results_dir}/")
            return result_3d, True

        else:
            if verbose:
                print(f"  ✗ Cache load failed, will recompute")

    # Compute from scratch
    if verbose:
        print(f"\nComputing 3D Morse graph (hash: {param_hash[:8]}...)...")
        print(f"  Will cache to: {cache_subdir}")

    result_3d = compute_morse_graph_3d(
        map_func,
        domain_bounds,
        subdiv_min=subdiv_min,
        subdiv_max=subdiv_max,
        subdiv_init=subdiv_init,
        subdiv_limit=subdiv_limit,
        padding=padding,
        cache_dir=None,  # Don't use internal caching
        use_cache=False,
        verbose=verbose
    )

    morse_graph_3d = result_3d['morse_graph']
    barycenters_3d = result_3d['barycenters']
    map_graph = result_3d.get('map_graph')

    # Save to cache using save_morse_graph_cache
    metadata = {
        'subdiv_min': subdiv_min,
        'subdiv_max': subdiv_max,
        'subdiv_init': subdiv_init,
        'subdiv_limit': subdiv_limit,
        'padding': padding,
        'domain_bounds': domain_bounds,
        'computation_time': result_3d['computation_time'],
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

    # Create results directory inside the hash directory
    os.makedirs(results_dir, exist_ok=True)

    # Generate and save visualizations
    if verbose:
        print(f"\n  Generating 3D visualizations...")

    # 1. Morse graph diagram
    plot_morse_graph_diagram(
        morse_graph_3d,
        output_path=f"{results_dir}/morse_graph_3d.png",
        title="3D Morse Graph Diagram"
    )
    if verbose:
        print(f"    ✓ Saved morse_graph_3d.png")

    # 2. 3D scatter plot with barycenters (marker size proportional to box volume)
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

    # 3. 2D projection plots using CMGDB.PlotMorseSets
    try:
        import CMGDB
        import matplotlib.pyplot as plt
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

            # Get axis labels for this projection
            xlabel_key = ['x', 'y', 'z'][proj_dims[0]]
            ylabel_key = ['x', 'y', 'z'][proj_dims[1]]

            # Use CMGDB's PlotMorseSets with projection and cool colormap
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

    if verbose:
        print(f"\n  ✓ Saved 3D Morse graph and visualizations to: {cache_subdir}")

    result_3d['cached'] = False
    result_3d['cache_path'] = cache_subdir
    result_3d['param_hash'] = param_hash

    return result_3d, False
