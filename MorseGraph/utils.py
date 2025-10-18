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

    return X, Y, trajectories


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

        # Model architecture
        input_dim: int = 3,
        latent_dim: int = 2,
        hidden_dim: int = 32,
        num_layers: int = 3,
        output_activation: Optional[str] = None,
        encoder_activation: Optional[str] = None,
        decoder_activation: Optional[str] = None,
        latent_dynamics_activation: Optional[str] = None,

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

        # Large sample for domain-restricted computation
        large_sample_size: Optional[int] = None,
        target_points_per_box: int = 2,
    ):
        """
        Initialize experiment configuration.

        Args:
            domain_bounds: [[lower_bounds], [upper_bounds]] for 3D domain
            subdiv_min, subdiv_max, subdiv_init, subdiv_limit: CMGDB subdivision parameters for 3D
            padding: Whether to use padding in BoxMap computation
            n_trajectories: Number of trajectories for training data
            n_points: Points per trajectory
            skip_initial: Initial iterations to skip
            random_seed: Random seed for reproducibility
            input_dim: Input dimension (should be 3)
            latent_dim: Latent space dimension (typically 2)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            output_activation: Default activation for all networks
            encoder_activation, decoder_activation, latent_dynamics_activation: Per-network overrides
            num_epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            early_stopping_patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            w_recon, w_dyn_recon, w_dyn_cons: Loss weights
            latent_subdiv_min, latent_subdiv_max: CMGDB parameters for 2D latent space
            latent_padding: Padding for latent BoxMap
            latent_bounds_padding: Padding factor for latent bounding box
            large_sample_size: Size of large sample for domain-restricted computation
            target_points_per_box: Target density for large sample estimation
        """
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

        # Architecture
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_activation = output_activation
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.latent_dynamics_activation = latent_dynamics_activation

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

        # Large sample
        self.large_sample_size = large_sample_size
        self.target_points_per_box = target_points_per_box

        # Map function (to be set)
        self.map_func = None

    def set_map_func(self, map_func: Callable):
        """Set the map function f: R^3 -> R^3."""
        self.map_func = map_func

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
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
            'large_sample_size': self.large_sample_size,
            'target_points_per_box': self.target_points_per_box,
        }


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


def get_cache_path(cache_dir: str, param_hash: str) -> Dict[str, str]:
    """
    Get cache file paths for a given parameter hash.

    Args:
        cache_dir: Base cache directory
        param_hash: Parameter hash from compute_parameter_hash()

    Returns:
        Dictionary with paths:
            - 'dir': Cache subdirectory for this hash
            - 'morse_graph': Path to morse_graph_data.mgdb
            - 'barycenters': Path to barycenters.npz
            - 'metadata': Path to metadata.json

    Example:
        >>> paths = get_cache_path('/path/to/cache', 'a1b2c3d4e5f6g7h8')
        >>> print(paths['morse_graph'])
        /path/to/cache/a1b2c3d4e5f6g7h8/morse_graph_data.mgdb
    """
    cache_subdir = os.path.join(cache_dir, param_hash)

    return {
        'dir': cache_subdir,
        'morse_graph': os.path.join(cache_subdir, 'morse_graph_data.mgdb'),
        'barycenters': os.path.join(cache_subdir, 'barycenters.npz'),
        'metadata': os.path.join(cache_subdir, 'metadata.json'),
    }


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
    1. morse_graph_data.mgdb: CMGDB native format with morse_graph + map_graph
    2. barycenters.npz: NumPy archive of barycenter coordinates
    3. metadata.json: Parameters and computation info

    Args:
        morse_graph: CMGDB MorseGraph object
        map_graph: CMGDB MapGraph object
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

    # Save CMGDB data
    CMGDB.SaveMorseGraphData(
        morse_graph,
        map_graph,
        paths['morse_graph'],
        metadata=metadata_with_timestamp
    )

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
            - 'morse_graph': CMGDB MorseGraph object
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

    try:
        import CMGDB
    except ImportError:
        if verbose:
            print("  WARNING: CMGDB not available, cannot load cache")
        return None

    # Get cache paths
    paths = get_cache_path(cache_dir, param_hash)

    # Check if cache exists
    if not os.path.exists(paths['morse_graph']):
        return None

    try:
        # Load morse graph
        morse_graph = CMGDB.MorseGraph(paths['morse_graph'])

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
            print(f"  WARNING: Failed to load cache: {e}")
        return None