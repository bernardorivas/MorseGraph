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
from typing import Callable, Dict, Tuple, List, Optional, Any, Union

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