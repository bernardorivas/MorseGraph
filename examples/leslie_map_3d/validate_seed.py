#!/usr/bin/env python3
"""
Seed Validation Script for Leslie Map 3D

This script validates that trajectory data generated with a given random seed
is rich enough to reproduce the ground-truth Morse graph structure using
CMGDB.BoxMapData (data-driven approach).

Ground truth structure: 4 nodes with edges: 3 -> 2, 2 -> 0, 2 -> 1
"""

import CMGDB
import numpy as np
import argparse
import time
import os
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from MorseGraph.systems import leslie_map_3d
from MorseGraph.utils import generate_map_trajectory_data


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Data generation parameters (matching cmgdb_single_run.py)
    N_TRAJECTORIES = 10000 # 1000
    N_POINTS = 10
    SKIP_INITIAL = 0
    DOMAIN_BOUNDS = [[-0.1, -0.1, -0.1], [90.0, 70.0, 70.0]]

    # CMGDB parameters for 3D Leslie map (data-driven validation)
    # Note: Data-driven needs much coarser subdivisions than function-based (36-42)
    # because with sparse data, fine subdivisions create mostly empty boxes
    # This validation uses coarser subdivisions to test if data CAN capture topology
    SUBDIV_MIN = 15
    SUBDIV_MAX = 18
    SUBDIV_INIT = 0
    SUBDIV_LIMIT = 10000

    # Output directory for validation results
    OUTPUT_DIR = 'validating_seeds'

    # Seed search range
    SEED_START = 0
    SEED_END = 1 # 100
    MAX_SEEDS_TO_TRY = 1


# ============================================================================
# Ground Truth Graph Structure
# ============================================================================

GROUND_TRUTH = {
    'num_nodes': 4,
    'edges': [(3, 2), (2, 0), (2, 1)]  # Directed edges
}


# ============================================================================
# Visualization Functions
# ============================================================================

def compute_data_bounds(data, padding_factor=0.05):
    """
    Compute the actual bounding box of the data with optional padding.

    Args:
        data: Array of shape (N, D) containing data points
        padding_factor: Fraction of range to add as padding (default 5%)

    Returns:
        [lower_bounds, upper_bounds] where each is a list of length D
    """
    if data is None or len(data) == 0:
        return None

    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_range = data_max - data_min

    # Add padding
    lower_bounds = (data_min - padding_factor * data_range).tolist()
    upper_bounds = (data_max + padding_factor * data_range).tolist()

    return [lower_bounds, upper_bounds]


def plot_morse_graph(morse_graph, output_path, title='Morse Graph'):
    """Plot and save the Morse graph diagram."""
    fig, ax = plt.subplots(figsize=(8, 8))

    if morse_graph is None or morse_graph.num_vertices() == 0:
        ax.text(0.5, 0.5, 'Empty graph', ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
    else:
        gv_source = CMGDB.PlotMorseGraph(morse_graph, cmap=matplotlib.cm.cool)
        img_data = gv_source.pipe(format='png')
        img = plt.imread(io.BytesIO(img_data))
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_barycenters_3d(morse_graph, domain_bounds, output_path, title='Barycenters of Morse Sets', data_overlay=None):
    """Compute and plot 3D barycenters of Morse sets."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    num_morse_sets = morse_graph.num_vertices()

    # Plot data overlay first (behind everything else) if provided
    if data_overlay is not None and len(data_overlay) > 0:
        ax.scatter(data_overlay[:, 0], data_overlay[:, 1], data_overlay[:, 2],
                  c='black', s=0.5, alpha=0.05, zorder=1)

    if num_morse_sets == 0:
        ax.text2D(0.5, 0.5, 'No Morse sets', ha='center', va='center', transform=ax.transAxes)
    else:
        colors = matplotlib.cm.cool(np.linspace(0, 1, num_morse_sets))

        for i in range(num_morse_sets):
            morse_set_boxes = morse_graph.morse_set_boxes(i)
            if morse_set_boxes:
                dim = len(morse_set_boxes[0]) // 2
                barycenters = []
                for box in morse_set_boxes:
                    barycenter = np.array([(box[j] + box[j + dim]) / 2.0 for j in range(dim)])
                    barycenters.append(barycenter)

                data = np.array(barycenters)
                ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                          c=[colors[i]], marker='s', s=10, label=f'Morse Set {i}', zorder=2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)

    if domain_bounds is not None:
        ax.set_xlim(domain_bounds[0][0], domain_bounds[1][0])
        ax.set_ylim(domain_bounds[0][1], domain_bounds[1][1])
        ax.set_zlim(domain_bounds[0][2], domain_bounds[1][2])

    ax.view_init(elev=30, azim=45)

    if num_morse_sets > 0 and num_morse_sets <= 10:
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_morse_sets_2d(morse_graph, output_path, title='Morse Sets (2D Projection)'):
    """Plot 2D projection of 3D Morse sets using CMGDB."""
    fig = CMGDB.PlotMorseSets(morse_graph, cmap=matplotlib.cm.cool, fig_w=7, fig_h=7)

    if fig is None:
        # Create a simple figure if PlotMorseSets returns None
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No Morse sets to plot', ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
    else:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Helper Functions
# ============================================================================

def extract_graph_structure(morse_graph):
    """
    Extract the graph structure (nodes and edges) from a CMGDB Morse graph.

    Args:
        morse_graph: CMGDB MorseGraph object

    Returns:
        dict with 'num_nodes' and 'edges' (list of tuples)
    """
    num_nodes = morse_graph.num_vertices()
    edges = []

    for i in range(num_nodes):
        adjacencies = morse_graph.adjacencies(i)
        for j in adjacencies:
            edges.append((i, j))

    return {
        'num_nodes': num_nodes,
        'edges': sorted(edges)
    }


def compare_graph_structures(actual, expected):
    """
    Compare actual graph structure to expected ground truth.

    Args:
        actual: dict with 'num_nodes' and 'edges'
        expected: dict with 'num_nodes' and 'edges'

    Returns:
        bool: True if structures match, False otherwise
    """
    if actual['num_nodes'] != expected['num_nodes']:
        return False

    # Sort edges for comparison
    actual_edges = sorted(actual['edges'])
    expected_edges = sorted(expected['edges'])

    return actual_edges == expected_edges


def compute_morse_graph_from_data(x_data, y_data, config):
    """
    Compute Morse graph using CMGDB.BoxMapData.

    Args:
        x_data: Input trajectory points (N x 3)
        y_data: Output trajectory points (N x 3)
        config: Configuration dict

    Returns:
        morse_graph: CMGDB MorseGraph object
    """
    # Create BoxMapData object
    # Use 'interp' mode for sparse data - it will interpolate when boxes have no points
    box_map_data = CMGDB.BoxMapData(
        X=x_data,
        Y=y_data,
        map_empty='outside',
        lower_bounds=config['DOMAIN_BOUNDS'][0],
        upper_bounds=config['DOMAIN_BOUNDS'][1],
        domain_padding=True,
        padding=True
    )

    # Define the map function wrapper
    def F(rect):
        return box_map_data.compute(rect)

    # Create CMGDB model
    model = CMGDB.Model(
        config['SUBDIV_MIN'],
        config['SUBDIV_MAX'],
        config['SUBDIV_INIT'],
        config['SUBDIV_LIMIT'],
        config['DOMAIN_BOUNDS'][0],
        config['DOMAIN_BOUNDS'][1],
        F
    )

    # Compute Morse graph
    morse_graph, _ = CMGDB.ComputeMorseGraph(model)

    return morse_graph


# ============================================================================
# Main Validation Function
# ============================================================================

def validate_seed(seed, config, output_base_dir=None, verbose=False):
    """
    Validate a single random seed by checking if the generated data
    reproduces the ground-truth Morse graph structure.

    Args:
        seed: Random seed to test
        config: Configuration dict
        output_base_dir: Base directory for saving outputs (if None, no saving)
        verbose: Print detailed information

    Returns:
        (bool, dict): (is_valid, structure_dict)
    """
    # Generate trajectory data with this seed
    x_t, x_t_plus_1, _ = generate_map_trajectory_data(
        map_func=leslie_map_3d,
        n_trajectories=config['N_TRAJECTORIES'],
        n_points=config['N_POINTS'],
        sampling_domain=np.array(config['DOMAIN_BOUNDS']),
        random_seed=seed,
        skip_initial=config['SKIP_INITIAL']
    )

    if verbose:
        print(f"  Generated {len(x_t)} data points")

    # Compute Morse graph from data
    morse_graph = compute_morse_graph_from_data(x_t, x_t_plus_1, config)

    # Extract structure
    actual_structure = extract_graph_structure(morse_graph)

    if verbose:
        print(f"  Computed Morse graph: {actual_structure['num_nodes']} nodes, {len(actual_structure['edges'])} edges")
        print(f"  Edges: {actual_structure['edges']}")

    # Compare to ground truth
    is_valid = compare_graph_structures(actual_structure, GROUND_TRUTH)

    # Save visualizations if output directory is provided
    if output_base_dir is not None:
        seed_dir = os.path.join(output_base_dir, f'seed_{seed:03d}')
        os.makedirs(seed_dir, exist_ok=True)

        if verbose:
            print(f"  Saving visualizations to {seed_dir}/")

        # 1. Morse graph diagram (no data overlay makes sense here)
        plot_morse_graph(
            morse_graph,
            os.path.join(seed_dir, 'morse_graph.png'),
            title=f'Morse Graph - Seed {seed}'
        )

        # 2. 3D barycenter scatterplot (without data)
        plot_barycenters_3d(
            morse_graph,
            config['DOMAIN_BOUNDS'],
            os.path.join(seed_dir, 'barycenters_3d.png'),
            title=f'Barycenters - Seed {seed}'
        )

        # 2b. 3D barycenter scatterplot WITH data overlay
        # Compute actual data bounds instead of using domain bounds
        data_bounds = compute_data_bounds(x_t, padding_factor=0.05)

        # Check if data goes outside the original domain
        domain_lower = np.array(config['DOMAIN_BOUNDS'][0])
        domain_upper = np.array(config['DOMAIN_BOUNDS'][1])
        data_lower = np.array(data_bounds[0])
        data_upper = np.array(data_bounds[1])

        outside_lower = data_lower < domain_lower
        outside_upper = data_upper > domain_upper

        if np.any(outside_lower) or np.any(outside_upper):
            if verbose:
                print(f"  ⚠ WARNING: Data goes outside your original domain!")
                print(f"    Original domain: {config['DOMAIN_BOUNDS']}")
                print(f"    Actual bounds:   {data_bounds}")
                if np.any(outside_lower):
                    dims = np.where(outside_lower)[0]
                    print(f"    Exceeds lower bound in dimension(s): {dims.tolist()}")
                if np.any(outside_upper):
                    dims = np.where(outside_upper)[0]
                    print(f"    Exceeds upper bound in dimension(s): {dims.tolist()}")

        plot_barycenters_3d(
            morse_graph,
            data_bounds,  # Use actual data bounds
            os.path.join(seed_dir, 'barycenters_3d_with_data.png'),
            title=f'Barycenters + Data - Seed {seed}',
            data_overlay=x_t  # Original trajectory data
        )

        # 3. 2D morse sets projection (without data)
        plot_morse_sets_2d(
            morse_graph,
            os.path.join(seed_dir, 'morse_sets_2d.png'),
            title=f'Morse Sets (2D) - Seed {seed}'
        )

        # Save metadata including actual data bounds
        metadata = {
            'seed': seed,
            'num_nodes': actual_structure['num_nodes'],
            'edges': actual_structure['edges'],
            'is_valid': is_valid,
            'ground_truth': GROUND_TRUTH,
            'n_trajectories': config['N_TRAJECTORIES'],
            'n_points': config['N_POINTS'],
            'n_data_points': len(x_t),
            'domain_bounds': config['DOMAIN_BOUNDS'],
            'actual_data_bounds': data_bounds
        }

        with open(os.path.join(seed_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    return is_valid, actual_structure


# ============================================================================
# Seed Search
# ============================================================================

def search_for_valid_seed(config, verbose=True):
    """
    Search for a random seed that produces data reproducing the ground truth.

    Args:
        config: Configuration dict
        verbose: Print progress

    Returns:
        int or None: Valid seed if found, None otherwise
    """
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, config['OUTPUT_DIR'])
    os.makedirs(output_dir, exist_ok=True)

    # Respect MAX_SEEDS_TO_TRY limit
    max_seed = min(config['SEED_END'] + 1, config['SEED_START'] + config['MAX_SEEDS_TO_TRY'])

    print(f"\n{'='*80}")
    print(f"Searching for valid random seed...")
    print(f"Ground truth: {GROUND_TRUTH['num_nodes']} nodes, edges: {GROUND_TRUTH['edges']}")
    print(f"Seed range: {config['SEED_START']} to {config['SEED_END']}")
    print(f"Max seeds to try: {config['MAX_SEEDS_TO_TRY']}")
    print(f"Will test seeds: {config['SEED_START']} to {max_seed - 1}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    valid_seeds = []

    for seed in range(config['SEED_START'], max_seed):
        if verbose:
            print(f"Testing seed {seed}...")

        start_time = time.time()
        is_valid, structure = validate_seed(seed, config, output_base_dir=output_dir, verbose=verbose)
        elapsed = time.time() - start_time

        if is_valid:
            print(f"  ✓ VALID! Seed {seed} reproduces ground truth (computation: {elapsed:.2f}s)")
            valid_seeds.append(seed)
        else:
            print(f"  ✗ Invalid. Got {structure['num_nodes']} nodes, edges: {structure['edges']} ({elapsed:.2f}s)")

        print()

    print(f"{'='*80}")
    if valid_seeds:
        print(f"Found {len(valid_seeds)} valid seed(s): {valid_seeds}")
        print(f"\nRecommendation: Use seed {valid_seeds[0]} for cmgdb_single_run.py")
    else:
        print(f"No valid seeds found in range [{config['SEED_START']}, {config['SEED_END']}]")
        print(f"Try expanding the search range or adjusting parameters.")
    print(f"{'='*80}\n")

    return valid_seeds[0] if valid_seeds else None


# ============================================================================
# Command-Line Interface
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate random seeds for Leslie map data-driven Morse graph'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Test a specific seed (if not provided, searches a range)'
    )
    parser.add_argument(
        '--seed-start', type=int, default=None,
        help='Start of seed search range (default: from Config)'
    )
    parser.add_argument(
        '--seed-end', type=int, default=None,
        help='End of seed search range (default: from Config)'
    )
    parser.add_argument(
        '--max-seeds', type=int, default=None,
        help='Maximum number of seeds to try (default: from Config)'
    )
    parser.add_argument(
        '--n-trajectories', type=int, default=None,
        help='Number of trajectories (default: from Config)'
    )
    parser.add_argument(
        '--n-points', type=int, default=None,
        help='Points per trajectory (default: from Config)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build configuration
    config = {k: v for k, v in vars(Config).items() if not k.startswith('__')}

    # Apply command-line overrides
    if args.seed_start is not None:
        config['SEED_START'] = args.seed_start
    if args.seed_end is not None:
        config['SEED_END'] = args.seed_end
    if args.max_seeds is not None:
        config['MAX_SEEDS_TO_TRY'] = args.max_seeds
    if args.n_trajectories is not None:
        config['N_TRAJECTORIES'] = args.n_trajectories
    if args.n_points is not None:
        config['N_POINTS'] = args.n_points

    verbose = not args.quiet

    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, config['OUTPUT_DIR'])
    os.makedirs(output_dir, exist_ok=True)

    # Test specific seed or search
    if args.seed is not None:
        print(f"\nTesting specific seed: {args.seed}")
        print(f"Output directory: {output_dir}\n")

        is_valid, structure = validate_seed(args.seed, config, output_base_dir=output_dir, verbose=True)

        if is_valid:
            print(f"\n✓ Seed {args.seed} is VALID!")
            print(f"  Reproduces ground truth: {GROUND_TRUTH}")
            print(f"  Visualizations saved to: {output_dir}/seed_{args.seed:03d}/")
        else:
            print(f"\n✗ Seed {args.seed} is INVALID")
            print(f"  Expected: {GROUND_TRUTH}")
            print(f"  Got: {structure}")
            print(f"  Visualizations saved to: {output_dir}/seed_{args.seed:03d}/")
    else:
        # Search for valid seed
        valid_seed = search_for_valid_seed(config, verbose=verbose)

        if valid_seed is not None:
            print(f"\n✓ Success! Recommended seed: {valid_seed}")
            return 0
        else:
            print(f"\n✗ No valid seed found")
            return 1


if __name__ == "__main__":
    exit(main())
