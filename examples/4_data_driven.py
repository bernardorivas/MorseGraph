#!/usr/bin/env python3
"""
Data-driven dynamics on a restricted subgrid.

This example demonstrates:
1. Generating trajectory data from the Henon map (known dynamics)
2. Using BoxMapData to construct data-driven dynamics
3. Restricting computation to a subgrid defined by where data exists
4. Comparing with ground truth (BoxMapFunction on full domain)

The key insight is that for data-driven dynamics, we only need to compute
on boxes that contain data - this can dramatically reduce computation time.

Generates figures:
- 4_morse_sets_data.png: Morse sets from data-driven dynamics
- 4_morse_sets_ground_truth.png: Morse sets from ground truth (full domain)
- 4_coverage.png: Data point density per box
- 4_raw_data.png: Raw trajectory data scatter plot
- 4_basins_data.png: Basins of attraction (data-driven)
- 4_comparison.png: Side-by-side comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapData, BoxMapFunction
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph, compute_all_morse_set_basins
from MorseGraph.plot import (
    plot_morse_sets, plot_morse_graph, plot_data_coverage,
    plot_basins_of_attraction, plot_data_points_overlay
)
from MorseGraph.systems import henon_map

# =============================================================================
# CONFIGURATION
# =============================================================================

# Henon map parameters
HENON_PARAMS = {'a': 1.4, 'b': 0.3}

# Domain - standard Henon attractor region
DOMAIN = np.array([[-2.5, -0.5], [2.5, 0.5]])

# Grid resolution
GRID_DIVISIONS = [256, 128]

# Data generation parameters
N_TRAJECTORIES = 500          # Number of random trajectories
TRAJECTORY_LENGTH = 200       # Steps per trajectory
TRANSIENT_SKIP = 50           # Skip initial transient

# Epsilon for bloating (in grid cell units)
EPSILON_CELLS = 1.0

# Random seed
RANDOM_SEED = 42

# =============================================================================
# Helper Functions
# =============================================================================

def generate_henon_data(n_trajectories, traj_length, skip_transient, domain, seed=None):
    """
    Generate trajectory data from the Henon map.

    Returns (X, Y) pairs where Y = f(X) for the Henon map.
    Only returns points that stay within the domain.
    """
    if seed is not None:
        np.random.seed(seed)

    henon = lambda x: henon_map(x, **HENON_PARAMS)

    X_list = []
    Y_list = []

    for _ in range(n_trajectories):
        # Random initial condition
        x = np.random.uniform(domain[0], domain[1])

        # Run trajectory
        for step in range(traj_length + skip_transient):
            x_next = henon(x)

            # After transient, collect data if both points are in domain
            if step >= skip_transient:
                if (np.all(x >= domain[0]) and np.all(x <= domain[1]) and
                    np.all(x_next >= domain[0]) and np.all(x_next <= domain[1])):
                    X_list.append(x.copy())
                    Y_list.append(x_next.copy())

            x = x_next

            # Stop if trajectory escapes
            if np.any(np.abs(x) > 10):
                break

    return np.array(X_list), np.array(Y_list)


def compute_data_containing_boxes(X, grid):
    """
    Find which grid boxes contain at least one data point.

    Returns set of box indices that contain data.
    """
    # Compute which box each point falls into
    cell_size = (grid.bounds[1] - grid.bounds[0]) / np.array(grid.divisions)

    # Clip points to domain
    X_clipped = np.clip(X, grid.bounds[0], grid.bounds[1] - 1e-10)

    # Compute box indices for each point
    indices = np.floor((X_clipped - grid.bounds[0]) / cell_size).astype(int)
    indices = np.clip(indices, 0, np.array(grid.divisions) - 1)

    # Convert to flat indices
    flat_indices = np.ravel_multi_index(indices.T, grid.divisions)

    return set(flat_indices)


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    print("Data-Driven Dynamics on Restricted Subgrid")
    print("=" * 60)
    print(f"System: Henon map (a={HENON_PARAMS['a']}, b={HENON_PARAMS['b']})")
    print(f"Domain: {DOMAIN[0]} to {DOMAIN[1]}")
    print(f"Grid: {GRID_DIVISIONS}")

    # Setup output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # =========================================================================
    # 1. Generate trajectory data from Henon map
    # =========================================================================
    print(f"\n1. Generating Henon map trajectory data...")
    print(f"   Trajectories: {N_TRAJECTORIES}")
    print(f"   Length: {TRAJECTORY_LENGTH} (skip first {TRANSIENT_SKIP})")

    t0 = time.time()
    X, Y = generate_henon_data(
        N_TRAJECTORIES, TRAJECTORY_LENGTH, TRANSIENT_SKIP, DOMAIN, RANDOM_SEED
    )
    print(f"   Generated {len(X)} (X, Y) pairs in {time.time()-t0:.2f}s")

    # Setup grid
    grid = UniformGrid(bounds=DOMAIN, divisions=GRID_DIVISIONS)
    cell_size = (DOMAIN[1] - DOMAIN[0]) / np.array(GRID_DIVISIONS)
    print(f"   Grid: {grid.divisions} divisions ({np.prod(grid.divisions)} total boxes)")
    print(f"   Cell size: {cell_size}")

    # Find data-containing boxes
    data_boxes = compute_data_containing_boxes(X, grid)
    print(f"   Data covers {len(data_boxes)} boxes ({100*len(data_boxes)/np.prod(grid.divisions):.1f}%)")

    # =========================================================================
    # 2. Visualize raw data
    # =========================================================================
    print("\n2. Visualizing raw data...")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X[:, 0], X[:, 1], c='blue', s=0.5, alpha=0.3, label='X')
    ax.scatter(Y[:, 0], Y[:, 1], c='red', s=0.5, alpha=0.1, label='Y=f(X)')
    ax.set_xlim(DOMAIN[0, 0], DOMAIN[1, 0])
    ax.set_ylim(DOMAIN[0, 1], DOMAIN[1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Henon Map Trajectory Data ({len(X)} points)')
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '4_raw_data.png'), dpi=150)
    plt.close()

    # =========================================================================
    # 3. Data-driven dynamics on RESTRICTED subgrid
    # =========================================================================
    print("\n3. Computing Morse graph (data-driven, restricted domain)...")

    epsilon = cell_size * EPSILON_CELLS

    # BoxMapData with map_empty='outside' restricts to data-containing boxes
    dynamics_data = BoxMapData(
        X, Y, grid,
        input_distance_metric='L1',
        output_distance_metric='L1',
        input_epsilon=epsilon,
        output_epsilon=epsilon,
        map_empty='outside'  # Boxes without data map outside domain
    )

    t0 = time.time()
    model_data = Model(grid, dynamics_data)
    box_map_data = model_data.compute_box_map()
    morse_graph_data = compute_morse_graph(box_map_data)
    t_data = time.time() - t0

    print(f"   Computed in {t_data:.2f}s")
    print(f"   BoxMap: {box_map_data.number_of_nodes()} nodes, {box_map_data.number_of_edges()} edges")
    print(f"   Found {len(morse_graph_data.nodes())} Morse sets")

    # Plot Morse sets
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_morse_sets(grid, morse_graph_data, ax=ax, box_map=box_map_data, show_outside=True)
    ax.scatter(X[:, 0], X[:, 1], c='black', s=0.1, alpha=0.1, zorder=0)
    ax.set_title(f'Morse Sets (Data-Driven, Restricted) - {len(morse_graph_data.nodes())} sets')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '4_morse_sets_data.png'), dpi=150)
    plt.close()

    # Data coverage visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data_coverage(grid, dynamics_data, ax=ax, colormap='plasma')
    ax.set_title('Data Coverage (points per box)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '4_coverage.png'), dpi=150)
    plt.close()

    # =========================================================================
    # 4. Ground truth: BoxMapFunction on full domain
    # =========================================================================
    print("\n4. Computing Morse graph (ground truth, full domain)...")

    henon_func = lambda x: henon_map(x, **HENON_PARAMS)
    dynamics_gt = BoxMapFunction(map_f=henon_func, epsilon=np.mean(epsilon))

    t0 = time.time()
    model_gt = Model(grid, dynamics_gt)
    box_map_gt = model_gt.compute_box_map()
    morse_graph_gt = compute_morse_graph(box_map_gt)
    t_gt = time.time() - t0

    print(f"   Computed in {t_gt:.2f}s")
    print(f"   BoxMap: {box_map_gt.number_of_nodes()} nodes, {box_map_gt.number_of_edges()} edges")
    print(f"   Found {len(morse_graph_gt.nodes())} Morse sets")

    # Plot ground truth Morse sets
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_morse_sets(grid, morse_graph_gt, ax=ax, box_map=box_map_gt, show_outside=True)
    ax.set_title(f'Morse Sets (Ground Truth, Full Domain) - {len(morse_graph_gt.nodes())} sets')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '4_morse_sets_ground_truth.png'), dpi=150)
    plt.close()

    # =========================================================================
    # 5. Basins of attraction (data-driven)
    # =========================================================================
    print("\n5. Computing basins of attraction...")

    basins_data = compute_all_morse_set_basins(morse_graph_data, box_map_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_basins_of_attraction(grid, basins_data, morse_graph=morse_graph_data, ax=ax, show_outside=True)
    ax.set_title('Basins of Attraction (Data-Driven)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '4_basins_data.png'), dpi=150)
    plt.close()

    # =========================================================================
    # 6. Side-by-side comparison
    # =========================================================================
    print("\n6. Creating comparison visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Data-driven (restricted)
    ax = axes[0]
    plot_morse_sets(grid, morse_graph_data, ax=ax, box_map=box_map_data, show_outside=True)
    ax.scatter(X[:, 0], X[:, 1], c='black', s=0.1, alpha=0.05, zorder=0)
    ax.set_title(f'Data-Driven (Restricted)\n{len(morse_graph_data.nodes())} Morse sets, {t_data:.2f}s')

    # Ground truth (full)
    ax = axes[1]
    plot_morse_sets(grid, morse_graph_gt, ax=ax, box_map=box_map_gt, show_outside=True)
    ax.set_title(f'Ground Truth (Full Domain)\n{len(morse_graph_gt.nodes())} Morse sets, {t_gt:.2f}s')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '4_comparison.png'), dpi=150)
    plt.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Data points:           {len(X)}")
    print(f"Data-containing boxes: {len(data_boxes)} / {np.prod(grid.divisions)} ({100*len(data_boxes)/np.prod(grid.divisions):.1f}%)")
    print()
    print("Data-Driven (Restricted):")
    print(f"  Morse sets: {len(morse_graph_data.nodes())}")
    print(f"  Edges:      {box_map_data.number_of_edges()}")
    print(f"  Time:       {t_data:.2f}s")
    print()
    print("Ground Truth (Full Domain):")
    print(f"  Morse sets: {len(morse_graph_gt.nodes())}")
    print(f"  Edges:      {box_map_gt.number_of_edges()}")
    print(f"  Time:       {t_gt:.2f}s")
    print()
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
