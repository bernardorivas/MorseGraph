#!/usr/bin/env python3
"""
Data-driven example using Van der Pol oscillator.

Computes
0. Data coverage visualization and raw data
1. BoxMapData with k-nearest points interpolation
2. MorseGraph and MorseSets
3. Basins of attraction computation

Generates 5 figures:
- 4_morse_sets.png: Morse sets in state space
- 4_morse_graph.png: Morse graph structure
- 4_coverage.png: Data point density per box
- 4_points.png: Raw trajectory data scatter plot
- 4_basins.png: Basins of attraction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapData
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph, compute_all_morse_set_basins
from MorseGraph.plot import plot_morse_sets, plot_morse_graph, plot_data_coverage, plot_basins_of_attraction
from MorseGraph.systems import van_der_pol_ode
from MorseGraph.utils import generate_trajectory_data

# =============================================================================
# CONFIGURATION - Edit this section to change the system
# =============================================================================

# System configuration
ODE_FUNCTION = van_der_pol_ode        # ODE Function (from MorseGraph.systems)
ODE_PARAMS = {'mu': 1.0}              # ODE Parameters
DOMAIN = np.array([[-4, -4], [4, 4]]) # State space (cubical domain)
GRID_DIVISIONS = [128, 128]           # Grid resolution

# BoxMapData configuration
_cell_sizes = (DOMAIN[1] - DOMAIN[0]) / np.array(GRID_DIVISIONS, dtype=float)
INPUT_EPSILON = _cell_sizes * 1.0
OUTPUT_EPSILON = _cell_sizes * 1.0

# Trajectory generation parameters
N_SAMPLES = 5000                      # Number of random initial conditions
TOTAL_TIME = 10.0                     # Total integration time per trajectory
N_POINTS_PER_TRAJECTORY = 10          # Number of time steps to force in integration

# Random seed
RANDOM_SEED = 42

# =============================================================================
# Analysis
# =============================================================================

# Set up output directory
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(output_dir, "figures")

def main():
    print("Data-Driven Example")
    print("=" * 60)
    print(f"System: {ODE_FUNCTION.__name__}")
    print(f"Domain: {DOMAIN[0]} to {DOMAIN[1]}")
    print(f"Grid: {GRID_DIVISIONS}")

    os.makedirs(figures_dir, exist_ok=True)

    # Generate training data
    print(f"\n1. Generating trajectory data...")
    print(f"   Initial conditions: {N_SAMPLES}")
    print(f"   Total time: {TOTAL_TIME}")
    print(f"   Points per trajectory: {N_POINTS_PER_TRAJECTORY}")

    X, Y, _ = generate_trajectory_data(
        ODE_FUNCTION,
        ODE_PARAMS,
        N_SAMPLES,
        TOTAL_TIME,
        N_POINTS_PER_TRAJECTORY,
        DOMAIN,
        RANDOM_SEED
    )
    print(f"   Generated {len(X)} data points")

    # Setup grid
    grid = UniformGrid(bounds=DOMAIN, divisions=GRID_DIVISIONS)
    print(f"   Grid: {grid.divisions} divisions ({np.prod(grid.divisions)} boxes)")

    # ========================================================================
    # Part 1: Computing morse sets with default settings
    # ========================================================================
    print("\n2. Computing Morse sets...")

    dynamics = BoxMapData(X, Y, grid,
                          input_distance_metric='L1',
                          output_distance_metric='L1',
                          input_epsilon=INPUT_EPSILON,
                          output_epsilon=OUTPUT_EPSILON)

    model = Model(grid, dynamics)
    box_map = model.compute_box_map()
    morse_graph = compute_morse_graph(box_map)
    print(f"   Found {len(morse_graph.nodes())} Morse sets")

    _, ax = plt.subplots(figsize=(8, 8))
    plot_morse_sets(grid, morse_graph, ax=ax, box_map=box_map, show_outside=True)
    ax.set_title(f'Morse sets')

    output_path = os.path.join(figures_dir, '4_morse_sets.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    # Plot Morse graph
    _, ax = plt.subplots(figsize=(8, 8))
    plot_morse_graph(morse_graph, ax=ax)
    ax.set_title('Morse Graph')

    output_path = os.path.join(figures_dir, '4_morse_graph.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    # ========================================================================
    # Part 2: Data-per-box visualization
    # ========================================================================
    print("\n3. Visualizing data coverage...")

    _, ax = plt.subplots(figsize=(8, 8))
    plot_data_coverage(grid, dynamics, ax=ax, colormap='plasma')
    ax.set_title('Data coverage per box')

    output_path = os.path.join(figures_dir, '4_coverage.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    # Scatter plot of the data (X and Y)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(X[:, 0], X[:, 1], 'b.', alpha=0.7, ms=2, label='X')
    ax.plot(Y[:, 0], Y[:, 1], 'r.', alpha=0.3, ms=2, label='Y=F(X)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(figures_dir, '4_raw_data.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    # ========================================================================
    # Part 3: Basins of Attraction
    # ========================================================================
    print("\n4. Computing basins of attraction...")

    # Compute basins for all Morse sets
    basins = compute_all_morse_set_basins(morse_graph, box_map)

    # Plot basins with grey boxes for points mapping outside
    _, ax = plt.subplots(figsize=(8, 8))
    plot_basins_of_attraction(grid, basins, morse_graph=morse_graph, ax=ax, show_outside=True)
    ax.set_title('Basins of Attraction')

    output_path = os.path.join(figures_dir, '4_basins.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("✓ Data-driven analysis complete!")
    print(f"✓ All figures saved to: {figures_dir}")

if __name__ == "__main__":
    main()
