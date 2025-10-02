#!/usr/bin/env python3
"""
MorseGraph Example 3: ODEs (Van der Pol)

This script shows how to compute a Morse Graph for the Van der Pol oscillator.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Import the necessary components
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapODE
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph, compute_all_morse_set_basins
from MorseGraph.plot import plot_morse_sets, plot_morse_graph, plot_basins_of_attraction
from MorseGraph.systems import van_der_pol_ode

# Set up output directory for figures
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(output_dir, "figures")

# =============================================================================
# CONFIGURATION - Edit this section to customize the analysis
# =============================================================================

# Domain and grid configuration
DOMAIN = np.array([[-4.0, -4.0], [4.0, 4.0]])  # State space bounds
GRID_RESOLUTION = 7  # Grid = 2^7 x 2^7 = 128 x 128 boxes

# Dynamics configuration
TAU = 2.0  # Integration time for ODE map
EPSILON_BLOAT = 0.05  # Bloating factor for rigorous outer approximation

# Van der Pol oscillator parameters (used by van_der_pol_ode from MorseGraph.systems)
# mu = 1.0 (default parameter)

# =============================================================================

# Alias for compatibility with existing code
van_der_pol = van_der_pol_ode

def main():
    print("MorseGraph Example 3: ODEs (Van der Pol)")
    print("=======================================")
    
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Set up the Morse Graph Computation
    print("\n1. Setting up Morse graph computation...")

    divisions = np.array([2**GRID_RESOLUTION, 2**GRID_RESOLUTION], dtype=int)
    dynamics = BoxMapODE(van_der_pol, tau=TAU, epsilon=EPSILON_BLOAT)

    grid = UniformGrid(bounds=DOMAIN, divisions=divisions)
    model = Model(grid, dynamics)

    print(f"Created grid with {divisions[0]}x{divisions[1]} boxes")
    print(f"Integration time tau: {TAU}")
    
    # 2. Compute the BoxMap
    print("\n2. Computing BoxMap...")
    box_map = model.compute_box_map()
    print(f"BoxMap computed with {len(box_map.nodes())} nodes and {len(box_map.edges())} edges")
    
    # 3. Compute the Morse graph
    print("\n3. Computing Morse graph...")
    morse_graph = compute_morse_graph(box_map)
    print(f"Morse graph has {len(morse_graph.nodes())} non-trivial Morse sets")

    # 4. Compute basins of attraction
    print("\n4. Computing basins of attraction...")
    basins = compute_all_morse_set_basins(morse_graph, box_map)
    
    # 5. Visualize the Results
    print("\n5. Creating visualizations...")

    # Plot Morse sets
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_morse_sets(grid, morse_graph, ax=ax)
    ax.set_title("Van der Pol: Morse Sets")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.savefig(os.path.join(figures_dir, "3_vdp_morse_sets.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot Morse graph
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_morse_graph(morse_graph, ax=ax)
    plt.savefig(os.path.join(figures_dir, "3_vdp_morse_graph.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot Basins of attraction
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_basins_of_attraction(grid, basins, morse_graph=morse_graph, ax=ax, show_outside=True)
    ax.set_title("Van der Pol: Basins of Attraction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.savefig(os.path.join(figures_dir, "3_vdp_basins.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nExample complete. Check the 'figures' directory.")

if __name__ == "__main__":
    main()
