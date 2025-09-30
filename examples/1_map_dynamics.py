#!/usr/bin/env python3
"""
MorseGraph Example 1: Maps

This script shows how to compute a Morse Graph for a discrete map.

The BoxMapFunction class uses epsilon-bloating on interval arithmetic
to define a rigorous outer approximation of the map's action on boxes.

As an example, we analyze the Hénon map, a classic chaotic
two-dimensional discrete dynamical system.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Import the necessary components
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapFunction
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph, compute_all_morse_set_basins
from MorseGraph.plot import plot_morse_sets, plot_morse_graph, plot_basins_of_attraction

# Set up output directory for figures
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(output_dir, "figures")

def henon_map(x, a=1.4, b=0.3):
    """Standard henon map"""
    x_next = 1 - a * x[0]**2 + x[1]
    y_next = b * x[0]
    return np.array([x_next, y_next])

def main():
    print("MorseGraph Example 1: Map Dynamics")
    print("==================================")
    
    # 1. Set up the Dynamics and Grid
    print("\n1. Setting up dynamics and grid...")
    
    # Define the domain for the grid
    domain = np.array([[-2.5, -0.5], [2.5, 0.5]])
    
    # Create the dynamics object
    dynamics = BoxMapFunction(henon_map, epsilon=0.01)
    
    # Create a grid
    grid_x, grid_y = 8, 8
    divisions = np.array([int(2**int(grid_x)), int(2**int(grid_y))], dtype=int)
    grid = UniformGrid(bounds=domain, divisions=divisions)
    
    # Create the model
    model = Model(grid, dynamics)
    
    print(f"Created grid with {len(grid.get_boxes())} boxes")
    print(f"Domain: {domain}")
    
    # 2. Compute the Morse Graph
    print("\n2. Computing Morse graph...")
    
    box_map = model.compute_box_map()
    morse_graph = compute_morse_graph(box_map)
    
    print(f"BoxMap has {len(box_map.nodes())} nodes and {len(box_map.edges())} edges")
    print(f"Morse graph has {len(morse_graph.nodes())} non-trivial Morse sets")
    
    # Print details of Morse sets
    for i, morse_set in enumerate(morse_graph.nodes()):
        print(f"  Morse set {i+1}: {len(morse_set)} boxes")

    # 3. Compute basins of attraction for all Morse sets
    print("\n3. Computing basins of attraction for all Morse sets...")
    basins = compute_all_morse_set_basins(morse_graph, box_map)

    # 4. Visualize the Results
    print("\n4. Creating visualizations...")
    
    # Plot and save the Morse sets figure
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_morse_sets(grid, morse_graph, ax=ax)
    ax.set_title("Morse Sets")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    figure_path = os.path.join(figures_dir, "1_henon_morse_sets.png")
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"Saved Morse sets plot to: {figure_path}")
    
    # Plot and save the Morse Graph
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_morse_graph(morse_graph, ax=ax)
    ax.set_title("Morse Graph")
    graph_figure_path = os.path.join(figures_dir, "1_henon_morse_graph.png")
    plt.savefig(graph_figure_path, dpi=150, bbox_inches='tight')
    print(f"Saved Morse graph plot to: {graph_figure_path}")

    # Plot and save the Basins of Attraction
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_basins_of_attraction(grid, basins, morse_graph=morse_graph, ax=ax, show_outside=True)
    ax.set_title("Basins of Attraction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    basins_plot_path = os.path.join(figures_dir, "1_henon_basins.png")
    plt.savefig(basins_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved basins of attraction plot to: {basins_plot_path}")

if __name__ == "__main__":
    main()