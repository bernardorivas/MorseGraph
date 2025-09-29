#!/usr/bin/env python3
"""
MorseGraph Example 1: Map Dynamics

This script demonstrates how to use MorseGraph with a simple 2D map, the Hénon map.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Import the necessary components
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapFunction
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph
from MorseGraph.plot import plot_morse_sets

# Set up output directory for figures
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = output_dir

def henon_map(x, a=1.4, b=0.3):
    """Standard Henon map, vectorized."""
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
    
    # 3. Visualize the Results
    print("\n3. Creating visualizations...")
    
    # Plot the Morse sets on the grid
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_morse_sets(grid, morse_graph, ax=ax)
    ax.set_title("Morse Sets for the Hénon Map")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Save the figure
    figure_path = os.path.join(figures_dir, "henon_morse_sets.png")
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"Saved Morse sets plot to: {figure_path}")
    
    # Show the plot
    plt.show()
    
    # Additional analysis
    print("\n4. Additional Analysis:")
    
    # Count attractors (nodes with no outgoing edges)
    attractors = [node for node in morse_graph.nodes() if morse_graph.out_degree(node) == 0]
    print(f"Number of attractors: {len(attractors)}")
    
    # Count sources (nodes with no incoming edges)  
    sources = [node for node in morse_graph.nodes() if morse_graph.in_degree(node) == 0]
    print(f"Number of sources: {len(sources)}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()