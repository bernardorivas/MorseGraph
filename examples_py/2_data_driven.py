#!/usr/bin/env python3
"""
MorseGraph Example 2: Data-Driven Dynamics

This script demonstrates how to compute a Morse graph from a dataset of input-output 
pairs (X, Y). This is useful when the dynamics are not known from a function or ODE, 
but are given by data (e.g., from a simulation or experiment).

We use the BoxMapData dynamics class. For this example, we first generate a dataset 
(X, Y) from a known map (the Henon map) so we can see how the data-driven approach works.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Import the necessary components from the MorseGraph library
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapData
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph
from MorseGraph.plot import plot_morse_sets

# Set up output directory for figures
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = output_dir

def henon_map(x, a=1.4, b=0.3):
    """Standard Henon map, vectorized."""
    x_next = 1 - a * x[:, 0]**2 + x[:, 1]
    y_next = b * x[:, 0]
    return np.column_stack([x_next, y_next])

def main():
    print("MorseGraph Example 2: Data-Driven Dynamics")
    print("==========================================")
    
    # 1. Generate Sample Data
    print("\n1. Generating sample data...")
    
    # Define the domain and number of sample points
    lower_bounds = np.array([-1.5, -0.4])
    upper_bounds = np.array([1.5, 0.4])
    num_points = 5000
    
    # Generate random points X and their images Y
    X = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(num_points, 2))
    Y = henon_map(X)
    
    print(f"Generated {num_points} data points")
    print(f"X range: [{X.min(axis=0)}, {X.max(axis=0)}]")
    print(f"Y range: [{Y.min(axis=0)}, {Y.max(axis=0)}]")
    
    # Plot the data to see what it looks like
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], s=5, c='blue', alpha=0.6, label='Original Points (X)')
    plt.scatter(Y[:, 0], Y[:, 1], s=5, c='red', alpha=0.6, label='Mapped Points (Y)')
    plt.title("Henon Map Dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the data plot
    data_plot_path = os.path.join(figures_dir, "henon_data.png")
    plt.savefig(data_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved data plot to: {data_plot_path}")
    plt.show()
    
    # 2. Set up the Morse Graph Computation
    print("\n2. Setting up Morse graph computation...")
    
    # Define the grid parameters
    divisions = np.array([32, 32])  # Smaller grid for faster computation
    domain = np.array([[-1.5, -0.4], [1.5, 0.4]])
    
    # Create the dynamics object from our data
    dynamics = BoxMapData(X, Y, epsilon=0.1)
    
    # Create the grid
    grid = UniformGrid(bounds=domain, divisions=divisions)
    
    # Create the model which connects the grid and dynamics
    model = Model(grid, dynamics)
    
    print(f"Created grid with {len(grid.get_boxes())} boxes")
    print(f"Using {len(X)} data points for dynamics")
    
    # 3. Compute the state transition graph (map graph)
    print("\n3. Computing map graph...")
    map_graph = model.compute_map_graph()
    print(f"Map graph computed with {len(map_graph.nodes())} nodes and {len(map_graph.edges())} edges")
    
    # 4. Compute the Morse graph from the map graph
    print("\n4. Computing Morse graph...")
    morse_graph = compute_morse_graph(map_graph)
    print(f"Morse graph has {len(morse_graph.nodes())} non-trivial Morse sets")
    
    # Print details of Morse sets
    for i, morse_set in enumerate(morse_graph.nodes()):
        print(f"  Morse set {i+1}: {len(morse_set)} boxes")
    
    # 5. Visualize the Results
    print("\n5. Creating visualizations...")
    
    # Plot the Morse sets on the grid
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_morse_sets(grid, morse_graph, ax=ax)
    ax.set_title("Morse Sets from Data-Driven Dynamics")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Save the Morse sets plot
    morse_plot_path = os.path.join(figures_dir, "henon_data_morse_sets.png")
    plt.savefig(morse_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved Morse sets plot to: {morse_plot_path}")
    plt.show()
    
    # 6. Analysis and Comparison
    print("\n6. Additional Analysis:")
    
    # Data coverage analysis
    boxes = grid.get_boxes()
    boxes_with_data = 0
    for i, box in enumerate(boxes):
        # Check if any data points fall in this box
        in_box = np.all((X >= box[0]) & (X <= box[1]), axis=1)
        if np.any(in_box):
            boxes_with_data += 1
    
    print(f"Grid boxes with data: {boxes_with_data}/{len(boxes)} ({100*boxes_with_data/len(boxes):.1f}%)")
    
    # Count attractors and sources
    attractors = [node for node in morse_graph.nodes() if morse_graph.out_degree(node) == 0]
    sources = [node for node in morse_graph.nodes() if morse_graph.in_degree(node) == 0]
    
    print(f"Number of attractors: {len(attractors)}")
    print(f"Number of sources: {len(sources)}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()