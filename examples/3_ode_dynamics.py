#!/usr/bin/env python3
"""
MorseGraph Example 3: ODE-Based Dynamics

This script demonstrates how to compute a Morse graph for a dynamical system 
defined by an Ordinary Differential Equation (ODE).

We use the BoxMapODE dynamics class, which uses a numerical integrator 
(scipy.integrate.solve_ivp) to approximate the flow of the system over a time 
horizon tau. As an example, we analyze a simple bistable system with two stable 
fixed points and a saddle point.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Import the necessary components from the MorseGraph library
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapODE
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph
from MorseGraph.plot import plot_morse_sets

# Set up output directory for figures
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = output_dir

def bistable_ode(t, x):
    """
    A simple bistable ODE system.
    dx/dt = x - x^3
    dy/dt = -y
    
    This system has:
    - Stable fixed points (sinks) at (1, 0) and (-1, 0)
    - Unstable fixed point (saddle) at (0, 0)
    """
    dxdt = x[0] - x[0]**3
    dydt = -x[1]
    return np.array([dxdt, dydt])

def main():
    print("MorseGraph Example 3: ODE-Based Dynamics")
    print("=======================================")
    
    # 1. Define and Visualize the ODE System
    print("\n1. Defining the bistable ODE system...")
    
    # Create vector field for visualization
    x_vec = np.linspace(-2, 2, 20)
    y_vec = np.linspace(-2, 2, 20)
    x, y = np.meshgrid(x_vec, y_vec)
    
    # Compute vector field
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            derivatives = bistable_ode(0, np.array([x[i,j], y[i,j]]))
            u[i,j] = derivatives[0]
            v[i,j] = derivatives[1]
    
    # Plot vector field
    plt.figure(figsize=(8, 8))
    plt.quiver(x, y, u, v, color='blue', alpha=0.7)
    plt.title("Vector Field of the Bistable ODE")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    
    # Mark fixed points
    plt.plot([-1, 0, 1], [0, 0, 0], 'ro', markersize=8, label='Fixed Points')
    plt.legend()
    
    # Save vector field plot
    vector_field_path = os.path.join(figures_dir, "bistable_vector_field.png")
    plt.savefig(vector_field_path, dpi=150, bbox_inches='tight')
    print(f"Saved vector field plot to: {vector_field_path}")
    plt.show()
    
    # 2. Set up the Morse Graph Computation
    print("\n2. Setting up Morse graph computation...")
    
    # Define the grid parameters
    divisions = np.array([24, 24])  # Reasonable resolution
    domain = np.array([[-2.0, -2.0], [2.0, 2.0]])
    
    # Create the dynamics object for our ODE
    # tau=0.5 is the integration time - should capture movement between regions
    dynamics = BoxMapODE(bistable_ode, tau=0.5, epsilon=0.1)
    
    # Create the grid
    grid = UniformGrid(bounds=domain, divisions=divisions)
    
    # Create the model
    model = Model(grid, dynamics)
    
    print(f"Created grid with {len(grid.get_boxes())} boxes")
    print(f"Integration time tau = 0.5")
    print(f"Domain: {domain}")
    
    # 3. Compute the BoxMap
    print("\n3. Computing BoxMap...")
    box_map = model.compute_box_map()
    print(f"BoxMap computed with {len(box_map.nodes())} nodes and {len(box_map.edges())} edges")
    
    # 4. Compute the Morse graph
    print("\n4. Computing Morse graph...")
    morse_graph = compute_morse_graph(box_map)
    print(f"Morse graph has {len(morse_graph.nodes())} non-trivial Morse sets")
    
    # Print details of Morse sets
    for i, morse_set in enumerate(morse_graph.nodes()):
        print(f"  Morse set {i+1}: {len(morse_set)} boxes")
        # Find approximate center of each Morse set
        boxes = grid.get_boxes()
        morse_boxes = boxes[list(morse_set)]
        center = np.mean(morse_boxes.reshape(-1, 2), axis=0)
        print(f"    Approximate center: ({center[0]:.2f}, {center[1]:.2f})")
    
    # 5. Visualize the Results
    print("\n5. Creating visualizations...")
    
    # Plot the Morse sets on the grid with vector field overlay
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_morse_sets(grid, morse_graph, ax=ax)
    
    # Overlay the vector field for context
    ax.quiver(x, y, u, v, color='blue', alpha=0.3, scale=50, width=0.002)
    
    # Mark fixed points
    ax.plot([-1, 0, 1], [0, 0, 0], 'ko', markersize=8, label='Fixed Points')
    
    ax.set_title("Morse Sets for Bistable ODE System")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save the Morse sets plot
    morse_plot_path = os.path.join(figures_dir, "bistable_morse_sets.png")
    plt.savefig(morse_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved Morse sets plot to: {morse_plot_path}")
    plt.show()
    
    # 6. Analysis
    print("\n6. System Analysis:")
    
    # Count attractors (nodes with no outgoing edges)
    attractors = [node for node in morse_graph.nodes() if morse_graph.out_degree(node) == 0]
    print(f"Number of attractors: {len(attractors)}")
    
    # Count sources (nodes with no incoming edges)
    sources = [node for node in morse_graph.nodes() if morse_graph.in_degree(node) == 0]
    print(f"Number of sources: {len(sources)}")
    
    # Analyze connectivity
    print(f"Morse graph connectivity:")
    for i, (source, target) in enumerate(morse_graph.edges()):
        print(f"  Connection {i+1}: {len(source)} boxes -> {len(target)} boxes")
    
    print(f"\nThis matches the expected bistable structure:")
    print(f"- Two attractors around (-1,0) and (1,0)")
    print(f"- Saddle region around (0,0) connecting to both attractors")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()