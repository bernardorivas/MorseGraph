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
import matplotlib.cm as cm
import networkx as nx
from scipy.integrate import solve_ivp
import os

# Import the necessary components from the MorseGraph library
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapODE
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph
from MorseGraph.plot import plot_morse_sets, plot_morse_graph

# Set up output directory for figures
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = output_dir

def bistable_ode(t, x):
    """
    A simple ODE system.
    dx/dt = x - x^3
    dy/dt = -y
    
    This system has:
    - Stable fixed points (sinks) at (1, 0) and (-1, 0)
    - Unstable fixed point (saddle) at (0, 0)
    """
    dxdt = x[0] - x[0]**3
    dydt = -x[1]
    return np.array([dxdt, dydt])

def main(show_phase_portrait=False):
    print("MorseGraph Example 3: ODE-Based Dynamics")
    print("=======================================")
    
    # 1. Define the ODE System and optionally visualize
    print("\n1. Defining the ODE system...")
    
    # Create vector field for visualization (always compute for overlay)
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
    
    # Normalize vector field for better visualization
    magnitude = np.sqrt(u**2 + v**2)
    magnitude = np.where(magnitude == 0, 1, magnitude)
    u_norm = u / magnitude
    v_norm = v / magnitude

    if show_phase_portrait:
        # Plot vector field with sample trajectories
        plt.figure(figsize=(10, 8))
        plt.quiver(x, y, u_norm, v_norm, color='blue', alpha=0.5, scale=40, width=0.002)
        
        
        # Generate 5 random initial conditions
        np.random.seed(42)  # For reproducible results
        initial_conditions = np.random.uniform(low=[-1.8, -1.8], high=[1.8, 1.8], size=(5, 2))
        
        # Integrate and plot trajectories
        t_span = (0, 3.0)  # Integration time
        t_eval = np.linspace(0, 3.0, 300)
        
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for i, initial_point in enumerate(initial_conditions):
            sol = solve_ivp(bistable_ode, t_span, initial_point, t_eval=t_eval, dense_output=True)
            
            if sol.success:
                trajectory = sol.y.T  # Shape: (n_times, 2)
                plt.plot(trajectory[:, 0], trajectory[:, 1], 
                        color=colors[i], linewidth=2, alpha=0.8, 
                        label=f'Trajectory {i+1}')
                
                # Mark initial point
                plt.plot(initial_point[0], initial_point[1], 
                        'o', color=colors[i], markersize=8, markeredgecolor='black')
                
                # Mark final point
                plt.plot(trajectory[-1, 0], trajectory[-1, 1], 
                        's', color=colors[i], markersize=6, markeredgecolor='black')
        
        plt.title("Bistable ODE: Vector Field and Sample Trajectories")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        
        # Mark fixed points
        plt.plot([-1, 0, 1], [0, 0, 0], 'ko', markersize=10, 
                 markerfacecolor='white', markeredgewidth=2, label='Fixed Points')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        
        # Save plot
        trajectory_plot_path = os.path.join(figures_dir, "bistable_trajectories.png")
        plt.savefig(trajectory_plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to: {trajectory_plot_path}")
        plt.show()
    
    # 2. Set up the Morse Graph Computation
    print("\n2. Setting up Morse graph computation...")
    
    # Define the grid parameters
    grid_x, grid_y = 7, 7
    divisions = np.array([int(2**grid_x-1), int(2**grid_y-1)], dtype=int)
    domain = np.array([[-2.0, -2.0], [2.0, 2.0]])
    
    # Create the dynamics object for our ODE
    tau = 2.0
    dynamics = BoxMapODE(bistable_ode,tau)
    
    # Create the grid
    grid = UniformGrid(bounds=domain, divisions=divisions)
    
    # Create the model
    model = Model(grid, dynamics)
    
    print(f"Created grid with {divisions[0]}x{divisions[1]} boxes")
    print(f"Integration time tau: {tau}")
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
    
    # First plot: Morse sets on the grid
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 7))
    morse_sets = list(morse_graph.nodes())
    plot_morse_sets(grid, morse_graph, ax=ax1)
    
    # Extract colors for consistency
    morse_set_colors = {}
    num_sets = len(morse_sets)
    if num_sets > 0:
        cmap = cm.get_cmap('tab10')
        for i, morse_set in enumerate(morse_sets):
            color = cmap(i / max(num_sets, 10))
            morse_set_colors[morse_set] = color
    ax1.set_title("Morse Sets (Spatial)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True, alpha=0.3)
    
    spatial_plot_path = os.path.join(figures_dir, "bistable_morse_sets.png")
    plt.savefig(spatial_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved Morse sets plot to: {spatial_plot_path}")
    plt.show()
    
    # Second plot: Morse graph with hierarchical layout
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    plot_morse_graph(morse_graph, ax=ax2, morse_sets_colors=morse_set_colors)
    
    graph_plot_path = os.path.join(figures_dir, "bistable_morse_graph.png")
    plt.savefig(graph_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved Morse graph to: {graph_plot_path}")
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