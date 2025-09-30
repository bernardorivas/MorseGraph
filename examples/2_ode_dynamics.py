#!/usr/bin/env python3
"""
MorseGraph Example 2: ODEs

This script shows how to compute a Morse Graph for a ODE.

The BoxMapODE class uses (scipy.integrate.solve_ivp) 
to define a map f(x) = \varphi(\tau,x) and then use BoxMap for f.

As an example, we analyze a genetic toggle switch - a bistable
biological system where two genes mutually inhibit each other.
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
from MorseGraph.analysis import compute_morse_graph, compute_all_morse_set_basins
from MorseGraph.plot import plot_morse_sets, plot_morse_graph, plot_basins_of_attraction

# Set up output directory for figures
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(output_dir, "figures")

def toggle_switch(t, x, alpha1=5.0, alpha2=5.0, beta=1.0, n=2.0):
    """
    Toggle switch: Models two genes that mutually inhibit each other's expression.

    Equations:
        dx/dt = alpha1/(1 + y^n) - beta*x
        dy/dt = alpha2/(1 + x^n) - beta*y

    Parameters:
        alpha1, alpha2: Maximum production rates for genes x and y
        beta: Degradation rate for both genes
        n: Hill coefficient
    """
    dxdt = alpha1 / (1 + x[1]**n) - beta * x[0]
    dydt = alpha2 / (1 + x[0]**n) - beta * x[1]
    return np.array([dxdt, dydt])

def main(show_phase_portrait=False):
    print("MorseGraph Example 2: ODEs")
    print("=======================================")
    
    # Ensure figures directory exists
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Define the ODE System and optionally visualize
    print("\n1. Defining the ODE system...")
    
    # Create vector field for visualization (always compute for overlay)
    x_vec = np.linspace(0, 6, 20)
    y_vec = np.linspace(0, 6, 20)
    x, y = np.meshgrid(x_vec, y_vec)
    
    # Compute vector field
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            derivatives = toggle_switch(0, np.array([x[i,j], y[i,j]]))
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
        initial_conditions = np.random.uniform(low=[0.5, 0.5], high=[5.5, 5.5], size=(5, 2))
        
        # Integrate and plot trajectories
        t_span = (0, 3.0)  # Integration time
        t_eval = np.linspace(0, 3.0, 300)
        
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for i, initial_point in enumerate(initial_conditions):
            sol = solve_ivp(toggle_switch, t_span, initial_point, t_eval=t_eval, dense_output=True)
            
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
        
        plt.title("Toggle Switch: Vector Field and Sample Trajectories")
        plt.xlabel("Gene X Expression")
        plt.ylabel("Gene Y Expression")
        plt.grid(True, alpha=0.3)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(0, 6)
        plt.ylim(0, 6)
        
        # Save plot
        trajectory_plot_path = os.path.join(figures_dir, "2_bistable_trajectories.png")
        plt.savefig(trajectory_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved trajectory plot to: {trajectory_plot_path}")
    
    # 2. Set up the Morse Graph Computation
    print("\n2. Setting up Morse graph computation...")
    
    # Define the grid parameters
    grid_x, grid_y = 7, 7
    divisions = np.array([int(2**grid_x-1), int(2**grid_y-1)], dtype=int)
    domain = np.array([[0.0, 0.0], [6.0, 6.0]])
    
    # Create the dynamics object for our ODE
    tau = 5.0
    dynamics = BoxMapODE(toggle_switch,tau)
    
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

    # 5. Compute basins of attraction for all Morse sets
    print("\n5. Computing basins of attraction for all Morse sets...")
    basins = compute_all_morse_set_basins(morse_graph, box_map)
    
    # 5. Visualize the Results
    print("\n5. Creating visualizations...")

    # First plot: Morse sets on the grid
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 7))
    plot_morse_sets(grid, morse_graph, ax=ax1)
    ax1.set_title("Morse Sets")
    ax1.set_xlabel("Gene X Expression")
    ax1.set_ylabel("Gene Y Expression")
    ax1.grid(True, alpha=0.3)

    spatial_plot_path = os.path.join(figures_dir, "2_bistable_morse_sets.png")
    plt.savefig(spatial_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Morse sets plot to: {spatial_plot_path}")

    # Second plot: Morse graph with hierarchical layout
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    plot_morse_graph(morse_graph, ax=ax2)

    graph_plot_path = os.path.join(figures_dir, "2_bistable_morse_graph.png")
    plt.savefig(graph_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Morse graph to: {graph_plot_path}")

    # Third plot: Basins of attraction
    fig, ax3 = plt.subplots(1, 1, figsize=(8, 7))
    plot_basins_of_attraction(grid, basins, morse_graph=morse_graph, ax=ax3, show_outside=True)
    ax3.set_title("Basins of Attraction")
    ax3.set_xlabel("Gene X Expression")
    ax3.set_ylabel("Gene Y Expression")
    ax3.grid(True, alpha=0.3)

    basins_plot_path = os.path.join(figures_dir, "2_bistable_basins.png")
    plt.savefig(basins_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved basins of attraction plot to: {basins_plot_path}")

if __name__ == "__main__":
    main()