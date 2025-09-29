#!/usr/bin/env python3
"""
MorseGraph Example 4: Iterative Refinement

This script demonstrates the iterative_morse_computation function, which allows for 
progressively refining the grid to get a more detailed Morse graph.

This process is key for adaptive analysis, where computational effort is focused on 
the most dynamically complex regions of the state space.

Note on AdaptiveGrid:
The project plan includes an AdaptiveGrid class for true adaptive refinement 
(i.e., subdividing only specific boxes). This example uses the actual AdaptiveGrid 
implementation to demonstrate local refinement capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Import the necessary components
from MorseGraph.grids import AdaptiveGrid
from MorseGraph.dynamics import BoxMapFunction
from MorseGraph.core import Model
from MorseGraph.analysis import iterative_morse_computation
from MorseGraph.plot import plot_morse_sets

# Set up output directory for figures
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = output_dir

def henon_map(x, a=1.4, b=0.3):
    """Standard Henon map."""
    x_next = 1 - a * x[0]**2 + x[1]
    y_next = b * x[0]
    return np.array([x_next, y_next])

def main():
    print("MorseGraph Example 4: Iterative Refinement")
    print("=========================================")
    
    # 1. Set up the Dynamics and Initial Grid
    print("\n1. Setting up dynamics and initial grid...")
    
    # Define the domain for the grid
    domain = np.array([[-1.5, -0.4], [1.5, 0.4]])
    
    # Create the dynamics object
    dynamics = BoxMapFunction(henon_map, epsilon=0.05)
    
    # Create an initial COARSE adaptive grid
    # We start with a single box and let the refinement process subdivide it
    grid = AdaptiveGrid(bounds=domain, max_depth=8)
    
    # Create the model
    model = Model(grid, dynamics)
    
    print(f"Initial grid: {len(grid.get_boxes())} boxes")
    print(f"Domain: {domain}")
    print(f"Max refinement depth: {grid.max_depth}")
    
    # 2. Run the Iterative Computation
    print("\n2. Running iterative refinement...")
    
    # Run the iterative computation for several refinement steps
    # The function will print its progress
    final_morse_graph, refinement_history = iterative_morse_computation(
        model=model,
        max_depth=4,  # 4 refinement iterations
        refinement_threshold=0.05  # Refine Morse sets with >5% of total boxes
    )
    
    print(f"\nFinal grid: {len(model.grid.get_boxes())} boxes")
    
    # 3. Analyze the Refinement Process
    print("\n3. Analyzing refinement history...")
    
    for i, iteration_info in enumerate(refinement_history):
        print(f"Iteration {iteration_info['iteration']}: "
              f"{iteration_info['num_boxes']} boxes, "
              f"{iteration_info['num_morse_sets']} Morse sets, "
              f"{iteration_info['num_recurrent_boxes']} boxes to refine")
    
    # 4. Visualize the Refinement Process
    print("\n4. Creating visualizations...")
    
    # Plot refinement convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot number of boxes over iterations
    iterations = [info['iteration'] for info in refinement_history]
    box_counts = [info['num_boxes'] for info in refinement_history]
    morse_counts = [info['num_morse_sets'] for info in refinement_history]
    
    ax1.plot(iterations, box_counts, 'bo-', label='Total Boxes')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Boxes')
    ax1.set_title('Grid Refinement Progress')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot number of Morse sets over iterations
    ax2.plot(iterations, morse_counts, 'ro-', label='Morse Sets')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Number of Morse Sets')
    ax2.set_title('Morse Set Discovery')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save refinement progress plot
    progress_plot_path = os.path.join(figures_dir, "refinement_progress.png")
    plt.savefig(progress_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved refinement progress plot to: {progress_plot_path}")
    plt.show()
    
    # 5. Visualize the Final Results
    print("\n5. Visualizing final results...")
    
    # Plot the final Morse sets on the adaptively refined grid
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_morse_sets(grid, final_morse_graph, ax=ax)
    ax.set_title(f"Final Morse Sets after Adaptive Refinement\n({len(grid.get_boxes())} boxes)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Save the final Morse sets plot
    final_plot_path = os.path.join(figures_dir, "adaptive_refinement_final.png")
    plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved final Morse sets plot to: {final_plot_path}")
    plt.show()
    
    # 6. Compare Initial vs Final
    print("\n6. Comparison: Initial vs Final")
    
    # Create initial computation for comparison
    initial_grid = AdaptiveGrid(bounds=domain, max_depth=8)
    initial_model = Model(initial_grid, dynamics)
    initial_box_map = initial_model.compute_box_map()
    
    from MorseGraph.analysis import compute_morse_graph
    initial_morse_graph = compute_morse_graph(initial_box_map)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot initial (single box)
    plot_morse_sets(initial_grid, initial_morse_graph, ax=ax1)
    ax1.set_title(f"Initial Grid\n({len(initial_grid.get_boxes())} box)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    
    # Plot final (refined)
    plot_morse_sets(grid, final_morse_graph, ax=ax2)
    ax2.set_title(f"Final Refined Grid\n({len(grid.get_boxes())} boxes)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_plot_path = os.path.join(figures_dir, "refinement_comparison.png")
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {comparison_plot_path}")
    plt.show()
    
    # 7. Final Analysis
    print("\n7. Final Analysis:")
    
    print(f"Grid refinement factor: {len(grid.get_boxes())} (final) / {len(initial_grid.get_boxes())} (initial) = {len(grid.get_boxes())}")
    print(f"Final Morse sets: {len(final_morse_graph.nodes())}")
    
    # Analyze final Morse structure
    attractors = [node for node in final_morse_graph.nodes() if final_morse_graph.out_degree(node) == 0]
    sources = [node for node in final_morse_graph.nodes() if final_morse_graph.in_degree(node) == 0]
    
    print(f"Attractors found: {len(attractors)}")
    print(f"Sources found: {len(sources)}")
    
    print(f"\nAdaptive refinement successfully focused computational effort")
    print(f"on the most dynamically interesting regions of the Henon map!")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()