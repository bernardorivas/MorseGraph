#!/usr/bin/env python3
"""
MorseGraph Example 3: Data-Driven Dynamics

This script shows how to compute a Morse Graph from trajectory data.

The BoxMapData class uses spatial indexing (cKDTree) to efficiently
find which boxes are reached from data points within each box.

As an example, we analyze trajectory data from the Hénon map,
demonstrating how to work with sampled observations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapData
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph
from MorseGraph.plot import plot_morse_sets

# Set up output directory for figures
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(output_dir, "figures")

def henon_map(x, a=1.4, b=0.3):
    """Standard henon map"""
    x_next = 1 - a * x[:, 0]**2 + x[:, 1]
    y_next = b * x[:, 0]
    return np.column_stack([x_next, y_next])

def analyze_configuration(X, Y, grid, config_name, dynamics_kwargs, figures_dir):
    """Analyze a specific BoxMapData configuration and return results."""
    print(f"\n=== {config_name} ===")
    
    start_time = time.time()
    
    # Create dynamics with specific configuration
    dynamics = BoxMapData(X, Y, grid, **dynamics_kwargs)
    model = Model(grid, dynamics)
    
    # Report configuration details
    print(f"Configuration: input_ε={dynamics.input_epsilon:.6f}, output_ε={dynamics.output_epsilon:.6f}, dilation={dynamics.dilation_radius}")
    
    # Compute BoxMap
    box_map = model.compute_box_map()
    active_boxes = len(dynamics.get_active_boxes(grid))
    
    # Compute Morse graph
    morse_graph = compute_morse_graph(box_map)
    
    compute_time = time.time() - start_time
    
    # Analysis
    attractors = [node for node in morse_graph.nodes() if morse_graph.out_degree(node) == 0]
    sources = [node for node in morse_graph.nodes() if morse_graph.in_degree(node) == 0]
    
    # Report results
    print(f"Active boxes: {active_boxes}/{len(grid.get_boxes())} ({100*active_boxes/len(grid.get_boxes()):.1f}%)")
    print(f"BoxMap: {box_map.number_of_nodes()} nodes, {box_map.number_of_edges()} edges")
    if box_map.number_of_nodes() > 0:
        print(f"Connectivity: {box_map.number_of_edges()/box_map.number_of_nodes():.2f} edges/node")
    print(f"Morse structure: {len(morse_graph.nodes())} sets, {len(attractors)} attractors, {len(sources)} sources")
    print(f"Computation time: {compute_time:.2f}s")
    
    # Visualize if requested
    if figures_dir:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_morse_sets(grid, morse_graph, ax=ax)
        ax.set_title(f"{config_name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        # Save plot
        filename = config_name.lower().replace(" ", "_")
        plot_path = os.path.join(figures_dir, f"3_henon_{filename}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")
    
    return {
        'name': config_name,
        'dynamics': dynamics,
        'box_map': box_map,
        'morse_graph': morse_graph,
        'active_boxes': active_boxes,
        'compute_time': compute_time,
        'attractors': len(attractors),
        'sources': len(sources)
    }

def main():
    print("MorseGraph Example 3: Data-Driven Dynamics")
    print("==========================================")
    print("Showcasing simplified and advanced BoxMapData APIs")
    
    # Ensure figures directory exists
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Generate Sample Data
    print("\n1. Generating sample data...")
    
    # Use smaller dataset and grid for faster demonstration
    lower_bounds = np.array([-2.5, -0.5])
    upper_bounds = np.array([2.5, 0.5])
    num_points = 2000  # Smaller for faster computation
    
    # Generate random points X and their images Y using Henon map
    np.random.seed(42)  # For reproducible results
    X = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(num_points, 2))
    Y = henon_map(X)
    
    print(f"Generated {num_points} data points")
    print(f"X range: [{X.min(axis=0)}, {X.max(axis=0)}]")
    print(f"Y range: [{Y.min(axis=0)}, {Y.max(axis=0)}]")
    
    # Quick demonstration of simplified API
    print("\n1.1 Quick start with simplified API...")
    simple_dynamics = BoxMapData.from_data(X, Y)
    simple_model = Model(simple_dynamics.grid, simple_dynamics)
    simple_morse = compute_morse_graph(simple_model.compute_box_map())
    print(f"Quick start: {simple_dynamics.grid.divisions} grid, {len(simple_morse.nodes())} Morse sets")
    
    # Plot the dataset
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], s=3, c='blue', alpha=0.7)
    plt.title("Input Points (X)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(Y[:, 0], Y[:, 1], s=3, c='red', alpha=0.7)
    plt.title("Output Points (Y = Henon(X))")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    data_plot_path = os.path.join(figures_dir, "3_henon_dataset.png")
    plt.savefig(data_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved dataset visualization: {data_plot_path}")
    
    # 2. Set up the Grid
    print("\n2. Setting up computational grid...")
    
    # Use moderate grid size for good balance of detail and speed
    grid_size = 7  # 2^7 = 128, so 128x128 = 16384 total boxes
    divisions = np.array([2**grid_size, 2**grid_size], dtype=int)
    domain = np.array([[-2.5, -0.5], [2.5, 0.5]])
    
    grid = UniformGrid(bounds=domain, divisions=divisions)
    
    print(f"Created {divisions[0]}×{divisions[1]} grid ({np.prod(divisions)} total boxes)")
    print(f"Box size: {grid.box_size}")
    print(f"Domain: {domain}")
    
    # 3. Demonstrate Different BoxMapData Configurations
    print("\n3. Comparing BoxMapData configurations...")
    
    configurations = [
        {
            'name': 'Default Epsilon',
            'kwargs': {},
            'description': 'Automatic epsilon = box size'
        },
        {
            'name': 'Custom Epsilon',
            'kwargs': {'input_epsilon': grid.box_size[0]/3, 'output_epsilon': grid.box_size[0]/2},
            'description': 'Manual epsilon control'
        },
        {
            'name': 'Grid Dilation',
            'kwargs': {'dilation_radius': 1},
            'description': 'Include neighboring boxes'
        },
        {
            'name': 'Sparse Epsilon',
            'kwargs': {'input_epsilon': 0, 'output_epsilon': grid.box_size[0]/10},
            'description': 'Zero input, minimal output epsilon'
        },
        {
            'name': 'Empty Box Outside',
            'kwargs': {'map_empty': 'outside'},
            'description': 'Map empty boxes outside domain'
        },
        {
            'name': 'Empty Box Interpolate',
            'kwargs': {'map_empty': 'interpolate'},
            'description': 'Interpolate from neighboring boxes'
        }
    ]
    
    results = []
    for config in configurations:
        print(f"\n{config['description']}")
        result = analyze_configuration(X, Y, grid, config['name'], config['kwargs'], figures_dir)
        results.append(result)
    
    # 4. Summary Comparison
    print("\n4. Configuration Summary:")
    print("=" * 80)
    print(f"{'Configuration':<25} {'Edges':<8} {'Conn.':<6} {'Morse':<6} {'Attr.':<5} {'Time':<6}")
    print("-" * 80)
    
    for result in results:
        connectivity = result['box_map'].number_of_edges() / result['active_boxes'] if result['active_boxes'] > 0 else 0
        print(f"{result['name']:<25} {result['box_map'].number_of_edges():<8} {connectivity:<6.2f} {len(result['morse_graph'].nodes()):<6} {result['attractors']:<5} {result['compute_time']:<6.2f}s")
    
    # 5. Configuration Usage and New Features
    print("\n5. When to use each configuration:")
    print("=" * 60)
    print("• Default Epsilon: Automatic epsilon based on grid size")
    print("• Custom Epsilon: Manual control of input/output epsilon")  
    print("• Grid Dilation: Expand search to neighboring grid boxes")
    print("• Sparse Epsilon: Minimal bloating for precise dynamics")
    print("• Empty Box Outside: Handle boundary effects gracefully")
    print("• Empty Box Interpolate: Fill gaps using neighbor information")
    
    print("\n6. New simplified API:")
    print("=" * 60)
    print("• Quick start: BoxMapData.from_data(X, Y)")
    print("• Custom domain: BoxMapData.from_data(X, Y, domain=bounds)")
    print("• Custom resolution: BoxMapData.from_data(X, Y, grid_resolution=8)")
    print("• Empty box handling: BoxMapData.from_data(X, Y, map_empty='outside')")
    
    print(f"\nAll visualizations saved to: {figures_dir}")
    print("Example completed.")

if __name__ == "__main__":
    main()