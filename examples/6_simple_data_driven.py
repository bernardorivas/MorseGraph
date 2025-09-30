#!/usr/bin/env python3
"""
MorseGraph Example 6: Simple Data-Driven API

This script shows how to compute a Morse Graph from data with minimal setup.

The BoxMapData.from_data() method automatically configures grid and dynamics
from trajectory data, with options for handling empty boxes.

As an example, we analyze a simple 2D map with different empty box
strategies (interpolate, outside, terminate).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from MorseGraph.dynamics import BoxMapData
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph
from MorseGraph.plot import plot_morse_sets, plot_morse_graph

# Set up output directory for figures
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

def simple_map(x):
    """Simple 2D map similar to CMGDB examples"""
    return [x[0] / (2.0 - x[0]), x[1] / (2.0 - x[1])]

def demonstrate_empty_box_handling(X, Y, strategy, title):
    """Demonstrate a specific empty box handling strategy."""
    print(f"\n=== {title} ===")
    
    try:
        # Use simplified API - no manual grid creation needed!
        dynamics = BoxMapData.from_data(X, Y, map_empty=strategy)
        model = Model(dynamics.grid, dynamics)
        
        print(f"Auto-created grid: {dynamics.grid.divisions} ({np.prod(dynamics.grid.divisions)} boxes)")
        print(f"Active boxes: {len(dynamics.get_active_boxes(dynamics.grid))}")
        print(f"Empty box strategy: {strategy}")
        
        # Compute Morse graph
        box_map = model.compute_box_map()
        morse_graph = compute_morse_graph(box_map)
        
        print(f"BoxMap: {box_map.number_of_nodes()} nodes, {box_map.number_of_edges()} edges")
        print(f"Morse graph: {len(morse_graph.nodes())} Morse sets")
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot Morse sets
        plot_morse_sets(dynamics.grid, morse_graph, ax=ax1)
        ax1.set_title(f"Morse Sets - {title}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        
        # Plot Morse graph
        plot_morse_graph(morse_graph, ax=ax2)
        ax2.set_title(f"Morse Graph - {title}")
        
        plt.tight_layout()
        
        # Save figure
        filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_path = os.path.join(figures_dir, f"6_simple_{filename}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")
        
        return True
        
    except Exception as e:
        print(f"Strategy '{strategy}' failed: {e}")
        return False

def main():
    print("MorseGraph Example 6: Simple Data-Driven Dynamics")
    print("=================================================")
    print("Demonstrating CMGDB-style simplified API")
    
    # 1. Generate sample data (similar to CMGDB example)
    print("\n1. Generating sample data...")
    
    np.random.seed(37)  # Same seed as CMGDB example
    lower_bounds = [0.0, 0.0]
    upper_bounds = [1.0, 1.0]
    num_pts = 2000
    
    # Generate random points and apply map
    X = np.random.uniform(lower_bounds, upper_bounds, size=(num_pts, 2))
    Y = np.array([simple_map(x) for x in X])
    
    print(f"Generated {num_pts} data points")
    print(f"Domain: [{lower_bounds}, {upper_bounds}]")
    
    # Plot the data
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(X[:, 0], X[:, 1], 'ro', markersize=2, alpha=0.6)
    plt.title("Input Points (X)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)  
    plt.plot(Y[:, 0], Y[:, 1], 'b*', markersize=2, alpha=0.6)
    plt.title("Output Points (Y = f(X))")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    data_path = os.path.join(figures_dir, "6_simple_data.png")
    plt.savefig(data_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved data plot: {data_path}")
    
    # 2. Demonstrate different empty box strategies
    print("\n2. Demonstrating empty box handling strategies...")
    
    strategies = [
        ('interpolate', 'Default (Interpolate)'),
        ('outside', 'Map Empty Outside'),
        ('terminate', 'Terminate on Empty')
    ]
    
    successful_strategies = []
    for strategy, title in strategies:
        success = demonstrate_empty_box_handling(X, Y, strategy, title)
        if success:
            successful_strategies.append((strategy, title))
    
    # 3. Compare with manual configuration
    print("\n3. Comparing with manual grid configuration...")
    
    # Manual approach (like current examples)
    from MorseGraph.grids import UniformGrid
    domain = np.array([[0.0, 0.0], [1.0, 1.0]])
    divisions = np.array([64, 64])  # 2^6 x 2^6
    manual_grid = UniformGrid(bounds=domain, divisions=divisions)
    manual_dynamics = BoxMapData(X, Y, manual_grid, map_empty='interpolate')
    
    # Automatic approach (new simplified API)
    auto_dynamics = BoxMapData.from_data(X, Y, map_empty='interpolate')
    
    print(f"Manual grid: {manual_grid.divisions} ({np.prod(manual_grid.divisions)} boxes)")
    print(f"Auto grid: {auto_dynamics.grid.divisions} ({np.prod(auto_dynamics.grid.divisions)} boxes)")
    print(f"Manual active: {len(manual_dynamics.get_active_boxes(manual_grid))}")
    print(f"Auto active: {len(auto_dynamics.get_active_boxes(auto_dynamics.grid))}")
    
    # 4. Usage recommendations
    print("\n4. Usage recommendations:")
    print("=" * 50)
    print("• Quick start: BoxMapData.from_data(X, Y)")
    print("• Control domain: BoxMapData.from_data(X, Y, domain=custom_domain)")
    print("• Control resolution: BoxMapData.from_data(X, Y, grid_resolution=8)")
    print("• Handle sparse data: map_empty='interpolate' (default)")
    print("• Handle boundary effects: map_empty='outside'")
    print("• Ensure data coverage: map_empty='terminate'")
    
    print(f"\nAll figures saved to: {figures_dir}")
    print("Example completed.")

if __name__ == "__main__":
    main()