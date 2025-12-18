#!/usr/bin/env python3
"""
MorseGraph Example 1: Maps

This script demonstrates computing a Morse Graph for a discrete map using both
CMGDB (C++ backend) and Python pipelines with the same resolution.

The BoxMapFunction class uses epsilon-bloating on interval arithmetic
to define a rigorous outer approximation of the map's action on boxes.

As an example, we analyze the Hénon map, a classic chaotic
two-dimensional discrete dynamical system.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import time
import argparse

# Import the necessary components
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapFunction
from MorseGraph.core import Model
from MorseGraph.analysis import (
    compute_morse_graph,
    compute_all_morse_set_basins,
    identify_attractors
)
from MorseGraph.plot import (
    plot_morse_sets,
    plot_morse_graph,
    plot_basins_of_attraction
)
from MorseGraph.systems import henon_map

# Set up output directory for figures
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "output", "1_map_dynamics")
os.makedirs(output_path, exist_ok=True)

# =============================================================================
# CONFIGURATION - Edit this section to customize the analysis
# =============================================================================

# Domain and grid configuration
DOMAIN = np.array([[-2.5, -0.5], [2.5, 0.5]])  # State space bounds

# Subdivision depth for CMGDB (2^depth boxes per dimension)
# Using depth=16 gives 2^16=65536 boxes per dimension
# For uniform grid matching Python, set subdiv_init=subdiv_min=subdiv_max
SUBDIV = 16

# Convert subdivision depth to grid divisions for Python pipeline
# CMGDB subdivision depth d → 2^d boxes per dimension
PYTHON_DIVISIONS = np.array([2**SUBDIV, 2**SUBDIV], dtype=int)

# Dynamics configuration
EPSILON_BLOAT = 0.01  # Bloating factor for rigorous outer approximation

# Henon map parameters
# a = 1.4, b = 0.3 (default parameters)

# =============================================================================
# Analysis Functions
# =============================================================================

def compute_statistics(box_map, morse_graph, computation_time, pipeline_name):
    """Compute statistics for a pipeline result."""
    num_boxes = len(box_map.nodes())
    num_edges = len(box_map.edges())
    num_morse_sets = len(morse_graph.nodes())
    
    # Find largest Morse set
    largest_morse_set_size = 0
    if num_morse_sets > 0:
        largest_morse_set_size = max(len(ms) for ms in morse_graph.nodes())
    
    # Count attractors
    attractors = identify_attractors(morse_graph)
    num_attractors = len(attractors)
    
    return {
        'pipeline': pipeline_name,
        'computation_time': computation_time,
        'num_boxes': num_boxes,
        'num_edges': num_edges,
        'num_morse_sets': num_morse_sets,
        'largest_morse_set_size': largest_morse_set_size,
        'num_attractors': num_attractors
    }

def print_statistics(stats):
    """Print statistics in a formatted way."""
    print(f"\n{stats['pipeline']} Pipeline Statistics:")
    print(f"  Computation time: {stats['computation_time']:.3f} seconds")
    print(f"  Number of boxes: {stats['num_boxes']:,}")
    print(f"  Number of edges: {stats['num_edges']:,}")
    print(f"  Number of Morse sets: {stats['num_morse_sets']}")
    print(f"  Largest Morse set size: {stats['largest_morse_set_size']:,} boxes")
    print(f"  Number of attractors: {stats['num_attractors']}")

def save_comparison_stats(stats_cmgdb, stats_python, output_path):
    """Save comparison statistics to a text file."""
    stats_file = os.path.join(output_path, "comparison_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("MorseGraph Pipeline Comparison\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CMGDB Pipeline:\n")
        f.write(f"  Computation time: {stats_cmgdb['computation_time']:.3f} seconds\n")
        f.write(f"  Number of boxes: {stats_cmgdb['num_boxes']:,}\n")
        f.write(f"  Number of edges: {stats_cmgdb['num_edges']:,}\n")
        f.write(f"  Number of Morse sets: {stats_cmgdb['num_morse_sets']}\n")
        f.write(f"  Largest Morse set size: {stats_cmgdb['largest_morse_set_size']:,} boxes\n")
        f.write(f"  Number of attractors: {stats_cmgdb['num_attractors']}\n\n")
        
        f.write("Python Pipeline:\n")
        f.write(f"  Computation time: {stats_python['computation_time']:.3f} seconds\n")
        f.write(f"  Number of boxes: {stats_python['num_boxes']:,}\n")
        f.write(f"  Number of edges: {stats_python['num_edges']:,}\n")
        f.write(f"  Number of Morse sets: {stats_python['num_morse_sets']}\n")
        f.write(f"  Largest Morse set size: {stats_python['largest_morse_set_size']:,} boxes\n")
        f.write(f"  Number of attractors: {stats_python['num_attractors']}\n\n")
        
        f.write("Comparison:\n")
        speedup = stats_python['computation_time'] / stats_cmgdb['computation_time']
        f.write(f"  Speedup (Python/CMGDB): {speedup:.2f}x\n")
        f.write(f"  Box count ratio (Python/CMGDB): {stats_python['num_boxes'] / stats_cmgdb['num_boxes']:.2f}\n")
        f.write(f"  Edge count ratio (Python/CMGDB): {stats_python['num_edges'] / stats_cmgdb['num_edges']:.2f}\n")
        f.write(f"  Morse set count ratio (Python/CMGDB): {stats_python['num_morse_sets'] / stats_cmgdb['num_morse_sets']:.2f}\n")
    
    print(f"\nSaved comparison statistics to: {stats_file}")


def plot_cmgdb_morse_sets_raw(morse_graph_obj, morse_set_boxes_dict, ax, domain, title="CMGDB Morse Sets"):
    """
    Plot CMGDB Morse sets directly from box data, mimicking CMGDB.PlotMorseSets approach.
    
    This plots the actual CMGDB boxes as rectangles without converting to grid indices.
    Each box is drawn at its true coordinates with size matching the box dimensions.
    
    :param morse_graph_obj: CMGDB MorseGraph object
    :param morse_set_boxes_dict: Dictionary mapping Morse set index to list of boxes
    :param ax: Matplotlib axis to plot on
    :param domain: numpy array of shape (2, D) with domain bounds
    :param title: Title for the plot
    """
    num_morse_sets = morse_graph_obj.num_vertices()
    
    # Use tab10 colormap for consistency with Python plotting
    cmap = plt.colormaps.get_cmap('tab10')
    
    # Set axis limits
    ax.set_xlim(domain[0, 0], domain[1, 0])
    ax.set_ylim(domain[0, 1], domain[1, 1])
    
    # Get figure and axis dimensions for proper sizing
    fig = ax.get_figure()
    x_axis_width = domain[1, 0] - domain[0, 0]
    y_axis_height = domain[1, 1] - domain[0, 1]
    
    # Compute scale factors for marker sizing (mimics CMGDB approach)
    # This ensures rectangles are sized correctly in data coordinates
    s0_x = (ax.get_window_extent().width / x_axis_width) * (72.0 / fig.dpi)
    s0_y = (ax.get_window_extent().height / y_axis_height) * (72.0 / fig.dpi)
    
    # Plot each Morse set
    for v in range(num_morse_sets):
        boxes = morse_set_boxes_dict[v]
        color = cmap(v / max(num_morse_sets, 10))
        
        X = []  # x-coordinates of box centers
        Y = []  # y-coordinates of box centers
        S = []  # sizes for scatter plot
        
        for cmgdb_box in boxes:
            # CMGDB box format: [x_min, y_min, ..., x_max, y_max, ...]
            dim = len(cmgdb_box) // 2
            
            # Extract x and y coordinates (assuming 2D for now)
            x_min, y_min = cmgdb_box[0], cmgdb_box[1]
            x_max, y_max = cmgdb_box[dim], cmgdb_box[dim + 1]
            
            # Center point
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            # Box dimensions
            width = x_max - x_min
            height = y_max - y_min
            
            # Compute marker size in scatter plot units
            # Use the maximum of x and y sizes (square markers)
            s_x = (s0_x * width) ** 2
            s_y = (s0_y * height) ** 2
            
            X.append(x_center)
            Y.append(y_center)
            S.append(max(s_x, s_y))
        
        # Plot all boxes for this Morse set at once
        if X:
            ax.scatter(X, Y, s=S, marker='s', c=[color], alpha=0.8, edgecolors='none')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')

# =============================================================================
# Main Analysis
# =============================================================================

def main(skip_cmgdb=False, skip_python=False):
    print("MorseGraph Example 1: Map Dynamics Comparison")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  Domain: {DOMAIN}")
    print(f"  Subdivision depth: {SUBDIV} (→ {2**SUBDIV} boxes per dimension)")
    print(f"  Python grid divisions: {PYTHON_DIVISIONS}")
    print(f"  Epsilon bloat: {EPSILON_BLOAT}")
    if skip_cmgdb:
        print(f"  [SKIPPING CMGDB pipeline]")
    if skip_python:
        print(f"  [SKIPPING Python pipeline]")
    
    # Create the dynamics object (same for both pipelines)
    dynamics = BoxMapFunction(henon_map, epsilon=EPSILON_BLOAT)
    
    # Initialize variables for results
    cmgdb_time = None
    python_time = None
    stats_cmgdb = None
    stats_python = None
    morse_graph_cmgdb = None
    morse_graph_cmgdb_obj = None
    morse_graph_python = None
    cmgdb_morse_set_boxes = None
    map_graph_cmgdb = None
    box_map_cmgdb = None
    box_map_python = None
    basins_cmgdb = None
    basins_python = None
    grid_python = None
    
    # ========================================================================
    # CMGDB Pipeline
    # ========================================================================
    if not skip_cmgdb:
        print("\n" + "=" * 50)
        print("CMGDB Pipeline (C++ Backend)")
        print("=" * 50)
        
        # Import CMGDB directly
        try:
            import CMGDB
        except ImportError:
            raise ImportError("CMGDB is required. Please install it via: pip install -e ./cmgdb")
        
        # Define box map function for CMGDB (following notebook pattern)
        # CMGDB expects rect as [x_min, y_min, ..., x_max, y_max, ...]
        # and returns [y0_min, y1_min, ..., y0_max, y1_max, ...]
        def F(rect):
            """Box map function for CMGDB - maps a rectangle to its image rectangle."""
            dim = len(rect) // 2
            # Convert CMGDB format [x_min, y_min, ..., x_max, y_max, ...] to numpy format
            box = np.array([rect[:dim], rect[dim:]])
            # Apply dynamics (returns [lower, upper] bounds)
            image_box = dynamics(box)
            # Return as flat list [min_x, ..., max_x, ...] matching CMGDB format
            return list(image_box[0]) + list(image_box[1])
        
        # Convert domain bounds to CMGDB format
        lower_bounds = DOMAIN[0].tolist()
        upper_bounds = DOMAIN[1].tolist()
        
        print(f"\nComputing Morse graph with CMGDB...")
        print(f"  Subdivision: init={SUBDIV}, min={SUBDIV}, max={SUBDIV}")
        print(f"  Domain: {lower_bounds} to {upper_bounds}")
        print(f"  Note: Using subdiv_init=subdiv_min=subdiv_max for uniform grid (matching Python)")
        
        # Create CMGDB model (similar to notebook, but with explicit subdiv_init for uniform grid)
        # Set subdiv_init=subdiv_min=subdiv_max to get uniform grid (no adaptive refinement)
        model = CMGDB.Model(
            SUBDIV,  # subdiv_min
            SUBDIV,  # subdiv_max
            SUBDIV,  # subdiv_init - start at same depth for uniform grid
            1000000, # subdiv_limit
            lower_bounds,
            upper_bounds,
            F
        )
        
        # Compute Morse graph (returns morse_graph and map_graph)
        start_time = time.time()
        morse_graph_cmgdb_obj, map_graph_cmgdb = CMGDB.ComputeMorseGraph(model)
        cmgdb_time = time.time() - start_time
        
        print(f"  Completed in {cmgdb_time:.3f} seconds")
        print(f"  Found {morse_graph_cmgdb_obj.num_vertices()} Morse sets")
        
        # Extract Morse set boxes
        cmgdb_morse_set_boxes = {}
        for v in range(morse_graph_cmgdb_obj.num_vertices()):
            boxes = morse_graph_cmgdb_obj.morse_set_boxes(v)
            cmgdb_morse_set_boxes[v] = boxes
            print(f"    Morse set {v+1}: {len(boxes)} boxes")

        # Build NetworkX Morse graph for graph structure visualization
        # Use vertex indices as node identifiers (simpler than grid mapping)
        morse_graph_cmgdb = nx.DiGraph()
        morse_graph_cmgdb.add_nodes_from(range(morse_graph_cmgdb_obj.num_vertices()))

        # Add edges from CMGDB Morse graph
        for v1 in range(morse_graph_cmgdb_obj.num_vertices()):
            for v2 in morse_graph_cmgdb_obj.adjacencies(v1):
                morse_graph_cmgdb.add_edge(v1, v2)

        # Assign colors to Morse sets
        num_sets = morse_graph_cmgdb_obj.num_vertices()
        if num_sets > 0:
            cmap = plt.colormaps.get_cmap('tab10')
            for v in range(num_sets):
                morse_graph_cmgdb.nodes[v]['color'] = cmap(v / max(num_sets, 10))

        print(f"  Created NetworkX Morse graph with {morse_graph_cmgdb_obj.num_vertices()} Morse sets")
        
        # Compute statistics
        num_boxes_cmgdb = map_graph_cmgdb.num_vertices()
        num_edges_cmgdb = 0  # Skip edge counting for large grids (too slow)
        
        stats_cmgdb = {
            'pipeline': 'CMGDB',
            'computation_time': cmgdb_time,
            'num_boxes': num_boxes_cmgdb,
            'num_edges': num_edges_cmgdb,
            'num_morse_sets': morse_graph_cmgdb_obj.num_vertices(),
            'largest_morse_set_size': max(len(boxes) for boxes in cmgdb_morse_set_boxes.values()) if cmgdb_morse_set_boxes else 0,
            'num_attractors': len(identify_attractors(morse_graph_cmgdb_obj))
        }
        print_statistics(stats_cmgdb)
        
        # Note: CMGDB doesn't store the full box map - it computes adjacencies on-demand
        # Converting to NetworkX for basin computation would be extremely slow for large grids
        print("\nNote: Skipping basin computation for CMGDB (MapGraph computes adjacencies on-demand)")
        basins_cmgdb = None
        box_map_cmgdb = None
    else:
        print("\n" + "=" * 50)
        print("CMGDB Pipeline - SKIPPED")
        print("=" * 50)
    
    # ========================================================================
    # Python Pipeline
    # ========================================================================
    if not skip_python:
        print("\n" + "=" * 50)
        print("Python Pipeline (Pure Python)")
        print("=" * 50)
        
        # Create grid for Python pipeline (same resolution)
        grid_python = UniformGrid(bounds=DOMAIN, divisions=PYTHON_DIVISIONS)
        model_python = Model(grid_python, dynamics)
        
        print(f"\nComputing BoxMap with Python backend...")
        print(f"  Grid divisions: {PYTHON_DIVISIONS}")
        print(f"  Total boxes: {len(grid_python.get_boxes()):,}")
        
        start_time = time.time()
        box_map_python = model_python._compute_box_map_python(n_jobs=-1)
        python_time = time.time() - start_time
        
        print(f"  Completed in {python_time:.3f} seconds")
        print(f"  BoxMap has {len(box_map_python.nodes())} nodes and {len(box_map_python.edges())} edges")
        
        print("\nComputing Morse graph from Python BoxMap...")
        morse_graph_python = compute_morse_graph(box_map_python)
        print(f"  Found {len(morse_graph_python.nodes())} Morse sets")
        
        for i, morse_set in enumerate(morse_graph_python.nodes()):
            print(f"    Morse set {i+1}: {len(morse_set)} boxes")
        
        print("\nComputing basins of attraction...")
        basins_python = compute_all_morse_set_basins(morse_graph_python, box_map_python)
        
        stats_python = compute_statistics(
            box_map_python, morse_graph_python, python_time, "Python"
        )
        print_statistics(stats_python)
    else:
        print("\n" + "=" * 50)
        print("Python Pipeline - SKIPPED")
        print("=" * 50)
    
    # ========================================================================
    # Comparison and Visualization
    # ========================================================================
    print("\n" + "=" * 50)
    print("Comparison and Visualization")
    print("=" * 50)
    
    # Save comparison statistics (if both pipelines ran)
    if not skip_cmgdb and not skip_python:
        save_comparison_stats(stats_cmgdb, stats_python, output_path)
    elif not skip_cmgdb:
        # Save CMGDB-only statistics
        stats_file = os.path.join(output_path, "cmgdb_stats.txt")
        with open(stats_file, 'w') as f:
            f.write("CMGDB Pipeline Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Computation time: {stats_cmgdb['computation_time']:.3f} seconds\n")
            f.write(f"Number of boxes: {stats_cmgdb['num_boxes']:,}\n")
            f.write(f"Number of edges: {stats_cmgdb['num_edges']:,}\n")
            f.write(f"Number of Morse sets: {stats_cmgdb['num_morse_sets']}\n")
            f.write(f"Largest Morse set size: {stats_cmgdb['largest_morse_set_size']:,} boxes\n")
            f.write(f"Number of attractors: {stats_cmgdb['num_attractors']}\n")
        print(f"\nSaved CMGDB statistics to: {stats_file}")
    elif not skip_python:
        # Save Python-only statistics
        stats_file = os.path.join(output_path, "python_stats.txt")
        with open(stats_file, 'w') as f:
            f.write("Python Pipeline Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Computation time: {stats_python['computation_time']:.3f} seconds\n")
            f.write(f"Number of boxes: {stats_python['num_boxes']:,}\n")
            f.write(f"Number of edges: {stats_python['num_edges']:,}\n")
            f.write(f"Number of Morse sets: {stats_python['num_morse_sets']}\n")
            f.write(f"Largest Morse set size: {stats_python['largest_morse_set_size']:,} boxes\n")
            f.write(f"Number of attractors: {stats_python['num_attractors']}\n")
        print(f"\nSaved Python statistics to: {stats_file}")
    
    # Note: CMGDB Morse sets are now mapped to Python grid indices
    # We can use the same plot_morse_sets function with the Python grid
    
    # Helper function to plot CMGDB basins (placeholder - basins not computed for CMGDB)
    def plot_cmgdb_basins(morse_graph_obj, morse_set_boxes_dict, basins, map_graph_obj, ax, title="Basins"):
        """Plot CMGDB basins of attraction (not available - CMGDB doesn't store full box map)."""
        ax.text(0.5, 0.5, 'Basins not computed for CMGDB\n(MapGraph computes adjacencies on-demand)',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlim(DOMAIN[0, 0], DOMAIN[1, 0])
        ax.set_ylim(DOMAIN[0, 1], DOMAIN[1, 1])
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
    
    # Create comparison plots
    print("\nGenerating plots...")
    
    if not skip_cmgdb and not skip_python:
        # Both pipelines: create comparison plots
        # 1. Morse Sets Comparison
        # Plot CMGDB Morse sets directly from box data, Python from grid
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        plot_cmgdb_morse_sets_raw(morse_graph_cmgdb_obj, cmgdb_morse_set_boxes, ax1, DOMAIN, "Morse Sets (CMGDB)")
        ax1.set_title("Morse Sets (CMGDB)", fontsize=14)
        
        plot_morse_sets(grid_python, morse_graph_python, ax=ax2)
        ax2.set_title("Morse Sets (Python)", fontsize=14)
        
        plt.tight_layout()
        morse_sets_path = os.path.join(output_path, "morse_sets_comparison.png")
        plt.savefig(morse_sets_path, dpi=150, bbox_inches='tight')
        print(f"  Saved Morse sets comparison to: {morse_sets_path}")
        plt.close()
        
        # 2. Basins Comparison (only if Python basins are available)
        if basins_python is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            plot_cmgdb_basins(morse_graph_cmgdb_obj, cmgdb_morse_set_boxes, basins_cmgdb, map_graph_cmgdb, ax1, "Basins of Attraction (CMGDB)")
            
            plot_basins_of_attraction(
                grid_python, basins_python, morse_graph=morse_graph_python,
                ax=ax2, show_outside=True
            )
            ax2.set_title("Basins of Attraction (Python)", fontsize=14)
            
            plt.tight_layout()
            basins_path = os.path.join(output_path, "basins_comparison.png")
            plt.savefig(basins_path, dpi=150, bbox_inches='tight')
            print(f"  Saved basins comparison to: {basins_path}")
            plt.close()
        
        # 3. Morse Graph Structure Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        plot_morse_graph(morse_graph_cmgdb, ax=ax1)
        ax1.set_title("Morse Graph Structure (CMGDB)", fontsize=14)
        
        plot_morse_graph(morse_graph_python, ax=ax2)
        ax2.set_title("Morse Graph Structure (Python)", fontsize=14)
        
        plt.tight_layout()
        graph_path = os.path.join(output_path, "morse_graph_comparison.png")
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  Saved Morse graph comparison to: {graph_path}")
        plt.close()
    
    # Individual plots for each pipeline
    if not skip_cmgdb:
        # CMGDB individual plots - plot raw boxes directly
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_cmgdb_morse_sets_raw(morse_graph_cmgdb_obj, cmgdb_morse_set_boxes, ax, DOMAIN, "Morse Sets (CMGDB)")
        plt.savefig(os.path.join(output_path, "cmgdb_morse_sets.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved CMGDB Morse sets plot")
        
        if basins_cmgdb is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_cmgdb_basins(morse_graph_cmgdb_obj, cmgdb_morse_set_boxes, basins_cmgdb, map_graph_cmgdb, ax, "Basins of Attraction (CMGDB)")
            plt.savefig(os.path.join(output_path, "cmgdb_basins.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved CMGDB basins plot")
        else:
            print(f"  Skipped CMGDB basins plot (not computed)")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_morse_graph(morse_graph_cmgdb, ax=ax)
        ax.set_title("Morse Graph Structure (CMGDB)")
        plt.savefig(os.path.join(output_path, "cmgdb_morse_graph.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved CMGDB Morse graph plot")
    
    if not skip_python:
        # Python individual plots
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_morse_sets(grid_python, morse_graph_python, ax=ax)
        ax.set_title("Morse Sets (Python)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.savefig(os.path.join(output_path, "python_morse_sets.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved Python Morse sets plot")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_basins_of_attraction(
            grid_python, basins_python, morse_graph=morse_graph_python,
            ax=ax, show_outside=True
        )
        ax.set_title("Basins of Attraction (Python)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.savefig(os.path.join(output_path, "python_basins.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved Python basins plot")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_morse_graph(morse_graph_python, ax=ax)
        ax.set_title("Morse Graph Structure (Python)")
        plt.savefig(os.path.join(output_path, "python_morse_graph.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved Python Morse graph plot")
    
    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print("=" * 50)
    print(f"\nAll outputs saved to: {output_path}")
    print(f"\nSummary:")
    if not skip_cmgdb:
        print(f"  CMGDB time: {cmgdb_time:.3f}s")
    if not skip_python:
        print(f"  Python time: {python_time:.3f}s")
    if not skip_cmgdb and not skip_python:
        print(f"  Speedup: {python_time/cmgdb_time:.2f}x (Python/CMGDB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare CMGDB and Python pipelines for Morse graph computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both pipelines (default)
  python 1_map_dynamics.py
  
  # Run only CMGDB
  python 1_map_dynamics.py --skip-python
  
  # Run only Python
  python 1_map_dynamics.py --skip-cmgdb
        """
    )
    parser.add_argument(
        '--skip-cmgdb',
        action='store_true',
        help='Skip CMGDB pipeline computation'
    )
    parser.add_argument(
        '--skip-python',
        action='store_true',
        help='Skip Python pipeline computation'
    )
    
    args = parser.parse_args()
    
    # Validate that at least one pipeline is enabled
    if args.skip_cmgdb and args.skip_python:
        parser.error("Cannot skip both CMGDB and Python pipelines. At least one must run.")
    
    main(skip_cmgdb=args.skip_cmgdb, skip_python=args.skip_python)