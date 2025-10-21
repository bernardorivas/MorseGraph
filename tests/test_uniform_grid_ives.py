#!/usr/bin/env python3
"""
Test UniformGrid-based Morse graph computation for Ives model.

This script explores different grid resolutions to determine what's needed
to separate the equilibrium from the period-7 orbit.

Key differences from CMGDB:
- UniformGrid uses `divisions` = number of boxes per dimension (e.g., [50, 50, 50])
- CMGDB uses `subdiv` = tree depth where depth d creates 2^d boxes per dimension

So divisions=50 ≈ subdiv=log2(50)≈5.6, or subdiv=6 creates 2^6=64 boxes per dimension.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapFunction
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph
from MorseGraph.systems import ives_model_log


def find_box_containing_point(grid, point):
    """Find which box contains a given point."""
    boxes = grid.get_boxes()

    for i, box in enumerate(boxes):
        lower = box[0]
        upper = box[1]
        if np.all(point >= lower) and np.all(point <= upper):
            return i

    return None


def analyze_morse_sets(morse_graph, grid, equilibrium, period_7_orbit):
    """Analyze which Morse sets contain the equilibrium and orbit."""

    # Find boxes containing equilibrium
    eq_box_idx = find_box_containing_point(grid, equilibrium)

    # Find boxes containing period-7 orbit points
    orbit_box_indices = []
    for pt in period_7_orbit:
        idx = find_box_containing_point(grid, pt)
        if idx is not None:
            orbit_box_indices.append(idx)

    print(f"\n  Equilibrium in box: {eq_box_idx}")
    print(f"  Period-7 orbit in boxes: {orbit_box_indices}")

    # Find which Morse sets contain these boxes
    eq_morse_set = None
    orbit_morse_sets = []

    for morse_set in morse_graph.nodes():
        if eq_box_idx in morse_set:
            eq_morse_set = morse_set

        for orbit_idx in orbit_box_indices:
            if orbit_idx in morse_set:
                orbit_morse_sets.append((orbit_idx, morse_set))

    print(f"\n  Equilibrium Morse set: {len(eq_morse_set) if eq_morse_set else 0} boxes")
    print(f"  Orbit points in Morse sets: {len(set([ms for _, ms in orbit_morse_sets]))} distinct set(s)")

    # Check separation
    orbit_morse_set_collection = set([ms for _, ms in orbit_morse_sets])

    if eq_morse_set and orbit_morse_set_collection:
        if eq_morse_set in orbit_morse_set_collection:
            print(f"\n  ✗ SAME Morse set - equilibrium and orbit NOT separated")
            return False
        else:
            print(f"\n  ✓ DIFFERENT Morse sets - equilibrium and orbit ARE separated!")
            return True
    else:
        print(f"\n  ? Could not determine separation (some points not in any Morse set)")
        return None


def test_grid_resolution(divisions, equilibrium, period_7_orbit, verbose=True):
    """Test a single grid resolution."""

    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing divisions = {divisions}")
        print(f"{'='*80}")

    # Domain bounds
    domain_bounds = np.array([[-1, -4, -1], [2, 1, 1]])

    # Create grid
    grid = UniformGrid(domain_bounds, divisions)
    total_boxes = np.prod(divisions)

    if verbose:
        print(f"  Total boxes: {total_boxes:,}")
        print(f"  Box size: {grid.box_size}")

    # Create dynamics (BoxMapFunction with bloating)
    dynamics = BoxMapFunction(
        map_f=ives_model_log,
        epsilon=0.01,  # Small bloating for outer approximation
        evaluation_method="corners"
    )

    # Create model
    model = Model(grid, dynamics)

    # Compute BoxMap
    if verbose:
        print(f"  Computing BoxMap...")
    box_map = model.compute_box_map(n_jobs=-1)

    num_nodes = box_map.number_of_nodes()
    num_edges = box_map.number_of_edges()

    if verbose:
        print(f"  BoxMap: {num_nodes} nodes, {num_edges} edges")

    # Compute Morse graph
    if verbose:
        print(f"  Computing Morse graph...")
    morse_graph = compute_morse_graph(box_map)

    num_morse_sets = morse_graph.number_of_nodes()
    num_mg_edges = morse_graph.number_of_edges()

    if verbose:
        print(f"  Morse graph: {num_morse_sets} Morse sets, {num_mg_edges} edges")

    # Analyze separation
    if verbose:
        print(f"\n  Analyzing separation...")

    separated = analyze_morse_sets(morse_graph, grid, equilibrium, period_7_orbit)

    return {
        'divisions': divisions,
        'total_boxes': total_boxes,
        'box_size': grid.box_size.copy(),
        'num_morse_sets': num_morse_sets,
        'num_mg_edges': num_mg_edges,
        'separated': separated
    }


def main():
    # Equilibrium and period-7 orbit (from basin analysis)
    equilibrium = np.array([0.7923, 0.2097, 0.3773])
    period_7_orbit = np.array([
        [0.3645, -2.1168, 0.1605],
        [-0.2800, -1.2874, -0.4924],
        [-0.5147, -0.4412, -0.8516],
        [-0.1614, 0.3895, -0.3733],
        [0.3307, 0.7944, 0.4252],
        [0.8057, 0.5918, 0.8763],
        [1.0425, -2.9998, 0.8328]
    ])

    print("="*80)
    print("UNIFORM GRID MORSE GRAPH ANALYSIS - IVES MODEL")
    print("="*80)
    print(f"\nGoal: Find minimum grid resolution to separate equilibrium from period-7 orbit")
    print(f"\nEquilibrium: {equilibrium}")
    print(f"Period-7 orbit: 7 points")

    # Test different resolutions
    # Start coarse, then refine
    test_configs = [
        np.array([10, 10, 10]),   # Very coarse
        np.array([20, 20, 20]),   # Coarse
        np.array([30, 30, 30]),   # Medium
        np.array([40, 40, 40]),   # Fine
        np.array([50, 50, 50]),   # Very fine
        # Anisotropic grids (finer in critical dimensions)
        np.array([50, 100, 50]),  # Finer in y (Algae) dimension
        np.array([100, 100, 50]), # Finer in x,y
    ]

    results = []

    for divisions in test_configs:
        result = test_grid_resolution(divisions, equilibrium, period_7_orbit, verbose=True)
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Divisions':<20} {'Total Boxes':<15} {'Morse Sets':<15} {'Separated?':<15}")
    print("-" * 70)

    for r in results:
        div_str = f"{r['divisions'][0]}×{r['divisions'][1]}×{r['divisions'][2]}"
        sep_str = "✓ Yes" if r['separated'] else ("✗ No" if r['separated'] is False else "? Unknown")
        print(f"{div_str:<20} {r['total_boxes']:<15,} {r['num_morse_sets']:<15} {sep_str:<15}")

    # Find minimum resolution that separates
    separated_results = [r for r in results if r['separated']]
    if separated_results:
        min_sep = min(separated_results, key=lambda r: r['total_boxes'])
        print(f"\nMinimum resolution for separation: {min_sep['divisions']}")
        print(f"  (Total boxes: {min_sep['total_boxes']:,})")
    else:
        print(f"\nNo configuration achieved separation! Need higher resolution.")


if __name__ == "__main__":
    main()
