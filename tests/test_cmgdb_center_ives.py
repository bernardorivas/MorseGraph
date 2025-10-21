#!/usr/bin/env python3
"""
Test CMGDB uniform grid with CENTER evaluation for Ives model.

Using center evaluation instead of corners should create a tighter BoxMap
with less bloating, which might help separate equilibrium from period-7 orbit.
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import CMGDB
from MorseGraph.systems import ives_model_log


def find_box_containing_point(morse_graph, point):
    """Find which Morse set contains a given point."""
    for morse_set_idx in range(morse_graph.num_vertices()):
        boxes = morse_graph.morse_set_boxes(morse_set_idx)
        dim = len(boxes[0]) // 2

        for box in boxes:
            lower = box[:dim]
            upper = box[dim:]

            if all(point[d] >= lower[d] and point[d] <= upper[d] for d in range(dim)):
                return morse_set_idx

    return None


def test_uniform_grid_center(subdiv, equilibrium, period_7_orbit, bloating_factor=1.0, verbose=True):
    """Test CMGDB uniform grid using CENTER evaluation."""

    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing subdiv = {subdiv} (CENTER evaluation, bloating = {bloating_factor}x)")
        print(f"{'='*80}")

    # Domain bounds
    domain_lower = [-1, -4, -1]
    domain_upper = [2, 1, 1]

    boxes_per_dim = 2**subdiv
    total_boxes = boxes_per_dim**3
    domain_width = np.array(domain_upper) - np.array(domain_lower)
    box_size = domain_width / boxes_per_dim

    if verbose:
        print(f"  Boxes per dimension: {boxes_per_dim}")
        print(f"  Total boxes: {total_boxes:,}")
        print(f"  Box size: {box_size}")

    # Define BoxMap function using CENTER evaluation
    def F_center(rect):
        """CMGDB BoxMap function with CENTER evaluation and bloating."""
        dim = 3

        # Evaluate at center only
        center = [(rect[d] + rect[d+dim]) / 2.0 for d in range(dim)]
        image = ives_model_log(np.array(center))

        # Bloating: add bloating_factor * box width
        padding_size = [bloating_factor * (rect[d + dim] - rect[d]) for d in range(dim)]

        Y_l = [image[d] - padding_size[d] for d in range(dim)]
        Y_u = [image[d] + padding_size[d] for d in range(dim)]

        return Y_l + Y_u

    # Build CMGDB uniform grid model
    if verbose:
        print(f"  Building CMGDB model...")

    model = CMGDB.Model(
        subdiv,
        subdiv,
        domain_lower,
        domain_upper,
        F_center
    )

    # Compute Morse graph
    if verbose:
        print(f"  Computing Morse graph...")

    start_time = time.time()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    comp_time = time.time() - start_time

    num_morse_sets = morse_graph.num_vertices()
    num_edges = len(morse_graph.edges())

    if verbose:
        print(f"  Completed in {comp_time:.2f}s")
        print(f"  Morse graph: {num_morse_sets} Morse sets, {num_edges} edges")

    # Find which Morse sets contain equilibrium and orbit
    if verbose:
        print(f"\n  Analyzing separation...")

    eq_morse_set = find_box_containing_point(morse_graph, equilibrium)

    orbit_morse_sets = []
    for i, pt in enumerate(period_7_orbit):
        ms = find_box_containing_point(morse_graph, pt)
        orbit_morse_sets.append(ms)
        if verbose:
            print(f"    Orbit point {i+1}: Morse set {ms}")

    if verbose:
        print(f"    Equilibrium: Morse set {eq_morse_set}")

    # Check separation
    unique_orbit_sets = set([ms for ms in orbit_morse_sets if ms is not None])

    if eq_morse_set is not None and unique_orbit_sets:
        if eq_morse_set in unique_orbit_sets:
            separated = False
            if verbose:
                print(f"\n  ✗ SAME Morse set - equilibrium and orbit NOT separated")
        else:
            separated = True
            if verbose:
                print(f"\n  ✓ DIFFERENT Morse sets - equilibrium and orbit ARE separated!")
                print(f"    Equilibrium in set {eq_morse_set}")
                print(f"    Orbit in set(s): {sorted(unique_orbit_sets)}")
    else:
        separated = None
        if verbose:
            print(f"\n  ? Could not determine (some points not in any Morse set)")

    return {
        'subdiv': subdiv,
        'bloating_factor': bloating_factor,
        'boxes_per_dim': boxes_per_dim,
        'total_boxes': total_boxes,
        'box_size': box_size.copy(),
        'num_morse_sets': num_morse_sets,
        'num_edges': num_edges,
        'computation_time': comp_time,
        'separated': separated,
        'eq_morse_set': eq_morse_set,
        'orbit_morse_sets': unique_orbit_sets
    }


def main():
    # Equilibrium and period-7 orbit
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
    print("CMGDB UNIFORM GRID - CENTER EVALUATION - IVES MODEL")
    print("="*80)
    print(f"\nGoal: Separate equilibrium from period-7 orbit using CENTER evaluation")
    print(f"\nEquilibrium: [{equilibrium[0]:.4f}, {equilibrium[1]:.4f}, {equilibrium[2]:.4f}]")
    print(f"Period-7 orbit: 7 points")

    # Test different subdivisions with center evaluation
    # Try different bloating factors too
    test_configs = [
        # (subdiv, bloating_factor)
        (8, 1.0),    # Standard bloating
        (9, 1.0),
        (10, 1.0),
        (11, 1.0),
        (10, 0.5),   # Reduced bloating
        (11, 0.5),
        (12, 0.5),
        (10, 0.1),   # Minimal bloating
        (11, 0.1),
    ]

    results = []

    for subdiv, bloat in test_configs:
        result = test_uniform_grid_center(subdiv, equilibrium, period_7_orbit,
                                         bloating_factor=bloat, verbose=True)
        results.append(result)

        # Stop early if we found separation
        if result['separated']:
            print(f"\n  ✓ Found separation! Testing one more to confirm...")
            break

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Subdiv':<10} {'Bloat':<8} {'Boxes/Dim':<12} {'Total Boxes':<15} "
          f"{'Morse Sets':<12} {'Time (s)':<10} {'Separated?':<15}")
    print("-" * 100)

    for r in results:
        sep_str = "✓ Yes" if r['separated'] else ("✗ No" if r['separated'] is False else "? Unknown")
        print(f"{r['subdiv']:<10} {r['bloating_factor']:<8.1f} {r['boxes_per_dim']:<12} "
              f"{r['total_boxes']:<15,} {r['num_morse_sets']:<12} "
              f"{r['computation_time']:<10.2f} {sep_str:<15}")

    # Find minimum resolution that separates
    separated_results = [r for r in results if r['separated']]
    if separated_results:
        min_sep = min(separated_results, key=lambda r: (r['subdiv'], r['bloating_factor']))
        print(f"\n✓ Separation achieved!")
        print(f"  Subdivision: {min_sep['subdiv']}")
        print(f"  Bloating factor: {min_sep['bloating_factor']}")
        print(f"  Boxes per dimension: {min_sep['boxes_per_dim']}")
        print(f"  Total boxes: {min_sep['total_boxes']:,}")
        print(f"  Box size: {min_sep['box_size']}")
        print(f"  Computation time: {min_sep['computation_time']:.2f}s")
    else:
        print(f"\n✗ No configuration achieved separation with center evaluation.")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
