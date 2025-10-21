#!/usr/bin/env python3
"""
Test CMGDB uniform grid for Ives model to find minimum resolution
needed to separate equilibrium from period-7 orbit.

CMGDB uniform grid uses: CMGDB.Model(subdiv, subdiv, lower, upper, F)
where subdiv creates 2^subdiv boxes per dimension.

For reference:
- subdiv=5  → 2^5 = 32 boxes/dim   → 32^3 = 32,768 total boxes
- subdiv=6  → 2^6 = 64 boxes/dim   → 64^3 = 262,144 total boxes
- subdiv=7  → 2^7 = 128 boxes/dim  → 128^3 = 2,097,152 total boxes
"""

import numpy as np
import sys
import os
import time
from itertools import product

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


def test_uniform_grid(subdiv, equilibrium, period_7_orbit, verbose=True):
    """Test CMGDB uniform grid at a given subdivision level."""

    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing subdiv = {subdiv}")
        print(f"{'='*80}")

    # Domain bounds
    domain_lower = [-1, -4, -1]
    domain_upper = [2, 1, 1]

    # CORRECTED: subdiv means 2^subdiv TOTAL boxes
    total_boxes = 2**subdiv
    boxes_per_dim = int(round(total_boxes**(1/3)))  # Cube root for 3D
    domain_width = np.array(domain_upper) - np.array(domain_lower)
    box_size = domain_width / boxes_per_dim

    if verbose:
        print(f"  Total boxes: {total_boxes:,}")
        print(f"  Boxes per dimension: {boxes_per_dim}")
        print(f"  Box size: {box_size}")

    # Define BoxMap function for CMGDB
    def F(rect):
        """CMGDB BoxMap function with bloating."""
        dim = 3
        # Evaluate at all corners
        corners = list(product(*[(rect[d], rect[d+dim]) for d in range(dim)]))
        corners = [list(c) for c in corners]
        corners_next = np.array([ives_model_log(np.array(c)) for c in corners])

        # Bloating: add box width
        padding_size = [(rect[d + dim] - rect[d]) for d in range(dim)]

        Y_l = [corners_next[:, d].min() - padding_size[d] for d in range(dim)]
        Y_u = [corners_next[:, d].max() + padding_size[d] for d in range(dim)]

        return Y_l + Y_u

    # Build CMGDB uniform grid model
    # Constructor: Model(subdiv, subdiv, lower, upper, F) for uniform grid
    if verbose:
        print(f"  Building CMGDB model...")

    model = CMGDB.Model(
        subdiv,      # Same value for min and max = uniform grid
        subdiv,
        domain_lower,
        domain_upper,
        F
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
        'total_boxes': total_boxes,
        'boxes_per_dim': boxes_per_dim,
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
    print("CMGDB UNIFORM GRID ANALYSIS - IVES MODEL")
    print("="*80)
    print("\nNOTE: subdiv=N means 2^N TOTAL boxes, not per dimension!")
    print("For 3D: boxes_per_dim ≈ (2^N)^(1/3) = 2^(N/3)")
    print(f"\nGoal: Find minimum subdivision to separate equilibrium from period-7 orbit")
    print(f"\nEquilibrium: [{equilibrium[0]:.4f}, {equilibrium[1]:.4f}, {equilibrium[2]:.4f}]")
    print(f"Period-7 orbit: 7 points")
    print(f"\nCritical separation (Midge dimension): 0.0134")
    print(f"Estimated subdiv needed: ~24")

    # Test at appropriate resolutions now that we understand subdiv semantics
    # subdiv=23 → 8.4M boxes, ~203 boxes/dim
    # subdiv=24 → 16.8M boxes, ~256 boxes/dim  
    # subdiv=25 → 33.5M boxes, ~323 boxes/dim
    test_subdivs = [23, 24, 25]

    results = []

    for subdiv in test_subdivs:
        result = test_uniform_grid(subdiv, equilibrium, period_7_orbit, verbose=True)
        results.append(result)

        # Stop if we found separation
        if result['separated']:
            print(f"\n  ✓ Found separation! Testing one more to confirm...")
            # Test next one to confirm
            if subdiv < max(test_subdivs):
                continue
            else:
                break

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Subdiv':<10} {'Total Boxes':<15} {'Boxes/Dim':<12} {'Morse Sets':<12} {'Time (s)':<10} {'Separated?':<15}")
    print("-" * 90)

    for r in results:
        sep_str = "✓ Yes" if r['separated'] else ("✗ No" if r['separated'] is False else "? Unknown")
        print(f"{r['subdiv']:<10} {r['total_boxes']:<15,} {r['boxes_per_dim']:<12} "
              f"{r['num_morse_sets']:<12} {r['computation_time']:<10.2f} {sep_str:<15}")

    # Find minimum resolution that separates
    separated_results = [r for r in results if r['separated']]
    if separated_results:
        min_sep = min(separated_results, key=lambda r: r['subdiv'])
        print(f"\n{'='*80}")
        print("✓ SEPARATION ACHIEVED!")
        print(f"{'='*80}")
        print(f"  Minimum subdivision: {min_sep['subdiv']}")
        print(f"  Total boxes: {min_sep['total_boxes']:,}")
        print(f"  Boxes per dimension: {min_sep['boxes_per_dim']}")
        print(f"  Box size: {min_sep['box_size']}")
        print(f"  Computation time: {min_sep['computation_time']:.2f}s")
        print(f"  Number of Morse sets: {min_sep['num_morse_sets']}")
        print(f"\n  Recommendation: Use subdiv_min={min_sep['subdiv']} in examples/7_ives_model.py")
    else:
        print(f"\n✗ No configuration achieved separation! Need higher subdivision.")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
