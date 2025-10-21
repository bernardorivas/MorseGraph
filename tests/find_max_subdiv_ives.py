#!/usr/bin/env python3
"""
Find the maximum subdivision where both attractors are captured in Morse sets.

Goal: Find the highest subdiv where:
- Equilibrium is in a Morse set
- ALL 7 period-7 orbit points are in Morse sets
- Ideally in separate Morse sets
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
        if not boxes:
            continue
        dim = len(boxes[0]) // 2

        for box in boxes:
            lower = box[:dim]
            upper = box[dim:]

            if all(point[d] >= lower[d] and point[d] <= upper[d] for d in range(dim)):
                return morse_set_idx

    return None


def test_subdiv(subdiv):
    """Quick test: are both attractors captured?"""

    # Equilibrium and period-7 orbit
    equilibrium = np.array([0.792107, 0.209010, 0.376449])
    period_7_orbit = np.array([
        [0.3645, -2.1168, 0.1605],
        [-0.2800, -1.2874, -0.4924],
        [-0.5147, -0.4412, -0.8516],
        [-0.1614, 0.3895, -0.3733],
        [0.3307, 0.7944, 0.4252],
        [0.8057, 0.5918, 0.8763],
        [1.0425, -2.9998, 0.8328]
    ])

    # Domain
    domain_lower = [-1, -4, -1]
    domain_upper = [2, 1, 1]

    # BoxMap with corners, no padding
    def F(rect):
        dim = 3
        corners = list(product(*[(rect[d], rect[d+dim]) for d in range(dim)]))
        corners = [list(c) for c in corners]
        corners_next = np.array([ives_model_log(np.array(c)) for c in corners])
        Y_l = [corners_next[:, d].min() for d in range(dim)]
        Y_u = [corners_next[:, d].max() for d in range(dim)]
        return Y_l + Y_u

    # Build model
    model = CMGDB.Model(subdiv, subdiv, domain_lower, domain_upper, F)

    # Compute
    start = time.time()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    comp_time = time.time() - start

    # Check attractor coverage
    eq_set = find_box_containing_point(morse_graph, equilibrium)

    orbit_sets = []
    for pt in period_7_orbit:
        ms = find_box_containing_point(morse_graph, pt)
        orbit_sets.append(ms)

    orbit_sets_unique = set([ms for ms in orbit_sets if ms is not None])
    n_orbit_captured = sum(1 for ms in orbit_sets if ms is not None)

    eq_captured = eq_set is not None
    all_orbit_captured = n_orbit_captured == 7
    separated = eq_captured and all_orbit_captured and (eq_set not in orbit_sets_unique)

    return {
        'subdiv': subdiv,
        'total_boxes': 2**subdiv,
        'boxes_per_dim': int(round((2**subdiv)**(1/3))),
        'num_morse_sets': morse_graph.num_vertices(),
        'comp_time': comp_time,
        'eq_captured': eq_captured,
        'eq_set': eq_set,
        'orbit_captured': n_orbit_captured,
        'orbit_sets': orbit_sets_unique,
        'all_orbit_captured': all_orbit_captured,
        'separated': separated,
    }


def main():
    print("="*80)
    print("FINDING MAXIMUM SUBDIVISION FOR IVES MODEL")
    print("="*80)
    print("\nSearching for highest subdiv where both attractors are captured...\n")

    # Test range from 24 to 42
    test_subdivs = [24, 27, 30, 33, 36, 39, 42]

    results = []

    for subdiv in test_subdivs:
        print(f"\nTesting subdiv={subdiv}...", end=" ", flush=True)
        result = test_subdiv(subdiv)
        results.append(result)

        # Quick status
        if result['all_orbit_captured'] and result['eq_captured']:
            print(f"✓ Both captured ({result['comp_time']:.1f}s, {result['num_morse_sets']} sets)")
        elif result['eq_captured']:
            print(f"⚠ Only equilibrium ({result['orbit_captured']}/7 orbit points)")
        else:
            print(f"✗ Failed to capture attractors")

    # Summary table
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"\n{'Subdiv':<8} {'Boxes/Dim':<12} {'Morse Sets':<12} {'Time(s)':<10} "
          f"{'Eq?':<6} {'Orbit':<8} {'Sep?':<6}")
    print("-" * 80)

    for r in results:
        eq_str = "✓" if r['eq_captured'] else "✗"
        orbit_str = f"{r['orbit_captured']}/7"
        sep_str = "✓" if r['separated'] else "✗"
        print(f"{r['subdiv']:<8} {r['boxes_per_dim']:<12} {r['num_morse_sets']:<12} "
              f"{r['comp_time']:<10.1f} {eq_str:<6} {orbit_str:<8} {sep_str:<6}")

    # Find maximum usable subdivision
    valid_results = [r for r in results if r['all_orbit_captured'] and r['eq_captured']]

    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    if valid_results:
        max_subdiv = max(valid_results, key=lambda r: r['subdiv'])
        print(f"\n✓ Maximum usable subdivision: {max_subdiv['subdiv']}")
        print(f"  Total boxes: {max_subdiv['total_boxes']:,}")
        print(f"  Boxes per dimension: {max_subdiv['boxes_per_dim']}")
        print(f"  Morse sets: {max_subdiv['num_morse_sets']}")
        print(f"  Computation time: {max_subdiv['comp_time']:.1f}s")
        print(f"  Equilibrium in set: {max_subdiv['eq_set']}")
        print(f"  Period-7 in set(s): {sorted(max_subdiv['orbit_sets'])}")
        print(f"  Separated: {'Yes' if max_subdiv['separated'] else 'No'}")

        print(f"\n  Recommended config:")
        print(f"    subdiv_min: {max_subdiv['subdiv']}")
        print(f"    subdiv_max: {max_subdiv['subdiv']}")

    else:
        print(f"\n✗ No valid configuration found!")
        print(f"  All tested subdivisions fail to capture both attractors.")

    print("="*80)


if __name__ == "__main__":
    main()
