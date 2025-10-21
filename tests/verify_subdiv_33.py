#!/usr/bin/env python3
"""
Verify subdiv=33 is the maximum usable resolution for Ives model.
Test subdiv 33, 34, 35, 36 to find exact cutoff.
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


def test_subdiv_detailed(subdiv):
    """Detailed test of subdivision level."""

    print(f"\n{'='*80}")
    print(f"Testing subdiv={subdiv}")
    print(f"{'='*80}")

    # Attractors
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

    total_boxes = 2**subdiv
    boxes_per_dim = int(round(total_boxes**(1/3)))

    print(f"\nGrid: {total_boxes:,} total boxes ({boxes_per_dim} per dimension)")

    # BoxMap
    def F(rect):
        dim = 3
        corners = list(product(*[(rect[d], rect[d+dim]) for d in range(dim)]))
        corners = [list(c) for c in corners]
        corners_next = np.array([ives_model_log(np.array(c)) for c in corners])
        Y_l = [corners_next[:, d].min() for d in range(dim)]
        Y_u = [corners_next[:, d].max() for d in range(dim)]
        return Y_l + Y_u

    # Build and compute
    print(f"Computing Morse graph...")
    model = CMGDB.Model(subdiv, subdiv, domain_lower, domain_upper, F)

    start = time.time()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    comp_time = time.time() - start

    print(f"  Completed in {comp_time:.1f}s")
    print(f"  Found {morse_graph.num_vertices()} Morse sets")

    # Check equilibrium
    eq_set = find_box_containing_point(morse_graph, equilibrium)
    print(f"\nEquilibrium: Morse set {eq_set}")

    # Check period-7
    print(f"\nPeriod-7 orbit:")
    orbit_sets = []
    for i, pt in enumerate(period_7_orbit):
        ms = find_box_containing_point(morse_graph, pt)
        orbit_sets.append(ms)
        status = "✓" if ms is not None else "✗ NOT IN ANY SET"
        print(f"  Point {i+1}: Morse set {ms} {status}")

    orbit_sets_unique = set([ms for ms in orbit_sets if ms is not None])
    n_captured = sum(1 for ms in orbit_sets if ms is not None)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    eq_ok = eq_set is not None
    orbit_ok = n_captured == 7

    print(f"  Equilibrium captured: {'✓ Yes' if eq_ok else '✗ No'}")
    print(f"  Period-7 captured: {n_captured}/7 {'✓' if orbit_ok else '✗'}")

    if eq_ok and orbit_ok:
        separated = eq_set not in orbit_sets_unique
        print(f"  Separated: {'✓ Yes' if separated else '✗ No'}")
        print(f"\n  ✓✓ VALID CONFIGURATION")
    else:
        print(f"\n  ✗✗ INVALID - Attractors not fully captured")

    return {
        'subdiv': subdiv,
        'valid': eq_ok and orbit_ok,
        'eq_set': eq_set,
        'orbit_sets': orbit_sets_unique,
        'n_captured': n_captured,
        'comp_time': comp_time,
        'num_morse_sets': morse_graph.num_vertices()
    }


def main():
    print("="*80)
    print("VERIFYING MAXIMUM USABLE SUBDIVISION FOR IVES MODEL")
    print("="*80)

    # Test around subdiv=33
    test_subdivs = [33, 34, 35, 36]

    results = []
    for subdiv in test_subdivs:
        result = test_subdiv_detailed(subdiv)
        results.append(result)

        if not result['valid']:
            print(f"\n⚠ Subdivision {subdiv} is NOT valid - stopping search")
            break

    # Final recommendation
    print(f"\n\n{'='*80}")
    print("FINAL RECOMMENDATION")
    print(f"{'='*80}")

    valid = [r for r in results if r['valid']]
    if valid:
        best = max(valid, key=lambda r: r['subdiv'])
        print(f"\n✓ Maximum usable subdivision: {best['subdiv']}")
        print(f"  Computation time: {best['comp_time']:.1f}s")
        print(f"  Morse sets: {best['num_morse_sets']}")
        print(f"  Equilibrium in set: {best['eq_set']}")
        print(f"  Period-7 in set(s): {sorted(best['orbit_sets'])}")

        print(f"\n  Update examples/configs/ives_default.yaml:")
        print(f"    subdiv_min: {best['subdiv']}")
        print(f"    subdiv_max: {best['subdiv']}")
    else:
        print(f"\n✗ No valid configuration found in tested range")

    print("="*80)


if __name__ == "__main__":
    main()
