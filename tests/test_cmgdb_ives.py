#!/usr/bin/env python3
"""
Test CMGDB Morse graph for Ives model - check proper separation of attractors.

For bistable system with equilibrium + period-7 orbit:
- Should have 2 minimal Morse sets (no outgoing edges)
- Equilibrium and period-7 should each be in their own minimal Morse set
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


def analyze_morse_graph_structure(morse_graph, verbose=True):
    """Analyze Morse graph structure - find minimal nodes, maximal nodes, etc."""
    num_sets = morse_graph.num_vertices()
    edges = morse_graph.edges()

    # Build adjacency structure
    outgoing = {i: set() for i in range(num_sets)}
    incoming = {i: set() for i in range(num_sets)}

    for src, dst in edges:
        outgoing[src].add(dst)
        incoming[dst].add(src)

    # Find minimal (sink) and maximal (source) nodes
    minimal_nodes = [i for i in range(num_sets) if len(outgoing[i]) == 0]
    maximal_nodes = [i for i in range(num_sets) if len(incoming[i]) == 0]

    if verbose:
        print(f"\nMorse Graph Structure:")
        print(f"  Total Morse sets: {num_sets}")
        print(f"  Total edges: {len(edges)}")
        print(f"  Minimal nodes (sinks): {minimal_nodes}")
        print(f"  Maximal nodes (sources): {maximal_nodes}")

        print(f"\nEdge details:")
        for src, dst in edges:
            print(f"    {src} → {dst}")

        print(f"\nOutgoing edges per Morse set:")
        for i in range(num_sets):
            print(f"    Morse set {i}: {len(outgoing[i])} outgoing edges → {sorted(outgoing[i])}")

    return {
        'num_sets': num_sets,
        'num_edges': len(edges),
        'minimal_nodes': minimal_nodes,
        'maximal_nodes': maximal_nodes,
        'outgoing': outgoing,
        'incoming': incoming,
        'edges': edges
    }


def test_ives_cmgdb(subdiv, use_center=False, verbose=True):
    """Test CMGDB for Ives model at given subdivision.

    Args:
        subdiv: Subdivision level (2^subdiv total boxes)
        use_center: If True, evaluate at box center. If False, evaluate at corners.
        verbose: Print detailed output
    """

    if verbose:
        print(f"\n{'='*80}")
        eval_method = "CENTER" if use_center else "CORNERS"
        print(f"Testing CMGDB for Ives Model - subdiv={subdiv} ({eval_method} evaluation)")
        print(f"{'='*80}")

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

    # Domain bounds
    domain_lower = [-1, -4, -1]
    domain_upper = [2, 1, 1]

    # Box map function
    def F(rect):
        """CMGDB BoxMap function."""
        dim = 3

        if use_center:
            # Evaluate at center only (minimizes bloating)
            center = [(rect[d] + rect[d+dim]) / 2.0 for d in range(dim)]
            image = ives_model_log(np.array(center))
            # Add small padding to avoid degenerate boxes
            # Use half the box width as padding (similar to 2D restricted code)
            padding_size = [(rect[d + dim] - rect[d]) / 2.0 for d in range(dim)]
            Y_l = [image[d] - padding_size[d] for d in range(dim)]
            Y_u = [image[d] + padding_size[d] for d in range(dim)]
        else:
            # Evaluate at all corners
            corners = list(product(*[(rect[d], rect[d+dim]) for d in range(dim)]))
            corners = [list(c) for c in corners]
            corners_next = np.array([ives_model_log(np.array(c)) for c in corners])

            # No padding (padding=false in config)
            Y_l = [corners_next[:, d].min() for d in range(dim)]
            Y_u = [corners_next[:, d].max() for d in range(dim)]

        return Y_l + Y_u

    if verbose:
        total_boxes = 2**subdiv
        boxes_per_dim = int(round(total_boxes**(1/3)))
        print(f"\nGrid parameters:")
        print(f"  Total boxes: {total_boxes:,}")
        print(f"  Boxes per dimension: {boxes_per_dim}")
        print(f"\nBuilding CMGDB model...")

    # Build model
    model = CMGDB.Model(
        subdiv,
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

    if verbose:
        print(f"  Completed in {comp_time:.2f}s")

    # Analyze structure
    structure = analyze_morse_graph_structure(morse_graph, verbose=verbose)

    # Find which Morse sets contain attractors
    if verbose:
        print(f"\n{'='*80}")
        print("ATTRACTOR LOCATIONS")
        print(f"{'='*80}")

    eq_morse_set = find_box_containing_point(morse_graph, equilibrium)
    if verbose:
        print(f"\nEquilibrium: Morse set {eq_morse_set}")

    orbit_morse_sets = []
    if verbose:
        print(f"\nPeriod-7 orbit:")
    for i, pt in enumerate(period_7_orbit):
        ms = find_box_containing_point(morse_graph, pt)
        orbit_morse_sets.append(ms)
        if verbose:
            print(f"  Point {i+1}: Morse set {ms}")

    unique_orbit_sets = set([ms for ms in orbit_morse_sets if ms is not None])

    # Check separation
    if verbose:
        print(f"\n{'='*80}")
        print("SEPARATION ANALYSIS")
        print(f"{'='*80}")

    if eq_morse_set is not None and unique_orbit_sets:
        separated = eq_morse_set not in unique_orbit_sets

        if separated:
            if verbose:
                print(f"\n✓ DIFFERENT Morse sets - equilibrium and orbit ARE separated!")
                print(f"  Equilibrium in set {eq_morse_set}")
                print(f"  Orbit in set(s): {sorted(unique_orbit_sets)}")
        else:
            if verbose:
                print(f"\n✗ SAME Morse set - equilibrium and orbit NOT separated")
            return {
                'subdiv': subdiv,
                'separated': False,
                'structure': structure
            }
    else:
        if verbose:
            print(f"\n? Could not determine separation (some points not in Morse sets)")
        return {
            'subdiv': subdiv,
            'separated': None,
            'structure': structure
        }

    # Check if both are in minimal nodes (as they should be for stable attractors)
    if verbose:
        print(f"\n{'='*80}")
        print("ATTRACTOR PROPERTIES CHECK")
        print(f"{'='*80}")

    eq_is_minimal = eq_morse_set in structure['minimal_nodes']
    orbit_sets_minimal = [ms in structure['minimal_nodes'] for ms in unique_orbit_sets]
    all_orbit_minimal = all(orbit_sets_minimal)

    if verbose:
        print(f"\nEquilibrium (Morse set {eq_morse_set}):")
        print(f"  Is minimal (sink)? {eq_is_minimal}")
        if not eq_is_minimal:
            print(f"  ⚠ WARNING: Equilibrium should be minimal!")
            print(f"  Outgoing edges: {structure['outgoing'][eq_morse_set]}")

        print(f"\nPeriod-7 orbit (Morse set(s) {sorted(unique_orbit_sets)}):")
        for ms in sorted(unique_orbit_sets):
            is_min = ms in structure['minimal_nodes']
            print(f"  Morse set {ms}: Is minimal (sink)? {is_min}")
            if not is_min:
                print(f"    ⚠ WARNING: Period-7 orbit should be minimal!")
                print(f"    Outgoing edges: {structure['outgoing'][ms]}")

    # Expected: 2 minimal nodes (eq + period-7)
    expected_minimal = 2
    if verbose:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"\nExpected minimal nodes: {expected_minimal} (equilibrium + period-7)")
        print(f"Actual minimal nodes: {len(structure['minimal_nodes'])}")

        if len(structure['minimal_nodes']) == expected_minimal and eq_is_minimal and all_orbit_minimal:
            print(f"\n✓ CORRECT structure: Both attractors are minimal nodes!")
        else:
            print(f"\n✗ INCORRECT structure:")
            if len(structure['minimal_nodes']) != expected_minimal:
                print(f"  - Wrong number of minimal nodes")
            if not eq_is_minimal:
                print(f"  - Equilibrium is not minimal")
            if not all_orbit_minimal:
                print(f"  - Period-7 orbit is not minimal")

    return {
        'subdiv': subdiv,
        'separated': separated,
        'structure': structure,
        'eq_morse_set': eq_morse_set,
        'orbit_morse_sets': unique_orbit_sets,
        'eq_is_minimal': eq_is_minimal,
        'orbit_is_minimal': all_orbit_minimal,
        'correct_structure': (
            len(structure['minimal_nodes']) == expected_minimal and
            eq_is_minimal and
            all_orbit_minimal
        )
    }


def investigate_morse_set(morse_graph, morse_set_idx, n_sample=10):
    """Sample some boxes from a Morse set to understand what's in it."""
    boxes = morse_graph.morse_set_boxes(morse_set_idx)
    if not boxes:
        return None

    dim = len(boxes[0]) // 2
    n_boxes = len(boxes)

    # Sample a few boxes
    sample_indices = np.linspace(0, n_boxes-1, min(n_sample, n_boxes), dtype=int)

    samples = []
    for idx in sample_indices:
        box = boxes[idx]
        lower = np.array(box[:dim])
        upper = np.array(box[dim:])
        center = (lower + upper) / 2
        size = upper - lower
        samples.append({
            'center': center,
            'size': size,
            'volume': np.prod(size)
        })

    return {
        'num_boxes': n_boxes,
        'samples': samples,
        'total_volume': sum(s['volume'] for s in samples) * (n_boxes / len(samples))
    }


def main():
    print("="*80)
    print("IVES MODEL CMGDB TEST - MORSE GRAPH STRUCTURE")
    print("="*80)

    # Test configurations
    configs = [
        {'subdiv': 24, 'use_center': False, 'name': 'subdiv=24, CORNERS'},
        {'subdiv': 33, 'use_center': False, 'name': 'subdiv=33, CORNERS (original config)'},
        {'subdiv': 36, 'use_center': False, 'name': 'subdiv=36, CORNERS (higher res)'},
    ]

    results = []

    for config in configs:
        print(f"\n\n{'#'*80}")
        print(f"# TEST: {config['name']}")
        print(f"{'#'*80}")

        result = test_ives_cmgdb(
            subdiv=config['subdiv'],
            use_center=config['use_center'],
            verbose=True
        )
        result['config_name'] = config['name']
        results.append(result)

        # If this configuration found Morse set 1, investigate it
        if result['structure']['num_sets'] >= 2:
            print(f"\n{'='*80}")
            print("INVESTIGATING MORSE SET 1 (intermediate set)")
            print(f"{'='*80}")

            # We need to re-compute to get the morse_graph object
            # For now, just note it in the output
            print("  Morse set 1 exists in this configuration")
            print(f"  Edges: {[f'{src}→{dst}' for src, dst in result['structure']['edges'] if src == 1 or dst == 1]}")

        # Stop early if we found correct structure
        if result.get('correct_structure'):
            print(f"\n  ✓✓ Found correct structure! Stopping tests.")
            break

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'='*80}")
    print(f"\n{'Config':<35} {'Separated':<12} {'Correct':<10} {'Morse Sets':<12} {'Minimal':<10}")
    print("-" * 85)

    for r in results:
        sep = "✓ Yes" if r.get('separated') else ("✗ No" if r.get('separated') is False else "? Unk")
        correct = "✓ Yes" if r.get('correct_structure') else "✗ No"
        print(f"{r['config_name']:<35} {sep:<12} {correct:<10} "
              f"{r['structure']['num_sets']:<12} {len(r['structure']['minimal_nodes']):<10}")

    # Final verdict
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print(f"{'='*80}")

    correct_results = [r for r in results if r.get('correct_structure')]
    if correct_results:
        best = correct_results[0]
        print(f"\n✓✓ SUCCESS: Found correct structure!")
        print(f"   Configuration: {best['config_name']}")
        print(f"   Both attractors are minimal nodes (sinks)")
    else:
        print(f"\n✗ NO CORRECT STRUCTURE FOUND")
        print(f"   All tested configurations have spurious connections")
        print(f"   Recommendation: Try even higher resolution or investigate bloating further")

    print("="*80)


if __name__ == "__main__":
    main()
