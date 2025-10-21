#!/usr/bin/env python3
"""
Test refined BoxMap evaluation strategies for CMGDB.

Instead of just evaluating at 8 corners, we sample more densely within each box
to get a tighter bounding box of the image, reducing bloating artifacts.

Strategies tested:
1. Corners only (8 points) - Current method
2. Face centers (6 points) + corners (8 points) = 14 points
3. Grid 3x3x3 (27 points) - Regular grid
4. Grid 5x5x5 (125 points) - Dense grid
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


def sample_box_corners(rect, dim):
    """Sample 8 corner points of a box."""
    corners = list(product(*[(rect[d], rect[d+dim]) for d in range(dim)]))
    return [np.array(c) for c in corners]


def sample_box_grid(rect, dim, n_per_dim=3):
    """Sample points on a regular grid within the box.

    Args:
        rect: Box specification [x_lower, y_lower, z_lower, x_upper, y_upper, z_upper]
        dim: Dimension (3 for 3D)
        n_per_dim: Number of sample points per dimension

    Returns:
        List of sample points
    """
    samples = []

    # Create grid for each dimension
    grids = []
    for d in range(dim):
        grids.append(np.linspace(rect[d], rect[d+dim], n_per_dim))

    # Generate all combinations
    for coords in product(*grids):
        samples.append(np.array(coords))

    return samples


def sample_box_face_centers(rect, dim):
    """Sample center points of all faces (6 for 3D) plus corners.

    For a 3D box, samples:
    - 8 corners
    - 6 face centers (one per face)
    Total: 14 points
    """
    samples = []

    # Add corners
    samples.extend(sample_box_corners(rect, dim))

    # Add face centers
    lower = np.array([rect[d] for d in range(dim)])
    upper = np.array([rect[d+dim] for d in range(dim)])
    center = (lower + upper) / 2

    # For each dimension, add two face centers (min and max face)
    for d in range(dim):
        # Min face (set dimension d to lower[d], others at center)
        pt_min = center.copy()
        pt_min[d] = lower[d]
        samples.append(pt_min)

        # Max face (set dimension d to upper[d], others at center)
        pt_max = center.copy()
        pt_max[d] = upper[d]
        samples.append(pt_max)

    return samples


def create_boxmap_function(strategy='corners', padding_factor=0.0):
    """
    Create BoxMap function with specified sampling strategy.

    Args:
        strategy: 'corners', 'faces', 'grid3', 'grid5'
        padding_factor: Multiplicative factor for padding (0.0 = no padding)

    Returns:
        BoxMap function F(rect)
    """

    def F(rect):
        dim = 3

        # Sample points based on strategy
        if strategy == 'corners':
            samples = sample_box_corners(rect, dim)
        elif strategy == 'faces':
            samples = sample_box_face_centers(rect, dim)
        elif strategy == 'grid3':
            samples = sample_box_grid(rect, dim, n_per_dim=3)
        elif strategy == 'grid5':
            samples = sample_box_grid(rect, dim, n_per_dim=5)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Evaluate map at all sample points
        images = np.array([ives_model_log(pt) for pt in samples])

        # Compute bounding box of images
        Y_l = images.min(axis=0)
        Y_u = images.max(axis=0)

        # Add padding if requested
        if padding_factor > 0:
            padding_size = [(rect[d + dim] - rect[d]) * padding_factor for d in range(dim)]
            Y_l = [Y_l[d] - padding_size[d] for d in range(dim)]
            Y_u = [Y_u[d] + padding_size[d] for d in range(dim)]
        else:
            Y_l = Y_l.tolist()
            Y_u = Y_u.tolist()

        return Y_l + Y_u

    return F


def analyze_morse_graph_structure(morse_graph):
    """Analyze Morse graph structure."""
    num_sets = morse_graph.num_vertices()
    edges = morse_graph.edges()

    outgoing = {i: set() for i in range(num_sets)}
    for src, dst in edges:
        outgoing[src].add(dst)

    minimal_nodes = [i for i in range(num_sets) if len(outgoing[i]) == 0]

    return {
        'num_sets': num_sets,
        'num_edges': len(edges),
        'minimal_nodes': minimal_nodes,
        'edges': edges
    }


def test_boxmap_strategy(subdiv, strategy, padding_factor=0.0, verbose=True):
    """Test CMGDB with specific BoxMap sampling strategy."""

    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing: {strategy.upper()} sampling, padding={padding_factor}x, subdiv={subdiv}")
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

    # Create BoxMap function
    F = create_boxmap_function(strategy, padding_factor)

    # Info
    n_samples_map = {
        'corners': 8,
        'faces': 14,
        'grid3': 27,
        'grid5': 125
    }

    if verbose:
        print(f"  Sample points per box: {n_samples_map.get(strategy, '?')}")

    # Build model
    model = CMGDB.Model(subdiv, subdiv, domain_lower, domain_upper, F)

    # Compute
    if verbose:
        print(f"  Computing Morse graph...")

    start = time.time()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    comp_time = time.time() - start

    if verbose:
        print(f"  Completed in {comp_time:.1f}s")

    # Analyze structure
    structure = analyze_morse_graph_structure(morse_graph)

    if verbose:
        print(f"  Morse sets: {structure['num_sets']}, Edges: {structure['num_edges']}")
        print(f"  Minimal nodes: {structure['minimal_nodes']}")

    # Check attractor locations
    eq_set = find_box_containing_point(morse_graph, equilibrium)

    orbit_sets = []
    n_captured = 0
    for pt in period_7_orbit:
        ms = find_box_containing_point(morse_graph, pt)
        orbit_sets.append(ms)
        if ms is not None:
            n_captured += 1

    orbit_sets_unique = set([ms for ms in orbit_sets if ms is not None])

    # Check properties
    eq_captured = eq_set is not None
    orbit_captured = n_captured == 7
    separated = eq_captured and orbit_captured and (eq_set not in orbit_sets_unique)

    eq_is_minimal = eq_set in structure['minimal_nodes'] if eq_set is not None else False
    orbit_is_minimal = all(ms in structure['minimal_nodes'] for ms in orbit_sets_unique if ms is not None)

    correct_structure = (
        eq_captured and orbit_captured and
        len(structure['minimal_nodes']) == 2 and
        eq_is_minimal and orbit_is_minimal
    )

    if verbose:
        print(f"\n  Equilibrium: set {eq_set} {'(minimal)' if eq_is_minimal else '(NOT minimal)'}")
        print(f"  Period-7: {n_captured}/7 points captured in set(s) {sorted(orbit_sets_unique)}")
        if orbit_sets_unique:
            print(f"           {'(minimal)' if orbit_is_minimal else '(NOT minimal)'}")

        if correct_structure:
            print(f"\n  ✓✓ CORRECT STRUCTURE - Both attractors are minimal nodes!")
        elif separated:
            print(f"\n  ⚠ Separated but incorrect structure")
        else:
            print(f"\n  ✗ Not properly separated or not captured")

    return {
        'strategy': strategy,
        'padding_factor': padding_factor,
        'subdiv': subdiv,
        'n_samples': n_samples_map.get(strategy, 0),
        'comp_time': comp_time,
        'structure': structure,
        'eq_captured': eq_captured,
        'orbit_captured': orbit_captured,
        'separated': separated,
        'correct_structure': correct_structure,
        'eq_set': eq_set,
        'orbit_sets': orbit_sets_unique
    }


def main():
    print("="*80)
    print("REFINED BOXMAP EVALUATION - IVES MODEL")
    print("="*80)
    print("\nTesting different sampling strategies to reduce bloating artifacts\n")

    # Test configurations
    # Start with subdiv=24 for reasonable computation time
    subdiv = 24

    configs = [
        # (strategy, padding_factor, description)
        ('corners', 0.0, 'Corners only, no padding'),
        ('faces', 0.0, 'Corners + face centers, no padding'),
        ('grid3', 0.0, 'Grid 3×3×3, no padding'),
        ('grid3', 0.1, 'Grid 3×3×3, 0.1× padding'),
        ('grid5', 0.0, 'Grid 5×5×5, no padding'),
        ('grid5', 0.05, 'Grid 5×5×5, 0.05× padding'),
    ]

    results = []

    for strategy, padding, desc in configs:
        print(f"\n{'#'*80}")
        print(f"# {desc}")
        print(f"{'#'*80}")

        result = test_boxmap_strategy(subdiv, strategy, padding, verbose=True)
        result['description'] = desc
        results.append(result)

        # Stop if we found correct structure
        if result['correct_structure']:
            print(f"\n  ✓✓ Found correct structure! Stopping tests.")
            break

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Strategy':<25} {'Samples':<10} {'Pad':<8} {'Sets':<6} {'Min':<6} "
          f"{'Time(s)':<10} {'Correct?':<10}")
    print("-" * 80)

    for r in results:
        correct = "✓✓ Yes" if r['correct_structure'] else ("✓ Sep" if r['separated'] else "✗ No")
        print(f"{r['description']:<25} {r['n_samples']:<10} {r['padding_factor']:<8.2f} "
              f"{r['structure']['num_sets']:<6} {len(r['structure']['minimal_nodes']):<6} "
              f"{r['comp_time']:<10.1f} {correct:<10}")

    # Recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    correct_results = [r for r in results if r['correct_structure']]
    if correct_results:
        best = correct_results[0]
        print(f"\n✓ Found correct structure with:")
        print(f"  Strategy: {best['description']}")
        print(f"  Sample points: {best['n_samples']}")
        print(f"  Padding factor: {best['padding_factor']}")
        print(f"  Computation time: {best['comp_time']:.1f}s")
        print(f"\n  This strategy should be used in the 3D CMGDB computation!")
    else:
        best_sep = [r for r in results if r['separated']]
        if best_sep:
            # Find the one with most samples (tightest approximation)
            best = max(best_sep, key=lambda r: r['n_samples'])
            print(f"\n⚠ No correct structure found, but best separation with:")
            print(f"  Strategy: {best['description']}")
            print(f"  Sample points: {best['n_samples']}")
            print(f"  Morse sets: {best['structure']['num_sets']} (expect 2 minimal)")
            print(f"  Minimal nodes: {len(best['structure']['minimal_nodes'])}")
        else:
            print(f"\n✗ No strategy achieved proper separation at subdiv={subdiv}")
            print(f"   May need higher resolution or different approach")

    print("="*80)


if __name__ == "__main__":
    main()
