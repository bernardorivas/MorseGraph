#!/usr/bin/env python3
"""
Measure the basin of attraction radius around each period-7 orbit point.

Find the minimum distance from each orbit point where equilibrium ICs appear.
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.systems import ives_model_log


# Period-7 orbit (from our search)
PERIOD_7_ORBIT = np.array([
    [1.042458, -2.999841, 0.832761],
    [0.364476, -2.116800, 0.160540],
    [-0.280030, -1.287354, -0.492422],
    [-0.514694, -0.441230, -0.851630],
    [-0.161418, 0.389527, -0.373334],
    [0.330669, 0.794379, 0.425250],
    [0.805739, 0.591782, 0.876293],
])

EQUILIBRIUM = np.array([0.792107, 0.209010, 0.376449])

DEFAULT_PARAMS = {
    'r1': 3.873,
    'r2': 11.746,
    'c': 10**-6.435,
    'd': 0.5517,
    'p': 0.06659,
    'q': 0.9026,
    'offset': 0.001
}


def classify_attractor(ic, n_steps=600, tol=1e-4, params=None):
    """Classify which attractor an IC converges to."""
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Simulate
    x = ic.copy()
    for _ in range(n_steps):
        x = ives_model_log(x, **params)
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            return 'diverged', None

    # Check if equilibrium
    x_next = ives_model_log(x, **params)
    if np.linalg.norm(x_next - x) < tol:
        return 'equilibrium', x

    # Check if period-7
    x_p7 = x.copy()
    for _ in range(7):
        x_p7 = ives_model_log(x_p7, **params)
    if np.linalg.norm(x_p7 - x) < tol:
        return 'period_7', x

    return 'other', x


def find_basin_radius(
    orbit_point: np.ndarray,
    distances: list,
    n_samples: int = 100,
    seed: int = 42
):
    """
    Find the radius where basin boundary occurs.

    Tests multiple distances and counts how many ICs go to each attractor.

    Returns:
        Dictionary mapping distance -> {equilibrium_frac, period7_frac, other_frac}
    """
    np.random.seed(seed)
    results = {}

    for dist in distances:
        counts = {'equilibrium': 0, 'period_7': 0, 'other': 0, 'diverged': 0}

        for _ in range(n_samples):
            # Random offset in cube
            offset = np.random.uniform(-dist, dist, 3)
            ic = orbit_point + offset

            attractor_type, _ = classify_attractor(ic)
            counts[attractor_type] += 1

        total = n_samples
        results[dist] = {
            'equilibrium': counts['equilibrium'] / total,
            'period_7': counts['period_7'] / total,
            'other': counts['other'] / total,
            'diverged': counts['diverged'] / total,
        }

    return results


def main():
    print("="*80)
    print("MEASURING BASIN RADIUS FOR PERIOD-7 ORBIT")
    print("="*80)

    # Test distances (in log space)
    distances = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]

    all_results = []

    print("\nTesting each point of period-7 orbit...")
    print("="*80)

    for i, orbit_point in enumerate(PERIOD_7_ORBIT):
        print(f"\n{'='*80}")
        print(f"POINT {i+1}/7: {orbit_point}")
        print(f"{'='*80}")

        results = find_basin_radius(orbit_point, distances, n_samples=100)
        all_results.append(results)

        # Find critical distance where equilibrium basin appears
        critical_dist = None
        for dist in distances:
            if results[dist]['equilibrium'] > 0:
                critical_dist = dist
                break

        if critical_dist is not None:
            print(f"\n✓ Basin boundary found at distance ≤ {critical_dist}")
            print(f"  At dist={critical_dist}:")
            print(f"    Equilibrium: {results[critical_dist]['equilibrium']*100:.1f}%")
            print(f"    Period-7:    {results[critical_dist]['period_7']*100:.1f}%")
        else:
            print(f"\n✓ No basin boundary found (tested up to {max(distances)})")
            print(f"  Entire neighborhood goes to period-7")

        # Show details for key distances
        print(f"\n  Detailed breakdown:")
        for dist in [0.01, 0.05, 0.1]:
            if dist in results:
                r = results[dist]
                print(f"    dist={dist:5.3f}: Eq={r['equilibrium']*100:5.1f}%, P7={r['period_7']*100:5.1f}%, Other={r['other']*100:5.1f}%")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    min_radii = []
    for i, results in enumerate(all_results):
        # Find smallest distance with equilibrium basin
        min_radius = None
        for dist in distances:
            if results[dist]['equilibrium'] > 0:
                min_radius = dist
                break

        min_radii.append(min_radius if min_radius is not None else max(distances))

        print(f"Point {i+1}: Basin radius ≈ {min_radii[-1]:.4f}")

    overall_min = min(min_radii)
    overall_max = max(min_radii)

    print(f"\nOverall basin radius:")
    print(f"  Minimum: {overall_min:.4f} (tightest constraint)")
    print(f"  Maximum: {overall_max:.4f} (loosest constraint)")
    print(f"\nConclusion:")
    if overall_min < 0.01:
        print(f"  → VERY SMALL basin! Points within {overall_min:.4f} of period-7 can escape.")
        print(f"  → This suggests a FRACTAL basin boundary (densely interleaved basins)")
        print(f"  → Period-7 is technically stable but FRAGILE")
    elif overall_min < 0.1:
        print(f"  → SMALL basin. Points within {overall_min:.4f} can escape.")
        print(f"  → Period-7 is stable but basin boundary is close")
    else:
        print(f"  → LARGE basin. Need to move >{overall_min:.4f} away to escape.")
        print(f"  → Period-7 is strongly stable")

    # Visualization
    print(f"\nCreating visualization...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(7):
        ax = axes[i]
        results = all_results[i]

        dists = sorted(results.keys())
        eq_fracs = [results[d]['equilibrium'] for d in dists]
        p7_fracs = [results[d]['period_7'] for d in dists]

        ax.plot(dists, eq_fracs, 'o-', label='→ Equilibrium', color='red', linewidth=2)
        ax.plot(dists, p7_fracs, 's-', label='→ Period-7', color='blue', linewidth=2)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Distance (log space)', fontsize=10)
        ax.set_ylabel('Fraction of ICs', fontsize=10)
        ax.set_title(f'Point {i+1}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    # Remove extra subplot
    axes[7].axis('off')

    plt.tight_layout()
    plt.savefig('results/basin_radius_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved to results/basin_radius_analysis.png")

    # Create a summary plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(7):
        results = all_results[i]
        dists = sorted(results.keys())
        eq_fracs = [results[d]['equilibrium'] for d in dists]

        ax.plot(dists, eq_fracs, 'o-', label=f'Point {i+1}', linewidth=2, alpha=0.7)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_xlabel('Distance from orbit point (log space)', fontsize=12)
    ax.set_ylabel('Fraction going to equilibrium', fontsize=12)
    ax.set_title('Basin Boundary Analysis: Period-7 Orbit', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig('results/basin_radius_summary.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved to results/basin_radius_summary.png")

    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
