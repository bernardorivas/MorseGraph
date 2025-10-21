#!/usr/bin/env python3
"""
Check if basins are truly interleaved or cleanly separated.

Test: Take a small neighborhood around a period-7 point and see if
nearby points go to equilibrium or all stay in period-7 basin.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.systems import ives_model_log


def classify_attractor(ic, n_steps=600, transient=200, tol=1e-4):
    """Classify which attractor an IC converges to."""

    params = {
        'r1': 3.873,
        'r2': 11.746,
        'c': 3.6e-7,
        'd': 0.5517,
        'p': 0.06659,
        'q': 0.9026,
        'offset': 0.001
    }

    # Simulate
    x = ic.copy()
    for _ in range(n_steps):
        x = ives_model_log(x, **params)

    # Check if equilibrium (period-1)
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


# Test neighborhood around a period-7 point
period_7_point = np.array([0.3645, -2.1168, 0.1605])

print("="*80)
print("BASIN BOUNDARY TEST")
print("="*80)
print(f"\nTesting neighborhood around period-7 point:")
print(f"  Center: {period_7_point}")

# Test points at different distances
distances = [0.001, 0.01, 0.05, 0.1]

for dist in distances:
    print(f"\nDistance = {dist}:")

    # Sample points in a small cube around the period-7 point
    n_samples = 20
    np.random.seed(42)

    results = {'equilibrium': 0, 'period_7': 0, 'other': 0}

    for _ in range(n_samples):
        # Random offset within distance
        offset = np.random.uniform(-dist, dist, 3)
        ic = period_7_point + offset

        attractor_type, _ = classify_attractor(ic)
        results[attractor_type] += 1

    print(f"  Equilibrium: {results['equilibrium']}/{n_samples} ({100*results['equilibrium']/n_samples:.1f}%)")
    print(f"  Period-7:    {results['period_7']}/{n_samples} ({100*results['period_7']/n_samples:.1f}%)")
    print(f"  Other:       {results['other']}/{n_samples} ({100*results['other']/n_samples:.1f}%)")

    if results['equilibrium'] > 0:
        print(f"  → Basin boundary is CLOSE (dist ≤ {dist})")
    else:
        print(f"  → No equilibrium points found at this distance")

print(f"\n{'='*80}")
print("CONCLUSION:")
print("="*80)

# Do finer test at the smallest distance where we found mixing
print("\nIf equilibrium points appear in small neighborhoods of period-7,")
print("then basins are densely interleaved (fractal boundary).")
print("\nIf they don't appear until large distances, basins are cleanly separated.")
print("="*80)
