#!/usr/bin/env python3
"""
Analyze the separation between equilibrium and period-7 orbit.
Determine what CMGDB subdivision is needed to isolate them.
"""

import numpy as np

# Equilibrium (from our basin analysis and config)
equilibrium = np.array([0.7923, 0.2097, 0.3773])

# Period-7 orbit points (from basin analysis at c=3.6e-7, d=0.5517)
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
domain_min = np.array([-1, -4, -1])
domain_max = np.array([2, 1, 1])
domain_width = domain_max - domain_min

print("="*80)
print("SEPARATION ANALYSIS: Equilibrium vs Period-7 Orbit")
print("="*80)

print(f"\nEquilibrium: [{equilibrium[0]:.4f}, {equilibrium[1]:.4f}, {equilibrium[2]:.4f}]")
print(f"\nPeriod-7 orbit (7 points):")

# Calculate distances
distances = []
for i, point in enumerate(period_7_orbit):
    dist = np.linalg.norm(point - equilibrium)
    distances.append(dist)
    print(f"  Point {i+1}: [{point[0]:7.4f}, {point[1]:7.4f}, {point[2]:7.4f}]  →  dist = {dist:.4f}")

min_distance = min(distances)
max_distance = max(distances)
avg_distance = np.mean(distances)

print(f"\nDistance statistics:")
print(f"  Minimum distance: {min_distance:.4f}")
print(f"  Maximum distance: {max_distance:.4f}")
print(f"  Average distance: {avg_distance:.4f}")

# Find closest and farthest points
closest_idx = np.argmin(distances)
farthest_idx = np.argmax(distances)
print(f"\nClosest point to equilibrium: Point {closest_idx + 1} (dist = {distances[closest_idx]:.4f})")
print(f"Farthest point from equilibrium: Point {farthest_idx + 1} (dist = {distances[farthest_idx]:.4f})")

# Calculate component-wise separations
print(f"\nComponent-wise analysis for CLOSEST point (Point {closest_idx + 1}):")
closest_point = period_7_orbit[closest_idx]
for dim, label in enumerate(['log(Midge)', 'log(Algae)', 'log(Detritus)']):
    separation = abs(closest_point[dim] - equilibrium[dim])
    print(f"  {label:15s}: |{closest_point[dim]:7.4f} - {equilibrium[dim]:7.4f}| = {separation:.4f}")

print("\n" + "="*80)
print("SUBDIVISION REQUIREMENTS")
print("="*80)

print(f"\nDomain: {domain_min.tolist()} to {domain_max.tolist()}")
print(f"Domain widths: {domain_width.tolist()}")

# For CMGDB, subdivision depth d creates boxes of approximate size:
# box_size ≈ domain_width / 2^d  (per dimension)
#
# To separate equilibrium from period-7, we need box size < min_distance
# But in practice, we want several boxes between them for reliability

safety_factors = [1, 2, 5, 10]

print(f"\nTo separate equilibrium (dist = {min_distance:.4f}):")
print(f"We need boxes smaller than {min_distance:.4f}")
print()

# For each dimension, calculate required subdivision
for dim, label in enumerate(['x (Midge)', 'y (Algae)', 'z (Detritus)']):
    print(f"{label}:")
    separation = abs(period_7_orbit[closest_idx, dim] - equilibrium[dim])

    for safety in safety_factors:
        # We want: domain_width[dim] / 2^d < separation / safety
        # So: 2^d > domain_width[dim] * safety / separation
        # So: d > log2(domain_width[dim] * safety / separation)

        required_subdiv = np.ceil(np.log2(domain_width[dim] * safety / separation))
        box_size = domain_width[dim] / (2**required_subdiv)

        print(f"  Safety factor {safety:2d}x: subdiv ≥ {int(required_subdiv):2d}  (box size ≈ {box_size:.4f})")

# Overall recommendation
max_separation = max([abs(period_7_orbit[closest_idx, dim] - equilibrium[dim]) for dim in range(3)])
overall_required = np.ceil(np.log2(max(domain_width) * 5 / max_separation))

print(f"\n" + "="*80)
print(f"RECOMMENDATION:")
print(f"="*80)
print(f"Current config: subdiv_min = 33, subdiv_max = 39")
print(f"\nTo reliably separate equilibrium from period-7 (5x safety):")
print(f"  Minimum subdivision: {int(overall_required)}")
print(f"\nThe current settings (33-39) are {'SUFFICIENT' if overall_required <= 33 else 'INSUFFICIENT'}!")

if overall_required <= 33:
    print(f"\n✓ Current subdivision (33) creates boxes of size ~{domain_width.max() / 2**33:.2e}")
    print(f"  This is much smaller than the separation ({min_distance:.4f})")
    print(f"  Should easily separate the attractors!")
else:
    print(f"\n✗ Need to increase subdiv_min to at least {int(overall_required)}")
