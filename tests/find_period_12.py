#!/usr/bin/env python3
"""
Quick script to find period-12 orbit by sweeping (c, d) parameter space.
Based on Figure 3a, period-12 should be at:
  c: 10^-7 to 10^-6
  d: 0.55 to 0.8
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.systems import ives_model_log


def check_period(c_val, d_val, max_period=15, n_steps=800, transient=300):
    """Check what period (if any) exists at given (c, d) parameters.

    Tests multiple initial conditions to find different attractors.
    Returns the maximum period found (to find periodic orbits, not just equilibria).
    """

    params = {
        'r1': 3.873,
        'r2': 11.746,
        'c': c_val,
        'd': d_val,
        'p': 0.06659,
        'q': 0.9026,
        'offset': 0.001
    }

    # Test multiple initial conditions
    test_ics = [
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, -2.0, 0.5]),
        np.array([-0.5, 0.5, -0.5]),
        np.array([0.5, -1.5, 0.2]),
        np.array([0.3, -2.5, 0.0]),
    ]

    periods_found = []

    for x0 in test_ics:
        # Integrate to attractor
        x = x0.copy()
        for _ in range(transient):
            x = ives_model_log(x, **params)

        # Collect trajectory
        trajectory = [x]
        for _ in range(n_steps):
            x = ives_model_log(x, **params)
            trajectory.append(x)
        trajectory = np.array(trajectory)

        # Check for periodicity
        for period in range(1, max_period + 1):
            is_periodic = True
            for t in range(n_steps // 2, n_steps - period):
                if np.linalg.norm(trajectory[t + period] - trajectory[t]) > 1e-4:
                    is_periodic = False
                    break

            if is_periodic:
                periods_found.append(period)
                break  # Found period for this IC

    # Return the maximum period found (prefer cycles over equilibria)
    return max(periods_found) if periods_found else None


# Grid search
print("Searching for period-12 orbit...")
print("Based on Figure 3a: c ∈ [10^-7, 10^-6], d ∈ [0.55, 0.8]")
print()

c_values = np.logspace(-7, -6, 20)  # 10^-7 to 10^-6
d_values = np.linspace(0.50, 0.85, 20)  # 0.50 to 0.85

print(f"Testing {len(c_values)} x {len(d_values)} = {len(c_values) * len(d_values)} parameter combinations")
print()

period_12_found = []

for i, c_val in enumerate(c_values):
    for j, d_val in enumerate(d_values):
        period = check_period(c_val, d_val)

        if period == 12:
            period_12_found.append((c_val, d_val))
            print(f"✓ PERIOD-12 at c={c_val:.3e}, d={d_val:.4f}")

        # Print progress every 50 combinations
        if (i * len(d_values) + j + 1) % 50 == 0:
            print(f"  Progress: {i * len(d_values) + j + 1}/{len(c_values) * len(d_values)}")

print()
print(f"Found {len(period_12_found)} parameter combinations with period-12")

if period_12_found:
    print("\nPeriod-12 parameters:")
    for c_val, d_val in period_12_found:
        print(f"  c = {c_val:.6e}, d = {d_val:.4f}")
else:
    print("\nNo period-12 found. Showing sample periods across the grid:")
    # Sample a few points
    for c_val in [2e-7, 4e-7, 6e-7, 8e-7]:
        for d_val in [0.55, 0.60, 0.65, 0.70, 0.75]:
            period = check_period(c_val, d_val)
            print(f"  c={c_val:.2e}, d={d_val:.2f} → period={period}")
