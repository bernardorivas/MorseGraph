#!/usr/bin/env python3
"""Test in linear space instead of log space to see if we get period-12"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MorseGraph.systems import ives_model  # LINEAR SPACE

params = {
    'r1': 3.873,
    'r2': 11.746,
    'c': 10**-6.435,
    'd': 0.5517,
    'p': 0.06659,
    'q': 0.9026,
}

print("Testing in LINEAR space (not log)")
print("="*60)

# Try several ICs in linear space
test_ics_linear = [
    np.array([0.23, 0.08, 0.49]),  # Near equilibrium
    np.array([1.0, 0.01, 0.1]),   # Random IC
    np.array([10.0, 0.001, 0.01]),  # Higher midge
]

for ic_idx, ic in enumerate(test_ics_linear):
    print(f"\nIC {ic_idx+1} (linear): {ic}")

    # Simulate
    traj = np.zeros((1000, 3))
    traj[0] = ic

    for t in range(999):
        traj[t+1] = ives_model(traj[t], **params)

        if np.any(traj[t+1] < 0) or np.any(np.isnan(traj[t+1])):
            print(f"  → Stopped at step {t+1}")
            traj = traj[:t+1]
            break

    # Check for periodicity after transient
    transient = 400
    if len(traj) > transient + 30:
        final_traj = traj[transient:]

        # Check periods 1-30
        for period in range(1, 31):
            if len(final_traj) < period + 50:
                continue

            # Check if periodic
            is_periodic = True
            for t in range(50, len(final_traj) - period):
                diff = np.linalg.norm(final_traj[t+period] - final_traj[t])
                if diff > 1e-4:
                    is_periodic = False
                    break

            if is_periodic:
                print(f"  → PERIOD-{period} detected!")

                # Print one cycle
                print(f"  → Orbit points (linear space):")
                for p in range(period):
                    pt = final_traj[-period + p]
                    print(f"      {p+1}: {pt}")

                break
        else:
            print(f"  → No clear period 1-30 detected")
            print(f"  → Final point: {traj[-1]}")

print("\n" + "="*60)
print("Now checking: do period-7 and period-12 coexist?")
print("="*60)

# Run many ICs and count what we find
periods_found = {}

for _ in range(100):
    # Random IC in reasonable range
    ic = np.array([
        np.random.uniform(0.01, 10.0),   # Midge
        np.random.uniform(0.001, 1.0),   # Algae
        np.random.uniform(0.001, 1.0),   # Detritus
    ])

    traj = np.zeros((800, 3))
    traj[0] = ic

    for t in range(799):
        traj[t+1] = ives_model(traj[t], **params)
        if np.any(traj[t+1] < 0) or np.any(np.isnan(traj[t+1])):
            break

    # Check for periodicity
    if len(traj) > 500:
        final_traj = traj[400:]

        for period in range(1, 31):
            if len(final_traj) < period + 50:
                continue

            is_periodic = True
            for t in range(20, len(final_traj) - period):
                diff = np.linalg.norm(final_traj[t+period] - final_traj[t])
                if diff > 1e-4:
                    is_periodic = False
                    break

            if is_periodic:
                if period not in periods_found:
                    periods_found[period] = 0
                periods_found[period] += 1
                break

print("\nPeriod counts from 100 random ICs:")
for period in sorted(periods_found.keys()):
    print(f"  Period-{period}: {periods_found[period]} ICs")
