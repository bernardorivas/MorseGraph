#!/usr/bin/env python3
"""Quick check of what happens with q=0.738"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MorseGraph.systems import ives_model_log

params_738 = {
    'r1': 3.873,
    'r2': 11.746,
    'c': 10**-6.435,
    'd': 0.5517,
    'p': 0.06659,
    'q': 0.738,
    'offset': 0.001
}

# Test a few ICs
test_ics = [
    np.array([0.8, 0.2, 0.4]),  # Near equilibrium (from q=0.9026)
    np.array([0.0, -2.0, 0.0]),  # Floor region
    np.array([0.5, -1.0, 0.5]),  # Middle of domain
]

print("Quick test with q=0.738")
print("="*60)

for i, ic in enumerate(test_ics):
    print(f"\nIC {i+1}: {ic}")

    # Simulate
    traj = np.zeros((1000, 3))
    traj[0] = ic

    for t in range(999):
        traj[t+1] = ives_model_log(traj[t], **params_738)

        if np.any(np.isnan(traj[t+1])) or np.any(np.isinf(traj[t+1])):
            print(f"  → DIVERGED at step {t+1}")
            traj = traj[:t+1]
            break
    else:
        # Check last 100 points - are they settling?
        last_100 = traj[-100:]
        variance = np.var(last_100, axis=0)
        print(f"  → Completed 1000 steps")
        print(f"  → Final point: {traj[-1]}")
        print(f"  → Variance (last 100): {variance}")

        # Check for periodicity
        found_period = False
        for period in range(1, 30):
            if len(traj) < 900 + period:
                continue
            diff = np.linalg.norm(traj[-1] - traj[-1-period])
            if diff < 1e-3:
                print(f"  → PERIOD-{period} found (tolerance 1e-3)")
                found_period = True
                break

        if not found_period:
            print(f"  → No clear period detected (checked 1-30)")

print("\n" + "="*60)
print("Creating trajectory plots...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for i, ic in enumerate(test_ics):
    traj = np.zeros((1000, 3))
    traj[0] = ic

    for t in range(999):
        traj[t+1] = ives_model_log(traj[t], **params_738)
        if np.any(np.isnan(traj[t+1])) or np.any(np.isinf(traj[t+1])):
            traj = traj[:t+1]
            break

    ax = axes[i]
    t_vals = np.arange(len(traj))
    ax.plot(t_vals, traj[:, 0], label='Midge', alpha=0.7)
    ax.plot(t_vals, traj[:, 1], label='Algae', alpha=0.7)
    ax.plot(t_vals, traj[:, 2], label='Detritus', alpha=0.7)
    ax.set_xlabel('Time step', fontsize=11)
    ax.set_ylabel('log₁₀(Population)', fontsize=11)
    ax.set_title(f'IC {i+1}: {ic}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/q738_trajectories.png', dpi=150)
print("✓ Saved to results/q738_trajectories.png")
