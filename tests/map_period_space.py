#!/usr/bin/env python3
"""
Map the full (c, d) parameter space to see where different periods occur.
This will help us understand Figure 3a.
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.systems import ives_model_log


def check_period_fast(c_val, d_val, max_period=20, n_steps=600, transient=200):
    """Quickly check period using one IC away from equilibrium."""

    params = {
        'r1': 3.873,
        'r2': 11.746,
        'c': c_val,
        'd': d_val,
        'p': 0.06659,
        'q': 0.9026,
        'offset': 0.001
    }

    # Use IC likely to hit periodic orbit
    x0 = np.array([0.5, -2.0, 0.2])

    # Integrate
    x = x0.copy()
    for _ in range(transient):
        x = ives_model_log(x, **params)

    trajectory = [x]
    for _ in range(n_steps):
        x = ives_model_log(x, **params)
        trajectory.append(x)
    trajectory = np.array(trajectory)

    # Check periods
    for period in range(1, max_period + 1):
        is_periodic = True
        for t in range(n_steps // 2, n_steps - period):
            if np.linalg.norm(trajectory[t + period] - trajectory[t]) > 1e-4:
                is_periodic = False
                break
        if is_periodic:
            return period

    return 0  # No period found (chaotic or > max_period)


# Create grid
n_c = 50
n_d = 50

c_values = np.logspace(-9, -3, n_c)  # 10^-9 to 10^-3
d_values = np.linspace(0.0, 1.0, n_d)  # 0 to 1

print(f"Mapping {n_c} x {n_d} = {n_c * n_d} parameter combinations")
print(f"c range: [{c_values[0]:.2e}, {c_values[-1]:.2e}]")
print(f"d range: [{d_values[0]:.2f}, {d_values[-1]:.2f}]")
print()

period_map = np.zeros((n_d, n_c))

total = n_c * n_d
for i, d_val in enumerate(d_values):
    for j, c_val in enumerate(c_values):
        period = check_period_fast(c_val, d_val)
        period_map[i, j] = period

        if (i * n_c + j + 1) % 250 == 0:
            print(f"  Progress: {i * n_c + j + 1}/{total}")

print("\nCreating visualization...")

# Plot
fig, ax = plt.subplots(figsize=(12, 8))

# Use discrete colormap for periods
im = ax.pcolormesh(c_values, d_values, period_map,
                   shading='auto', cmap='tab20', vmin=0, vmax=20)

ax.set_xscale('log')
ax.set_xlabel('c (Resource input)', fontsize=12)
ax.set_ylabel('d (Detritus retention)', fontsize=12)
ax.set_title('Period Map of Ives Model', fontsize=14, fontweight='bold')

# Mark fitted parameters
c_fitted = 10**(-6.435)
d_fitted = 0.5517
ax.plot(c_fitted, d_fitted, 'kx', markersize=15, markeredgewidth=3,
        label=f'Fitted params\n(c={c_fitted:.2e}, d={d_fitted:.3f})')

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Period', fontsize=12)

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/period_map.png', dpi=150, bbox_inches='tight')
print("✓ Saved to results/period_map.png")

# Print summary
print("\nPeriod summary:")
unique_periods = np.unique(period_map)
for p in sorted(unique_periods):
    if p == 0:
        count = np.sum(period_map == 0)
        print(f"  No period/chaotic: {count} points ({100*count/period_map.size:.1f}%)")
    else:
        count = np.sum(period_map == int(p))
        print(f"  Period {int(p)}: {count} points ({100*count/period_map.size:.1f}%)")

# Find period-12 locations
if 12 in unique_periods:
    print("\n✓ Period-12 found at:")
    idx = np.where(period_map == 12)
    for i, j in zip(idx[0][:10], idx[1][:10]):  # Show first 10
        print(f"    c = {c_values[j]:.3e}, d = {d_values[i]:.3f}")
else:
    print("\n✗ Period-12 NOT found in this parameter range")

print(f"\nAt fitted parameters (c={c_fitted:.2e}, d={d_fitted:.3f}):")
# Find closest grid point
i_fit = np.argmin(np.abs(d_values - d_fitted))
j_fit = np.argmin(np.abs(c_values - c_fitted))
print(f"  Period = {int(period_map[i_fit, j_fit])}")
