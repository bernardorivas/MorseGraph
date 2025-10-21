#!/usr/bin/env python3
"""
Visualize basins of attraction for the Ives model with default parameters.
Color trajectories by which attractor they converge to: blue = equilibrium, red = period-12.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.systems import ives_model


def classify_attractor(trajectory, max_period=15, tolerance=1e-6):
    """
    Classify the attractor type.
    Returns: ('equilibrium', 1) or ('periodic', period) or ('unknown', None)
    """
    n = len(trajectory)
    test_start = n // 2

    # Check for equilibrium
    if np.allclose(trajectory[test_start:], trajectory[-1], atol=tolerance):
        return 'equilibrium', 1

    # Check for periodic orbits
    for period in range(2, max_period + 1):
        is_periodic = True
        for t in range(test_start, n - period):
            if np.linalg.norm(trajectory[t + period] - trajectory[t]) > tolerance:
                is_periodic = False
                break
        if is_periodic:
            return 'periodic', period

    return 'unknown', None


def run_trajectory(x0, n_steps=1000, transient=500):
    """Run trajectory using default parameters."""
    trajectory = [x0]
    x = x0.copy()

    for _ in range(n_steps):
        x = ives_model(x)
        trajectory.append(x)

    trajectory = np.array(trajectory)
    return trajectory, trajectory[transient:]


def main():
    print("=" * 70)
    print("IVES MODEL - BASINS OF ATTRACTION")
    print("Default parameters: c = 10^-6.435, d = 0.5517")
    print("=" * 70)

    # Generate 100 random initial conditions
    np.random.seed(42)
    n_samples = 100

    print(f"\nGenerating {n_samples} random initial conditions...")
    initial_conditions = []
    for _ in range(n_samples):
        # Sample from a broad region of state space
        ic = np.random.uniform(0.01, 3.0, size=3)
        initial_conditions.append(ic)

    # Run all trajectories
    print("Running trajectories and classifying attractors...")

    equilibrium_ics = []
    period12_ics = []
    other_ics = []

    for i, x0 in enumerate(initial_conditions):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{n_samples}")

        full_traj, steady_traj = run_trajectory(x0, n_steps=1000, transient=500)
        att_type, period = classify_attractor(steady_traj, max_period=15)

        if att_type == 'equilibrium':
            equilibrium_ics.append((x0, full_traj))
        elif att_type == 'periodic' and period == 12:
            period12_ics.append((x0, full_traj))
        else:
            other_ics.append((x0, full_traj, att_type, period))

    # Print results
    print(f"\n" + "=" * 70)
    print("RESULTS:")
    print(f"  Converge to EQUILIBRIUM: {len(equilibrium_ics)} / {n_samples} ({100*len(equilibrium_ics)/n_samples:.1f}%)")
    print(f"  Converge to PERIOD-12:   {len(period12_ics)} / {n_samples} ({100*len(period12_ics)/n_samples:.1f}%)")
    if other_ics:
        print(f"  Other attractors:        {len(other_ics)} / {n_samples}")
        for ic, traj, att_type, period in other_ics:
            print(f"    - {att_type} (period={period})")
    print("=" * 70)

    # Create visualization
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: 3D phase space with trajectories
    ax1 = fig.add_subplot(131, projection='3d')

    # Plot equilibrium trajectories in blue
    for x0, traj in equilibrium_ics:
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color='blue', alpha=0.2, linewidth=0.5)

    # Plot period-12 trajectories in red
    for x0, traj in period12_ics:
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color='red', alpha=0.3, linewidth=0.8)

    # Plot other trajectories in gray
    for x0, traj, _, _ in other_ics:
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color='gray', alpha=0.3, linewidth=0.5)

    ax1.set_xlabel('Midge')
    ax1.set_ylabel('Algae')
    ax1.set_zlabel('Detritus')
    ax1.set_title(f'Phase Space Trajectories\n(Blue={len(equilibrium_ics)} equilibrium, Red={len(period12_ics)} period-12)')

    # Plot 2: Initial conditions colored by attractor
    ax2 = fig.add_subplot(132, projection='3d')

    # Plot initial conditions
    if equilibrium_ics:
        eq_x0s = np.array([x0 for x0, _ in equilibrium_ics])
        ax2.scatter(eq_x0s[:, 0], eq_x0s[:, 1], eq_x0s[:, 2],
                   c='blue', s=50, alpha=0.6, label=f'→ Equilibrium ({len(equilibrium_ics)})')

    if period12_ics:
        p12_x0s = np.array([x0 for x0, _ in period12_ics])
        ax2.scatter(p12_x0s[:, 0], p12_x0s[:, 1], p12_x0s[:, 2],
                   c='red', s=50, alpha=0.6, label=f'→ Period-12 ({len(period12_ics)})')

    if other_ics:
        other_x0s = np.array([x0 for x0, _, _, _ in other_ics])
        ax2.scatter(other_x0s[:, 0], other_x0s[:, 1], other_x0s[:, 2],
                   c='gray', s=50, alpha=0.6, label=f'→ Other ({len(other_ics)})')

    ax2.set_xlabel('Midge')
    ax2.set_ylabel('Algae')
    ax2.set_zlabel('Detritus')
    ax2.set_title('Initial Conditions\n(colored by final attractor)')
    ax2.legend()

    # Plot 3: 2D projection (Midge vs Algae) showing basins
    ax3 = fig.add_subplot(133)

    if equilibrium_ics:
        eq_x0s = np.array([x0 for x0, _ in equilibrium_ics])
        ax3.scatter(eq_x0s[:, 0], eq_x0s[:, 1],
                   c='blue', s=60, alpha=0.6, edgecolors='darkblue', linewidths=0.5,
                   label=f'→ Equilibrium ({len(equilibrium_ics)})')

    if period12_ics:
        p12_x0s = np.array([x0 for x0, _ in period12_ics])
        ax3.scatter(p12_x0s[:, 0], p12_x0s[:, 1],
                   c='red', s=60, alpha=0.6, edgecolors='darkred', linewidths=0.5,
                   label=f'→ Period-12 ({len(period12_ics)})')

    if other_ics:
        other_x0s = np.array([x0 for x0, _, _, _ in other_ics])
        ax3.scatter(other_x0s[:, 0], other_x0s[:, 1],
                   c='gray', s=60, alpha=0.6, edgecolors='black', linewidths=0.5,
                   label=f'→ Other ({len(other_ics)})')

    ax3.set_xlabel('Midge (initial)')
    ax3.set_ylabel('Algae (initial)')
    ax3.set_title('Basins of Attraction\n(Midge vs Algae projection)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()

    # Save figure
    filename = "ives_basins_of_attraction.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")

    plt.show()

    print("\nDone!")


if __name__ == '__main__':
    main()
