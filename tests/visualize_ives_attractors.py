#!/usr/bin/env python3
"""
Visualize Ives model trajectories with DEFAULT parameters.
Shows how different initial conditions converge to attractors.

Default parameters from ives_model:
  r1: 3.873
  r2: 11.746
  c: 10^-6.435 ≈ 3.67e-7
  d: 0.5517
  p: 0.06659
  q: 0.9026
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
    Classify the attractor from a trajectory.

    Returns:
        (attractor_type, period, attractor_points)
        where attractor_type is 'equilibrium', 'periodic', or 'unknown'
    """
    # Check last portion of trajectory for periodicity
    n = len(trajectory)
    test_start = n // 2

    # Check for equilibrium first (period-1)
    if np.allclose(trajectory[test_start:], trajectory[-1], atol=tolerance):
        return 'equilibrium', 1, np.array([trajectory[-1]])

    # Check for periodic orbits
    for period in range(2, max_period + 1):
        is_periodic = True

        # Check if trajectory repeats with this period
        for t in range(test_start, n - period):
            if np.linalg.norm(trajectory[t + period] - trajectory[t]) > tolerance:
                is_periodic = False
                break

        if is_periodic:
            # Extract one cycle of the periodic orbit
            attractor_points = trajectory[-period:]
            return 'periodic', period, attractor_points

    return 'unknown', None, None


def run_trajectory(x0, n_steps=500, transient=200):
    """
    Run trajectory from initial condition using DEFAULT parameters.

    Returns:
        full_trajectory: array of shape (n_steps, 3)
        steady_trajectory: array of shape (n_steps - transient, 3) after transient
    """
    trajectory = [x0]
    x = x0.copy()

    for i in range(n_steps):
        x = ives_model(x)  # Using default parameters
        trajectory.append(x)

    trajectory = np.array(trajectory)
    return trajectory, trajectory[transient:]


def main():
    print("=" * 70)
    print("IVES MODEL ATTRACTOR VISUALIZATION (DEFAULT PARAMETERS)")
    print("=" * 70)
    print("\nDefault parameters:")
    print("  r1 = 3.873")
    print("  r2 = 11.746")
    print("  c  = 10^-6.435 ≈ 3.67e-7")
    print("  d  = 0.5517")
    print("  p  = 0.06659")
    print("  q  = 0.9026")
    print()

    # Test initial conditions spread across state space
    np.random.seed(42)
    n_ics = 40

    # Generate initial conditions in different regions
    initial_conditions = []

    # Around potential equilibrium regions
    for _ in range(n_ics // 4):
        ic = np.array([
            np.random.uniform(0.1, 0.5),   # midge
            np.random.uniform(0.05, 0.15), # algae
            np.random.uniform(0.2, 0.6),   # detritus
        ])
        initial_conditions.append(ic)

    # Random throughout state space (smaller values)
    for _ in range(n_ics // 4):
        ic = np.random.uniform(0.01, 1.0, size=3)
        initial_conditions.append(ic)

    # Random throughout state space (larger values)
    for _ in range(n_ics // 4):
        ic = np.random.uniform(0.5, 2.0, size=3)
        initial_conditions.append(ic)

    # Some very small initial conditions
    for _ in range(n_ics - 3 * (n_ics // 4)):
        ic = np.random.uniform(0.001, 0.1, size=3)
        initial_conditions.append(ic)

    print(f"Running {len(initial_conditions)} trajectories...")

    trajectories = []
    attractor_types = []
    periods = []
    attractor_points_list = []

    for i, x0 in enumerate(initial_conditions):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(initial_conditions)}")

        full_traj, steady_traj = run_trajectory(x0, n_steps=1000, transient=500)
        att_type, period, att_points = classify_attractor(steady_traj, max_period=15)

        trajectories.append(full_traj)
        attractor_types.append(att_type)
        periods.append(period)
        attractor_points_list.append(att_points)

    # Count attractors
    eq_count = sum(1 for t in attractor_types if t == 'equilibrium')
    periodic_counts = {}
    for att_type, period in zip(attractor_types, periods):
        if att_type == 'periodic':
            periodic_counts[period] = periodic_counts.get(period, 0) + 1
    unknown_count = sum(1 for t in attractor_types if t == 'unknown')

    print(f"\nResults from {len(initial_conditions)} initial conditions:")
    print(f"  Equilibria: {eq_count}")
    for period in sorted(periodic_counts.keys()):
        print(f"  Period-{period}: {periodic_counts[period]}")
    if unknown_count > 0:
        print(f"  Unknown/Chaotic: {unknown_count}")

    # Print example attractor points
    if eq_count > 0:
        eq_idx = attractor_types.index('equilibrium')
        print(f"\nExample equilibrium: {attractor_points_list[eq_idx][0]}")

    for period in sorted(periodic_counts.keys()):
        per_idx = [i for i, (t, p) in enumerate(zip(attractor_types, periods))
                   if t == 'periodic' and p == period][0]
        print(f"\nExample period-{period} orbit (first 3 points):")
        for pt in attractor_points_list[per_idx][:3]:
            print(f"  {pt}")

    # Create visualization
    fig = plt.figure(figsize=(18, 5))

    # Color map
    colors_map = {
        'equilibrium': 'blue',
        'periodic': 'red',
        'unknown': 'gray'
    }

    # Get period-based colors for periodic orbits
    unique_periods = sorted(set(p for p in periods if p is not None and p > 1))
    period_colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_periods), 1)))

    # 3D phase space
    ax1 = fig.add_subplot(131, projection='3d')

    for traj, att_type, period, att_points in zip(trajectories, attractor_types,
                                                   periods, attractor_points_list):
        if att_type == 'equilibrium':
            color = 'blue'
            alpha = 0.4
        elif att_type == 'periodic':
            period_idx = unique_periods.index(period) if period in unique_periods else 0
            color = period_colors[period_idx]
            alpha = 0.7
        else:
            color = 'gray'
            alpha = 0.3

        # Plot trajectory
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=color, alpha=alpha, linewidth=0.8)

        # Mark attractor points
        if att_points is not None and len(att_points) > 0:
            marker = 'o' if att_type == 'equilibrium' else '*'
            size = 80 if att_type == 'equilibrium' else 150
            ax1.scatter(att_points[:, 0], att_points[:, 1], att_points[:, 2],
                       color=color, marker=marker, s=size,
                       edgecolors='black', linewidths=1, zorder=10)

    ax1.set_xlabel('Midge')
    ax1.set_ylabel('Algae')
    ax1.set_zlabel('Detritus')
    ax1.set_title('3D Phase Space (Default Parameters)')

    # 2D projection: Midge vs Algae
    ax2 = fig.add_subplot(132)

    # Track which labels we've added
    labels_added = set()

    for idx, (traj, att_type, period, att_points) in enumerate(zip(trajectories, attractor_types,
                                                                     periods, attractor_points_list)):
        if att_type == 'equilibrium':
            color = 'blue'
            alpha = 0.4
            label = 'Equilibrium' if 'Equilibrium' not in labels_added else None
            if label:
                labels_added.add('Equilibrium')
        elif att_type == 'periodic':
            period_idx = unique_periods.index(period) if period in unique_periods else 0
            color = period_colors[period_idx]
            alpha = 0.7
            label_key = f'Period-{period}'
            label = label_key if label_key not in labels_added else None
            if label:
                labels_added.add(label_key)
        else:
            color = 'gray'
            alpha = 0.3
            label = None

        ax2.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=0.8)

        if att_points is not None and len(att_points) > 0:
            marker = 'o' if att_type == 'equilibrium' else '*'
            size = 80 if att_type == 'equilibrium' else 150
            ax2.scatter(att_points[:, 0], att_points[:, 1],
                       color=color, marker=marker, s=size,
                       edgecolors='black', linewidths=1, zorder=10, label=label)

    ax2.set_xlabel('Midge')
    ax2.set_ylabel('Algae')
    ax2.set_title('Midge vs Algae Projection')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 2D projection: Midge vs Detritus
    ax3 = fig.add_subplot(133)

    for traj, att_type, period, att_points in zip(trajectories, attractor_types,
                                                   periods, attractor_points_list):
        if att_type == 'equilibrium':
            color = 'blue'
            alpha = 0.4
        elif att_type == 'periodic':
            period_idx = unique_periods.index(period) if period in unique_periods else 0
            color = period_colors[period_idx]
            alpha = 0.7
        else:
            color = 'gray'
            alpha = 0.3

        ax3.plot(traj[:, 0], traj[:, 2], color=color, alpha=alpha, linewidth=0.8)

        if att_points is not None and len(att_points) > 0:
            marker = 'o' if att_type == 'equilibrium' else '*'
            size = 80 if att_type == 'equilibrium' else 150
            ax3.scatter(att_points[:, 0], att_points[:, 2],
                       color=color, marker=marker, s=size,
                       edgecolors='black', linewidths=1, zorder=10)

    ax3.set_xlabel('Midge')
    ax3.set_ylabel('Detritus')
    ax3.set_title('Midge vs Detritus Projection')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    filename = "ives_attractors_default_params.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")

    plt.show()

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
