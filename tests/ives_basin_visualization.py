#!/usr/bin/env python3
"""
Basin of attraction visualization for the Ives model.

Given a fixed parameter value c, this script:
1. Samples ~50k points uniformly in the domain
2. Integrates each for N steps
3. Identifies where trajectories accumulate (attractors)
4. Visualizes the basin structure

This helps identify:
- Equilibrium points
- Periodic orbits (e.g., period-12)
- Basin boundaries
- Multi-stability

Usage:
    python tests/ives_basin_visualization.py --c 3.5e-7
    python tests/ives_basin_visualization.py --c 3.6e-7 --n-samples 100000
"""

import numpy as np
import os
import sys
import argparse
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.systems import ives_model_log


def generate_uniform_grid(domain_bounds: Tuple[List, List],
                         n_samples: int,
                         seed: int = 42) -> np.ndarray:
    """
    Generate uniform random samples in the domain.

    Args:
        domain_bounds: ((x_min, y_min, z_min), (x_max, y_max, z_max))
        n_samples: Number of samples
        seed: Random seed

    Returns:
        Array of shape (n_samples, 3)
    """
    np.random.seed(seed)

    samples = np.zeros((n_samples, 3))
    for i in range(3):
        samples[:, i] = np.random.uniform(
            domain_bounds[0][i],
            domain_bounds[1][i],
            n_samples
        )

    return samples


def integrate_to_attractor(initial_conditions: np.ndarray,
                           c_value: float,
                           n_steps: int = 600,
                           transient: int = 200,
                           **model_params) -> np.ndarray:
    """
    Integrate initial conditions and return final states.

    Args:
        initial_conditions: Array of shape (N, 3)
        c_value: Parameter value
        n_steps: Total integration steps
        transient: Steps to discard
        **model_params: Additional Ives parameters

    Returns:
        Array of shape (N, 3) with final states
    """
    params = {
        'r1': 3.873,
        'r2': 11.746,
        'c': c_value,
        'd': 0.5517,
        'p': 0.06659,
        'q': 0.9026,
        'offset': 0.001
    }
    params.update(model_params)

    n_samples = len(initial_conditions)
    final_states = np.zeros((n_samples, 3))

    print(f"Integrating {n_samples} initial conditions for {n_steps} steps...")

    for i in tqdm(range(n_samples), desc="Integrating"):
        x = initial_conditions[i]

        # Integrate
        for _ in range(n_steps):
            x = ives_model_log(x, **params)

        final_states[i] = x

    return final_states


def cluster_attractors(final_states: np.ndarray,
                      eps: float = 0.05,
                      min_samples: int = 10) -> Tuple[np.ndarray, Dict]:
    """
    Cluster final states to identify attractors using DBSCAN.

    Args:
        final_states: Array of shape (N, 3)
        eps: DBSCAN neighborhood radius
        min_samples: Minimum points for a cluster

    Returns:
        (labels, attractor_info) where labels[i] is the cluster ID for point i,
        and attractor_info is dict with cluster statistics
    """
    print(f"\nClustering final states (eps={eps}, min_samples={min_samples})...")

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(final_states)
    labels = clustering.labels_

    # Analyze clusters
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)

    print(f"Found {n_clusters} attractors")
    print(f"Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")

    attractor_info = {}
    for label in unique_labels:
        if label == -1:  # Noise
            continue

        mask = labels == label
        cluster_points = final_states[mask]

        attractor_info[label] = {
            'n_points': int(np.sum(mask)),
            'centroid': cluster_points.mean(axis=0),
            'std': cluster_points.std(axis=0),
            'points': cluster_points[:100]  # Store first 100 for visualization
        }

        print(f"\nAttractor {label}:")
        print(f"  Basin size: {attractor_info[label]['n_points']} points "
              f"({100*attractor_info[label]['n_points']/len(labels):.1f}%)")
        print(f"  Centroid: [{attractor_info[label]['centroid'][0]:.4f}, "
              f"{attractor_info[label]['centroid'][1]:.4f}, "
              f"{attractor_info[label]['centroid'][2]:.4f}]")
        print(f"  Std dev:  [{attractor_info[label]['std'][0]:.4f}, "
              f"{attractor_info[label]['std'][1]:.4f}, "
              f"{attractor_info[label]['std'][2]:.4f}]")

        # Check if it might be periodic
        if attractor_info[label]['std'].max() > 0.01:
            print(f"  → Likely PERIODIC ORBIT (high variance)")
        else:
            print(f"  → Likely EQUILIBRIUM (low variance)")

    return labels, attractor_info


def check_periodicity(c_value: float,
                     attractor_center: np.ndarray,
                     max_period: int = 20,
                     n_steps: int = 500,
                     tolerance: float = 1e-4) -> Tuple[bool, int, np.ndarray]:
    """
    Check if an attractor is periodic and find the period.

    Args:
        c_value: Parameter value
        attractor_center: Point on/near the attractor
        max_period: Maximum period to check
        n_steps: Steps to integrate
        tolerance: Period detection tolerance

    Returns:
        (is_periodic, period, orbit_points)
    """
    params = {
        'r1': 3.873,
        'r2': 11.746,
        'c': c_value,
        'd': 0.5517,
        'p': 0.06659,
        'q': 0.9026,
        'offset': 0.001
    }

    # Integrate to get on attractor
    x = attractor_center.copy()
    for _ in range(200):  # Transient
        x = ives_model_log(x, **params)

    # Collect trajectory
    trajectory = [x]
    for _ in range(n_steps):
        x = ives_model_log(x, **params)
        trajectory.append(x)
    trajectory = np.array(trajectory)

    # Check for periodicity
    for period in range(1, max_period + 1):
        # Check if x(t+p) ≈ x(t) for all t in second half
        is_periodic = True
        for t in range(n_steps // 2, n_steps - period):
            if np.linalg.norm(trajectory[t + period] - trajectory[t]) > tolerance:
                is_periodic = False
                break

        if is_periodic:
            # Extract one cycle
            orbit_points = trajectory[-period:]
            return True, period, orbit_points

    return False, 0, np.array([])


def visualize_basins(initial_conditions: np.ndarray,
                    final_states: np.ndarray,
                    labels: np.ndarray,
                    attractor_info: Dict,
                    c_value: float,
                    domain_bounds: Tuple[List, List],
                    output_path: str):
    """
    Create comprehensive basin visualization.
    """
    n_attractors = len(attractor_info)

    # Color map for basins
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_attractors)))

    fig = plt.figure(figsize=(20, 12))

    # ========================================================================
    # Row 1: Initial conditions colored by basin (3D + 2D projections)
    # ========================================================================

    # 3D scatter
    ax = fig.add_subplot(2, 4, 1, projection='3d')
    for label in sorted(attractor_info.keys()):
        mask = labels == label
        ax.scatter(initial_conditions[mask, 0],
                  initial_conditions[mask, 1],
                  initial_conditions[mask, 2],
                  c=[colors[label]], s=1, alpha=0.3)
    ax.set_xlabel('log(Midge)')
    ax.set_ylabel('log(Algae)')
    ax.set_zlabel('log(Detritus)')
    ax.set_title(f'Basins of Attraction (c={c_value:.2e})\n{n_attractors} attractors',
                 fontweight='bold')

    # 2D projections of basins
    projections = [
        (0, 1, 'log(Midge)', 'log(Algae)', 2),
        (0, 2, 'log(Midge)', 'log(Detritus)', 3),
        (1, 2, 'log(Algae)', 'log(Detritus)', 4)
    ]

    for dim1, dim2, label1, label2, subplot_idx in projections:
        ax = fig.add_subplot(2, 4, subplot_idx)
        for label in sorted(attractor_info.keys()):
            mask = labels == label
            ax.scatter(initial_conditions[mask, dim1],
                      initial_conditions[mask, dim2],
                      c=[colors[label]], s=1, alpha=0.3,
                      label=f'Attractor {label}')
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.set_title(f'Basin Projection ({label1} vs {label2})')
        ax.grid(True, alpha=0.3)

    # ========================================================================
    # Row 2: Attractors in state space
    # ========================================================================

    # 3D attractors
    ax = fig.add_subplot(2, 4, 5, projection='3d')
    for label in sorted(attractor_info.keys()):
        info = attractor_info[label]
        # Plot sample points from attractor
        pts = info['points']
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                  c=[colors[label]], s=20, alpha=0.6,
                  label=f"Attr {label} (n={info['n_points']})")
        # Mark centroid
        c = info['centroid']
        ax.scatter([c[0]], [c[1]], [c[2]],
                  c='black', s=200, marker='*', edgecolors='white', linewidths=2)
    ax.set_xlabel('log(Midge)')
    ax.set_ylabel('log(Algae)')
    ax.set_zlabel('log(Detritus)')
    ax.set_title('Attractors in State Space', fontweight='bold')
    ax.legend(fontsize=8)

    # 2D projections of attractors
    for dim1, dim2, label1, label2, subplot_idx in [
        (0, 1, 'log(Midge)', 'log(Algae)', 6),
        (0, 2, 'log(Midge)', 'log(Detritus)', 7),
        (1, 2, 'log(Algae)', 'log(Detritus)', 8)
    ]:
        ax = fig.add_subplot(2, 4, subplot_idx)
        for label in sorted(attractor_info.keys()):
            info = attractor_info[label]
            pts = info['points']
            ax.scatter(pts[:, dim1], pts[:, dim2],
                      c=[colors[label]], s=20, alpha=0.6,
                      label=f'Attractor {label}')
            # Mark centroid
            c = info['centroid']
            ax.scatter([c[dim1]], [c[dim2]],
                      c='black', s=200, marker='*', edgecolors='white', linewidths=2)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.set_title(f'Attractors ({label1} vs {label2})')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved basin visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Basin of attraction visualization for Ives model'
    )

    parser.add_argument('--c', type=float, required=True,
                       help='Parameter c value to analyze')
    parser.add_argument('--n-samples', type=int, default=50000,
                       help='Number of initial conditions (default: 50000)')
    parser.add_argument('--n-steps', type=int, default=600,
                       help='Integration steps (default: 600)')
    parser.add_argument('--domain-min', type=float, nargs=3, default=[-1, -4, -1],
                       help='Domain minimum [x y z] (default: -1 -4 -1)')
    parser.add_argument('--domain-max', type=float, nargs=3, default=[2, 1, 1],
                       help='Domain maximum [x y z] (default: 2 1 1)')
    parser.add_argument('--eps', type=float, default=0.05,
                       help='DBSCAN clustering radius (default: 0.05)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory (default: results)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    domain_bounds = (args.domain_min, args.domain_max)

    print(f"\n{'='*80}")
    print(f"BASIN OF ATTRACTION ANALYSIS")
    print(f"{'='*80}")
    print(f"Parameter c: {args.c:.6e}")
    print(f"Domain: {args.domain_min} to {args.domain_max}")
    print(f"Samples: {args.n_samples:,}")
    print(f"Integration steps: {args.n_steps}")
    print(f"{'='*80}\n")

    # Generate initial conditions
    print("Generating uniform initial conditions...")
    initial_conditions = generate_uniform_grid(domain_bounds, args.n_samples)

    # Integrate to attractors
    final_states = integrate_to_attractor(
        initial_conditions,
        args.c,
        n_steps=args.n_steps
    )

    # Cluster attractors
    labels, attractor_info = cluster_attractors(final_states, eps=args.eps)

    # Check each attractor for periodicity
    print(f"\n{'='*80}")
    print("PERIODICITY ANALYSIS")
    print(f"{'='*80}")

    for label in sorted(attractor_info.keys()):
        info = attractor_info[label]
        print(f"\nAttractor {label}:")

        is_periodic, period, orbit = check_periodicity(
            args.c,
            info['centroid'],
            max_period=20
        )

        if is_periodic:
            print(f"  ✓ PERIODIC ORBIT DETECTED - Period {period}")
            print(f"    Orbit points:")
            for i, pt in enumerate(orbit):
                print(f"      Point {i+1}: [{pt[0]:7.4f}, {pt[1]:7.4f}, {pt[2]:7.4f}]")

            # Save orbit to file
            orbit_file = os.path.join(args.output_dir, f"periodic_orbit_c{args.c:.2e}.txt")
            np.savetxt(orbit_file, orbit,
                      header=f"Period-{period} orbit for c={args.c:.6e}\nColumns: log(Midge), log(Algae), log(Detritus)",
                      fmt='%.8f')
            print(f"    Saved orbit to {orbit_file}")
        else:
            print(f"  Fixed point (equilibrium)")

    # Visualize
    output_path = os.path.join(args.output_dir, f"basins_c{args.c:.2e}.png")
    visualize_basins(
        initial_conditions,
        final_states,
        labels,
        attractor_info,
        args.c,
        domain_bounds,
        output_path
    )

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
