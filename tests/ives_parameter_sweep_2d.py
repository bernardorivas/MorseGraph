#!/usr/bin/env python3
"""
Recreate Figure 3a from Ives et al. (2008) Nature paper.

Parameter space sweep over:
- x-axis: d (detritus retention) in [0, 1]
- y-axis: c (resource input) in [10^-9, 10^-3] (log scale)

Color coding:
- Black: Single stability (equilibrium only)
- Color: Bistability (equilibrium + periodic orbit)
- Gradient: Different periods (darker = higher period)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import sys
import os
from typing import Tuple, Optional, List
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.systems import ives_model_log


def estimate_domain_bounds(params: dict, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate domain bounds by sampling random trajectories.

    Returns:
        lower_bounds, upper_bounds (both in log scale)
    """
    # Sample random initial conditions in a broad range
    np.random.seed(42)

    all_points = []
    for _ in range(n_samples):
        # Start with broad initial guess
        ic = np.array([
            np.random.uniform(-2, 2),   # log(Midge)
            np.random.uniform(-5, 2),   # log(Algae)
            np.random.uniform(-2, 2)    # log(Detritus)
        ])

        # Simulate
        trajectory = [ic]
        for _ in range(100):
            try:
                next_pt = ives_model_log(trajectory[-1], **params)
                if np.any(np.isnan(next_pt)) or np.any(np.isinf(next_pt)):
                    break
                trajectory.append(next_pt)
            except:
                break

        all_points.extend(trajectory[-50:])  # Keep last 50 points

    if not all_points:
        # Fallback to default
        return np.array([-1, -4, -1]), np.array([2, 1, 1])

    all_points = np.array(all_points)

    # Compute bounds with some padding
    lower = all_points.min(axis=0) - 0.5
    upper = all_points.max(axis=0) + 0.5

    return lower, upper


def detect_attractors(params: dict,
                     n_ics: int = 30,
                     n_steps: int = 1000,
                     transient: int = 500,
                     tol: float = 1e-4,
                     max_period: int = 20) -> dict:
    """
    Detect attractors for given parameters.

    Returns:
        {
            'equilibrium': True/False,
            'equilibrium_point': np.array or None,
            'periodic': True/False,
            'period': int or None,
            'orbit': np.array or None,
            'type': 'equilibrium', 'periodic', 'bistable', or 'none'
        }
    """
    # Estimate domain bounds for this parameter set
    lower, upper = estimate_domain_bounds(params, n_samples=10)

    # Sample initial conditions
    np.random.seed(None)  # Different ICs each time
    ics = []
    for _ in range(n_ics):
        ic = np.array([
            np.random.uniform(lower[0], upper[0]),
            np.random.uniform(lower[1], upper[1]),
            np.random.uniform(lower[2], upper[2])
        ])
        ics.append(ic)

    found_equilibria = []
    found_periodic = []

    for ic in ics:
        # Simulate
        x = ic.copy()
        try:
            for _ in range(n_steps):
                x = ives_model_log(x, **params)
                if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                    break
        except:
            continue

        # Check for equilibrium (period-1)
        x_next = ives_model_log(x, **params)
        if np.linalg.norm(x_next - x) < tol:
            # Found equilibrium
            # Check if we already have this one
            is_new = True
            for eq in found_equilibria:
                if np.linalg.norm(eq - x) < 0.1:  # Same equilibrium
                    is_new = False
                    break
            if is_new:
                found_equilibria.append(x.copy())
            continue

        # Check for periodic orbit
        for period in range(2, max_period + 1):
            x_test = x.copy()
            for _ in range(period):
                x_test = ives_model_log(x_test, **params)

            if np.linalg.norm(x_test - x) < tol:
                # Found periodic orbit
                # Check if new
                is_new = True
                for orb_info in found_periodic:
                    if orb_info['period'] == period:
                        if np.linalg.norm(orb_info['orbit'][0] - x) < 0.1:
                            is_new = False
                            break

                if is_new:
                    # Extract full orbit
                    orbit_points = [x.copy()]
                    x_orb = x.copy()
                    for _ in range(period - 1):
                        x_orb = ives_model_log(x_orb, **params)
                        orbit_points.append(x_orb.copy())

                    found_periodic.append({
                        'period': period,
                        'orbit': np.array(orbit_points)
                    })
                break

    # Classify
    has_eq = len(found_equilibria) > 0
    has_periodic = len(found_periodic) > 0

    if has_eq and has_periodic:
        attractor_type = 'bistable'
    elif has_eq:
        attractor_type = 'equilibrium'
    elif has_periodic:
        attractor_type = 'periodic'
    else:
        attractor_type = 'none'

    # Get representative periodic orbit (lowest period)
    period = None
    orbit = None
    if has_periodic:
        found_periodic.sort(key=lambda x: x['period'])
        period = found_periodic[0]['period']
        orbit = found_periodic[0]['orbit']

    return {
        'equilibrium': has_eq,
        'equilibrium_point': found_equilibria[0] if has_eq else None,
        'periodic': has_periodic,
        'period': period,
        'orbit': orbit,
        'type': attractor_type,
        'n_equilibria': len(found_equilibria),
        'n_periodic': len(found_periodic)
    }


def parameter_sweep(
    d_range: Tuple[float, float],
    c_range: Tuple[float, float],
    n_d: int,
    n_c: int,
    fixed_params: dict
) -> np.ndarray:
    """
    Sweep parameter space and detect attractors.

    Returns:
        results: array of shape (n_d, n_c) with entries containing attractor info
    """
    d_values = np.linspace(d_range[0], d_range[1], n_d)
    c_values = np.logspace(np.log10(c_range[0]), np.log10(c_range[1]), n_c)

    results = np.empty((n_d, n_c), dtype=object)

    total = n_d * n_c
    with tqdm(total=total, desc="Parameter sweep") as pbar:
        for i, d in enumerate(d_values):
            for j, c in enumerate(c_values):
                # Update parameters
                params = fixed_params.copy()
                params['d'] = d
                params['c'] = c

                # Detect attractors
                result = detect_attractors(params)
                results[i, j] = result

                pbar.update(1)
                pbar.set_postfix({
                    'd': f'{d:.3f}',
                    'c': f'{c:.2e}',
                    'type': result['type'][:4]
                })

    return results, d_values, c_values


def plot_parameter_space(
    results: np.ndarray,
    d_values: np.ndarray,
    c_values: np.ndarray,
    output_path: str
):
    """
    Plot parameter space similar to Figure 3a.
    """
    n_d, n_c = results.shape

    # Create arrays for plotting
    attractor_type = np.zeros((n_d, n_c))  # 0=none, 1=eq, 2=periodic, 3=bistable
    period_value = np.zeros((n_d, n_c))

    for i in range(n_d):
        for j in range(n_c):
            r = results[i, j]
            if r['type'] == 'equilibrium':
                attractor_type[i, j] = 1
                period_value[i, j] = 1
            elif r['type'] == 'periodic':
                attractor_type[i, j] = 2
                period_value[i, j] = r['period'] if r['period'] else 0
            elif r['type'] == 'bistable':
                attractor_type[i, j] = 3
                period_value[i, j] = r['period'] if r['period'] else 0
            else:
                attractor_type[i, j] = 0
                period_value[i, j] = 0

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each category
    D, C = np.meshgrid(d_values, c_values, indexing='ij')

    # 1. Equilibrium only (black)
    mask_eq = attractor_type == 1
    if mask_eq.any():
        ax.scatter(D[mask_eq], C[mask_eq], c='black', s=30,
                  label='Equilibrium only', marker='s', alpha=0.8)

    # 2. Bistable (colored by period)
    mask_bistable = attractor_type == 3
    if mask_bistable.any():
        periods_bistable = period_value[mask_bistable]
        sc = ax.scatter(D[mask_bistable], C[mask_bistable],
                       c=periods_bistable, s=50, cmap='coolwarm',
                       vmin=2, vmax=20, marker='o',
                       edgecolors='black', linewidths=0.5,
                       label='Bistable')

        # Add colorbar for period
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Period of periodic orbit', fontsize=11)

    # 3. Periodic only (lighter color)
    mask_periodic = attractor_type == 2
    if mask_periodic.any():
        periods_periodic = period_value[mask_periodic]
        ax.scatter(D[mask_periodic], C[mask_periodic],
                  c=periods_periodic, s=30, cmap='coolwarm',
                  vmin=2, vmax=20, marker='^', alpha=0.6,
                  label='Periodic only')

    # Labels and formatting
    ax.set_xlabel('Detritus retention (d)', fontsize=12)
    ax.set_ylabel('Resource input (c)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Ives Model Parameter Space (Recreating Figure 3a)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Mark the default parameter value
    default_d = 0.5517
    default_c = 3.67e-7
    ax.plot(default_d, default_c, 'w*', markersize=20,
           markeredgecolor='black', markeredgewidth=2,
           label=f'Default (d={default_d:.4f}, c={default_c:.2e})')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved parameter space plot to: {output_path}")

    return fig


def main():
    print("="*80)
    print("IVES MODEL - PARAMETER SPACE SWEEP")
    print("Recreating Figure 3a from Ives et al. (2008) Nature")
    print("="*80)

    # Fixed parameters
    fixed_params = {
        'r1': 3.873,
        'r2': 11.746,
        'p': 0.06659,
        'q': 0.9026,
        'offset': 0.001
    }

    # Parameter ranges
    d_range = (0.0, 1.0)        # Detritus retention
    c_range = (1e-9, 1e-3)      # Resource input (log scale)

    # Grid resolution
    n_d = 100   # 100 points in d
    n_c = 100   # 100 points in c (log-spaced)

    print(f"\nParameter ranges:")
    print(f"  d (detritus retention): [{d_range[0]}, {d_range[1]}]")
    print(f"  c (resource input): [{c_range[0]:.1e}, {c_range[1]:.1e}] (log scale)")
    print(f"\nGrid resolution: {n_d} × {n_c} = {n_d * n_c} parameter combinations")
    print(f"\nThis will take approximately {n_d * n_c * 3 / 60:.1f} minutes...")

    # Run sweep
    results, d_values, c_values = parameter_sweep(
        d_range, c_range, n_d, n_c, fixed_params
    )

    # Count results
    n_eq = sum(1 for i in range(n_d) for j in range(n_c) if results[i,j]['type'] == 'equilibrium')
    n_per = sum(1 for i in range(n_d) for j in range(n_c) if results[i,j]['type'] == 'periodic')
    n_bi = sum(1 for i in range(n_d) for j in range(n_c) if results[i,j]['type'] == 'bistable')
    n_none = sum(1 for i in range(n_d) for j in range(n_c) if results[i,j]['type'] == 'none')

    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  Equilibrium only: {n_eq}/{n_d*n_c} ({100*n_eq/(n_d*n_c):.1f}%)")
    print(f"  Periodic only:    {n_per}/{n_d*n_c} ({100*n_per/(n_d*n_c):.1f}%)")
    print(f"  Bistable:         {n_bi}/{n_d*n_c} ({100*n_bi/(n_d*n_c):.1f}%)")
    print(f"  None/Other:       {n_none}/{n_d*n_c} ({100*n_none/(n_d*n_c):.1f}%)")

    # Periods found in bistable regions
    bistable_periods = [results[i,j]['period'] for i in range(n_d) for j in range(n_c)
                       if results[i,j]['type'] == 'bistable' and results[i,j]['period']]
    if bistable_periods:
        unique_periods = sorted(set(bistable_periods))
        print(f"\n  Periods found in bistable regions: {unique_periods}")
        for p in unique_periods:
            count = bistable_periods.count(p)
            print(f"    Period-{p}: {count} parameter combinations")

    # Plot
    output_path = os.path.join(os.path.dirname(__file__), 'ives_parameter_space_2d.png')
    plot_parameter_space(results, d_values, c_values, output_path)

    # Save data
    data_path = os.path.join(os.path.dirname(__file__), 'ives_parameter_space_2d.npz')
    np.savez(data_path,
             results=results,
             d_values=d_values,
             c_values=c_values,
             fixed_params=fixed_params)
    print(f"✓ Saved data to: {data_path}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
