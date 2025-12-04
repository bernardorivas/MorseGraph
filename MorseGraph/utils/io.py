"""
Data I/O utilities for trajectory data and other resources.

This module provides functions for loading and saving trajectory data,
models, training history, and other data structures.
"""

import os
import numpy as np
from typing import Dict, Tuple, List, Optional, Any


def _save_trajectory_data_file(
    filepath: str,
    x_t: np.ndarray,
    x_t_plus_1: np.ndarray,
    trajectories: List[np.ndarray],
    metadata: Dict[str, Any]
) -> None:
    """
    Save trajectory data and metadata to disk in compressed NPZ format.

    Args:
        filepath: Path to save the .npz file
        x_t: Current states array
        x_t_plus_1: Next states array
        trajectories: List of trajectory arrays
        metadata: Dictionary of metadata to save

    Example:
        >>> _save_trajectory_data_file(
        ...     'data.npz', X, Y, trajs,
        ...     {'n_trajectories': 100, 'total_time': 10.0}
        ... )
    """
    save_dict = {
        'x_t': x_t,
        'x_t_plus_1': x_t_plus_1,
        'n_trajectories': len(trajectories),
        **metadata
    }

    # Save trajectories as separate arrays
    for i, traj in enumerate(trajectories):
        save_dict[f'trajectory_{i}'] = traj

    np.savez_compressed(filepath, **save_dict)
    print(f"  Saved training data to: {filepath}")


def _load_trajectory_data_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Dict[str, Any]]:
    """
    Load trajectory data and metadata from disk.

    Args:
        filepath: Path to the .npz file

    Returns:
        x_t: Current states array
        x_t_plus_1: Next states array
        trajectories: List of trajectory arrays
        metadata: Dictionary of metadata

    Example:
        >>> X, Y, trajs, meta = _load_trajectory_data_file('data.npz')
    """
    data = np.load(filepath)

    x_t = data['x_t']
    x_t_plus_1 = data['x_t_plus_1']
    n_trajectories = int(data['n_trajectories'])

    # Reconstruct trajectories list
    trajectories = []
    for i in range(n_trajectories):
        trajectories.append(data[f'trajectory_{i}'])

    # Extract metadata
    metadata = {
        'n_trajectories': n_trajectories,
        'n_points': int(data['n_points']),
        'random_seed': int(data['random_seed'])
    }
    if 'total_time' in data:
        metadata['total_time'] = float(data['total_time'])
    if 'sampling_time' in data:
        metadata['sampling_time'] = float(data['sampling_time'])
    if 'timeskip' in data:
        metadata['timeskip'] = float(data['timeskip'])

    print(f"  Loaded training data from: {filepath}")
    print(f"  Metadata: {metadata}")

    return x_t, x_t_plus_1, trajectories, metadata


def get_next_run_number(base_dir: str) -> int:
    """
    Find the next available run number in a directory containing run_XXX folders.

    Scans the base directory for existing folders named 'run_001', 'run_002', etc.,
    and returns the next available number in the sequence.

    Args:
        base_dir: Directory to scan for existing run folders

    Returns:
        Next available run number (starting from 1)

    Example:
        >>> # Directory contains run_001, run_002, run_005
        >>> get_next_run_number('/path/to/experiments')
        6
        >>> # Empty directory
        >>> get_next_run_number('/path/to/new_experiments')
        1
    """
    if not os.path.exists(base_dir):
        return 1

    existing_runs = [
        d for d in os.listdir(base_dir)
        if d.startswith('run_') and os.path.isdir(os.path.join(base_dir, d))
    ]
    if not existing_runs:
        return 1

    # Extract run numbers
    run_numbers = []
    for run_dir in existing_runs:
        try:
            num = int(run_dir.split('_')[1])
            run_numbers.append(num)
        except (IndexError, ValueError):
            continue

    return max(run_numbers) + 1 if run_numbers else 1


__all__ = [
    '_save_trajectory_data_file',
    '_load_trajectory_data_file',
    'get_next_run_number',
]

