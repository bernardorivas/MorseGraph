"""
Ground Truth Morse Graph Computation for Leslie Map 3D

This module handles computation of the ground truth Morse graph
for the Leslie map 3D system using CMGDB.
"""

import CMGDB
import numpy as np
import os
import time
from functools import partial
from MorseGraph.systems import leslie_map_3d
from .plotting import plot_morse_graph, plot_barycenters_3d
from .config import Config


def compute_ground_truth_morse_graph(
        theta_1=Config.THETA_1,
        theta_2=Config.THETA_2,
        theta_3=Config.THETA_3,
        survival_1=Config.SURVIVAL_1,
        survival_2=Config.SURVIVAL_2,
        lower_bounds=Config.LOWER_BOUNDS,
        upper_bounds=Config.UPPER_BOUNDS,
        subdiv_min=Config.SUBDIV_MIN,
        subdiv_max=Config.SUBDIV_MAX,
        subdiv_init=Config.SUBDIV_INIT,
        subdiv_limit=Config.SUBDIV_LIMIT,
        output_dir=None,
        verbose=True):
    """
    Compute the ground truth Morse graph for the Leslie map.

    Args:
        theta_1: Fertility for age class 0
        theta_2: Fertility for age class 1
        theta_3: Fertility for age class 2
        survival_1: Survival rate from age class 0 to 1
        survival_2: Survival rate from age class 1 to 2
        lower_bounds: Domain lower bounds
        upper_bounds: Domain upper bounds
        subdiv_min, subdiv_max, subdiv_init, subdiv_limit: CMGDB parameters
        output_dir: Directory to save results (if None, uses Config default)
        verbose: Print progress messages

    Returns:
        dict: Results including morse_graph, map_graph, computation_time, barycenters, etc.
    """
    # Configure leslie_map_3d
    f = partial(leslie_map_3d, theta_1=theta_1, theta_2=theta_2, theta_3=theta_3,
                survival_1=survival_1, survival_2=survival_2)

    def F(rect):
        return CMGDB.BoxMap(f, rect)

    # Set output directory
    if output_dir is None:
        output_dir = Config.get_ground_truth_dir()
    os.makedirs(output_dir, exist_ok=True)

    # Build model and compute Morse graph
    model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit,
                       lower_bounds, upper_bounds, F)

    if verbose:
        print(f"Computing 3D Morse graph...")
        print(f"  theta=({theta_1:.1f}, {theta_2:.1f}, {theta_3:.1f})")
        print(f"  survival=({survival_1}, {survival_2})")
        print(f"  subdivisions: min={subdiv_min}, max={subdiv_max}, init={subdiv_init}")

    start_time = time.time()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    end_time = time.time()
    computation_time = end_time - start_time

    if verbose:
        print(f"  Computation completed in {computation_time:.2f} seconds")
        print(f"  Found {morse_graph.num_vertices()} Morse sets")

    # Compute barycenters
    barycenters = {}
    for i in range(morse_graph.num_vertices()):
        morse_set_boxes = morse_graph.morse_set_boxes(i)
        barycenters[i] = []
        if morse_set_boxes:
            dim = len(morse_set_boxes[0]) // 2
            for box in morse_set_boxes:
                barycenter = np.array([(box[j] + box[j + dim]) / 2.0 for j in range(dim)])
                barycenters[i].append(barycenter)

    # Save results if output directory provided
    if output_dir:
        # Save Morse Sets to CSV
        morse_sets_fname = os.path.join(output_dir, "morse_sets.csv")
        CMGDB.SaveMorseData.SaveMorseSets(morse_graph, morse_sets_fname)

        # Save barycenters
        barycenters_fname = os.path.join(output_dir, "barycenters.npz")
        barycenters_to_save = {f'morse_set_{k}': np.array(v) for k, v in barycenters.items() if v}
        if barycenters_to_save:
            np.savez(barycenters_fname, **barycenters_to_save)

        # Save complete Morse graph data
        model_params = {
            'theta_1': theta_1,
            'theta_2': theta_2,
            'theta_3': theta_3,
            'survival_1': survival_1,
            'survival_2': survival_2,
            'subdiv_min': subdiv_min,
            'subdiv_max': subdiv_max,
            'subdiv_init': subdiv_init,
            'subdiv_limit': subdiv_limit,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
        }
        runtime_info = {
            'computation_time_seconds': computation_time
        }
        metadata = {
            'model_params': model_params,
            'runtime_info': runtime_info,
            'barycenters': {k: [pt.tolist() for pt in v] for k, v in barycenters.items()}
        }
        morse_graph_data_fname = os.path.join(output_dir, "morse_graph_data.mgdb")
        CMGDB.SaveMorseGraphData(morse_graph, map_graph, morse_graph_data_fname, metadata=metadata)

        # Generate visualizations
        if verbose:
            print(f"  Generating visualizations...")

        # Morse graph diagram
        plot_morse_graph(
            morse_graph,
            os.path.join(output_dir, "morse_graph.png"),
            title=f'Morse Graph (θ=({theta_1:.1f}, {theta_2:.1f}, {theta_3:.1f}))'
        )

        # Barycenters 3D plot
        plot_barycenters_3d(
            morse_graph,
            [lower_bounds, upper_bounds],
            os.path.join(output_dir, "barycenters_scatterplot.png"),
            title=f"Barycenters of Morse Sets (θ=({theta_1:.1f}, {theta_2:.1f}, {theta_3:.1f}))"
        )

        if verbose:
            print(f"  Results saved to: {output_dir}")

    # Return results
    results = {
        'morse_graph': morse_graph,
        'map_graph': map_graph,
        'theta_1': theta_1,
        'theta_2': theta_2,
        'theta_3': theta_3,
        'survival_1': survival_1,
        'survival_2': survival_2,
        'num_morse_sets': morse_graph.num_vertices(),
        'computation_time': computation_time,
        'barycenters': barycenters,
        'output_dir': output_dir
    }

    return results


def load_ground_truth_barycenters(ground_truth_dir=None):
    """
    Load precomputed ground truth barycenters.

    Args:
        ground_truth_dir: Directory containing barycenters.npz (if None, uses Config default)

    Returns:
        numpy archive with barycenter data or None if not found
    """
    if ground_truth_dir is None:
        ground_truth_dir = Config.get_ground_truth_dir()

    barycenters_path = os.path.join(ground_truth_dir, "barycenters.npz")

    if not os.path.exists(barycenters_path):
        return None

    return np.load(barycenters_path, allow_pickle=True)


def load_ground_truth_morse_graph(ground_truth_dir=None):
    """
    Load precomputed ground truth Morse graph.

    Args:
        ground_truth_dir: Directory containing morse_graph_data.mgdb (if None, uses Config default)

    Returns:
        CMGDB MorseGraph object or None if not found
    """
    if ground_truth_dir is None:
        ground_truth_dir = Config.get_ground_truth_dir()

    morse_graph_path = os.path.join(ground_truth_dir, "morse_graph_data.mgdb")

    if not os.path.exists(morse_graph_path):
        return None

    return CMGDB.MorseGraph(morse_graph_path)


def extract_graph_structure(morse_graph):
    """
    Extract the graph structure (nodes and edges) from a CMGDB Morse graph.

    Args:
        morse_graph: CMGDB MorseGraph object

    Returns:
        dict with 'num_nodes' and 'edges' (list of tuples)
    """
    num_nodes = morse_graph.num_vertices()
    edges = []

    for i in range(num_nodes):
        adjacencies = morse_graph.adjacencies(i)
        for j in adjacencies:
            edges.append((i, j))

    return {
        'num_nodes': num_nodes,
        'edges': sorted(edges)
    }
