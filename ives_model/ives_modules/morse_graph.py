"""
3D Morse Graph Computation for Ives Model

This module handles the computation of the 3D Morse graph
for the Ives midge-algae-detritus ecological model using CMGDB.
"""

import CMGDB
import numpy as np
import os
import time
import matplotlib.cm
import matplotlib.pyplot as plt
from functools import partial
from MorseGraph.systems import ives_model_log
from .plotting import plot_morse_graph, plot_barycenters_3d
from .config import Config


def compute_morse_graph(
        r1=Config.R1,
        r2=Config.R2,
        c=Config.C,
        d=Config.D,
        p=Config.P,
        q=Config.Q,
        log_offset=Config.LOG_OFFSET,
        lower_bounds=Config.LOWER_BOUNDS,
        upper_bounds=Config.UPPER_BOUNDS,
        subdiv_min=Config.SUBDIV_MIN,
        subdiv_max=Config.SUBDIV_MAX,
        subdiv_init=Config.SUBDIV_INIT,
        subdiv_limit=Config.SUBDIV_LIMIT,
        output_dir=None,
        verbose=True):
    """
    Compute the 3D Morse graph for the Ives model.

    Args:
        r1: Midge reproduction rate
        r2: Algae growth rate
        c: Constant input of algae and detritus
        d: Detritus decay rate
        p: Relative palatability of detritus
        q: Exponent in midge consumption
        log_offset: Offset added beforelog transform
        lower_bounds: Domain lower bounds (inlog scale)
        upper_bounds: Domain upper bounds (inlog scale)
        subdiv_min, subdiv_max, subdiv_init, subdiv_limit: CMGDB parameters
        output_dir: Directory to save results (if None, uses Config default)
        verbose: Print progress messages

    Returns:
        dict: Results including morse_graph, map_graph, computation_time, barycenters, etc.
    """
    # Configure ives_model_log
    f = partial(ives_model_log, r1=r1, r2=r2, c=c, d=d, p=p, q=q, offset=log_offset)

    def F(rect):
        return CMGDB.BoxMap(f, rect)

    # Set output directory
    if output_dir is None:
        output_dir = Config.get_morse_graph_dir()
    os.makedirs(output_dir, exist_ok=True)

    # Build model and compute Morse graph
    model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit,
                       lower_bounds, upper_bounds, F)

    if verbose:
        print(f"Computing 3D Morse graph (log scale)...")
        print(f"  r1={r1:.3f}, r2={r2:.3f}, c={c:.2e}")
        print(f"  d={d:.4f}, p={p:.5f}, q={q:.3f}")
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
            'r1': r1,
            'r2': r2,
            'c': c,
            'd': d,
            'p': p,
            'q': q,
            'log_offset': log_offset,
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
            title=f'Morse Graph (Ives Model,log scale)'
        )

        # Barycenters 3D plot
        plot_barycenters_3d(
            morse_graph,
            [lower_bounds, upper_bounds],
            os.path.join(output_dir, "barycenters_scatterplot.png"),
            title=f"Barycenters of Morse Sets (Ives Model,log scale)"
        )

        # Morse sets 2D projections (manually plot projected boxes)
        def plot_morse_sets_projection(morse_graph, proj_dims, output_path, cmap, bounds=None, equilibrium_point=None):
            """Plot 2D projection of 3D Morse sets."""
            from matplotlib.patches import Rectangle

            fig, ax = plt.subplots(figsize=(7, 7))

            num_morse_sets = morse_graph.num_vertices()
            colors = cmap(np.linspace(0, 1, num_morse_sets))

            # Extract and project boxes
            for morse_idx in range(num_morse_sets):
                boxes = morse_graph.morse_set_boxes(morse_idx)
                if boxes:
                    for box in boxes:
                        # box format: [x_min, y_min, z_min, x_max, y_max, z_max]
                        # Project to 2D using proj_dims
                        d1, d2 = proj_dims
                        x_min, y_min = box[d1], box[d2]
                        x_max, y_max = box[d1 + 3], box[d2 + 3]

                        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                       facecolor=colors[morse_idx],
                                       edgecolor='none', alpha=0.7)
                        ax.add_patch(rect)

            # Plot equilibrium point if provided
            if equilibrium_point is not None:
                d1, d2 = proj_dims
                ax.plot(equilibrium_point[d1], equilibrium_point[d2],
                       marker='*', markersize=10, color='red',
                       markeredgecolor='darkred', markeredgewidth=1.5,
                       label='Equilibrium', zorder=10)

            # Set bounds and labels
            # ax.set_xlim(bounds[0], bounds[1])
            # ax.set_ylim(bounds[2], bounds[3])

            # Add labels based on projection
            labels = ['log(midge)', 'log(algae)', 'log(detritus)']
            ax.set_xlabel(labels[proj_dims[0]])
            ax.set_ylabel(labels[proj_dims[1]])
            ax.set_aspect('equal')

            if equilibrium_point is not None:
                ax.legend(loc='best')

            plt.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Equilibrium point (stable point in log10 scale)
        # Found via numerical optimization (fsolve) - verified to machine precision
        equilibrium_point = np.array([0.792107, 0.209010, 0.376449])

        # Projection 0,1 (midge vs algae)
        plot_morse_sets_projection(
            morse_graph, [0, 1],
            os.path.join(output_dir, "morse_sets_dim_0_1.png"),
            matplotlib.cm.cool,
            equilibrium_point=equilibrium_point
        )

        # Projection 0,2 (midge vs detritus)
        plot_morse_sets_projection(
            morse_graph, [0, 2],
            os.path.join(output_dir, "morse_sets_dim_0_2.png"),
            matplotlib.cm.cool,
            equilibrium_point=equilibrium_point
        )

        # Projection 1,2 (algae vs detritus)
        plot_morse_sets_projection(
            morse_graph, [1, 2],
            os.path.join(output_dir, "morse_sets_dim_1_2.png"),
            matplotlib.cm.cool,
            equilibrium_point=equilibrium_point
        )

        if verbose:
            print(f"  Results saved to: {output_dir}")

    # Return results
    results = {
        'morse_graph': morse_graph,
        'map_graph': map_graph,
        'r1': r1,
        'r2': r2,
        'c': c,
        'd': d,
        'p': p,
        'q': q,
        'log_offset': log_offset,
        'num_morse_sets': morse_graph.num_vertices(),
        'computation_time': computation_time,
        'barycenters': barycenters,
        'output_dir': output_dir
    }

    return results


def load_morse_sets_barycenters(morse_graph_dir=None):
    """
    Load precomputed barycenters.

    Args:
        morse_graph_dir: Directory containing barycenters.npz (if None, uses Config default)

    Returns:
        numpy archive with barycenter data or None if not found
    """
    if morse_graph_dir is None:
        morse_graph_dir = Config.get_morse_graph_dir()

    barycenters_path = os.path.join(morse_graph_dir, "barycenters.npz")

    if not os.path.exists(barycenters_path):
        return None

    return np.load(barycenters_path, allow_pickle=True)


def load_morse_graph(morse_graph_dir=None):
    """
    Load precomputed Morse graph.

    Args:
        morse_graph_dir: Directory containing morse_graph_data.mgdb (if None, uses Config default)

    Returns:
        CMGDB MorseGraph object or None if not found
    """
    if morse_graph_dir is None:
        morse_graph_dir = Config.get_morse_graph_dir()

    morse_graph_path = os.path.join(morse_graph_dir, "morse_graph_data.mgdb")

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
