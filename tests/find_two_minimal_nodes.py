#!/usr/bin/env python3
"""
Find subdivision parameters that produce 2 minimal nodes in the Morse graph.

This script tries different subdivision levels to find which ones produce
exactly 2 minimal elements (sources) in the Morse graph.
"""

import numpy as np
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.systems import ives_model_log
from MorseGraph.core import compute_morse_graph_3d
import networkx as nx


def analyze_morse_graph_structure(morse_graph):
    """Count sources (minimal nodes) and sinks (maximal nodes) from CMGDB MorseGraph."""
    vertices = list(range(morse_graph.num_vertices()))
    edges = morse_graph.edges()

    # Build adjacency structure to find sources/sinks
    has_incoming = set()
    has_outgoing = set()

    for edge_idx in range(len(edges)):
        u, v = edges[edge_idx]
        has_outgoing.add(u)
        has_incoming.add(v)

    sources = [v for v in vertices if v not in has_incoming]
    sinks = [v for v in vertices if v not in has_outgoing]

    return {
        'num_nodes': len(vertices),
        'num_edges': len(edges),
        'num_sources': len(sources),
        'num_sinks': len(sinks),
        'sources': sources,
        'sinks': sinks,
    }


def test_subdivision(subdiv, padding=False):
    """Test a specific subdivision level."""
    # Ives model parameters
    R1, R2, C, D, P, Q, LOG_OFFSET = 3.873, 11.746, 3.67e-07, 0.5517, 0.06659, 0.9026, 0.001
    DOMAIN_BOUNDS = [[-5, -20, -8], [10, 20, 8]]

    ives_map = partial(ives_model_log, r1=R1, r2=R2, c=C, d=D, p=P, q=Q, offset=LOG_OFFSET)

    print(f"\n{'='*80}")
    print(f"Testing subdiv={subdiv}, padding={padding}")
    print(f"{'='*80}")

    result = compute_morse_graph_3d(
        ives_map,
        DOMAIN_BOUNDS,
        subdiv_min=subdiv,
        subdiv_max=subdiv,
        subdiv_init=0,
        subdiv_limit=100000,
        padding=padding,
        cache_dir=None,  # Don't cache
        use_cache=False,
        verbose=True
    )

    morse_graph = result['morse_graph']
    analysis = analyze_morse_graph_structure(morse_graph)

    print(f"\nResults:")
    print(f"  Total Morse Sets: {analysis['num_nodes']}")
    print(f"  Edges: {analysis['num_edges']}")
    print(f"  Sources (minimal nodes): {analysis['num_sources']}")
    print(f"  Sinks (maximal nodes): {analysis['num_sinks']}")

    if analysis['num_sources'] <= 5:
        print(f"  Source nodes: {analysis['sources']}")
    if analysis['num_sinks'] <= 5:
        print(f"  Sink nodes: {analysis['sinks']}")

    if analysis['num_sources'] == 2:
        print(f"\n  ⭐ FOUND: This configuration has exactly 2 minimal nodes!")
        return True

    return False


def main():
    print("\n" + "="*80)
    print("SEARCHING FOR MORSE GRAPH WITH 2 MINIMAL NODES")
    print("="*80)
    print("\nIves Model: Midge-Algae-Detritus dynamics")
    print("Domain: [[-5, -20, -8], [10, 20, 8]]")

    # Test a range of subdivision levels
    found_configs = []

    # Test a range of subdivisions with both padding on and off
    # subdiv is PER-DIMENSION, so subdiv=11 gives 2^11 = 2048 boxes per dim
    # In 3D: 2^11 × 2^11 × 2^11 = 2^33 total boxes
    # subdiv=33 means 2^33 boxes PER dimension (way too fine!)

    test_subdivs = [10, 11, 12, 13, 14]  # More reasonable range

    for subdiv in test_subdivs:
        # Try without padding
        if test_subdivision(subdiv, padding=False):
            found_configs.append(('uniform', subdiv, False))

        # Try with padding
        if test_subdivision(subdiv, padding=True):
            found_configs.append(('uniform', subdiv, True))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if found_configs:
        print(f"\nFound {len(found_configs)} configuration(s) with 2 minimal nodes:")
        for grid_type, subdiv, padding in found_configs:
            print(f"  - subdiv={subdiv}, padding={padding}")
            print(f"    Command: python tests/compute_ives_3d_morse_graph.py --subdiv {subdiv}")
    else:
        print("\nNo configurations with exactly 2 minimal nodes found in tested range.")
        print("Try testing more subdivision levels or different parameters.")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
