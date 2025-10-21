#!/usr/bin/env python3
"""
Analyze cached Morse graphs to find their structure (sources, sinks, etc.)
"""

import os
import sys
import json
import pickle
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_morse_graph(morse_graph):
    """Analyze Morse graph structure."""
    sources = [n for n in morse_graph.nodes() if morse_graph.in_degree(n) == 0]
    sinks = [n for n in morse_graph.nodes() if morse_graph.out_degree(n) == 0]
    is_dag = nx.is_directed_acyclic_graph(morse_graph)

    return {
        'num_nodes': morse_graph.number_of_nodes(),
        'num_edges': morse_graph.number_of_edges(),
        'num_sources': len(sources),
        'num_sinks': len(sinks),
        'sources': sources,
        'sinks': sinks,
        'is_dag': is_dag
    }

def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'examples', 'ives_model_output')
    cmgdb_3d_dir = os.path.join(base_dir, 'cmgdb_3d')

    if not os.path.exists(cmgdb_3d_dir):
        print(f"No cache directory found at: {cmgdb_3d_dir}")
        return

    print("\n" + "="*80)
    print("MORSE GRAPH STRUCTURE ANALYSIS")
    print("="*80)

    # Get all cached results
    cache_folders = []
    for item in os.listdir(cmgdb_3d_dir):
        item_path = os.path.join(cmgdb_3d_dir, item)
        if os.path.isdir(item_path) and item != 'results':
            metadata_path = os.path.join(item_path, 'metadata.json')
            morse_graph_path = os.path.join(item_path, 'morse_graph_data.mgdb')

            if os.path.exists(metadata_path) and os.path.exists(morse_graph_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    # Load Morse graph
                    import CMGDB
                    morse_graph = CMGDB.MorseGraph(morse_graph_path)

                    analysis = analyze_morse_graph(morse_graph)

                    cache_folders.append({
                        'metadata': metadata,
                        'analysis': analysis,
                        'hash': item,
                        'path': item_path
                    })
                except Exception as e:
                    print(f"Warning: Skipping corrupted cache {item}: {e}")

    # Sort by number of sources (to highlight 2-source configs)
    cache_folders.sort(key=lambda x: (
        -x['analysis']['num_sources'],  # More sources first
        x['metadata'].get('cached_at', '')  # Then by date
    ))

    for i, entry in enumerate(cache_folders, 1):
        meta = entry['metadata']
        analysis = entry['analysis']

        subdiv_min = meta.get('subdiv_min', '?')
        subdiv_max = meta.get('subdiv_max', '?')
        subdiv_init = meta.get('subdiv_init', '?')
        padding = meta.get('padding', '?')

        if subdiv_min == subdiv_max:
            grid_type = f"UNIFORM (subdiv={subdiv_min})"
        else:
            grid_type = f"ADAPTIVE (min={subdiv_min}, max={subdiv_max}, init={subdiv_init})"

        # Highlight if 2 sources
        if analysis['num_sources'] == 2:
            marker = " ⭐ TARGET"
        else:
            marker = ""

        print(f"\n{i}. {grid_type}{marker}")
        print(f"   Total Morse Sets: {analysis['num_nodes']}")
        print(f"   Edges: {analysis['num_edges']}")
        print(f"   Sources (minimal nodes): {analysis['num_sources']}")
        print(f"   Sinks (maximal nodes): {analysis['num_sinks']}")
        print(f"   Is DAG: {analysis['is_dag']}")
        print(f"   Padding: {padding}")
        print(f"   Hash: {entry['hash']}")

        if analysis['num_sources'] <= 5:
            print(f"   Source nodes: {analysis['sources']}")
        if analysis['num_sinks'] <= 5:
            print(f"   Sink nodes: {analysis['sinks']}")

    # Summary
    two_source_configs = [e for e in cache_folders if e['analysis']['num_sources'] == 2]

    print("\n" + "="*80)
    print(f"Total cached results: {len(cache_folders)}")
    print(f"Configurations with 2 minimal nodes: {len(two_source_configs)}")

    if two_source_configs:
        print("\nTo recompute a configuration with 2 minimal nodes:")
        for entry in two_source_configs:
            meta = entry['metadata']
            subdiv_min = meta.get('subdiv_min')
            subdiv_max = meta.get('subdiv_max')
            padding_flag = '--padding' if meta.get('padding') else ''

            if subdiv_min == subdiv_max:
                print(f"  python tests/compute_ives_3d_morse_graph.py --subdiv {subdiv_min} {padding_flag}")
            else:
                print(f"  python tests/compute_ives_3d_morse_graph.py --subdiv-min {subdiv_min} --subdiv-max {subdiv_max} {padding_flag}")

    print("="*80 + "\n")

if __name__ == "__main__":
    main()
