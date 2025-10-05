# SaveMorseGraphData.py
# Save and load Morse graph computation results
# MIT LICENSE

import pickle
import numpy as np
from pathlib import Path
import CMGDB

def SaveMorseGraphData(morse_graph, map_graph, filename, metadata=None):
    """
    Save complete Morse graph computation results to a pickle file.

    Extracts all data from the C++ objects and saves to a .mgdb (Morse Graph DataBase) file.

    :param morse_graph: CMGDB.MorseGraph object from ComputeMorseGraph or ComputeConleyMorseGraph
    :param map_graph: MapGraph object from the same computation
    :param filename: Output filename (will add .mgdb extension if not present)
    :param metadata: Optional dictionary of metadata to save (e.g., parameters, runtime info)
    """
    # Ensure filename has .mgdb extension
    filepath = Path(filename)
    if filepath.suffix != '.mgdb':
        filepath = filepath.with_suffix('.mgdb')

    # Extract morse graph data
    morse_data = _extract_morse_graph_data(morse_graph)

    # Extract map graph data
    map_data = _extract_map_graph_data(map_graph, morse_graph)

    # Create metadata if not provided
    if metadata is None:
        metadata = {}

    # Add automatic metadata
    metadata['num_morse_sets'] = morse_graph.num_vertices()
    metadata['phase_space_size'] = map_graph.num_vertices()

    # Bundle all data
    save_data = {
        'morse_graph': morse_data,
        'map_graph': map_data,
        'metadata': metadata,
        'version': '1.0'  # For future compatibility
    }

    # Save to pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Morse graph data saved to {filepath}")
    print(f"  - {morse_data['num_vertices']} Morse sets")
    print(f"  - {map_data['num_vertices']} phase space boxes")

def LoadMorseGraphData(filename):
    """
    Load saved Morse graph computation results.

    Returns Python dictionaries containing the Morse graph and map graph data.
    Note: Returns data dictionaries, not C++ objects (which cannot be reconstructed).

    :param filename: Input .mgdb filename
    :return: Dictionary with keys 'morse_graph', 'map_graph', 'metadata'
    """
    filepath = Path(filename)
    if not filepath.exists():
        # Try adding .mgdb extension
        if filepath.with_suffix('.mgdb').exists():
            filepath = filepath.with_suffix('.mgdb')
        else:
            raise FileNotFoundError(f"File not found: {filename}")

    # Load pickle file
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Verify version
    if 'version' not in data:
        print("Warning: Loading old format file without version info")

    print(f"Loaded Morse graph data from {filepath}")
    if 'metadata' in data and 'num_morse_sets' in data['metadata']:
        print(f"  - {data['metadata']['num_morse_sets']} Morse sets")
        print(f"  - {data['metadata']['phase_space_size']} phase space boxes")

    return data

def _extract_morse_graph_data(morse_graph):
    """
    Extract all data from a CMGDB.MorseGraph C++ object into Python structures.

    :param morse_graph: CMGDB.MorseGraph object
    :return: Dictionary containing all morse graph data
    """
    num_vertices = morse_graph.num_vertices()

    data = {
        'num_vertices': num_vertices,
        'vertices': [],
        'edges': [],
        'morse_sets': [],  # List of lists of boxes for each morse set
        'annotations': [],  # Annotations for each vertex
    }

    # Extract vertices
    data['vertices'] = list(morse_graph.vertices())

    # Extract edges (reachability relations)
    data['edges'] = list(morse_graph.edges())

    # Extract Morse sets (the actual grid boxes)
    for vertex in range(num_vertices):
        morse_set_boxes = morse_graph.morse_set_boxes(vertex)
        data['morse_sets'].append(morse_set_boxes)

        # Extract annotations if available
        try:
            annotations = morse_graph.annotations(vertex)
            data['annotations'].append(annotations)
        except:
            data['annotations'].append([])

    # Try to extract phase space information
    try:
        # Get phase space bounds by examining first box from first morse set
        if data['morse_sets'] and len(data['morse_sets'][0]) > 0:
            first_box = data['morse_sets'][0][0]
            dim = len(first_box) // 2
            data['dimension'] = dim
        else:
            data['dimension'] = None
    except:
        data['dimension'] = None

    return data

def _extract_map_graph_data(map_graph, morse_graph):
    """
    Extract data from a MapGraph C++ object.

    :param map_graph: MapGraph object
    :param morse_graph: MorseGraph object (for phase space info)
    :return: Dictionary containing map graph data
    """
    num_vertices = map_graph.num_vertices()

    data = {
        'num_vertices': num_vertices,
        'adjacencies': {},  # Sparse representation: vertex -> list of adjacent vertices
    }

    # Extract adjacencies for a sample of vertices (storing all can be huge)
    # For now, just store the count and structure info
    # Full adjacency lists can be very large, so we'll be selective

    # Store adjacencies for vertices in Morse sets (these are the important ones)
    morse_set_vertices = set()
    for vertex in range(morse_graph.num_vertices()):
        morse_set_vertices.update(morse_graph.morse_set(vertex))

    # Sample adjacencies (limit to avoid huge files)
    max_adjacencies_to_store = min(10000, len(morse_set_vertices))
    vertices_to_sample = list(morse_set_vertices)[:max_adjacencies_to_store]

    for vertex in vertices_to_sample:
        try:
            adj = map_graph.adjacencies(vertex)
            if len(adj) > 0:  # Only store if has adjacencies
                data['adjacencies'][int(vertex)] = list(adj)
        except:
            pass

    data['num_adjacencies_stored'] = len(data['adjacencies'])

    return data

def SaveComputationResults(morse_graph, map_graph, filename,
                          model_params=None, runtime_info=None):
    """
    High-level convenience function to save Morse graph computation with metadata.

    :param morse_graph: CMGDB.MorseGraph object
    :param map_graph: MapGraph object
    :param filename: Output filename
    :param model_params: Dictionary of model parameters (subdiv_min, subdiv_max, bounds, etc.)
    :param runtime_info: Dictionary of runtime information (computation time, etc.)
    """
    metadata = {}

    if model_params is not None:
        metadata['model_params'] = model_params

    if runtime_info is not None:
        metadata['runtime_info'] = runtime_info

    SaveMorseGraphData(morse_graph, map_graph, filename, metadata)

def GetMorseSetsFromData(data):
    """
    Extract Morse sets in the format expected by PlotMorseSets from loaded data.

    :param data: Data dictionary from LoadMorseGraphData
    :return: List of boxes in format [x_min, y_min, ..., x_max, y_max, ..., node_index]
    """
    morse_graph_data = data['morse_graph']
    morse_sets_list = []

    for node_index, morse_set_boxes in enumerate(morse_graph_data['morse_sets']):
        for box in morse_set_boxes:
            # Add node index at the end
            morse_sets_list.append(box + [node_index])

    return morse_sets_list
