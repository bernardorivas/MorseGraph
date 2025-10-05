# CMGDB 3D Features Guide

This document describes the new 3D visualization and data persistence features added to CMGDB.

## New Features

### 1. 3D Morse Set Visualization

**Function**: `CMGDB.PlotMorseSets3D()`

Visualize 3D Morse sets using cuboid rendering with `Poly3DCollection`. This provides true 3D visualization instead of 2D projections.

**Key Features**:
- Automatic cuboid face generation from grid boxes
- Configurable transparency to see interior structure
- Adjustable viewing angles (elevation and azimuth)
- Same interface as 2D `PlotMorseSets` for easy migration

### 2. Computation Result Persistence

**Functions**:
- `CMGDB.SaveMorseGraphData()` - Save morse_graph and map_graph to file
- `CMGDB.LoadMorseGraphData()` - Load saved computation results
- `CMGDB.SaveComputationResults()` - High-level save with metadata
- `CMGDB.GetMorseSetsFromData()` - Extract Morse sets for plotting

**Key Features**:
- Pickle-based serialization with `.mgdb` file format
- Preserves all Morse graph data (vertices, edges, annotations, boxes)
- Stores metadata (parameters, runtime info, etc.)
- Can reload and re-plot without recomputing

---

## Usage Examples

### Example 1: 3D Visualization

```python
import CMGDB
import matplotlib

# Compute Morse graph (3D system)
morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)

# Plot with default view (elev=30°, azim=45°)
CMGDB.PlotMorseSets3D(
    morse_graph,
    cmap=matplotlib.cm.cool,
    fig_w=10,
    fig_h=10,
    alpha=0.3,  # Transparency (0.0-1.0)
    fig_fname="morse_3d_default.png"
)

# Plot from different angle (top view)
CMGDB.PlotMorseSets3D(
    morse_graph,
    cmap=matplotlib.cm.cool,
    elev=90,    # View from top
    azim=0,
    alpha=0.3,
    fig_fname="morse_3d_top.png"
)

# Plot with custom axis limits
CMGDB.PlotMorseSets3D(
    morse_graph,
    cmap=matplotlib.cm.viridis,
    xlim=[0, 90],
    ylim=[0, 70],
    zlim=[0, 70],
    xlabel='$x_1$',
    ylabel='$x_2$',
    zlabel='$x_3$',
    fig_fname="morse_3d_custom.png"
)
```

### Example 2: Save Computation Results

```python
import CMGDB

# Compute Morse graph
morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)

# Option 1: Simple save
CMGDB.SaveMorseGraphData(morse_graph, map_graph, "my_results.mgdb")

# Option 2: Save with metadata
metadata = {
    'model': 'Leslie 3-age class',
    'parameters': {
        'theta_1': 28.9,
        'theta_2': 29.8,
        'theta_3': 22.0
    },
    'computation_time': 45.2,
    'description': 'Population dynamics analysis'
}

CMGDB.SaveMorseGraphData(
    morse_graph,
    map_graph,
    "my_results.mgdb",
    metadata=metadata
)

# Option 3: High-level save (automatic metadata)
CMGDB.SaveComputationResults(
    morse_graph,
    map_graph,
    "my_results.mgdb",
    model_params={'subdiv_min': 36, 'subdiv_max': 42},
    runtime_info={'time': 45.2, 'date': '2025-10-05'}
)
```

### Example 3: Load and Visualize Saved Results

```python
import CMGDB
import matplotlib

# Load saved data
data = CMGDB.LoadMorseGraphData("my_results.mgdb")

# Inspect loaded data
print(f"Morse sets: {data['morse_graph']['num_vertices']}")
print(f"Phase space boxes: {data['map_graph']['num_vertices']}")
print(f"Metadata: {data['metadata']}")

# Extract Morse sets for plotting
morse_sets = CMGDB.GetMorseSetsFromData(data)

# Plot loaded data (2D projection)
CMGDB.PlotMorseSets(
    morse_sets,
    proj_dims=[0, 1],
    cmap=matplotlib.cm.cool,
    fig_fname="loaded_2d.png"
)

# Plot loaded data (3D)
CMGDB.PlotMorseSets3D(
    morse_sets,
    cmap=matplotlib.cm.viridis,
    alpha=0.3,
    fig_fname="loaded_3d.png"
)
```

---

## API Reference

### PlotMorseSets3D

```python
CMGDB.PlotMorseSets3D(
    morse_sets,          # MorseGraph object, filename, or list of boxes
    morse_nodes=None,    # List of node indices to plot (default: all)
    cmap=None,           # Matplotlib colormap (default: tab20)
    clist=None,          # List of colors for custom colormap
    fig_w=10,            # Figure width in inches
    fig_h=10,            # Figure height in inches
    alpha=0.3,           # Transparency (0.0-1.0, lower shows interior)
    elev=30,             # Elevation angle in degrees
    azim=45,             # Azimuth angle in degrees
    xlim=None,           # X-axis limits [x_min, x_max]
    ylim=None,           # Y-axis limits [y_min, y_max]
    zlim=None,           # Z-axis limits [z_min, z_max]
    axis_labels=True,    # Whether to show axis labels
    xlabel='$x$',        # X-axis label
    ylabel='$y$',        # Y-axis label
    zlabel='$z$',        # Z-axis label
    fontsize=15,         # Font size for labels
    fig_fname=None,      # Filename to save (optional)
    dpi=300              # DPI for saved figure
)
```

**Parameters**:
- `morse_sets`: Can be:
  - `CMGDB.MorseGraph` object from `ComputeMorseGraph()`
  - String filename of saved morse sets
  - List of boxes `[x_min, y_min, z_min, x_max, y_max, z_max, node]`

- `alpha`: Controls transparency (0.0 = fully transparent, 1.0 = opaque)
  - Lower values (0.2-0.3) good for seeing interior structure
  - Higher values (0.5-0.7) better for surface visualization

- `elev`, `azim`: Viewing angles
  - `elev=90, azim=0`: Top view (looking down z-axis)
  - `elev=0, azim=0`: Side view (looking along y-axis)
  - `elev=30, azim=45`: Default oblique view

### SaveMorseGraphData

```python
CMGDB.SaveMorseGraphData(
    morse_graph,     # MorseGraph object from ComputeMorseGraph()
    map_graph,       # MapGraph object from ComputeMorseGraph()
    filename,        # Output filename (adds .mgdb if missing)
    metadata=None    # Optional dictionary of metadata
)
```

**Saves**:
- All Morse graph vertices, edges, boxes
- Annotations for each Morse set
- Map graph adjacency information (sampled for large graphs)
- Metadata (user-provided + automatic stats)

**File Format**:
- Extension: `.mgdb` (Morse Graph DataBase)
- Format: Pickle (Python pickle protocol)
- Cross-platform compatible

### LoadMorseGraphData

```python
data = CMGDB.LoadMorseGraphData(filename)
```

**Returns**: Dictionary with keys:
- `'morse_graph'`: Dictionary containing:
  - `'num_vertices'`: Number of Morse sets
  - `'vertices'`: List of vertex indices
  - `'edges'`: List of (source, target) edge pairs
  - `'morse_sets'`: List of boxes for each Morse set
  - `'annotations'`: Annotations for each vertex
  - `'dimension'`: Spatial dimension

- `'map_graph'`: Dictionary containing:
  - `'num_vertices'`: Number of phase space boxes
  - `'adjacencies'`: Sampled adjacency lists
  - `'num_adjacencies_stored'`: Count of stored adjacencies

- `'metadata'`: User-provided and automatic metadata

- `'version'`: File format version (for compatibility)

### GetMorseSetsFromData

```python
morse_sets = CMGDB.GetMorseSetsFromData(data)
```

**Input**: Dictionary from `LoadMorseGraphData()`

**Returns**: List of boxes in format expected by plotting functions:
```
[x_min, y_min, z_min, x_max, y_max, z_max, node_index]
```

---

## Comparison: 2D vs 3D Plotting

| Feature | PlotMorseSets (2D) | PlotMorseSets3D (3D) |
|---------|-------------------|----------------------|
| **Method** | Scatter plot with sized markers | Poly3DCollection cuboids |
| **Dimensions** | 1D, 2D (with projections) | 3D only |
| **Projections** | Via `proj_dims` parameter | Built-in 3D view |
| **Transparency** | Implicit via alpha on markers | Explicit cuboid transparency |
| **View Angles** | N/A | `elev` and `azim` parameters |
| **Performance** | Very fast (2D scatter) | Moderate (3D rendering) |

**When to use each**:
- Use `PlotMorseSets` for:
  - 2D systems
  - 3D projections onto 2D planes
  - Quick visualization
  - Publication-ready 2D figures

- Use `PlotMorseSets3D` for:
  - True 3D visualization
  - Understanding spatial structure
  - Interactive exploration (different angles)
  - 3D presentation/demo

---

## File Structure

New files added to CMGDB:

```
cmgdb/src/CMGDB/
├── PlotMorseSets.py           # Updated with 3D support
│   ├── PlotMorseSets()        # Existing 2D function
│   ├── PlotBoxesScatter()     # Existing 2D helper
│   ├── PlotMorseSets3D()      # NEW: 3D visualization
│   └── _box_to_cuboid_faces() # NEW: 3D geometry helper
├── SaveMorseGraphData.py      # NEW: Save/load module
│   ├── SaveMorseGraphData()
│   ├── LoadMorseGraphData()
│   ├── SaveComputationResults()
│   ├── GetMorseSetsFromData()
│   └── _extract_morse_graph_data()
└── __init__.py                # Updated exports
```

---

## Tips and Best Practices

### 3D Visualization

1. **Start with low transparency** (`alpha=0.2-0.3`) to see interior structure
2. **Try multiple view angles** to understand geometry:
   ```python
   # Default oblique view
   PlotMorseSets3D(..., elev=30, azim=45)

   # Top view
   PlotMorseSets3D(..., elev=90, azim=0)

   # Side views
   PlotMorseSets3D(..., elev=0, azim=0)    # Front
   PlotMorseSets3D(..., elev=0, azim=90)   # Side
   ```

3. **Use different colormaps** for different purposes:
   - `matplotlib.cm.cool`: Blue to magenta (good default)
   - `matplotlib.cm.viridis`: Perceptually uniform
   - `matplotlib.cm.tab20`: Distinct colors for many sets

4. **Save at high DPI** for publication: `dpi=300` or `dpi=600`

### Data Persistence

1. **Always include metadata** when saving:
   ```python
   metadata = {
       'model': 'Description of system',
       'parameters': {...},
       'date': '2025-10-05',
       'notes': 'Any relevant information'
   }
   ```

2. **Use descriptive filenames**:
   ```python
   "leslie_theta1_28.9_subdiv_36.mgdb"
   ```

3. **Check file size** for large computations:
   - Map graph adjacencies are sampled to limit file size
   - For very large graphs, consider saving only Morse sets

4. **Version control** your `.mgdb` files if needed:
   - Files are binary but deterministic
   - Can be committed to git if not too large

---

## Examples and Tests

Three test files are provided:

1. **tests/test_cmgdb_imports.py**
   - Quick validation that new functions are available
   - Run first to verify installation

2. **tests/test_cmgdb_3d_features.py**
   - Comprehensive demo of all features
   - Computes 3D Leslie model
   - Generates multiple visualizations
   - Tests save/load cycle
   - Creates 2D projections for comparison

3. **tests/test_cmgdb_ode_examples.py**
   - ODE-based examples (2D and 3D)
   - Converted from notebooks

**Run tests**:
```bash
# Quick import check
python tests/test_cmgdb_imports.py

# Full feature demo (~1-2 minutes with computation)
python tests/test_cmgdb_3d_features.py

# ODE examples
python tests/test_cmgdb_ode_examples.py      # Quick demo
python tests/test_cmgdb_ode_examples.py --all  # All examples
```

---

## Migration Guide

### From Notebooks to Scripts

**Before** (notebook):
```python
%%time
morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
CMGDB.PlotMorseGraph(morse_graph, cmap=matplotlib.cm.cool)
CMGDB.PlotMorseSets(morse_graph, cmap=matplotlib.cm.cool, proj_dims=[0, 2])
```

**After** (script with new features):
```python
import time

start = time.time()
morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
elapsed = time.time() - start

# Save results
CMGDB.SaveComputationResults(
    morse_graph, map_graph,
    "my_results.mgdb",
    runtime_info={'time': elapsed}
)

# Plot Morse graph (unchanged)
CMGDB.PlotMorseGraph(morse_graph, cmap=matplotlib.cm.cool)

# Plot 2D projection (unchanged)
CMGDB.PlotMorseSets(morse_graph, cmap=matplotlib.cm.cool, proj_dims=[0, 2])

# NEW: Plot true 3D view
CMGDB.PlotMorseSets3D(morse_graph, cmap=matplotlib.cm.cool, alpha=0.3)
```

### From 2D to 3D Plotting

Minimal changes needed:

```python
# 2D plotting (projection)
CMGDB.PlotMorseSets(morse_graph, proj_dims=[0, 2])

# 3D plotting (full visualization)
CMGDB.PlotMorseSets3D(morse_graph, alpha=0.3, elev=30, azim=45)
```

---

## Troubleshooting

### Import Errors

**Problem**: `AttributeError: module 'CMGDB' has no attribute 'PlotMorseSets3D'`

**Solution**: Reinstall CMGDB package
```bash
pip install -e ./cmgdb --force-reinstall
```

### 3D Plotting Errors

**Problem**: `ValueError: PlotMorseSets3D requires 3D data, got 2D`

**Solution**: Use `PlotMorseSets` for 2D data or 3D projections

**Problem**: Empty or strange 3D plot

**Solution**: Check axis limits, try different `alpha` values and viewing angles

### Save/Load Issues

**Problem**: `FileNotFoundError` when loading

**Solution**: Check filename, `.mgdb` extension is automatic:
```python
# These are equivalent:
LoadMorseGraphData("results.mgdb")
LoadMorseGraphData("results")
```

**Problem**: Large `.mgdb` files

**Solution**: This is normal for large computations. The file includes:
- All Morse set boxes
- Sampled adjacency information
- Consider compressing with gzip if needed

---

## Performance Notes

### 3D Rendering

- **Time complexity**: O(n_boxes * n_faces) where n_faces = 6 per box
- **Typical performance**:
  - 1000 boxes: < 1 second
  - 10000 boxes: 1-5 seconds
  - 100000 boxes: 5-30 seconds

**Optimization tips**:
- Use `fig_fname` to save without displaying (faster)
- Reduce `dpi` for faster rendering during development
- Plot subsets with `morse_nodes` parameter

### Save/Load

- **Save time**: ~0.1-1.0 seconds (pickle serialization)
- **Load time**: ~0.1-0.5 seconds (pickle deserialization)
- **File size**: Roughly 1-10 KB per Morse set box

**Storage tips**:
- Map graph adjacencies are sampled (only relevant boxes)
- Full adjacency lists can be reconstructed from map if needed
- Compress `.mgdb` files with gzip for long-term storage

---

## See Also

- `CLAUDE.md` - General MorseGraph development guide
- `README.md` - CMGDB package overview
- `tests/test_cmgdb_3d_features.py` - Comprehensive usage examples
- Original papers on Morse decomposition and combinatorial methods
