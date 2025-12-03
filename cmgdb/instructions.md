# CMGDB (Conley Morse Graph Database) - Complete Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Python API Reference](#python-api-reference)
4. [C++ Backend Components](#c-backend-components)
   - [Database Components](#database-components)
   - [CHomP Components](#chomp-components)
5. [Data Structures](#data-structures)
6. [Algorithms](#algorithms)
7. [File Structure](#file-structure)
8. [Usage Patterns](#usage-patterns)

---

## Overview

**CMGDB** (Conley Morse Graph Database) is a C++ library with Python bindings for rigorous analysis of discrete dynamical systems using combinatorial and topological methods. It computes the **Conley-Morse graph** of a system by discretizing the state space and analyzing the resulting transition graph.

### Mathematical Foundation

- **State Space Discretization**: The state space $X$ is partitioned into hyperrectangles (boxes), forming a grid
- **Box Map**: A multi-valued map $F$ on the grid where an edge exists from box $B_i$ to $B_j$ if $f(B_i)$ intersects $B_j$
- **Morse Graph**: The condensation graph of strongly connected components (SCCs) of the transition graph
- **Morse Sets**: Box-regions corresponding to SCCs representing recurrent/chain-recurrent behavior
- **Conley-Morse Graph**: Morse Graph annotated with Conley Index for each Morse set

### Key Capabilities

- Compute Morse decompositions for discrete dynamical systems
- Compute Conley indices for Morse sets
- Visualize Morse graphs and Morse sets (2D and 3D)
- Support for data-driven and function-based dynamics
- Adaptive mesh refinement for efficient computation
- Save/load computation results

---

## Architecture

CMGDB consists of three main layers:

### 1. **CHomP (Computational Homology Project) Library**
Location: `src/CMGDB/_cmgdb/include/chomp/`

Provides computational topology algorithms:
- Homology computation
- Conley index computation
- Morse complex construction
- Chain complexes and boundary operators
- Sparse matrix operations

### 2. **Database Layer (CMGDB Core)**
Location: `src/CMGDB/_cmgdb/include/database/`

Core data structures and algorithms:
- Grid representations (uniform, tree-based, succinct)
- Map representations (interval maps, point maps)
- Morse graph data structure
- Graph algorithms (SCC computation, reachability)
- Model configuration

### 3. **Python Bindings**
Location: `src/CMGDB/`

User-facing Python API:
- High-level computation functions
- Visualization utilities
- Data persistence
- Helper functions for box maps

---

## Python API Reference

### Core Computation Functions

#### `CMGDB.Model`

**Purpose**: Encapsulates all parameters for Morse graph computation.

**Constructors**:
```python
Model(subdiv_min, subdiv_max, lower_bounds, upper_bounds)
Model(subdiv_min, subdiv_max, lower_bounds, upper_bounds, periodic)
Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit, lower_bounds, upper_bounds, periodic)
Model(subdiv, lower_bounds, upper_bounds, F)
Model(subdiv_min, subdiv_max, lower_bounds, upper_bounds, F)
# ... (with periodic variants)
```

**Parameters**:
- `subdiv_min` (int): Minimum subdivision level
- `subdiv_max` (int): Maximum subdivision level
- `subdiv_init` (int): Initial subdivision level (optional)
- `subdiv_limit` (int): Complexity limit for subdivision (optional)
- `lower_bounds` (list[float]): Lower corner of domain
- `upper_bounds` (list[float]): Upper corner of domain
- `periodic` (list[bool]): Periodic boundary conditions per dimension (optional)
- `F` (callable): Python function `rect -> rect` for box map (optional)

**Methods**:
- `parameterSpace()`: Get parameter space object
- `phaseSpace()`: Get phase space grid object
- `param_dim()`: Get parameter dimension
- `phase_dim()`: Get phase space dimension
- `phase_subdiv_min()`: Get minimum subdivision
- `phase_subdiv_max()`: Get maximum subdivision
- `setmap(parameter)`: Set map for given parameter

#### `CMGDB.ComputeMorseGraph(model)`

**Purpose**: Compute the Morse graph (without Conley indices).

**Parameters**:
- `model` (Model): Model object specifying system and parameters

**Returns**:
- `morse_graph` (MorseGraph): The computed Morse graph
- `map_graph` (MapGraph): The underlying transition graph

**Example**:
```python
model = CMGDB.Model(20, 30, [0, 0], [10, 10], F)
morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
```

#### `CMGDB.ComputeConleyMorseGraph(model)`

**Purpose**: Compute the Conley-Morse graph (with Conley indices).

**Parameters**:
- `model` (Model): Model object

**Returns**:
- `morse_graph` (MorseGraph): Annotated Morse graph with Conley indices
- `map_graph` (MapGraph): The underlying transition graph

**Note**: Requires CHomP library. Computes homological Conley index for each Morse set.

#### `CMGDB.ComputeConleyIndex(...)`

**Purpose**: Compute Conley index from combinatorial index pair.

**Parameters**:
- `X_cubes` (list[uint64]): Index pair X
- `A_cubes` (list[uint64]): Index pair A
- `sizes` (list[uint64]): Grid sizes per dimension
- `periodic` (list[bool]): Periodic flags
- `F` (dict): Combinatorial map as adjacency dictionary
- `acyclic_check` (bool): Whether to check acyclicity (default: True)

**Returns**:
- `list[str]`: Conley index strings

### Box Map Functions

#### `CMGDB.BoxMap(f, rect, mode='corners', padding=False, num_pts=10)`

**Purpose**: Create a box map from a point map function.

**Parameters**:
- `f` (callable): Point map function `x -> f(x)`
- `rect` (list[float]): Input rectangle `[x_min, y_min, ..., x_max, y_max, ...]`
- `mode` (str): Evaluation mode
  - `'corners'`: Evaluate at all corner points (default)
  - `'center'`: Evaluate at center point
  - `'random'`: Evaluate at random points
- `padding` (bool): Add padding equal to box size (default: False)
- `num_pts` (int): Number of random points if `mode='random'` (default: 10)

**Returns**:
- `list[float]`: Output rectangle containing image

**Example**:
```python
def f(x):
    return [x[0]**2, x[1]**2]

def F(rect):
    return CMGDB.BoxMap(f, rect, padding=True)
```

#### `CMGDB.BoxMapData`

**Purpose**: Create box map from datasets (data-driven dynamics).

**Class**: `CMGDB.BoxMapData`

**Constructor**:
```python
BoxMapData(X, Y, map_empty='interp', lower_bounds=None, upper_bounds=None, 
           domain_padding=True, padding=False)
```

**Parameters**:
- `X` (array): Input points (N×d array)
- `Y` (array): Output points (N×d array), images of X
- `map_empty` (str): Behavior when no points in rect
  - `'interp'`: Interpolate by expanding rectangle (default)
  - `'outside'`: Map to rectangle outside domain
  - `'terminate'`: Raise exception
- `lower_bounds` (list[float]): Domain lower bounds (required if `map_empty='outside'`)
- `upper_bounds` (list[float]): Domain upper bounds (required if `map_empty='outside'`)
- `domain_padding` (bool): Expand domain rectangle before mapping (default: True)
- `padding` (bool): Add padding to output rectangle (default: False)

**Methods**:
- `__call__(rect)`: Compute image rectangle (alias for `compute`)
- `compute(rect)`: Compute image rectangle
- `map_points(rect)`: Return points in Y that are images of points in X inside rect
- `interpolate(rect)`: Expand rectangle until non-empty

**Example**:
```python
X = np.array([[0, 0], [1, 1], [2, 2]])
Y = np.array([[1, 1], [2, 2], [3, 3]])
box_map = CMGDB.BoxMapData(X, Y, map_empty='interp')
rect_image = box_map([0.5, 0.5, 1.5, 1.5])
```

### Visualization Functions

#### `CMGDB.PlotMorseGraph(morse_graph, cmap=None, clist=None, shape=None, margin=None)`

**Purpose**: Visualize the Morse graph as a directed graph using Graphviz.

**Parameters**:
- `morse_graph` (MorseGraph): Morse graph to plot
- `cmap` (matplotlib.colors.Colormap): Colormap for vertices (default: tab20)
- `clist` (list[str]): List of colors (alternative to cmap)
- `shape` (str): Graphviz node shape (default: 'ellipse')
- `margin` (str): Node margin (default: '0.11, 0.055')

**Returns**:
- `graphviz.Source`: Graphviz object (renders in Jupyter)

**Features**:
- Colors vertices by index
- Shows annotations if present
- Ranks attractors (nodes without children) at same level

#### `CMGDB.PlotMorseSets(morse_sets, morse_nodes=None, proj_dims=None, cmap=None, clist=None, fig_w=8, fig_h=8, xlim=None, ylim=None, axis_labels=True, xlabel='$x$', ylabel='$y$', fontsize=15, fig_fname=None, dpi=300)`

**Purpose**: Plot Morse sets in phase space (2D scatter plot).

**Parameters**:
- `morse_sets`: MorseGraph object, filename, or list of boxes
- `morse_nodes` (list[int]): Indices of Morse sets to plot (default: all)
- `proj_dims` (list[int]): Dimensions to project onto for 3D+ systems (default: [0, 1])
- `cmap`: Matplotlib colormap
- `clist`: List of colors
- `fig_w`, `fig_h` (float): Figure dimensions in inches
- `xlim`, `ylim` (list[float]): Axis limits
- `axis_labels` (bool): Show axis labels
- `xlabel`, `ylabel` (str): Axis labels
- `fontsize` (int): Font size
- `fig_fname` (str): Filename to save figure
- `dpi` (int): DPI for saved figure

**Returns**: None (displays plot)

**Note**: For 1D systems, adds fake second dimension. For 3D+ systems, projects onto specified dimensions.

#### `CMGDB.PlotMorseSets3D(morse_sets, morse_nodes=None, cmap=None, clist=None, fig_w=10, fig_h=10, alpha=0.3, elev=30, azim=45, xlim=None, ylim=None, zlim=None, axis_labels=True, xlabel='$x$', ylabel='$y$', zlabel='$z$', fontsize=15, fig_fname=None, dpi=300)`

**Purpose**: Plot 3D Morse sets using cuboid visualization.

**Parameters**:
- `morse_sets`: MorseGraph object, filename, or list of boxes
- `morse_nodes` (list[int]): Indices to plot (default: all)
- `cmap`: Matplotlib colormap
- `clist`: List of colors
- `fig_w`, `fig_h` (float): Figure dimensions
- `alpha` (float): Transparency (0.0-1.0, default: 0.3)
- `elev` (float): Elevation angle in degrees (default: 30)
- `azim` (float): Azimuth angle in degrees (default: 45)
- `xlim`, `ylim`, `zlim` (list[float]): Axis limits
- `axis_labels` (bool): Show labels
- `xlabel`, `ylabel`, `zlabel` (str): Axis labels
- `fontsize` (int): Font size
- `fig_fname` (str): Save filename
- `dpi` (int): DPI for saved figure

**Returns**: None (displays 3D plot)

**Note**: Requires 3D data. Uses `Poly3DCollection` for cuboid rendering.

### Data Persistence Functions

#### `CMGDB.SaveMorseGraphData(morse_graph, map_graph, filename, metadata=None)`

**Purpose**: Save complete Morse graph computation results.

**Parameters**:
- `morse_graph` (MorseGraph): Morse graph object
- `map_graph` (MapGraph): Map graph object
- `filename` (str): Output filename (adds .mgdb extension if missing)
- `metadata` (dict): Optional metadata dictionary

**Saves**:
- All Morse graph vertices, edges, boxes
- Annotations for each Morse set
- Map graph adjacency information (sampled)
- Metadata (user-provided + automatic stats)

**File Format**: Pickle (.mgdb extension)

#### `CMGDB.LoadMorseGraphData(filename)`

**Purpose**: Load saved Morse graph data.

**Parameters**:
- `filename` (str): Input .mgdb filename

**Returns**:
- `dict`: Dictionary with keys:
  - `'morse_graph'`: Dictionary with vertices, edges, morse_sets, annotations
  - `'map_graph'`: Dictionary with adjacencies
  - `'metadata'`: Metadata dictionary
  - `'version'`: File format version

**Note**: Returns Python dictionaries, not C++ objects.

#### `CMGDB.SaveComputationResults(morse_graph, map_graph, filename, model_params=None, runtime_info=None)`

**Purpose**: High-level save function with automatic metadata.

**Parameters**:
- `morse_graph` (MorseGraph): Morse graph
- `map_graph` (MapGraph): Map graph
- `filename` (str): Output filename
- `model_params` (dict): Model parameters (subdiv_min, subdiv_max, etc.)
- `runtime_info` (dict): Runtime information (time, date, etc.)

#### `CMGDB.GetMorseSetsFromData(data)`

**Purpose**: Extract Morse sets from loaded data for plotting.

**Parameters**:
- `data` (dict): Data dictionary from `LoadMorseGraphData`

**Returns**:
- `list`: List of boxes `[x_min, y_min, ..., x_max, y_max, ..., node_index]`

#### `CMGDB.SaveMorseSets(morse_graph, morse_graph_fname)`

**Purpose**: Save Morse sets to CSV file.

**Parameters**:
- `morse_graph` (MorseGraph): Morse graph object
- `morse_graph_fname` (str): Output CSV filename

**Format**: CSV with columns: `x_min, y_min, ..., x_max, y_max, ..., node_index`

#### `CMGDB.LoadMorseSetFile(morse_set_fname)`

**Purpose**: Load Morse sets from CSV file.

**Parameters**:
- `morse_set_fname` (str): Input CSV filename

**Returns**:
- `list`: List of boxes `[x_min, y_min, ..., x_max, y_max, ..., node_index]`

### Helper Functions

#### `CMGDB.CornerPoints(rect)`

**Purpose**: Get corner points of a rectangle.

**Parameters**:
- `rect` (list[float]): Rectangle `[x_min, y_min, ..., x_max, y_max, ...]`

**Returns**:
- `list`: List of corner points (2^d points for d-dimensional rectangle)

#### `CMGDB.CenterPoint(rect)`

**Purpose**: Get center point of a rectangle.

**Parameters**:
- `rect` (list[float]): Rectangle

**Returns**:
- `list`: Single-element list containing center point

#### `CMGDB.SamplePoints(lower_bounds, upper_bounds, num_pts)`

**Purpose**: Sample random points in a hyperrectangle.

**Parameters**:
- `lower_bounds` (list[float]): Lower bounds
- `upper_bounds` (list[float]): Upper bounds
- `num_pts` (int): Number of points to sample

**Returns**:
- `list`: List of sampled points

### MorseGraph Object Methods

The `MorseGraph` object returned by computation functions provides:

**Methods**:
- `num_vertices()`: Number of Morse sets (vertices)
- `vertices()`: List of vertex indices
- `edges()`: List of edges `[(source, target), ...]`
- `adjacencies(vertex)`: List of adjacent vertices (children)
- `morse_set(vertex)`: List of grid element indices in Morse set
- `morse_set_boxes(vertex)`: List of boxes `[[x_min, y_min, ..., x_max, y_max, ...], ...]`
- `phase_space_box(index)`: Get box geometry for grid element index
- `annotations(vertex)`: List of annotation strings for vertex

**Properties**:
- Vertices are integers starting from 0
- Edges represent reachability relation
- Morse sets are collections of grid boxes

### MapGraph Object Methods

The `MapGraph` object provides:

**Methods**:
- `num_vertices()`: Number of vertices (grid elements)
- `adjacencies(vertex)`: List of adjacent vertices (image boxes)

**Note**: Adjacencies are computed on-demand to avoid storing large adjacency lists.

---

## C++ Backend Components

### Database Components

Location: `src/CMGDB/_cmgdb/include/database/`

#### Core Classes

##### `Model` (`Model.h`)

**Purpose**: Encapsulates system configuration and parameters.

**Key Members**:
- `Configuration config_`: Configuration object
- `std::shared_ptr<EuclideanParameterSpace> parameter_space_`: Parameter space
- `std::shared_ptr<Map> map_`: Map function object

**Methods**:
- Constructors: Multiple variants for different parameter combinations
- `initialize(...)`: Initialize with parameters
- `parameterSpace()`: Get parameter space
- `phaseSpace()`: Get phase space grid
- `map()`: Get map function
- `annotate(MorseGraph*)`: Annotate Morse graph

##### `MorseGraph` (`MorseGraph.h`)

**Purpose**: Directed acyclic graph representing Morse decomposition.

**Key Members**:
- `int num_vertices_`: Number of vertices
- `std::unordered_set<Edge> edges_`: Edge set
- `std::shared_ptr<Grid> phasespace_`: Phase space grid
- `std::vector<std::shared_ptr<Grid>> grids_`: Grid for each vertex
- `std::vector<std::shared_ptr<chomp::ConleyIndex_t>> conleyindexes_`: Conley indices
- `std::vector<std::set<std::string>> annotation_by_vertex_`: Annotations

**Methods**:
- `AddVertex()`: Create new vertex
- `AddEdge(from, to)`: Add edge
- `RemoveEdge(from, to)`: Remove edge
- `NumVertices()`: Get vertex count
- `Vertices()`: Get vertex iterator pair
- `Edges()`: Get edge iterator pair
- `grid(vertex)`: Get grid for vertex
- `conleyIndex(vertex)`: Get Conley index for vertex
- `annotation(vertex)`: Get annotations for vertex
- `clearGrids()`: Remove grids (save memory)

**Python Bindings**:
- `num_vertices()`, `vertices()`, `edges()`, `adjacencies()`
- `morse_set(vertex)`, `morse_set_boxes(vertex)`
- `phase_space_box(index)`, `annotations(vertex)`

##### `MapGraph` (`MapGraph.h`)

**Purpose**: Graph representation of multi-valued map on grid.

**Key Members**:
- `std::shared_ptr<const Grid> grid_`: Grid
- `std::shared_ptr<const Map> f_`: Map function
- `bool stored_graph`: Whether graph is stored in memory
- `std::vector<std::vector<Vertex>> adjacency_lists_`: Stored adjacencies (if stored)

**Methods**:
- `adjacencies(vertex)`: Get out-neighbors of vertex
- `num_vertices()`: Get vertex count
- `compute_adjacencies(vertex)`: Compute adjacencies on-demand

**Note**: By default, adjacencies computed on-demand. Can store graph if `CMDB_STORE_GRAPH` defined.

##### `Grid` (`Grid.h`)

**Purpose**: Abstract base class for grid representations.

**Key Methods** (virtual):
- `clone()`: Create copy
- `subdivide()`: Subdivide grid
- `subgrid(elements)`: Create subgrid
- `subset(other)`: Find subset in other grid
- `geometry(element)`: Get geometry object for element
- `cover(geo)`: Find grid elements covering geometry
- `memory()`: Memory usage

**Subclasses**:
- `UniformGrid`: Uniform rectangular grid
- `TreeGrid`: Tree-based adaptive grid
- `PointerGrid`: Pointer-based tree grid
- `SuccinctGrid`: Succinct tree grid

##### `Map` (`Map.h`)

**Purpose**: Abstract base class for multi-valued maps.

**Key Methods** (virtual):
- `operator()(geo)`: Apply map to geometry, return geometry

**Subclasses**:
- `ModelMap`: Map from model file
- `ModelMapF`: Map from function object
- `ModelPointMapF`: Point map from function object
- `ChompMap`: Map adapter for CHomP

##### `TreeGrid` (`TreeGrid.h`)

**Purpose**: Tree-based adaptive grid with hierarchical subdivision.

**Key Features**:
- Supports adaptive mesh refinement
- Efficient storage using tree structure
- Can interface with CHomP library
- Methods for depth, subdivision, subset operations

**Methods**:
- `getDepth(subset)`: Get depth of subset
- `subset(other_grid)`: Find subset in other grid
- `subdivide()`: Subdivide all leaves
- `cover(geo)`: Find covering elements

##### `UniformGrid` (`UniformGrid.h`)

**Purpose**: Uniform rectangular grid.

**Key Features**:
- Simple uniform subdivision
- Efficient for regular domains
- Fixed resolution

##### `PointerGrid` (`PointerGrid.h`)

**Purpose**: Pointer-based tree grid implementation.

**Key Features**:
- Uses pointer tree structure
- Default grid type for phase space (`PHASE_GRID`)

##### `SuccinctGrid` (`SuccinctGrid.h`)

**Purpose**: Succinct tree grid using SDSL library.

**Key Features**:
- Memory-efficient representation
- Uses succinct data structures
- Good for large grids

##### `Compute_Morse_Graph` (`Compute_Morse_Graph.h`, `.hpp`)

**Purpose**: Core algorithm for computing Morse graph.

**Function Signature**:
```cpp
void Compute_Morse_Graph(
    MorseGraph* MG,
    std::shared_ptr<Grid> phase_space,
    std::shared_ptr<const Map> interval_map,
    unsigned int Init,
    unsigned int Min,
    unsigned int Max,
    unsigned int Limit
);
```

**Algorithm**:
1. Initialize grid at `Init` subdivision level
2. Build transition graph by evaluating map on each box
3. Find strongly connected components (SCCs)
4. For each SCC:
   - If size > threshold, subdivide and recurse
   - Otherwise, create Morse set vertex
5. Compute reachability relation between Morse sets
6. Stop when `Max` subdivision reached or `Limit` exceeded

**Key Classes**:
- `MorseDecomposition`: Manages SCC computation and subdivision
- Uses graph algorithms for SCC finding
- Adaptive refinement based on SCC size

##### `ParameterSpace` (`ParameterSpace.h`)

**Purpose**: Manages parameter space for parameterized systems.

**Key Classes**:
- `Parameter`: Abstract parameter point
- `ParameterPatch`: Parameter region
- `ParameterSpace`: Collection of parameters

**Methods**:
- `size()`: Number of parameters
- `parameter(index)`: Get parameter at index
- Iterators for parameter enumeration

##### `EuclideanParameterSpace` (`EuclideanParameterSpace.h`)

**Purpose**: Euclidean parameter space implementation.

**Key Features**:
- Rectangular parameter domains
- Grid-based parameter sampling
- Integration with `Model` class

##### Geometry Classes

**`Geo`** (`Geo.h`): Abstract base for geometric objects

**`RectGeo`** (`RectGeo.h`): Rectangle geometry
- Stores lower and upper bounds
- Methods for intersection, union, etc.

**`PrismGeo`** (`PrismGeo.h`): Prism geometry (for ODEs)
- Represents flow tubes
- Used for continuous-time systems

**`UnionGeo`** (`UnionGeo.h`): Union of geometries
- Container for multiple geometries

**`IntersectionGeo`** (`IntersectionGeo.h`): Intersection of geometries

##### Utility Classes

**`Tree`** (`Tree.h`): Tree data structure
- Base class for tree grids
- Methods for navigation, insertion, deletion

**`PointerTree`** (`PointerTree.h`): Pointer-based tree
- Standard tree implementation

**`SuccinctTree`** (`SuccinctTree.h`): Succinct tree
- Memory-efficient using SDSL

**`RankSelect`** (`RankSelect.h`): Rank/select operations
- Used in succinct data structures

**`simple_interval`** (`simple_interval.h`): Interval arithmetic
- Template class for intervals
- Operations: +, -, *, /, exp, log, etc.

**`conleyIndexString`** (`conleyIndexString.h`): Conley index formatting
- Converts Conley index to string representation
- Handles polynomial invariants

**`GraphTheory`** (`GraphTheory.h`, `.hpp`): Graph algorithms
- SCC computation
- Reachability analysis
- Graph traversal

**`Configuration`** (`Configuration.h`): Configuration management
- Stores computation parameters
- Serialization support

**`SingleOutput`** (`SingleOutput.h`): Output formatting
- Formats computation results
- Graphviz output
- Text output

### CHomP Components

Location: `src/CMGDB/_cmgdb/include/chomp/`

#### Core Classes

##### `ConleyIndex_t` (`ConleyIndex.h`)

**Purpose**: Stores Conley index data.

**Key Members**:
- `std::vector<SparseMatrix<Ring>> data_`: Homology data
- `bool undefined_`: Whether index is undefined

**Methods**:
- `data()`: Get/set homology data
- `undefined()`: Get/set undefined flag

**Computation Function**:
```cpp
template<class Grid, class Subset, class Map>
void ConleyIndex(
    ConleyIndex_t* output,
    const Grid& grid,
    const Subset& S,
    Map& F
)
```

**Algorithm**:
1. Construct combinatorial index pair (X, A) from Morse set S
2. Build restricted map G: (X, A) -> (X, A)
3. Compute relative map homology G_*
4. Return homology groups

##### `MorseComplex` (`MorseComplex.h`)

**Purpose**: Morse complex for homology computation.

**Key Features**:
- Derived from `Complex`
- Uses decomposition (e.g., `CoreductionDecomposer`)
- Provides Morse theory operations

**Methods**:
- `boundary(chain, dim)`: Compute boundary
- `coboundary(chain, dim)`: Compute coboundary
- `include(chain)`: Include chain from base complex
- `project(chain)`: Project chain to Morse complex
- `lift(chain)`: Lift chain from Morse to base
- `lower(chain)`: Lower chain from base to Morse

##### `Complex` (`Complex.h`)

**Purpose**: Abstract chain complex.

**Key Methods**:
- `size(dim)`: Number of cells in dimension
- `dimension()`: Maximum dimension
- `boundary(chain, dim)`: Compute boundary
- `coboundary(chain, dim)`: Compute coboundary

##### `Chain` (`Chain.h`)

**Purpose**: Represents a chain (formal sum of cells).

**Key Features**:
- Stores terms with coefficients
- Operations: +, -, scalar multiplication
- Dimension tracking

##### `SparseMatrix` (`SparseMatrix.h`)

**Purpose**: Sparse matrix for boundary/coboundary operators.

**Key Features**:
- Efficient storage of sparse matrices
- Operations: multiplication, transpose, etc.
- Used in homology computation

##### `RelativeMapHomology` (`RelativeMapHomology.h`)

**Purpose**: Compute relative map homology.

**Function**:
```cpp
int RelativeMapHomology(
    std::vector<SparseMatrix<Ring>>* output,
    const Grid& grid_X,
    const Subset& X,
    const Subset& A,
    const Grid& grid_Y,
    const Subset& Y,
    const Subset& B,
    const Map& F,
    int depth
)
```

**Algorithm**:
1. Build relative chain complexes
2. Compute induced map on homology
3. Return homology groups with maps

##### `CubicalComplex` (`CubicalComplex.h`)

**Purpose**: Cubical complex for grid-based homology.

**Key Features**:
- Represents grid as cubical complex
- Computes boundary operators
- Integrates with grid structures

##### `Decomposer` (`Decomposer.h`)

**Purpose**: Abstract base for decomposition algorithms.

**Subclasses**:
- `CoreductionDecomposer`: Coreduction-based decomposition

##### `CoreductionDecomposer` (`CoreductionDecomposer.h`)

**Purpose**: Coreduction algorithm for Morse complex.

**Key Features**:
- Reduces complex while preserving homology
- Identifies critical cells
- Builds Morse complex

##### `Field` (`Field.h`)

**Purpose**: Field for homology coefficients.

**Key Types**:
- Integer fields (Z/pZ)
- Rational fields

##### `Ring` (`Ring.h`)

**Purpose**: Ring for homology coefficients.

**Key Features**:
- Supports various coefficient rings
- Used in matrix operations

##### `PolyRing` (`PolyRing.h`)

**Purpose**: Polynomial ring.

**Key Features**:
- Used for Conley index invariants
- Polynomial operations

##### `Matrix` (`Matrix.h`)

**Purpose**: Dense matrix operations.

**Key Features**:
- Used in homology computation
- Matrix operations

##### `SmithNormalForm` (`SmithNormalForm.h`)

**Purpose**: Compute Smith normal form.

**Key Features**:
- Used in homology computation
- Diagonalization of integer matrices

##### `FrobeniusNormalForm` (`FrobeniusNormalForm.h`)

**Purpose**: Compute Frobenius normal form.

**Key Features**:
- Used for Conley index invariants
- Polynomial invariants

##### `Generators` (`Generators.h`)

**Purpose**: Compute homology generators.

**Key Features**:
- Finds representative cycles
- Used in Conley index computation

##### `FiberComplex` (`FiberComplex.h`)

**Purpose**: Fiber complex for relative homology.

**Key Features**:
- Used in relative map homology
- Fiber product construction

##### `RelativePair` (`RelativePair.h`)

**Purpose**: Represents relative pair (X, A).

**Key Features**:
- Used in relative homology
- Pair operations

##### `Preboundary` (`Preboundary.h`)

**Purpose**: Preboundary operator.

**Key Features**:
- Used in decomposition algorithms
- Boundary preimage computation

##### `Prism` (`Prism.h`)

**Purpose**: Prism operator for homotopy.

**Key Features**:
- Used in homology computation
- Homotopy operations

##### `BitmapSubcomplex` (`BitmapSubcomplex.h`)

**Purpose**: Bitmap-based subcomplex.

**Key Features**:
- Efficient subcomplex representation
- Used in large-scale computations

##### `Algebra` (`Algebra.h`)

**Purpose**: Algebraic operations.

**Key Features**:
- Ring/field operations
- Used throughout CHomP

##### `Closure` (`Closure.h`)

**Purpose**: Closure operations.

**Key Features**:
- Topological closure
- Used in complex operations

##### `Rect` (`Rect.h`)

**Purpose**: Rectangle operations.

**Key Features**:
- Used in geometry
- Rectangle arithmetic

##### `Real` (`Real.h`)

**Purpose**: Real number type.

**Key Features**:
- Typedef for double
- Used throughout

---

## Data Structures

### Grid Element Representation

Grid elements are represented as `uint64_t` indices. The grid provides:
- Mapping from index to geometry (box)
- Mapping from geometry to indices (cover)

### Box Representation

Boxes are represented as:
- `[x_min, y_min, z_min, ..., x_max, y_max, z_max, ...]` (2d elements)
- For Morse sets: `[x_min, y_min, ..., x_max, y_max, ..., node_index]` (with node index)

### Geometry Representation

Geometries are represented as shared pointers to `Geo` objects:
- `RectGeo`: Rectangle with lower/upper bounds
- `PrismGeo`: Prism for flow tubes
- `UnionGeo`: Union of geometries
- `IntersectionGeo`: Intersection of geometries

### Graph Representation

**Morse Graph**:
- Vertices: Integers (Morse set indices)
- Edges: Unordered set of pairs (source, target)
- Each vertex has associated grid (Morse set boxes)

**Map Graph**:
- Vertices: Grid element indices
- Edges: Computed on-demand from map evaluation
- Adjacencies: List of target vertices

---

## Algorithms

### Morse Graph Computation

**Algorithm**: `Compute_Morse_Graph`

1. **Initialization**:
   - Create grid at initial subdivision level
   - Initialize empty Morse graph

2. **Transition Graph Construction**:
   - For each grid box:
     - Evaluate map on box geometry
     - Find covering boxes (image)
     - Add edges to transition graph

3. **Strongly Connected Component (SCC) Computation**:
   - Find all SCCs in transition graph
   - Identify trivial SCCs (single vertex, no self-loop)
   - Identify non-trivial SCCs (Morse sets)

4. **Adaptive Refinement**:
   - For each non-trivial SCC:
     - If size > threshold and subdivision < max:
       - Subdivide boxes in SCC
       - Recurse on subgrid
     - Otherwise:
       - Create Morse set vertex
       - Store grid for vertex

5. **Reachability Computation**:
   - Compute reachability between Morse sets
   - Add edges to Morse graph

6. **Termination**:
   - Stop when max subdivision reached or complexity limit exceeded

### Conley Index Computation

**Algorithm**: `ConleyIndex`

1. **Index Pair Construction**:
   - Given Morse set S:
     - X = S ∪ F(S) (forward image)
     - A = F(S) \ S (forward image minus set)

2. **Restricted Map Construction**:
   - G: (X, A) -> (X, A)
   - G(x) = F(x) ∩ A

3. **Relative Map Homology**:
   - Compute relative chain complexes
   - Compute induced map on homology
   - Return homology groups with maps

4. **Invariant Computation**:
   - Compute polynomial invariants
   - Format as strings

### SCC Computation

Uses standard graph algorithms:
- Tarjan's algorithm or similar
- Efficient for large graphs
- Handles cycles and strongly connected components

### Adaptive Mesh Refinement

**Strategy**:
- Subdivide boxes in large SCCs
- Refine until SCC size below threshold
- Balance between resolution and complexity

**Criteria**:
- SCC size > threshold
- Subdivision level < maximum
- Complexity < limit

---

## File Structure

```
cmgdb/
├── README.md                    # Package overview
├── CMGDB_3D_FEATURES.md        # 3D features documentation
├── instructions.md             # This file
├── setup.py                     # Python package setup
├── pyproject.toml              # Package metadata
├── CMakeLists.txt              # CMake build configuration
├── install.sh                  # Installation script
├── LICENSE                     # License file
├── MANIFEST.in                 # Package manifest
│
├── src/CMGDB/                  # Python package
│   ├── __init__.py            # Package initialization
│   ├── BoxMapData.py          # Data-driven box map
│   ├── ComputeBoxMap.py       # Box map helpers
│   ├── PlotMorseGraph.py      # Graph visualization
│   ├── PlotMorseSets.py       # Phase space visualization (2D/3D)
│   ├── SaveMorseGraphData.py  # Data persistence
│   ├── SaveMorseData.py       # CSV save
│   ├── LoadMorseSetFile.py    # CSV load
│   │
│   └── _cmgdb/                # C++ extension module
│       ├── CMGDB.cpp          # Main binding file
│       └── include/
│           ├── chomp/         # CHomP library headers
│           │   ├── ConleyIndex.h
│           │   ├── MorseComplex.h
│           │   ├── Complex.h
│           │   ├── Chain.h
│           │   ├── SparseMatrix.h
│           │   ├── RelativeMapHomology.h
│           │   ├── CubicalComplex.h
│           │   ├── Decomposer.h
│           │   ├── CoreductionDecomposer.h
│           │   ├── Field.h
│           │   ├── Ring.h
│           │   ├── PolyRing.h
│           │   ├── Matrix.h
│           │   ├── SmithNormalForm.h
│           │   ├── FrobeniusNormalForm.h
│           │   ├── Generators.h
│           │   ├── FiberComplex.h
│           │   ├── RelativePair.h
│           │   ├── Preboundary.h
│           │   ├── Prism.h
│           │   ├── BitmapSubcomplex.h
│           │   ├── Algebra.h
│           │   ├── Closure.h
│           │   ├── Rect.h
│           │   ├── Real.h
│           │   └── ...
│           │
│           └── database/       # CMGDB core headers
│               ├── Model.h
│               ├── MorseGraph.h
│               ├── MapGraph.h
│               ├── Grid.h
│               ├── Map.h
│               ├── TreeGrid.h
│               ├── UniformGrid.h
│               ├── PointerGrid.h
│               ├── SuccinctGrid.h
│               ├── Compute_Morse_Graph.h
│               ├── Compute_Morse_Graph.hpp
│               ├── ParameterSpace.h
│               ├── EuclideanParameterSpace.h
│               ├── Geo.h
│               ├── RectGeo.h
│               ├── PrismGeo.h
│               ├── UnionGeo.h
│               ├── IntersectionGeo.h
│               ├── Tree.h
│               ├── PointerTree.h
│               ├── SuccinctTree.h
│               ├── RankSelect.h
│               ├── simple_interval.h
│               ├── conleyIndexString.h
│               ├── GraphTheory.h
│               ├── GraphTheory.hpp
│               ├── Configuration.h
│               ├── SingleOutput.h
│               ├── ModelMap.h
│               ├── ModelMapF.h
│               ├── ModelPointMapF.h
│               ├── ChompMap.h
│               ├── Atlas.h
│               ├── AtlasGeo.h
│               ├── EdgeGrid.h
│               ├── CompressedTree.h
│               ├── CompressedTreeGrid.h
│               ├── join.h
│               ├── Map.h
│               ├── Real.h
│               └── ...
│
├── examples/                   # Example notebooks
│   ├── BoxMap_Data_Examples.ipynb
│   ├── Conley_Index_Examples.ipynb
│   ├── Example_Graph_Map.ipynb
│   ├── Example_Leslie_model.ipynb
│   ├── Examples.ipynb
│   ├── Gaussian_Process_Example.ipynb
│   ├── Interval_Example.ipynb
│   ├── ODE_Computations_Examples.ipynb
│   ├── Order_Retraction_Example.ipynb
│   ├── Periodic_Domain_Examples.ipynb
│   └── ...
│
└── tests/                      # Test files
    └── test_basic.py
```

---

## Usage Patterns

### Basic Usage: Function-Based Dynamics

```python
import CMGDB
import math

# Define point map
def f(x):
    return [x[0]**2, 0.5*x[1]]

# Define box map
def F(rect):
    return CMGDB.BoxMap(f, rect, padding=True)

# Create model
model = CMGDB.Model(
    subdiv_min=20,
    subdiv_max=30,
    lower_bounds=[0, 0],
    upper_bounds=[10, 10],
    F=F
)

# Compute Morse graph
morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)

# Visualize
CMGDB.PlotMorseGraph(morse_graph)
CMGDB.PlotMorseSets(morse_graph)
```

### Data-Driven Dynamics

```python
import CMGDB
import numpy as np

# Load or generate data
X = np.array([[0, 0], [1, 1], [2, 2], ...])  # Input points
Y = np.array([[1, 1], [2, 2], [3, 3], ...])  # Output points

# Create box map from data
box_map = CMGDB.BoxMapData(
    X, Y,
    map_empty='interp',
    lower_bounds=[0, 0],
    upper_bounds=[10, 10]
)

# Create model
model = CMGDB.Model(
    subdiv_min=20,
    subdiv_max=30,
    lower_bounds=[0, 0],
    upper_bounds=[10, 10],
    F=box_map
)

# Compute
morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
```

### Conley Index Computation

```python
# Compute Conley-Morse graph (with indices)
morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(model)

# Access annotations (Conley index strings)
for vertex in morse_graph.vertices():
    annotations = morse_graph.annotations(vertex)
    print(f"Vertex {vertex}: {annotations}")
```

### 3D Visualization

```python
# Plot 3D Morse sets
CMGDB.PlotMorseSets3D(
    morse_graph,
    alpha=0.3,
    elev=30,
    azim=45,
    fig_fname="morse_3d.png"
)

# Different viewing angles
CMGDB.PlotMorseSets3D(morse_graph, elev=90, azim=0)  # Top view
CMGDB.PlotMorseSets3D(morse_graph, elev=0, azim=0)   # Side view
```

### Data Persistence

```python
# Save computation results
CMGDB.SaveMorseGraphData(
    morse_graph,
    map_graph,
    "results.mgdb",
    metadata={
        'model': 'Leslie map',
        'parameters': {'theta1': 19.6, 'theta2': 23.68},
        'computation_time': 45.2
    }
)

# Load results
data = CMGDB.LoadMorseGraphData("results.mgdb")
morse_sets = CMGDB.GetMorseSetsFromData(data)
CMGDB.PlotMorseSets(morse_sets)
```

### Advanced: Custom Box Map

```python
def custom_box_map(rect):
    dim = len(rect) // 2
    lower = rect[:dim]
    upper = rect[dim:]
    
    # Custom logic to compute image
    # Must return rectangle [x_min, y_min, ..., x_max, y_max, ...]
    image_lower = [f(x) for x in corners]  # Evaluate at corners
    image_upper = [max(...), max(...)]      # Compute bounds
    
    return image_lower + image_upper

model = CMGDB.Model(20, 30, [0, 0], [10, 10], custom_box_map)
```

### Periodic Boundaries

```python
# Periodic boundaries in first dimension
model = CMGDB.Model(
    subdiv_min=20,
    subdiv_max=30,
    lower_bounds=[0, 0],
    upper_bounds=[2*math.pi, 10],
    periodic=[True, False],  # Periodic in x, not y
    F=F
)
```

### Selective Plotting

```python
# Plot only specific Morse sets
CMGDB.PlotMorseSets(
    morse_graph,
    morse_nodes=[0, 2, 5],  # Only plot these sets
    cmap=matplotlib.cm.viridis
)

# 3D projection for higher-dimensional systems
CMGDB.PlotMorseSets(
    morse_graph,
    proj_dims=[0, 2]  # Project onto dimensions 0 and 2
)
```

---

## Key Design Decisions

### Grid Types

- **PointerGrid**: Default for phase space, good balance of performance and memory
- **UniformGrid**: Simple, efficient for uniform domains
- **SuccinctGrid**: Memory-efficient for large grids

### Map Evaluation

- **On-demand**: Adjacencies computed when needed (default)
- **Stored**: Can store graph if `CMDB_STORE_GRAPH` defined (for small graphs)

### Adaptive Refinement

- Subdivides large SCCs automatically
- Balances resolution with computational cost
- Stops at max subdivision or complexity limit

### Conley Index

- Computed only when `ComputeConleyMorseGraph` called
- Requires CHomP library
- Can be expensive for large Morse sets

---

## Dependencies

### Python
- `numpy>=1.19`: Numerical arrays
- `matplotlib>=3.3`: Visualization
- `graphviz>=0.16`: Graph visualization

### C++
- **C++11** compatible compiler
- **Boost**: Serialization, iterators, hash
- **GMP**: Arbitrary precision arithmetic
- **SDSL**: Succinct data structures (optional, for SuccinctGrid)
- **pybind11**: Python bindings

### External Libraries
- **CHomP**: Computational Homology Project (included)
- **CMGDB Core**: Database layer (included)

---

## Performance Considerations

### Grid Size
- Small grids (< 10K boxes): Fast, can store graph
- Medium grids (10K-1M boxes): On-demand adjacencies recommended
- Large grids (> 1M boxes): Use adaptive refinement, SuccinctGrid

### Subdivision Levels
- `subdiv_min`: Start resolution (lower = coarser)
- `subdiv_max`: Maximum resolution (higher = finer)
- Balance: Higher max gives better accuracy but more computation

### Conley Index
- Expensive for large Morse sets
- Consider computing only for interesting sets
- Can be parallelized (future work)

### Memory
- Grids: O(n) where n = number of boxes
- Graphs: O(n + m) where m = number of edges
- Morse graph: O(k) where k = number of Morse sets (usually small)

---

## Limitations

1. **Discrete Systems Only**: Designed for discrete-time dynamics
2. **Rectangular Domains**: Phase space must be rectangular
3. **Deterministic Maps**: Handles deterministic and multi-valued maps, not stochastic
4. **Memory**: Large grids can be memory-intensive
5. **Conley Index**: Requires CHomP, can be slow for large sets

---

## Future Extensions

Potential areas for extension:
- Continuous-time systems (ODEs) - partial support via PrismGeo
- Stochastic systems
- Parallel computation
- GPU acceleration
- More visualization options
- Interactive exploration tools

---

## References

- Conley-Morse theory
- Computational topology
- Combinatorial dynamics
- CHomP library documentation
- Original CMGDB papers

---

## Version Information

- **Current Version**: 1.3.1 (from setup.py)
- **Python Bindings**: pybind11
- **C++ Standard**: C++11
- **License**: See LICENSE file

---

## Contact and Support

- **Author**: Marcio Gameiro
- **Repository**: https://github.com/marciogameiro/CMGDB
- **Documentation**: See README.md and example notebooks

---

*This document serves as a comprehensive reference for the CMGDB package. For specific usage examples, see the `examples/` directory.*

