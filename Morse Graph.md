## Implementation Roadmap

### Milestone 1: Core Workflow (Non-adaptive Core) - COMPLETE ✅

Goal: Implement the foundational classes to compute a basic Morse graph from an explicit function on a uniform grid.

#### 1. Project Infrastructure (Setup & Dependencies)

- [x] Implement pyproject.toml. Define core dependencies (numpy, scipy, networkx, matplotlib) and the optional ML group ([ml] for torch).
- [x] Set up the pytest framework in the tests/ directory.

#### 2. Dynamics Abstraction (MorseGraph/dynamics.py)

- [x] Implement Dynamics(ABC) with the abstract method __call__(self, box: np.ndarray) -> np.ndarray.
- [x] Implement BoxMapFunction(Dynamics).
- [x] Critical Detail (Rigor): In __call__, implement the "bloating" mechanism. The function must compute the bounding box of the image (e.g., by sampling corners and the center) and then slightly expand it by a configurable epsilon to guarantee an outer approximation.

#### 3. Grid Abstraction (MorseGraph/grids.py)

- [x] Implement AbstractGrid(ABC). Define the interface for get_boxes(), box_to_indices() (finding intersections), and subdivide().
- [x] Implement UniformGrid(AbstractGrid). Implement initialization, coordinate/index mappings, and the global subdivide() method.

#### 4. The Core Engine (MorseGraph/core.py)

- [x] Implement the Model class, initialized with Dynamics and AbstractGrid.
- [x] Implement compute_map_graph(self) -> nx.DiGraph. This iterates over boxes, computes their image, finds intersections, and builds the graph.
- [x] Performance Consideration: Integrate parallel processing (e.g., using joblib) for the loop that computes box images, as this is typically the main computational bottleneck.

#### 5. Analysis (MorseGraph/analysis.py)

- [x] Implement compute_morse_graph(map_graph). Use networkx.strongly_connected_components and networkx.condensation. Ensure the resulting Morse graph nodes store the list of grid indices they contain.

#### 6. Testing (M1)

- [x] tests/test_grids.py: Validate UniformGrid indexing, coordinate conversion, and subdivision.
- [x] tests/test_dynamics.py: Test BoxMapFunction with a known simple map (e.g., identity or translation) to verify bounding box calculation and bloating.

### Milestone 2: Expanded Capabilities - COMPLETE ✅

Goal: Support Data and ODE dynamics, compute basins of attraction, and implement visualization tools.

#### 1. Expanding Dynamics (MorseGraph/dynamics.py)

- [x] Implement BoxMapData(Dynamics).
- [x] Performance Consideration: In the initializer, build a spatial index on the input data X using scipy.spatial.cKDTree. This is crucial for efficient lookups in the __call__ method.
- [x] Implement BoxMapODE(Dynamics).
- [x] Use scipy.integrate.solve_ivp to integrate sampled points (corners, center) for time tau. Compute the bounding box and apply bloating.

#### 2. Advanced Analysis (MorseGraph/analysis.py)

- [x] Implement compute_basins_of_attraction(morse_graph, map_graph).
- [x] Identify attractors (terminal nodes in the morse_graph).
- [x] Perform backward graph traversal (e.g., Breadth-First Search) on the reversed map_graph starting from the attractor indices.

#### 3. Visualization (MorseGraph/plot.py)

- [x] Implement plot_morse_sets(...) and plot_basins_of_attraction(...).
- [x] Performance Consideration: Use matplotlib.collections.PatchCollection for efficient rendering of a large number of rectangles.
- [x] Implement plot_morse_graph(...) using networkx.draw.

#### 4. Testing and Examples (M2)

- [x] Add unit tests for BoxMapData (spatial lookup accuracy) and BoxMapODE.
- [x] Develop examples/1_map_dynamics.ipynb, examples/2_data_driven.ipynb, and examples/3_ode_dynamics.ipynb.

### Milestone 3: Advanced Features (Adaptivity) - COMPLETE ✅

Goal: Implement adaptive grid structures and the iterative refinement process.

#### 1. Adaptive Grids (MorseGraph/grids.py)

- [x] Implement AdaptiveGrid(AbstractGrid).
- [x] Critical Detail: Use a tree-based structure (e.g., Quadtree/Octree). The leaves represent the active boxes. Implementing efficient spatial indexing (box_to_indices) and neighbor-finding algorithms within the tree structure is complex but essential.
- [x] Implement subdivide(indices) to perform local refinement (splitting specific leaves).

#### 2. Iterative Computation (MorseGraph/analysis.py)

- [x] Implement iterative_morse_computation(model, max_depth).
- [x] The loop should: Compute Morse Graph -> Identify "recurrent" boxes (those in non-trivial Morse sets) -> Subdivide the grid locally.
- [x] Optimization: When the grid is refined locally, the map graph should ideally be updated locally as well. Only recompute the dynamics for the newly subdivided regions and their immediate neighbors, rather than recalculating the entire graph at each step.

#### 3. Testing and Examples (M3)

- [ ] tests/test_grids.py: Rigorously test AdaptiveGrid index management, tree structure integrity after local subdivision, and neighbor finding.
- [x] Develop examples/4_adaptive_refinement.ipynb.

### Milestone 4: ML Integration (Optional) - COMPLETE ✅

Goal: Implement the infrastructure for learning dynamics using PyTorch and analyzing the learned latent space.

#### 1. PyTorch Models (MorseGraph/models.py)

- [x] Define Encoder, Decoder (Autoencoder), and LatentDynamics (e.g., MLP) using torch.nn.Module.

#### 2. Training Utilities (MorseGraph/training.py)

- [x] Implement a training class or function.
- [x] Define the combined loss function: Reconstruction Loss (Autoencoder fidelity) + Dynamics Prediction Loss (accuracy in latent space).

#### 3. Learned Dynamics Wrapper (MorseGraph/dynamics.py)

- [x] Implement LearnedMapDynamics(Dynamics).
- [x] Critical Detail (Design): This class represents the dynamics in the latent space. The associated AbstractGrid used in the Model must also be defined over the latent space dimensions.
- [x] Implement __call__(self, latent_box). Apply the trained LatentDynamics model to the input latent box (using sampling/bloating as in BoxMapFunction).

#### 4. Testing and Examples (M4)

- [ ] Create tests/test_ml.py to verify model forward passes and basic training steps.
- [x] Develop examples/5_learned_dynamics.ipynb, demonstrating the full workflow from data generation and training to latent space analysis.

---

## Rigorous Outer Approximation Strategies

The MorseGraph framework requires rigorous outer approximations to ensure mathematical correctness of the computed Morse graphs. There are two complementary approaches for achieving this rigor:

### 1. Geometric Bloating (Phase Space Expansion) - IMPLEMENTED ✅

**Current Approach**: Expand box boundaries in the continuous phase space by an epsilon parameter.

**Implementation**: All dynamics classes (BoxMapFunction, BoxMapData, BoxMapODE) use this method:
- Sample points from the input box (corners, center, etc.)
- Apply the dynamics to get image points  
- Compute bounding box of image points
- Expand the bounding box by epsilon: `[min_bounds - ε, max_bounds + ε]`

**Advantages**:
- Mathematically rigorous outer approximation
- Simple to implement and understand
- Works with any dynamics function
- Parameter ε directly controls approximation quality vs. computational cost

**Limitations**:
- May be overly conservative for smooth dynamics
- ε parameter requires careful tuning for different systems
- Can lead to rapid expansion in chaotic regions

### 2. Grid Dilation (Discrete Space Expansion) - IMPLEMENTED ✅

**Proposed Approach**: Expand in the discrete grid space by including neighboring boxes around target boxes.

**Implementation Strategy**: 
```python
# In AbstractGrid or specific grid implementations
def dilate_indices(self, indices: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Expand a set of box indices to include their spatial neighbors.
    
    :param indices: Original box indices
    :param radius: Neighborhood radius (1 = immediate neighbors, 2 = second-order, etc.)
    :return: Expanded set of indices including neighbors
    """
    # For UniformGrid: use spatial indexing to find neighboring boxes
    # For AdaptiveGrid: use tree structure to find adjacent leaves
    pass

# Usage in dynamics or analysis
class GridDilatedDynamics(Dynamics):
    def __init__(self, base_dynamics: Dynamics, grid: AbstractGrid, dilation_radius: int = 1):
        self.base_dynamics = base_dynamics
        self.grid = grid
        self.radius = dilation_radius
    
    def __call__(self, box: np.ndarray) -> List[int]:
        # 1. Apply base dynamics to get target boxes
        # 2. Apply grid dilation to expand the result
        target_indices = self.base_dynamics.box_to_indices(self.base_dynamics(box))
        return self.grid.dilate_indices(target_indices, self.radius)
```

**Advantages**:
- Accounts for discretization error in a grid-aware manner
- More computationally efficient than geometric bloating for complex dynamics
- Natural integration with adaptive refinement algorithms
- Intuitive parameter (radius = number of neighboring layers)

**Limitations**:
- Requires grid-specific implementation
- Less direct mathematical interpretation than geometric bloating
- May not capture continuous dynamics behavior between grid scales

### 3. Hybrid Approach - FUTURE WORK ⚠️

**Concept**: Combine both strategies for maximum rigor:
- Apply geometric bloating (small ε) for continuous approximation
- Apply grid dilation (small radius) for discretization robustness  
- User can tune the balance between the two approaches

**Use Cases**:
- **Geometric bloating only**: Smooth, well-behaved dynamics with fine grids
- **Grid dilation only**: Rapid prototyping, coarse grids, or when dynamics are already discrete
- **Hybrid approach**: Critical applications requiring maximum rigor, chaotic dynamics, or adaptive grids

### Implementation Priority

1. **Immediate**: Document and validate current geometric bloating approach
2. **Next Phase**: Implement grid dilation for UniformGrid 
3. **Advanced**: Extend grid dilation to AdaptiveGrid and implement hybrid strategies
4. **Research**: Develop automatic parameter selection methods for both approaches

---

## Revised ML Architecture Design

This design strategy focuses on generalizing the MorseGraph architecture to support diverse dimension reduction (DR) techniques and dynamics learning methods. This is achieved by decoupling the process of dimension reduction from the process of learning the dynamics in the reduced space, allowing users to mix and match techniques seamlessly.

### Design Philosophy: Separation of Concerns

The strategy separates the data-driven analysis into three distinct stages:

1. **Reduction (Geometry)**: Learning the mapping between the high-dimensional space X and the low-dimensional latent space Z.

2. **Dynamics Learning (The Map)**: Learning the evolution rule g: Z → Z in the latent space.

3. **Morse Graph Computation (Analysis)**: Applying the rigorous BoxMap framework to the latent dynamics.

We will adopt interfaces inspired by scikit-learn (fit, transform, predict) for maximum interoperability.

### Architectural Modifications

The machine learning components will be reorganized into a dedicated learning subpackage, replacing the previous models.py and training.py.

```
MorseGraph/
├── morsegraph/
│   ├── ... (core.py, analysis.py, dynamics.py, grids.py)
│   └── learning/
│       ├── __init__.py
│       ├── reduction.py         # DR Abstractions and Implementations
│       ├── latent_dynamics.py   # Dynamics Learning Abstractions
│       └── pipelines.py         # Complex training workflows (e.g., joint AE training)
```

#### 1. morsegraph/learning/reduction.py

This module handles the geometry (encoding/decoding).

**Abstract Base Class: AbstractReducer**

Defines the interface for any DR technique.

```python
class AbstractReducer(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray = None):
        # Learns the transformation. Y is optional for dynamics-aware DR.
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        # Encodes X into the latent space Z.
        pass

    # inverse_transform (decoding) is optional
```

**Concrete Implementations:**

- **SklearnReducer**: A wrapper for standard sklearn models (e.g., PCA, KernelPCA, Isomap).
- **AEReducer**: PyTorch implementation (Encoder/Decoder).
- **IdentityReducer**: A pass-through (Z=X), useful for baseline comparisons.

#### 2. morsegraph/learning/latent_dynamics.py

This module handles learning the map g: Z → Z from latent data transitions (Z_X, Z_Y).

**Abstract Base Class: AbstractLatentDynamics**

```python
class AbstractLatentDynamics(ABC):
    @abstractmethod
    def fit(self, Z_X: np.ndarray, Z_Y: np.ndarray):
        # Trains the model on latent transitions.
        pass

    @abstractmethod
    def predict(self, Z: np.ndarray) -> np.ndarray:
        # Predicts the next state g(Z).
        pass
```

**Concrete Implementations:**

- **SklearnRegressionDynamics**: Wrapper for sklearn regressors (e.g., KNeighborsRegressor, GaussianProcessRegressor).
- **MLPDynamics**: PyTorch implementation of a Neural Network map.
- **DMDDynamics**: Dynamic Mode Decomposition (learns the best linear operator A).

#### 3. morsegraph/learning/pipelines.py

This module manages scenarios where reduction and dynamics are learned simultaneously (end-to-end).

- **AutoencoderPipeline**: Manages the joint training of an AEReducer and MLPDynamics. It optimizes a combined loss (Reconstruction Loss + Dynamics Prediction Loss) using the original data (X, Y).

#### 4. Integration with morsegraph/dynamics.py

We must integrate these learned models back into the core Dynamics framework.

**Strategy A: Data-Driven Latent Dynamics**

Users can apply a reducer and then use the existing BoxMapData directly on the transformed latent data points (Z_X, Z_Y). This requires no explicit dynamics learning.

**Strategy B: Learned Function Dynamics**

When an explicit function g is learned (via AbstractLatentDynamics), we need a wrapper to convert it into a rigorous BoxMap.

**LearnedMapDynamics(Dynamics)** (New Class):

This replaces the previous, specialized LearnedDynamics.

```python
# In morsegraph/dynamics.py
class LearnedMapDynamics(Dynamics):
    """Wraps a learned function g: Z -> Z into a rigorous BoxMap."""
    def __init__(self, dynamics_model: AbstractLatentDynamics, dimension, bloating=1e-6):
        self.g = dynamics_model.predict
        # ...

    # This __call__ takes a box IN THE LATENT SPACE (Z)
    def __call__(self, latent_box: np.ndarray) -> np.ndarray:
        # 1. Sample the latent_box (corners, center, etc.)
        # 2. Apply the learned map 'g'
        # 3. Compute the bounding box of the images and apply bloating
        # ...
```

**Optimization: LinearMapDynamics(Dynamics)** (New Class):

A specialization for linear models (like DMD). Since the dynamics are defined by a matrix A, the __call__ method can be optimized by only mapping the corners of the input box.

### Revised Implementation Tasks (Milestone 4)

- [x] **Dependencies**: Add scikit-learn to the project dependencies (likely within the optional [ml] group alongside torch).
- [x] **Restructure**: Create the morsegraph/learning package with the three new modules.
- [x] **Implement Abstractions**: Define AbstractReducer and AbstractLatentDynamics.
- [x] **Implement Sklearn Wrappers**: Implement SklearnReducer and SklearnRegressionDynamics to enable immediate use of PCA, k-NN, etc.
- [x] **Implement Core Dynamics Wrappers**: Implement LearnedMapDynamics and LinearMapDynamics in dynamics.py.
- [x] **Implement DMD**: Implement DMDDynamics.
- [x] **Refactor AE (PyTorch)**: Implement AEReducer, MLPDynamics, and the AutoencoderPipeline, migrating the logic from the original plan.
- [ ] **Expand Examples**: Update examples/5_learned_dynamics.ipynb to demonstrate different workflows (e.g., Workflow 1: PCA + Data-Driven; Workflow 2: PCA + KNN Regression; Workflow 3: Autoencoder Pipeline).