# MorseGraph TODO

## Caching & Enhanced Visualization Implementation

### ✅ Completed

**Phase 1: CMGDB Caching System**
- ✅ Core caching utilities in `MorseGraph/utils.py`:
  - `compute_parameter_hash()` - SHA256 hash of function + parameters
  - `get_cache_path()` - Organized cache directory structure
  - `save_morse_graph_cache()` - Save MorseGraph to `.mgdb` format
  - `load_morse_graph_cache()` - Load from cache
- ✅ Integrated caching into `MorseGraph/core.py`:
  - `compute_morse_graph_3d()` - Added `cache_dir`, `use_cache`, `force_recompute` params
  - `compute_morse_graph_2d_data()` - Added caching with `model_hash` param
  - `compute_morse_graph_2d_restricted()` - Added caching with `model_hash` param

**Phase 2: Enhanced Plotting Helpers**
- ✅ Added helper functions to `MorseGraph/plot.py`:
  - `get_morse_set_colors()` - Consistent color schemes
  - `create_latent_grid()` - Generate sampling grids
  - `compute_encoded_barycenters()` - Project 3D→2D barycenters
  - `classify_points_to_morse_sets()` - Classify points by Morse set membership
  - `plot_data_boxes()` - Plot white boxes showing data coverage

---

### 🔧 In Progress / TODO

#### 1. Debug Cache Loading Issue
**Status**: Minor bug
**Location**: `MorseGraph/utils.py` - `load_morse_graph_cache()`
**Issue**: Cache saves successfully but fails to load with "input stream error" from CMGDB
**Action needed**:
- Investigate CMGDB.MorseGraph() constructor compatibility
- Check if we need to use CMGDB.LoadMorseGraphData() instead
- Verify .mgdb file format version compatibility
- Add error handling and fallback to recomputation

#### 2. Add Comprehensive Plotting Functions
**Status**: Not started
**Location**: `MorseGraph/plot.py`

Add three multi-panel visualization functions from original `ives_model/`:

##### A. `plot_latent_transformation_analysis()`
- **Layout**: 3x2 grid (6 panels)
- **Top row (3D)**:
  - Original Data
  - Decoded Grid (D applied to latent grid)
  - Reconstructed Data (D(E(X)))
- **Bottom row (2D)**:
  - Encoded Data (E(X))
  - Latent Grid (regular grid in latent space)
  - Re-encoded Grid (E(D(latent grid)))
- **Purpose**: Visualize encoder/decoder quality and round-trip transformations

##### B. `plot_morse_graph_comparison()`
- **Layout**: 2x5 grid with 5 variants
- **Variants**: default, grey, clean, minimal, no_overlay
- **Features**:
  - Grey background showing domain bounds
  - White boxes showing data coverage (`plot_data_boxes()`)
  - Points colored by Morse set membership (`classify_points_to_morse_sets()`)
  - Encoded barycenters overlaid (`compute_encoded_barycenters()`)
- **Top row**: Morse graph diagrams
- **Bottom row**: 3D barycenters or 2D latent space visualizations

##### C. `plot_preimage_classification()`
- **Layout**: 3x4 grid (12 panels)
- **Columns**: Full dataset, Train, Val, Large sample
- **Rows**:
  - 3D Preimages: E^-1(Morse Sets) - which 3D regions map to each latent Morse set
  - Morse Graph: Diagram for each method
  - Latent Morse Sets: 2D latent space with Morse sets
- **Purpose**: Show relationship between 3D dynamics and latent Morse structure

#### 3. Add Basins of Attraction Visualization
**Status**: Not started
**Location**: `MorseGraph/plot.py`

##### `plot_basins_of_attraction()`
- Integrate with existing `compute_basins_of_attraction()` in `MorseGraph/analysis.py`
- Support both 2D and 3D visualizations
- Color points by which attractor basin they belong to
- Variants for original space and latent space

**Reference**: Check `ives_model/ives_modules/plotting.py` for basin visualization patterns

#### 4. Update Example: `examples/7_ives_model.py`
**Status**: Not started

**Changes needed**:

##### A. Enable Caching
```python
cache_dir = dirs['base'] + '/cache'

# 3D Morse graph with caching
result_3d = compute_morse_graph_3d(
    config.map_func,
    config.domain_bounds,
    cache_dir=cache_dir,
    use_cache=True,
    ...
)

# 2D Morse graphs with model hash
# Compute model_hash from training config
import hashlib
import json
model_params = {
    'latent_dim': config.latent_dim,
    'hidden_dim': config.hidden_dim,
    'num_epochs': config.num_epochs,
    'random_seed': config.random_seed,
}
model_hash = hashlib.sha256(json.dumps(model_params, sort_keys=True).encode()).hexdigest()[:16]

result_2d_data = compute_morse_graph_2d_data(
    ...,
    cache_dir=cache_dir,
    model_hash=model_hash,
    use_cache=True
)
```

##### B. Use Enhanced Plotting
Replace current simplified plots with:
```python
# Transformation analysis
plot_latent_transformation_analysis(
    X, encoder, decoder, latent_dynamics, device,
    config.domain_bounds, latent_bounds,
    output_path=f"{dirs['results']}/latent_transformation_analysis.png"
)

# Morse graph comparison (all 5 variants)
for variant in ['', '_grey', '_clean', '_minimal', '_no_overlay']:
    plot_morse_graph_comparison(
        morse_graph_3d, morse_graph_2d_data, morse_graph_2d_restricted,
        barycenters_3d, z_train, z_large,
        latent_bounds, config,
        variant=variant,
        output_path=f"{dirs['results']}/morse_graph_comparison{variant}.png"
    )

# Preimage classification
plot_preimage_classification(
    [result_2d_data, result_2d_restricted],
    encoder, X, x_train, x_val, X_large,
    config.domain_bounds, latent_bounds,
    output_path=f"{dirs['results']}/preimage_classification.png"
)

# Basins of attraction
from MorseGraph.analysis import compute_basins_of_attraction
basins_3d = compute_basins_of_attraction(result_3d['morse_graph'], ...)
plot_basins_of_attraction(
    basins_3d, barycenters_3d, config.domain_bounds,
    output_path=f"{dirs['results']}/basins_of_attraction_3d.png"
)
```

##### C. Update Output Structure
```
ives_model_output/
├── cache/                                      # NEW: Cached CMGDB results
│   ├── {hash_3d}/
│   │   ├── morse_graph_data.mgdb
│   │   ├── barycenters.npz
│   │   └── metadata.json
│   ├── {hash_2d_data}/
│   └── {hash_2d_restricted}/
├── metadata.json
├── training_data/
├── models/
└── results/
    ├── latent_transformation_analysis.png      # NEW: 3x2 grid
    ├── morse_graph_comparison.png              # NEW: 2x5 grid (default)
    ├── morse_graph_comparison_grey.png         # NEW: variant
    ├── morse_graph_comparison_clean.png        # NEW: variant
    ├── morse_graph_comparison_minimal.png      # NEW: variant
    ├── morse_graph_comparison_no_overlay.png   # NEW: variant
    ├── preimage_classification.png             # NEW: 3x4 grid
    ├── basins_of_attraction_3d.png             # NEW
    └── basins_of_attraction_latent.png         # NEW
```

#### 5. Update Documentation
**Status**: Not started

##### A. `examples/README.md`
- Document caching feature
- Update example 7 output structure
- Add performance notes (caching saves 2-5 minutes)
- Document enhanced visualizations

##### B. `examples/7_ives_model_NOTES.md`
- Document new comprehensive plotting functions
- Explain each multi-panel figure
- Add cache management examples:
  ```bash
  # Check cache contents
  ls examples/ives_model_output/cache/

  # Force recomputation
  python 7_ives_model.py  # edit force_recompute=True

  # Clear cache
  rm -rf examples/ives_model_output/cache/
  ```

##### C. `examples/generic_3d_pipeline/README.md`
- Add caching usage examples
- Document enhanced plotting function signatures
- Add troubleshooting section for cache issues

#### 6. Optional: Cache Management Utilities
**Status**: Nice-to-have
**Location**: `MorseGraph/utils.py`

Add convenience functions:
```python
def list_cached_morse_graphs(cache_dir):
    """List all cached results with metadata."""
    pass

def clear_cache(cache_dir, older_than_days=None):
    """Clear cache directory, optionally filtering by age."""
    pass

def get_cache_info(cache_dir):
    """Get cache statistics (size, number of entries, age)."""
    pass
```

---

### 📋 Implementation Priority

**High Priority** (blocking user's workflow):
1. Fix cache loading bug (#1)
2. Add comprehensive plotting functions (#2)
3. Update `7_ives_model.py` (#4)

**Medium Priority** (improves usability):
4. Update documentation (#5)

**Low Priority** (nice-to-have):
5. Cache management utilities (#6)

---

### 🎯 Goal

Provide users with:
1. **Fast iteration**: Caching avoids recomputing 3D CMGDB (saves 2-5 min/run)
2. **Better visualizations**: Multi-panel figures showing comprehensive analysis
3. **Easy replication**: Clear examples in `7_ives_model.py` and `generic_3d_pipeline/`

---

### 📝 Notes

- **Cache format**: Uses CMGDB native `.mgdb` format + NumPy `.npz` for barycenters
- **Model hashing**: 2D caching requires `model_hash` parameter to avoid caching across different trained models
- **Original reference**: `ives_model/ives_modules/plotting.py` (1114 lines) has full implementation details
- **Design pattern**: All new plotting functions follow existing signature patterns (accept data + config, return nothing, save to `output_path`)
