import numpy as np
import pytest

from MorseGraph.grids import UniformGrid

def test_uniform_grid_init():
    bounds = np.array([[0., 0.], [1., 1.]])
    divisions = np.array([10, 10])
    grid = UniformGrid(bounds, divisions)
    assert grid.dim == 2
    assert np.all(grid.divisions == divisions)
    assert np.allclose(grid.box_size, np.array([0.1, 0.1]))
    assert grid.get_boxes().shape == (100, 2, 2)

def test_uniform_grid_subdivide():
    bounds = np.array([[0., 0.], [1., 1.]])
    divisions = np.array([10, 10])
    grid = UniformGrid(bounds, divisions)
    grid.subdivide()
    assert np.all(grid.divisions == np.array([20, 20]))
    assert np.allclose(grid.box_size, np.array([0.05, 0.05]))
    assert grid.get_boxes().shape == (400, 2, 2)

def test_box_to_indices_single():
    bounds = np.array([[0., 0.], [1., 1.]])
    divisions = np.array([10, 10])
    grid = UniformGrid(bounds, divisions)
    
    # Box entirely within one grid cell
    box = np.array([[0.15, 0.15], [0.18, 0.18]])
    indices = grid.box_to_indices(box)
    assert len(indices) == 1
    assert indices[0] == 11 # index for (1,1)

def test_box_to_indices_multiple():
    bounds = np.array([[0., 0.], [1., 1.]])
    divisions = np.array([10, 10])
    grid = UniformGrid(bounds, divisions)

    # Box spanning four grid cells
    box = np.array([[0.15, 0.15], [0.25, 0.25]])
    indices = grid.box_to_indices(box)
    assert len(indices) == 4
    expected_indices = [11, 12, 21, 22] # (1,1), (1,2), (2,1), (2,2)
    assert set(indices) == set(expected_indices)

def test_box_to_indices_boundary():
    bounds = np.array([[0., 0.], [1., 1.]])
    divisions = np.array([10, 10])
    grid = UniformGrid(bounds, divisions)

    # Box that aligns with grid lines
    box = np.array([[0.1, 0.1], [0.3, 0.3]])
    indices = grid.box_to_indices(box)
    # Should cover cells (1,1), (1,2), (2,1), (2,2)
    assert len(indices) == 4
    expected_indices = [11, 12, 21, 22]
    assert set(indices) == set(expected_indices)

def test_box_to_indices_outside():
    bounds = np.array([[0., 0.], [1., 1.]])
    divisions = np.array([10, 10])
    grid = UniformGrid(bounds, divisions)

    # Box completely outside the grid
    box = np.array([[1.1, 1.1], [1.2, 1.2]])
    indices = grid.box_to_indices(box)
    assert len(indices) == 0

def test_box_to_indices_partially_outside():
    bounds = np.array([[0., 0.], [1., 1.]])
    divisions = np.array([10, 10])
    grid = UniformGrid(bounds, divisions)

    # Box partially outside the grid
    box = np.array([[0.95, 0.95], [1.05, 1.05]])
    indices = grid.box_to_indices(box)
    assert len(indices) == 1
    assert indices[0] == 99 # index for (9,9)
