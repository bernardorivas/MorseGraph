import numpy as np
import pytest

from MorseGraph.dynamics import BoxMapFunction

def test_boxmapfunction_identity():
    # Identity map
    map_f = lambda x: x
    dyn = BoxMapFunction(map_f, epsilon=0.0)
    
    box = np.array([[0.1, 0.1], [0.2, 0.2]])
    image_box = dyn(box)
    
    assert np.allclose(image_box, box)

def test_boxmapfunction_identity_with_bloating():
    # Identity map
    map_f = lambda x: x
    epsilon = 0.1
    dyn = BoxMapFunction(map_f, epsilon=epsilon)
    
    box = np.array([[0.1, 0.1], [0.2, 0.2]])
    image_box = dyn(box)
    
    expected_image_box = np.array([
        [0.1 - epsilon, 0.1 - epsilon],
        [0.2 + epsilon, 0.2 + epsilon]
    ])
    
    assert np.allclose(image_box, expected_image_box)

def test_boxmapfunction_translation():
    # Translation map
    translation = np.array([0.5, -0.5])
    map_f = lambda x: x + translation
    dyn = BoxMapFunction(map_f, epsilon=0.0)
    
    box = np.array([[0.1, 0.1], [0.2, 0.2]])
    image_box = dyn(box)
    
    expected_image_box = np.array([
        [0.1 + translation[0], 0.1 + translation[1]],
        [0.2 + translation[0], 0.2 + translation[1]]
    ])
    
    assert np.allclose(image_box, expected_image_box)

def test_boxmapfunction_scaling():
    # Scaling map
    scaling = np.array([2.0, 0.5])
    map_f = lambda x: x * scaling
    dyn = BoxMapFunction(map_f, epsilon=0.0)
    
    box = np.array([[0.1, 0.2], [0.3, 0.4]])
    image_box = dyn(box)
    
    # The corners of the box are (0.1, 0.2), (0.1, 0.4), (0.3, 0.2), (0.3, 0.4)
    # The images of the corners are:
    # (0.2, 0.1), (0.2, 0.2), (0.6, 0.1), (0.6, 0.2)
    # The center is (0.2, 0.3), image is (0.4, 0.15)
    # Bounding box of image points:
    # min: (0.2, 0.1)
    # max: (0.6, 0.2)
    
    expected_image_box = np.array([
        [0.2, 0.1],
        [0.6, 0.2]
    ])
    
    assert np.allclose(image_box, expected_image_box)
