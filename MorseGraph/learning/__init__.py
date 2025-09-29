"""
MorseGraph learning subpackage.

This subpackage provides machine learning capabilities for the MorseGraph library,
including dimension reduction, dynamics learning, and training pipelines.
"""

# Import abstractions
from .reduction import AbstractReducer
from .latent_dynamics import AbstractLatentDynamics

# Import concrete implementations (with conditional imports)
try:
    from .reduction import SklearnReducer, AEReducer, IdentityReducer
    from .latent_dynamics import SklearnRegressionDynamics, MLPDynamics, DMDDynamics
    from .pipelines import AutoencoderPipeline
except ImportError:
    # Some implementations may not be available depending on dependencies
    pass

__all__ = [
    'AbstractReducer',
    'AbstractLatentDynamics',
    'SklearnReducer',
    'AEReducer', 
    'IdentityReducer',
    'SklearnRegressionDynamics',
    'MLPDynamics',
    'DMDDynamics',
    'AutoencoderPipeline'
]