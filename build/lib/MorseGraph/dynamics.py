'''
Defines the dynamics abstraction for the MorseGraph library.

This module provides the `Dynamics` abstract base class (ABC) and several
concrete implementations for different ways of defining a dynamical system:
- `BoxMapFunction`: For dynamics defined by an explicit Python function f(x).
- `BoxMapODE`: For dynamics defined by an ordinary differential equation dx/dt = f(t, x).
- `BoxMapData`: For dynamics learned from a dataset of input/output points (x, f(x)).
- `LearnedDynamics`: Placeholder for dynamics from a trained neural network.
'''

from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp


class Dynamics(ABC):
    """
    Abstract base class for dynamical systems.

    The core purpose of a Dynamics object is to take a box in the state space
    and return an outer approximation of where that box maps to under the
    system's dynamics.
    """

    @abstractmethod
    def __call__(self, box: np.ndarray) -> np.ndarray:
        """
        Maps a box to its image, returning an outer-approximating bounding box.

        :param box: A numpy array of shape (D, 2) representing the box,
                    where D is the dimension, and each row is [min, max].
        :return: A numpy array for the image box in the same format.
        """
        pass


class BoxMapFunction(Dynamics):
    """
    Represents dynamics defined by a mathematical function f: R^D -> R^D.

    The box mapping is estimated by sampling points within the box, applying the
    function, and computing the bloated bounding box of the image points.
    """

    def __init__(self, func: callable, dimension: int, sample_points: int = 100, bloat_factor: float = 0.1):
        """
        :param func: A callable, vectorized function that takes a numpy array (N, D)
                     and returns a numpy array (N, D).
        :param dimension: The dimension D of the state space.
        :param sample_points: Number of points to sample inside a box to estimate its image.
        :param bloat_factor: Factor by which to bloat the computed image box to ensure
                             it's an outer approximation.
        """
        self.func = func
        self.D = dimension
        self.sample_points = sample_points
        self.bloat_factor = bloat_factor

    def __call__(self, box: np.ndarray) -> np.ndarray:
        # 1. Generate 'sample_points' random points uniformly within the input 'box'.
        random_samples = np.random.rand(self.sample_points, self.D)
        box_widths = (box[:, 1] - box[:, 0]).reshape(1, -1)
        box_mins = box[:, 0].reshape(1, -1)
        samples_in_box = random_samples * box_widths + box_mins

        # 2. Apply the function f to the samples.
        try:
            image_points = self.func(samples_in_box)
        except Exception as e:
            raise RuntimeError(f"Error applying function to samples: {e}")

        if image_points.shape != samples_in_box.shape:
            raise ValueError(
                f"Function output shape {image_points.shape} does not match expected shape {samples_in_box.shape}."
            )

        # 3. Compute the minimum bounding box of the image points.
        image_box = np.array([
            [np.min(image_points[:, d]), np.max(image_points[:, d])]
            for d in range(self.D)
        ])

        # 4. Add bloating to create an outer approximation.
        box_sizes = image_box[:, 1] - image_box[:, 0]
        bloat_amount = box_sizes * self.bloat_factor / 2.0
        image_box[:, 0] -= bloat_amount
        image_box[:, 1] += bloat_amount

        return image_box


class BoxMapODE(Dynamics):
    """
    Represents dynamics defined by an ordinary differential equation dx/dt = f(t, x).

    The box mapping is estimated by integrating sample points over a time horizon tau.
    """

    def __init__(self, ode_func: callable, dimension: int, tau: float, sample_points: int = 10, bloat_factor: float = 0.1):
        """
        :param ode_func: A callable function f(t, x) for the ODE.
        :param dimension: The dimension D of the state space.
        :param tau: The time horizon for integration.
        :param sample_points: Number of points to sample inside a box.
        :param bloat_factor: Factor by which to bloat the computed image box.
        """
        self.ode_func = ode_func
        self.D = dimension
        self.tau = tau
        self.sample_points = sample_points
        self.bloat_factor = bloat_factor

    def __call__(self, box: np.ndarray) -> np.ndarray:
        # 1. Sample points in the box (for now, just corners).
        # A more robust implementation might use random sampling as in BoxMapFunction.
        num_corners = 2 ** self.D
        corner_indices = np.array(list(np.binary_repr(i, width=self.D) for i in range(num_corners))).astype(int)
        corners = np.array([box[d, corner_indices[:, d]] for d in range(self.D)]).T

        # 2. Integrate each sample point for time tau.
        final_points = []
        for point in corners:
            try:
                sol = solve_ivp(self.ode_func, (0, self.tau), point, dense_output=False, max_step=self.tau / 10)
                final_points.append(sol.y[:, -1])
            except Exception as e:
                # If integration fails, we can't determine the image; return invalid box.
                return np.full((self.D, 2), np.nan)

        image_points = np.array(final_points)

        # 3. Compute the minimum bounding box of the image points.
        image_box = np.array([
            [np.min(image_points[:, d]), np.max(image_points[:, d])]
            for d in range(self.D)
        ])

        # 4. Add bloating.
        box_sizes = image_box[:, 1] - image_box[:, 0]
        bloat_amount = box_sizes * self.bloat_factor / 2.0
        image_box[:, 0] -= bloat_amount
        image_box[:, 1] += bloat_amount

        return image_box


class BoxMapData(Dynamics):
    """
    Represents dynamics from a dataset of input-output pairs (X, Y).
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, bloat_factor: float = 0.1):
        """
        :param X: Input data points, shape (N, D).
        :param Y: Output data points, shape (N, D), where Y[i] = f(X[i]).
        :param bloat_factor: Factor by which to bloat the computed image box.
        """
        if X.shape != Y.shape:
            raise ValueError("Input X and output Y must have the same shape.")
        self.X = X
        self.Y = Y
        self.D = X.shape[1]
        self.bloat_factor = bloat_factor

    def __call__(self, box: np.ndarray) -> np.ndarray:
        # 1. Find all points in X that are inside the given box.
        lower_bounds = box[:, 0]
        upper_bounds = box[:, 1]
        mask = np.all((self.X >= lower_bounds) & (self.X <= upper_bounds), axis=1)

        image_points = self.Y[mask]

        if image_points.shape[0] == 0:
            # Handle case where no data points are in the box.
            # Return an invalid box of NaNs.
            return np.full((self.D, 2), np.nan)

        # 2. Compute the minimum bounding box of their images in Y.
        image_box = np.array([
            [np.min(image_points[:, d]), np.max(image_points[:, d])]
            for d in range(self.D)
        ])

        # 3. Add bloating.
        box_sizes = image_box[:, 1] - image_box[:, 0]
        bloat_amount = box_sizes * self.bloat_factor / 2.0
        image_box[:, 0] -= bloat_amount
        image_box[:, 1] += bloat_amount

        return image_box


try:
    import torch

    class LearnedDynamics(Dynamics):
        """
        Represents dynamics from a trained autoencoder and latent dynamics model.
        """

        def __init__(self, encoder, dynamics_model, decoder, bloat_factor: float = 0.1):
            """
            :param encoder: Trained PyTorch model for encoding states.
            :param dynamics_model: Trained PyTorch model for latent space dynamics.
            :param decoder: Trained PyTorch model for decoding states.
            :param bloat_factor: Factor for bloating.
            """
            if not hasattr(torch, 'nn'):
                raise ImportError("PyTorch is required for LearnedDynamics. Please install it.")
            self.encoder = encoder
            self.dynamics_model = dynamics_model
            self.decoder = decoder
            self.bloat_factor = bloat_factor

        def __call__(self, box: np.ndarray) -> np.ndarray:
            # Ensure models are in evaluation mode
            self.encoder.eval()
            self.dynamics_model.eval()
            self.decoder.eval()

            D = box.shape[0]

            # 1. Sample points (corners) from the input box.
            num_corners = 2 ** D
            corner_indices = np.array(list(np.binary_repr(i, width=D) for i in range(num_corners))).astype(int)
            corners = np.array([box[d, corner_indices[:, d]] for d in range(D)]).T
            
            with torch.no_grad():
                # Convert to tensor
                corners_tensor = torch.from_numpy(corners).float()

                # 2. Encode these points into the latent space.
                latent_points = self.encoder(corners_tensor)

                # 3. Compute the bounding box of the encoded points.
                latent_dim = latent_points.shape[1]
                latent_box = np.array([
                    [torch.min(latent_points[:, d]).item(), torch.max(latent_points[:, d]).item()]
                    for d in range(latent_dim)
                ])

                # 4. Apply the latent dynamics model to the corners of this latent box.
                latent_corners_indices = np.array(list(np.binary_repr(i, width=latent_dim) for i in range(2**latent_dim)))
                latent_corners_indices = latent_corners_indices.astype(int)
                latent_corners = np.array([latent_box[d, latent_corners_indices[:, d]] for d in range(latent_dim)]).T
                latent_corners_tensor = torch.from_numpy(latent_corners).float()

                image_latent_points = self.dynamics_model(latent_corners_tensor)

                # 5. Compute the bounding box of the resulting latent points.
                image_latent_box = np.array([
                    [torch.min(image_latent_points[:, d]).item(), torch.max(image_latent_points[:, d]).item()]
                    for d in range(latent_dim)
                ])

                # 6. Decode the corners of this latent image box back to the original space.
                image_latent_corners_indices = np.array(list(np.binary_repr(i, width=latent_dim) for i in range(2**latent_dim)))
                image_latent_corners_indices = image_latent_corners_indices.astype(int)
                image_latent_corners = np.array([image_latent_box[d, image_latent_corners_indices[:, d]] for d in range(latent_dim)]).T
                image_latent_corners_tensor = torch.from_numpy(image_latent_corners).float()
                
                decoded_points = self.decoder(image_latent_corners_tensor)

            # 7. Compute the final bounding box of the decoded points and apply bloating.
            final_box = np.array([
                [torch.min(decoded_points[:, d]).item(), torch.max(decoded_points[:, d]).item()]
                for d in range(D)
            ])

            box_sizes = final_box[:, 1] - final_box[:, 0]
            bloat_amount = box_sizes * self.bloat_factor / 2.0
            final_box[:, 0] -= bloat_amount
            final_box[:, 1] += bloat_amount

            return final_box


except ImportError:
    # If torch is not installed, create a dummy class that raises an error upon instantiation.
    class LearnedDynamics:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LearnedDynamics. Please install it via `pip install torch` or `pip install morsegraph[ml]`.")
