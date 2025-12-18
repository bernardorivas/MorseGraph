import numpy as np
from MORALS.systems.system import BaseSystem

class Leslie_map_3d(BaseSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "leslie_map_3d"
        self.state_bounds = np.array([
            [-0.01, 120.0],
            [-0.01, 75.0],
            [-0.01, 70.0]
        ])
        self.theta_1 = 28.9
        self.theta_2 = 29.8
        self.theta_3 = 22.0
        self.survival_1 = 0.7
        self.survival_2 = 0.7

    def f(self, s):
        x0, x1, x2 = s
        x0_next = (self.theta_1 * x0 + self.theta_2 * x1 + self.theta_3 * x2) * np.exp(-0.1 * (x0 + x1 + x2))
        x1_next = self.survival_1 * x0
        x2_next = self.survival_2 * x1
        return np.array([x0_next, x1_next, x2_next])

    def get_true_bounds(self):
        return self.state_bounds

    def get_bounds(self):
        return self.state_bounds