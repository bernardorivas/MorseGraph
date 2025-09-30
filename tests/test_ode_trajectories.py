#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def bistable_ode(t, x):
    """
    A simple bistable ODE system.
    dx/dt = x - x^3
    dy/dt = -y
    """
    dxdt = x[0] - x[0]**3
    dydt = -x[1]
    return np.array([dxdt, dydt])

# Create vector field for visualization
x_vec = np.linspace(-2, 2, 20)
y_vec = np.linspace(-2, 2, 20)
x, y = np.meshgrid(x_vec, y_vec)

# Compute vector field
u = np.zeros_like(x)
v = np.zeros_like(y)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        derivatives = bistable_ode(0, np.array([x[i,j], y[i,j]]))
        u[i,j] = derivatives[0]
        v[i,j] = derivatives[1]

# Plot vector field with sample trajectories
plt.figure(figsize=(10, 8))
plt.quiver(x, y, u, v, color='blue', alpha=0.5, scale=20, width=0.003)

# Generate 5 random initial conditions
np.random.seed(42)  # For reproducible results
initial_conditions = np.random.uniform(low=[-1.8, -1.8], high=[1.8, 1.8], size=(5, 2))

# Integrate and plot trajectories
t_span = (0, 3.0)  # Integration time
t_eval = np.linspace(0, 3.0, 300)

colors = ['red', 'green', 'orange', 'purple', 'brown']

for i, initial_point in enumerate(initial_conditions):
    sol = solve_ivp(bistable_ode, t_span, initial_point, t_eval=t_eval, dense_output=True)
    
    if sol.success:
        trajectory = sol.y.T  # Shape: (n_times, 2)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 
                color=colors[i], linewidth=2, alpha=0.8, 
                label=f'Trajectory {i+1}')
        
        # Mark initial point
        plt.plot(initial_point[0], initial_point[1], 
                'o', color=colors[i], markersize=8, markeredgecolor='black')
        
        # Mark final point
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 
                's', color=colors[i], markersize=6, markeredgecolor='black')

plt.title("Bistable ODE: Vector Field and Sample Trajectories")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.3)

# Mark fixed points
plt.plot([-1, 0, 1], [0, 0, 0], 'ko', markersize=10, 
         markerfacecolor='white', markeredgewidth=2, label='Fixed Points')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.tight_layout()
plt.show()

print("Trajectory visualization completed!")