"""
Standard dynamical systems for testing and examples.

This module provides canonical implementations of well-known dynamical systems
commonly used in testing and demonstrating Morse graph computations.
"""

import numpy as np


# =============================================================================
# Discrete Maps
# =============================================================================

def henon_map(x: np.ndarray, a: float = 1.4, b: float = 0.3) -> np.ndarray:
    """
    Henon map: A canonical chaotic discrete dynamical system.

    The Henon map is defined by:
        x_{n+1} = 1 - a*x_n^2 + y_n
        y_{n+1} = b*x_n

    Args:
        x: State vector of shape (2,) with [x, y]
        a: Parameter a (default: 1.4)
        b: Parameter b (default: 0.3)

    Returns:
        Next state vector of shape (2,)

    Example:
        >>> from MorseGraph.systems import henon_map
        >>> x = np.array([0.0, 0.0])
        >>> x_next = henon_map(x)
    """
    x_val, y_val = x
    x_next = 1 - a * x_val**2 + y_val
    y_next = b * x_val
    return np.array([x_next, y_next])


def leslie_map(x: np.ndarray, th1: float = 19.6, th2: float = 23.68,
               mortality: float = 0.7) -> np.ndarray:
    """
    Leslie matrix population model map.

    A discrete-time population model with two age classes. The map captures
    reproduction and survival dynamics in structured populations.

    The Leslie map is defined by:
        x_{n+1} = (th1*x_n + th2*y_n) * exp(-0.1*(x_n + y_n))
        y_{n+1} = mortality * x_n

    Args:
        x: State vector of shape (2,) with [x, y] representing two age classes
        th1: Fertility parameter for age class 1 (default: 19.6)
        th2: Fertility parameter for age class 2 (default: 23.68)
        mortality: Survival rate from class 1 to class 2 (default: 0.7)

    Returns:
        Next state vector of shape (2,)

    Example:
        >>> from MorseGraph.systems import leslie_map
        >>> x = np.array([10.0, 5.0])
        >>> x_next = leslie_map(x)
    """
    x_val, y_val = x
    x_next = (th1 * x_val + th2 * y_val) * np.exp(-0.1 * (x_val + y_val))
    y_next = mortality * x_val
    return np.array([x_next, y_next])


def leslie_map_3d(x: np.ndarray,
                  theta_1: float = 28.9,
                  theta_2: float = 29.8,
                  theta_3: float = 22.0,
                  mortality: float = 0.7) -> np.ndarray:
    """
    Three-dimensional Leslie population model map.

    Extended Leslie matrix model with three age classes. This map captures
    reproduction and survival dynamics in structured populations with an
    additional age class.

    The 3D Leslie map is defined by:
        x_{n+1} = (th1*x_n + th2*y_n + th3*z_n) * exp(-0.1*(x_n + y_n + z_n))
        y_{n+1} = mortality * x_n
        z_{n+1} = mortality * y_n

    Args:
        x: State vector of shape (3,) with [x, y, z] representing three age classes
        theta_1: Fertility parameter for age class 1 (default: 28.9)
        theta_2: Fertility parameter for age class 2 (default: 29.8)
        theta_3: Fertility parameter for age class 3 (default: 22.0)
        mortality: Survival rate between consecutive age classes (default: 0.7)

    Returns:
        Next state vector of shape (3,)

    Example:
        >>> from MorseGraph.systems import leslie_map_3d
        >>> x = np.array([10.0, 5.0, 2.0])
        >>> x_next = leslie_map_3d(x)

    Reference:
        Leslie, P.H., "On the use of matrices in certain population mathematics"
        Biometrika 33 (3): 183-212 (1945)
    """
    x0, x1, x2 = x
    x0_next = (theta_1 * x0 + theta_2 * x1 + theta_3 * x2) * np.exp(-0.1 * (x0 + x1 + x2))
    x1_next = mortality * x0
    x2_next = mortality * x1
    return np.array([x0_next, x1_next, x2_next])


# =============================================================================
# ODEs (Continuous Systems)
# =============================================================================

def van_der_pol_ode(t: float, y: np.ndarray, mu: float = 1.0) -> list:
    """
    Van der Pol oscillator ODE.

    The Van der Pol oscillator is a non-conservative oscillator with
    non-linear damping. It exhibits limit cycle behavior.

    Equations:
        dx/dt = v
        dv/dt = mu*(1 - x^2)*v - x

    Args:
        t: Time (not used, but required for scipy.integrate.solve_ivp)
        y: State vector [x, v]
        mu: Damping parameter (default: 1.0)

    Returns:
        Time derivatives [dx/dt, dv/dt]

    Example:
        >>> from scipy.integrate import solve_ivp
        >>> from MorseGraph.systems import van_der_pol_ode
        >>> sol = solve_ivp(van_der_pol_ode, [0, 10], [0.1, 0.1], args=(1.0,))
    """
    x, v = y
    return [v, mu * (1 - x**2) * v - x]


def toggle_switch_ode(t: float, y: np.ndarray,
                     alpha1: float = 156.25, alpha2: float = 156.25,
                     beta: float = 2.5, gamma: float = 2.0,
                     n: int = 4) -> list:
    """
    Genetic toggle switch ODE.

    Models a synthetic genetic regulatory network consisting of two mutually
    repressing genes. Exhibits bistability.

    Equations:
        du/dt = alpha1 / (1 + v^n) - beta * u
        dv/dt = alpha2 / (1 + u^n) - beta * v

    Args:
        t: Time (not used, but required for scipy.integrate.solve_ivp)
        y: State vector [u, v] (protein concentrations)
        alpha1: Production rate of protein 1 (default: 156.25)
        alpha2: Production rate of protein 2 (default: 156.25)
        beta: Degradation rate (default: 2.5)
        gamma: Cooperativity parameter (default: 2.0)
        n: Hill coefficient (default: 4)

    Returns:
        Time derivatives [du/dt, dv/dt]

    Reference:
        Gardner et al., "Construction of a genetic toggle switch in Escherichia coli"
        Nature 403, 339-342 (2000)

    Example:
        >>> from scipy.integrate import solve_ivp
        >>> from MorseGraph.systems import toggle_switch_ode
        >>> sol = solve_ivp(toggle_switch_ode, [0, 10], [1.0, 1.0])
    """
    u, v = y
    du_dt = alpha1 / (1 + v**n) - beta * u
    dv_dt = alpha2 / (1 + u**n) - beta * v
    return [du_dt, dv_dt]


def lorenz_ode(t: float, state: np.ndarray,
               sigma: float = 10.0, rho: float = 28.0,
               beta: float = 8.0/3.0) -> list:
    """
    Lorenz system ODE.

    The Lorenz system is a canonical example of a chaotic dynamical system,
    originally derived as a simplified model of atmospheric convection.

    Equations:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Args:
        t: Time (not used, but required for scipy.integrate.solve_ivp)
        state: State vector [x, y, z]
        sigma: Prandtl number (default: 10.0)
        rho: Rayleigh number (default: 28.0)
        beta: Geometric parameter (default: 8.0/3.0)

    Returns:
        Time derivatives [dx/dt, dy/dt, dz/dt]

    Reference:
        Lorenz, E.N., "Deterministic Nonperiodic Flow"
        Journal of the Atmospheric Sciences 20 (2): 130-141 (1963)

    Example:
        >>> from scipy.integrate import solve_ivp
        >>> from MorseGraph.systems import lorenz_ode
        >>> sol = solve_ivp(lorenz_ode, [0, 10], [1.0, 1.0, 1.0])
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]
