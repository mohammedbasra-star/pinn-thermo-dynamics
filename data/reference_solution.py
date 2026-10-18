"""
Physics-Informed Neural Networks for High-Temperature Turbomachinery & Thermodynamic Dynamics
Module: data/reference_solution.py

Classical numerical PDE solver using the second-order Crank-Nicolson Finite Difference method.
Serves as an independent high-fidelity numerical ground-truth generator to cross-verify
PINN predictions and manufactured analytical solutions.
"""

from typing import Tuple, Callable, Optional
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


class FiniteDifferenceHeatSolver:
    r"""
    Solves the 1D unsteady heat equation using the implicit Crank-Nicolson finite difference scheme:
        .. math::
            \frac{T_i^{n+1} - T_i^n}{\Delta t} = \frac{\alpha}{2} \left[ \frac{T_{i+1}^{n+1} - 2T_i^{n+1} + T_{i-1}^{n+1}}{\Delta x^2} + \frac{T_{i+1}^n - 2T_i^n + T_{i-1}^n}{\Delta x^2} \right] + \frac{Q_i^{n+1} + Q_i^n}{2}

    Second-order accurate in both space and time :math:`\mathcal{O}(\Delta x^2, \Delta t^2)`
    and unconditionally stable.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        x_span: Tuple[float, float] = (0.0, 1.0),
        t_span: Tuple[float, float] = (0.0, 1.0),
        nx: int = 101,
        nt: int = 201,
    ):
        self.alpha = float(alpha)
        self.x_min, self.x_max = x_span
        self.t_min, self.t_max = t_span
        self.nx = nx
        self.nt = nt

        self.x = np.linspace(self.x_min, self.x_max, nx)
        self.t = np.linspace(self.t_min, self.t_max, nt)
        self.dx = (self.x_max - self.x_min) / (nx - 1)
        self.dt = (self.t_max - self.t_min) / (nt - 1)
        self.r = self.alpha * self.dt / (2.0 * self.dx ** 2)

    def solve(
        self,
        initial_func: Callable[[np.ndarray], np.ndarray],
        bc_left_func: Callable[[float], float],
        bc_right_func: Callable[[float], float],
        source_func: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes time integration from t_min to t_max.

        Returns:
            X, T_grid, T_solution (matrices of shape [nt, nx])
        """
        nx, nt, r = self.nx, self.nt, self.r
        T_matrix = np.zeros((nt, nx))

        # Initial condition
        T_matrix[0, :] = initial_func(self.x)

        # Boundary conditions at t=0
        T_matrix[0, 0] = bc_left_func(self.t[0])
        T_matrix[0, -1] = bc_right_func(self.t[0])

        # Tridiagonal system matrix A for interior nodes (indices 1 to nx-2)
        m = nx - 2
        main_diag = (1.0 + 2.0 * r) * np.ones(m)
        off_diag = -r * np.ones(m - 1)
        A = sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(m, m), format="csr")

        # Explicit operator B for right-hand side
        main_diag_b = (1.0 - 2.0 * r) * np.ones(m)
        off_diag_b = r * np.ones(m - 1)
        B = sparse.diags([off_diag_b, main_diag_b, off_diag_b], [-1, 0, 1], shape=(m, m), format="csr")

        for n in range(0, nt - 1):
            t_curr = self.t[n]
            t_next = self.t[n + 1]

            # Boundary values
            bc_left_next = bc_left_func(t_next)
            bc_right_next = bc_right_func(t_next)
            bc_left_curr = bc_left_func(t_curr)
            bc_right_curr = bc_right_func(t_curr)

            # Compute internal RHS
            rhs = B.dot(T_matrix[n, 1:-1])

            # Boundary flux contributions to interior equations
            rhs[0] += r * (bc_left_curr + bc_left_next)
            rhs[-1] += r * (bc_right_curr + bc_right_next)

            # Heat source contribution
            if source_func is not None:
                q_curr = source_func(self.x[1:-1], t_curr)
                q_next = source_func(self.x[1:-1], t_next)
                rhs += 0.5 * self.dt * (q_curr + q_next)

            # Solve linear system A * T_next = rhs
            T_interior_next = spsolve(A, rhs)

            T_matrix[n + 1, 0] = bc_left_next
            T_matrix[n + 1, 1:-1] = T_interior_next
            T_matrix[n + 1, -1] = bc_right_next

        X, T_grid = np.meshgrid(self.x, self.t)
        return X, T_grid, T_matrix
