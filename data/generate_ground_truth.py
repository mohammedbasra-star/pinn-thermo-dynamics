"""
Physics-Informed Neural Networks for High-Temperature Turbomachinery & Thermodynamic Dynamics
Module: data/generate_ground_truth.py

Generates high-resolution 256x256 spatiotemporal ground-truth benchmark datasets.
Combines:
  1. Exact analytical Fourier-series / manufactured thermal solution.
  2. Implicit Crank-Nicolson Finite Difference numerical solver.
  3. Sparse thermocouple sensor measurement generation with controllable Gaussian noise.
"""

from typing import Tuple, Callable, Optional, Dict, Union
import os
import sys
import argparse
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Ensure root repository directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class AnalyticalHeatConduction:
    r"""
    Exact analytical benchmark for 1D unsteady aerothermal heat conduction.

    Governing PDE:
        .. math::
            \frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2} + Q(x, t),
            \quad x \in [0, 1], \quad t \in [0, 1]

    Closed-form analytical solution:
        .. math::
            T^*(x, t) = \sin(\pi x) e^{-\lambda t} + T_{hot}(1 - x) + T_{cool} x

    Exact source term satisfying the PDE:
        .. math::
            Q(x, t) = (\alpha \pi^2 - \lambda) \sin(\pi x) e^{-\lambda t}
    """

    def __init__(
        self,
        alpha_true: float = 0.05,
        decay_lambda: float = 0.5,
        T_hot: float = 1.0,
        T_cool: float = 0.2,
    ):
        self.alpha_true = float(alpha_true)
        self.decay_lambda = float(decay_lambda)
        self.T_hot = float(T_hot)
        self.T_cool = float(T_cool)

    def temperature(
        self, x: Union[np.ndarray, float], t: Union[np.ndarray, float]
    ) -> np.ndarray:
        """Evaluates exact temperature field T*(x, t)."""
        x_arr = np.asarray(x)
        t_arr = np.asarray(t)
        sin_term = np.sin(np.pi * x_arr) * np.exp(-self.decay_lambda * t_arr)
        linear_profile = self.T_hot * (1.0 - x_arr) + self.T_cool * x_arr
        return sin_term + linear_profile

    def heat_source(
        self, x: Union[np.ndarray, float], t: Union[np.ndarray, float]
    ) -> np.ndarray:
        """Evaluates exact source term Q(x, t)."""
        x_arr = np.asarray(x)
        t_arr = np.asarray(t)
        factor = (self.alpha_true * (np.pi ** 2)) - self.decay_lambda
        return factor * np.sin(np.pi * x_arr) * np.exp(-self.decay_lambda * t_arr)


class CrankNicolsonHeatSolver:
    r"""
    Solves 1D transient heat conduction using the second-order implicit Crank-Nicolson method:
        .. math::
            \frac{T_i^{n+1} - T_i^n}{\Delta t} = \frac{\alpha}{2} \left[ \frac{T_{i+1}^{n+1} - 2T_i^{n+1} + T_{i-1}^{n+1}}{\Delta x^2} + \frac{T_{i+1}^n - 2T_i^n + T_{i-1}^n}{\Delta x^2} \right] + \frac{Q_i^{n+1} + Q_i^n}{2}

    Unconditionally stable and O(dx^2, dt^2) accurate.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        x_span: Tuple[float, float] = (0.0, 1.0),
        t_span: Tuple[float, float] = (0.0, 1.0),
        nx: int = 256,
        nt: int = 256,
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
        initial_fn: Callable[[np.ndarray], np.ndarray],
        bc_left_fn: Callable[[float], float],
        bc_right_fn: Callable[[float], float],
        source_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs time marching integration from t_min to t_max."""
        nx, nt, r = self.nx, self.nt, self.r
        T_matrix = np.zeros((nt, nx))

        # Initial condition
        T_matrix[0, :] = initial_fn(self.x)
        T_matrix[0, 0] = bc_left_fn(self.t[0])
        T_matrix[0, -1] = bc_right_fn(self.t[0])

        m = nx - 2
        main_diag_a = (1.0 + 2.0 * r) * np.ones(m)
        off_diag_a = -r * np.ones(m - 1)
        A = sparse.diags([off_diag_a, main_diag_a, off_diag_a], [-1, 0, 1], shape=(m, m), format="csr")

        main_diag_b = (1.0 - 2.0 * r) * np.ones(m)
        off_diag_b = r * np.ones(m - 1)
        B = sparse.diags([off_diag_b, main_diag_b, off_diag_b], [-1, 0, 1], shape=(m, m), format="csr")

        for n in range(0, nt - 1):
            t_curr = self.t[n]
            t_next = self.t[n + 1]

            bc_left_next = bc_left_fn(t_next)
            bc_right_next = bc_right_fn(t_next)
            bc_left_curr = bc_left_fn(t_curr)
            bc_right_curr = bc_right_fn(t_curr)

            rhs = B.dot(T_matrix[n, 1:-1])
            rhs[0] += r * (bc_left_curr + bc_left_next)
            rhs[-1] += r * (bc_right_curr + bc_right_next)

            if source_fn is not None:
                q_curr = source_fn(self.x[1:-1], t_curr)
                q_next = source_fn(self.x[1:-1], t_next)
                rhs += 0.5 * self.dt * (q_curr + q_next)

            T_interior_next = spsolve(A, rhs)
            T_matrix[n + 1, 0] = bc_left_next
            T_matrix[n + 1, 1:-1] = T_interior_next
            T_matrix[n + 1, -1] = bc_right_next

        X, T_grid = np.meshgrid(self.x, self.t)
        return X, T_grid, T_matrix


def generate_benchmark(
    nx: int = 256,
    nt: int = 256,
    alpha: float = 0.05,
    decay_lambda: float = 0.5,
    n_sensors: int = 200,
    noise_std: float = 0.01,
    output_path: Optional[str] = "data/ground_truth_256x256.npz",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Synthesizes a 256x256 spatiotemporal ground-truth dataset.
    Cross-verifies Crank-Nicolson against analytical solution and saves dataset to disk.
    """
    np.random.seed(seed)
    analytical = AnalyticalHeatConduction(alpha_true=alpha, decay_lambda=decay_lambda)

    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    X, T_grid = np.meshgrid(x, t)

    T_exact = analytical.temperature(X, T_grid)
    Q_exact = analytical.heat_source(X, T_grid)

    # Numerical verification via Crank-Nicolson
    solver = CrankNicolsonHeatSolver(alpha=alpha, nx=nx, nt=nt)
    init_fn = lambda x_arr: analytical.temperature(x_arr, 0.0)
    bc_l_fn = lambda t_val: float(analytical.temperature(0.0, t_val))
    bc_r_fn = lambda t_val: float(analytical.temperature(1.0, t_val))
    src_fn = lambda x_arr, t_val: analytical.heat_source(x_arr, t_val)

    _, _, T_fd = solver.solve(init_fn, bc_l_fn, bc_r_fn, src_fn)
    fd_rel_l2 = np.linalg.norm(T_fd - T_exact) / np.linalg.norm(T_exact)

    # Sample sparse sensor points for Inverse problem
    idx_x = np.random.choice(nx, size=n_sensors, replace=True)
    idx_t = np.random.choice(nt, size=n_sensors, replace=True)
    sensor_x = x[idx_x].reshape(-1, 1)
    sensor_t = t[idx_t].reshape(-1, 1)
    sensor_clean = T_exact[idx_t, idx_x].reshape(-1, 1)
    sensor_noisy = sensor_clean + np.random.normal(0, noise_std, sensor_clean.shape)

    dataset = {
        "x": x,
        "t": t,
        "X": X,
        "T_grid": T_grid,
        "T_exact": T_exact,
        "Q_exact": Q_exact,
        "T_fd": T_fd,
        "fd_relative_l2_error": np.array([fd_rel_l2]),
        "sensor_x": sensor_x,
        "sensor_t": sensor_t,
        "sensor_clean": sensor_clean,
        "sensor_noisy": sensor_noisy,
        "alpha": np.array([alpha]),
        "decay_lambda": np.array([decay_lambda]),
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez_compressed(output_path, **dataset)
        print(f"[Ground Truth Generated] Saved to {output_path}")
        print(f"  Grid resolution: {nx} x {nt} (Total points: {nx * nt:,})")
        print(f"  Crank-Nicolson vs Analytical Relative L2 Error: {fd_rel_l2:.3e}")
        print(f"  Sparse sensor observations: {n_sensors} points (noise std: {noise_std})")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 256x256 ground truth dataset.")
    parser.add_argument("--nx", type=int, default=256, help="Spatial resolution")
    parser.add_argument("--nt", type=int, default=256, help="Temporal resolution")
    parser.add_argument("--alpha", type=float, default=0.05, help="Thermal diffusivity")
    parser.add_argument("--output", type=str, default="data/ground_truth_256x256.npz")
    args = parser.parse_args()

    generate_benchmark(nx=args.nx, nt=args.nt, alpha=args.alpha, output_path=args.output)
