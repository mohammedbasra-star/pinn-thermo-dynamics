"""
Physics-Informed Neural Networks for High-Temperature Turbomachinery & Thermodynamic Dynamics
Module: data/generate_data.py

Script and utility functions to synthesize high-fidelity aerothermal benchmark datasets,
generate simulated embedded thermocouple sensor observations, and verify numerical
against analytical solutions.
"""

import os
import sys
import argparse
import numpy as np

# Ensure root repository directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from physics.pde_residuals import ManufacturedHeatConduction
from data.reference_solution import FiniteDifferenceHeatSolver


def generate_and_save_benchmark(
    output_path: str = "data/benchmark_turbomachinery_heat.npz",
    alpha: float = 0.05,
    decay_lambda: float = 0.5,
    nx: int = 101,
    nt: int = 101,
    noise_std: float = 0.01,
    seed: int = 42,
) -> str:
    """
    Generates analytical and finite difference solutions, adds sensor noise, and saves to .npz
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.random.seed(seed)

    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    X, T_grid = np.meshgrid(x, t)

    # 1. Analytical Manufactured Solution
    benchmark = ManufacturedHeatConduction(alpha_true=alpha, decay_lambda=decay_lambda)
    T_exact = benchmark.exact_temperature(X, T_grid)
    Q_exact = benchmark.exact_source(X, T_grid)

    # 2. Finite Difference Crank-Nicolson verification
    fd_solver = FiniteDifferenceHeatSolver(alpha=alpha, x_span=(0.0, 1.0), t_span=(0.0, 1.0), nx=nx, nt=nt)
    init_fn = lambda x_arr: benchmark.exact_temperature(x_arr, np.zeros_like(x_arr))
    bc_left_fn = lambda t_val: float(benchmark.exact_temperature(0.0, t_val))
    bc_right_fn = lambda t_val: float(benchmark.exact_temperature(1.0, t_val))
    source_fn = lambda x_arr, t_val: benchmark.exact_source(x_arr, np.full_like(x_arr, t_val))

    _, _, T_fd = fd_solver.solve(init_fn, bc_left_fn, bc_right_fn, source_fn)

    # Compute cross-verification error between Crank-Nicolson and Analytical
    fd_error_l2 = np.linalg.norm(T_fd - T_exact) / np.linalg.norm(T_exact)

    # 3. Simulated Sparse Embedded Sensor Readings
    n_sensors = 200
    idx_x = np.random.choice(nx, size=n_sensors, replace=True)
    idx_t = np.random.choice(nt, size=n_sensors, replace=True)
    sensor_x = x[idx_x].reshape(-1, 1)
    sensor_t = t[idx_t].reshape(-1, 1)
    sensor_exact = T_exact[idx_t, idx_x].reshape(-1, 1)
    sensor_noisy = sensor_exact + np.random.normal(0, noise_std, sensor_exact.shape)

    # Save to disk
    np.savez_compressed(
        output_path,
        x=x,
        t=t,
        X=X,
        T_grid=T_grid,
        T_exact=T_exact,
        Q_exact=Q_exact,
        T_fd=T_fd,
        fd_relative_l2_error=fd_error_l2,
        sensor_x=sensor_x,
        sensor_t=sensor_t,
        sensor_clean=sensor_exact,
        sensor_noisy=sensor_noisy,
        alpha=alpha,
        decay_lambda=decay_lambda,
    )

    print(f"[Dataset Generator] Successfully generated benchmark dataset:")
    print(f"  - File path: {output_path}")
    print(f"  - Grid dimensions: {nx} x {nt} (Total points: {nx * nt})")
    print(f"  - Crank-Nicolson vs Analytical Relative L2 Error: {fd_error_l2:.2e}")
    print(f"  - Simulated sensor observations: {n_sensors} points (Noise std: {noise_std})")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate turbomachinery heat benchmark dataset.")
    parser.add_argument("--output", type=str, default="data/benchmark_turbomachinery_heat.npz")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--decay-lambda", type=float, default=0.5)
    parser.add_argument("--nx", type=int, default=101)
    parser.add_argument("--nt", type=int, default=101)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    generate_and_save_benchmark(
        output_path=args.output,
        alpha=args.alpha,
        decay_lambda=args.decay_lambda,
        nx=args.nx,
        nt=args.nt,
        noise_std=args.noise_std,
        seed=args.seed,
    )
