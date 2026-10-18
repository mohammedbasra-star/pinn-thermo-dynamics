"""
Physics-Informed Neural Networks (PINNs) Benchmark Suite - PyTorch 2.x
Module: data/dataset.py

Spatiotemporal dataset representations and collocation samplers for:
1. Burgers' Equation benchmark (continuous-time forward problem)
2. Nonlinear Schrödinger (NLS) equation benchmark
3. Generic PINN PyTorch Dataset and Latin Hypercube Sampling (LHS) utilities
"""

import os
from typing import Tuple, Optional, Dict, Union
import numpy as np
import scipy.io as sio
from scipy.stats import qmc
import torch
from torch.utils.data import Dataset

from data.download_benchmarks import download_dataset


class CollocationDataset(Dataset):
    """
    Standard PyTorch Dataset for spatio-temporal collocation coordinates (x, t).
    Supports mini-batching or full-batch continuous physics evaluation.
    """

    def __init__(self, x: torch.Tensor, t: torch.Tensor):
        assert x.shape == t.shape, f"Shape mismatch: x={x.shape}, t={t.shape}"
        self.x = x.clone().detach().requires_grad_(True)
        self.t = t.clone().detach().requires_grad_(True)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.t[idx]


class BoundaryDataset(Dataset):
    """
    PyTorch Dataset for labeled boundary / initial condition measurements (x, t, u).
    """

    def __init__(self, x: torch.Tensor, t: torch.Tensor, u: torch.Tensor):
        assert x.shape == t.shape == u.shape, f"Mismatch: x={x.shape}, t={t.shape}, u={u.shape}"
        self.x = x.clone().detach()
        self.t = t.clone().detach()
        self.u = u.clone().detach()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.t[idx], self.u[idx]


class BurgersBenchmarkData:
    r"""
    Data manager for the historic 1D Burgers' equation benchmark from Raissi et al. (2019):
        .. math::
            \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0,
            \quad x \in [-1, 1], \quad t \in [0, 0.99], \quad \nu = \frac{0.01}{\pi}

    Boundary and initial conditions:
        .. math::
            u(0, x) = -\sin(\pi x), \quad u(t, -1) = u(t, 1) = 0

    Ground truth resolution:
        Spatial grid: 256 points in [-1, 1]
        Temporal grid: 100 points in [0, 0.99]
        Total evaluation points: 25,600
    """

    def __init__(
        self,
        mat_path: Optional[str] = None,
        n_u: int = 100,
        n_f: int = 10000,
        sampling_method: str = "lhs",
        seed: int = 1234,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = seed
        self.n_u = n_u
        self.n_f = n_f
        self.sampling_method = sampling_method

        # Resolve dataset file
        if mat_path is None or not os.path.exists(mat_path):
            mat_path = download_dataset("burgers", target_dir="data/benchmarks")
        self.mat_path = mat_path

        self._load_and_prepare()

    def _load_and_prepare(self):
        """Loads .mat file and prepares training and evaluation tensors faithfully matching Raissi 2019."""
        rng = np.random.RandomState(self.seed)

        data = sio.loadmat(self.mat_path)
        t = data["t"].flatten()[:, None]  # (100, 1)
        x = data["x"].flatten()[:, None]  # (256, 1)
        Exact = np.real(data["usol"]).T   # (100, 256)

        self.x_grid = x
        self.t_grid = t
        self.usol = np.real(data["usol"])  # (256, 100)
        self.exact_transposed = Exact

        # Meshgrid: X, T shape (100, 256)
        X, T = np.meshgrid(x, t)

        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]

        self.x_star = X_star[:, 0:1]
        self.t_star = X_star[:, 1:2]
        self.u_star = u_star

        self.lb = X_star.min(0)
        self.ub = X_star.max(0)

        # Candidate initial and boundary points
        xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))  # t = 0 initial condition (256, 2)
        uu1 = Exact[0:1, :].T                        # (256, 1)
        xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))      # x = -1 boundary condition (100, 2)
        uu2 = Exact[:, 0:1]                          # (100, 1)
        xx3 = np.hstack((X[:, -1:], T[:, -1:]))      # x = 1 boundary condition (100, 2)
        uu3 = Exact[:, -1:]                          # (100, 1)

        X_u_candidates = np.vstack([xx1, xx2, xx3])  # (456, 2)
        u_u_candidates = np.vstack([uu1, uu2, uu3])  # (456, 1)

        # Subsample N_u points randomly without replacement
        idx_u = rng.choice(X_u_candidates.shape[0], min(self.n_u, X_u_candidates.shape[0]), replace=False)
        X_u_train = X_u_candidates[idx_u, :]
        u_train = u_u_candidates[idx_u, :]

        self.x_u = X_u_train[:, 0:1]
        self.t_u = X_u_train[:, 1:2]
        self.u_train = u_train

        # Collocation points: N_f samples via LHS in [lb, ub] stacked with candidates
        if self.sampling_method == "lhs":
            sampler = qmc.LatinHypercube(d=2, seed=self.seed)
            lhs_pts = qmc.scale(sampler.random(n=self.n_f), self.lb, self.ub)
            X_f_train = np.vstack([lhs_pts, X_u_candidates])
            X_f_train = X_f_train[: self.n_f]
        else:
            x_f_rand = rng.uniform(self.lb[0], self.ub[0], (self.n_f, 1))
            t_f_rand = rng.uniform(self.lb[1], self.ub[1], (self.n_f, 1))
            X_f_train = np.hstack([x_f_rand, t_f_rand])

        self.x_f = X_f_train[:, 0:1]
        self.t_f = X_f_train[:, 1:2]

    def get_torch_data(self) -> Dict[str, torch.Tensor]:
        """
        Returns PyTorch tensors mapped to the target device and dtype:
        - 'x_u', 't_u', 'u_train': boundary / initial condition training data
        - 'x_f', 't_f': collocation coordinates with requires_grad=True
        - 'x_star', 't_star', 'u_star': full 256x100 evaluation benchmark grid
        """
        return {
            "x_u": torch.tensor(self.x_u, dtype=self.dtype, device=self.device),
            "t_u": torch.tensor(self.t_u, dtype=self.dtype, device=self.device),
            "u_train": torch.tensor(self.u_train, dtype=self.dtype, device=self.device),
            "x_f": torch.tensor(self.x_f, dtype=self.dtype, device=self.device, requires_grad=True),
            "t_f": torch.tensor(self.t_f, dtype=self.dtype, device=self.device, requires_grad=True),
            "x_star": torch.tensor(self.x_star, dtype=self.dtype, device=self.device),
            "t_star": torch.tensor(self.t_star, dtype=self.dtype, device=self.device),
            "u_star": torch.tensor(self.u_star, dtype=self.dtype, device=self.device),
        }

    def get_evaluation_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (x_grid, t_grid, usol) as original numpy arrays for publication plotting."""
        return self.x_grid, self.t_grid, self.usol


class SchrodingerBenchmarkData:
    r"""
    Data manager for the 1D Nonlinear Schrödinger (NLS) soliton benchmark from Raissi et al. (2019):
        .. math::
            i \psi_t + 0.5 \psi_{xx} + |\psi|^2 \psi = 0, \quad x \in [-5, 5], \quad t \in [0, \pi/2]
        with initial condition:
            .. math::
                \psi(0, x) = 2 \operatorname{sech}(x)
        and periodic boundary conditions at :math:`x = -5, 5`.
    """

    def __init__(
        self,
        mat_path: Optional[str] = None,
        n_0: int = 50,
        n_b: int = 50,
        n_f: int = 20000,
        seed: int = 1234,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = seed
        self.n_0 = n_0
        self.n_b = n_b
        self.n_f = n_f

        if mat_path is None or not os.path.exists(mat_path):
            mat_path = download_dataset("schrodinger", target_dir="data/benchmarks")
        self.mat_path = mat_path

        self._load_and_prepare()

    def _load_and_prepare(self):
        rng = np.random.RandomState(self.seed)
        data = sio.loadmat(self.mat_path)

        self.x_grid = data["x"].flatten()[:, None]  # (256, 1)
        self.t_grid = data["tt"].flatten()[:, None]  # (201, 1)
        self.exact_sol = data["uu"]  # (256, 201) complex

        self.exact_u = np.real(self.exact_sol)
        self.exact_v = np.imag(self.exact_sol)
        self.exact_h = np.sqrt(self.exact_u**2 + self.exact_v**2)

        X, T = np.meshgrid(self.x_grid, self.t_grid)
        self.x_star = X.T.flatten()[:, None]
        self.t_star = T.T.flatten()[:, None]
        self.h_star = self.exact_h.flatten()[:, None]
        self.u_star = self.exact_u.flatten()[:, None]
        self.v_star = self.exact_v.flatten()[:, None]

        self.lb = np.array([-5.0, 0.0])
        self.ub = np.array([5.0, np.pi / 2.0])

        # Initial condition (t = 0)
        idx_0 = rng.choice(self.x_grid.shape[0], self.n_0, replace=False)
        self.x_0 = self.x_grid[idx_0]
        self.t_0 = np.zeros_like(self.x_0)
        self.u_0 = self.exact_u[idx_0, 0:1]
        self.v_0 = self.exact_v[idx_0, 0:1]

        # Periodic boundary samples: (x = -5, t) and (x = 5, t)
        idx_b = rng.choice(self.t_grid.shape[0], self.n_b, replace=False)
        self.t_b = self.t_grid[idx_b]
        self.x_b_lower = np.full_like(self.t_b, -5.0)
        self.x_b_upper = np.full_like(self.t_b, 5.0)

        # Collocation interior
        sampler = qmc.LatinHypercube(d=2, seed=self.seed)
        lhs_samples = sampler.random(n=self.n_f)
        coords_f = qmc.scale(lhs_samples, self.lb, self.ub)
        self.x_f = coords_f[:, 0:1]
        self.t_f = coords_f[:, 1:2]

    def get_torch_data(self) -> Dict[str, torch.Tensor]:
        return {
            "x_0": torch.tensor(self.x_0, dtype=self.dtype, device=self.device),
            "t_0": torch.tensor(self.t_0, dtype=self.dtype, device=self.device),
            "u_0": torch.tensor(self.u_0, dtype=self.dtype, device=self.device),
            "v_0": torch.tensor(self.v_0, dtype=self.dtype, device=self.device),
            "x_b_lower": torch.tensor(self.x_b_lower, dtype=self.dtype, device=self.device, requires_grad=True),
            "x_b_upper": torch.tensor(self.x_b_upper, dtype=self.dtype, device=self.device, requires_grad=True),
            "t_b": torch.tensor(self.t_b, dtype=self.dtype, device=self.device, requires_grad=True),
            "x_f": torch.tensor(self.x_f, dtype=self.dtype, device=self.device, requires_grad=True),
            "t_f": torch.tensor(self.t_f, dtype=self.dtype, device=self.device, requires_grad=True),
            "x_star": torch.tensor(self.x_star, dtype=self.dtype, device=self.device),
            "t_star": torch.tensor(self.t_star, dtype=self.dtype, device=self.device),
            "h_star": torch.tensor(self.h_star, dtype=self.dtype, device=self.device),
        }
