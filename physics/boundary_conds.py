"""
Physics-Informed Neural Networks for High-Temperature Turbomachinery & Thermodynamic Dynamics
Module: physics/boundary_conds.py

Defines boundary and initial conditions, spatial-temporal domain definitions,
and Latin Hypercube / Sobol / uniform point sampling for scientific machine learning.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
from scipy.stats import qmc

from .pde_residuals import ManufacturedHeatConduction


class BoundaryConditionManager:
    r"""
    Manages Dirichlet and Neumann boundary conditions, as well as Initial Conditions (IC),
    for unsteady aerothermal turbine blade simulations.

    Domain:
        :math:`x \in [0, L]`, :math:`t \in [0, t_{max}]`
        Here :math:`L=1.0` (normalized blade wall thickness), :math:`t_{max}=1.0`.

    Boundary Conditions:
        - Exterior Hot Gas Interface (:math:`x=0`): :math:`T(0, t) = T_{hot}(t)`
        - Internal Cooling Channel (:math:`x=1`): :math:`T(1, t) = T_{cool}(t)`
        - Initial Condition (:math:`t=0`): :math:`T(x, 0) = T_0(x)`
    """

    def __init__(
        self,
        benchmark: ManufacturedHeatConduction,
        x_min: float = 0.0,
        x_max: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ):
        self.benchmark = benchmark
        self.x_min = x_min
        self.x_max = x_max
        self.t_min = t_min
        self.t_max = t_max

    def get_initial_condition_values(self, x: torch.Tensor) -> torch.Tensor:
        """Returns exact analytical temperature at t = t_min."""
        t_zero = torch.full_like(x, self.t_min)
        return self.benchmark.exact_temperature(x, t_zero)

    def get_boundary_left_values(self, t: torch.Tensor) -> torch.Tensor:
        """Returns exact analytical temperature at x = x_min (hot gas side)."""
        x_left = torch.full_like(t, self.x_min)
        return self.benchmark.exact_temperature(x_left, t)

    def get_boundary_right_values(self, t: torch.Tensor) -> torch.Tensor:
        """Returns exact analytical temperature at x = x_max (coolant side)."""
        x_right = torch.full_like(t, self.x_max)
        return self.benchmark.exact_temperature(x_right, t)


class CollocationSampler:
    r"""
    Generates training collocation points across interior and boundaries using
    Quasi-Monte Carlo (Sobol / Latin Hypercube) or uniform sampling.
    """

    def __init__(
        self,
        bc_manager: BoundaryConditionManager,
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        self.bc_manager = bc_manager
        self.device = device
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def sample_interior(
        self, n_collocation: int, method: str = "lhs"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples interior spatiotemporal points (x_f, t_f) where PDE residual is enforced.
        """
        if method == "lhs":
            sampler = qmc.LatinHypercube(d=2, seed=self.seed)
            sample = sampler.random(n=n_collocation)
            x_np = self.bc_manager.x_min + (self.bc_manager.x_max - self.bc_manager.x_min) * sample[:, 0:1]
            t_np = self.bc_manager.t_min + (self.bc_manager.t_max - self.bc_manager.t_min) * sample[:, 1:2]
        else:
            x_np = np.random.uniform(self.bc_manager.x_min, self.bc_manager.x_max, (n_collocation, 1))
            t_np = np.random.uniform(self.bc_manager.t_min, self.bc_manager.t_max, (n_collocation, 1))

        x = torch.tensor(x_np, dtype=torch.float32, device=self.device, requires_grad=True)
        t = torch.tensor(t_np, dtype=torch.float32, device=self.device, requires_grad=True)
        return x, t

    def sample_initial(
        self, n_initial: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples initial condition points (x_ic, 0) and computes target T(x_ic, 0).
        """
        x_np = np.linspace(self.bc_manager.x_min, self.bc_manager.x_max, n_initial).reshape(-1, 1)
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        t = torch.full_like(x, self.bc_manager.t_min)
        T_target = self.bc_manager.get_initial_condition_values(x)
        return x, t, T_target

    def sample_boundary(
        self, n_boundary: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples boundary points for both left (hot gas side) and right (coolant side).

        Returns:
            x_left, t_left, T_target_left, x_right, t_right, T_target_right
        """
        t_np = np.linspace(self.bc_manager.t_min, self.bc_manager.t_max, n_boundary).reshape(-1, 1)
        t = torch.tensor(t_np, dtype=torch.float32, device=self.device)

        x_left = torch.full_like(t, self.bc_manager.x_min)
        T_left = self.bc_manager.get_boundary_left_values(t)

        x_right = torch.full_like(t, self.bc_manager.x_max)
        T_right = self.bc_manager.get_boundary_right_values(t)

        return x_left, t, T_left, x_right, t, T_right

    def sample_sparse_observations(
        self, n_sensors: int, noise_std: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates sparse, noisy measurement data inside the domain for the Inverse problem
        (simulating embedded thermal sensors in a turbine blade).
        """
        sampler = qmc.LatinHypercube(d=2, seed=self.seed + 100)
        sample = sampler.random(n=n_sensors)
        x_np = self.bc_manager.x_min + (self.bc_manager.x_max - self.bc_manager.x_min) * sample[:, 0:1]
        t_np = self.bc_manager.t_min + (self.bc_manager.t_max - self.bc_manager.t_min) * sample[:, 1:2]

        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        t = torch.tensor(t_np, dtype=torch.float32, device=self.device)

        # Ground truth values
        T_clean = self.bc_manager.benchmark.exact_temperature(x, t)

        # Add Gaussian sensor measurement noise
        if noise_std > 0.0:
            noise = torch.randn_like(T_clean) * noise_std
            T_noisy = T_clean + noise
        else:
            T_noisy = T_clean

        return x, t, T_noisy
