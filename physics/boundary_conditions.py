"""
Physics-Informed Neural Networks for High-Temperature Turbomachinery & Thermodynamic Dynamics
Module: physics/boundary_conditions.py

Evaluates boundary condition losses (Dirichlet and Neumann) and initial conditions (IC).
Implements Latin Hypercube Sampling (LHS) and spatiotemporal point generation for SciML.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc

from .heat_pde import HeatConductionPDE


class BoundaryConditionEvaluator:
    r"""
    Enforces Dirichlet and Neumann boundary conditions, as well as Initial Conditions (IC),
    for transient aerothermal turbine blade wall dynamics.

    Domain:
        :math:`x \in [0, 1]`, :math:`t \in [0, 1]`

    Conditions:
        - Exterior boundary (:math:`x=0`): :math:`T(0, t) = T_{hot}(t)`
        - Interior boundary (:math:`x=1`): :math:`T(1, t) = T_{cool}(t)`
        - Initial state (:math:`t=0`): :math:`T(x, 0) = T_{init}(x)`
    """

    def __init__(
        self,
        pde: HeatConductionPDE,
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        self.pde = pde
        self.device = device
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def sample_interior(
        self, n_points: int = 2500, method: str = "lhs"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates interior spatiotemporal collocation points (x, t) with requires_grad=True."""
        if method == "lhs":
            sampler = qmc.LatinHypercube(d=2, seed=self.seed)
            sample = sampler.random(n=n_points)
            x_np = sample[:, 0:1]
            t_np = sample[:, 1:2]
        else:
            x_np = np.random.uniform(0.0, 1.0, (n_points, 1))
            t_np = np.random.uniform(0.0, 1.0, (n_points, 1))

        x = torch.tensor(x_np, dtype=torch.float32, device=self.device, requires_grad=True)
        t = torch.tensor(t_np, dtype=torch.float32, device=self.device, requires_grad=True)
        return x, t

    def sample_initial(
        self, n_points: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates initial condition points (x, 0) and analytical target temperatures."""
        x_np = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        t = torch.zeros_like(x)
        T_target = self.pde.exact_temperature(x, t)
        return x, t, T_target

    def sample_boundaries(
        self, n_points: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates left boundary (x=0) and right boundary (x=1) points with analytical values.

        Returns:
            x_left, t_left, T_target_left, x_right, t_right, T_target_right
        """
        t_np = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)
        t = torch.tensor(t_np, dtype=torch.float32, device=self.device)

        x_left = torch.zeros_like(t)
        T_left = self.pde.exact_temperature(x_left, t)

        x_right = torch.ones_like(t)
        T_right = self.pde.exact_temperature(x_right, t)

        return x_left, t, T_left, x_right, t, T_right

    def sample_sensor_data(
        self, n_sensors: int = 150, noise_std: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates sparse thermocouple sensor measurement points inside the domain."""
        sampler = qmc.LatinHypercube(d=2, seed=self.seed + 200)
        sample = sampler.random(n=n_sensors)
        x_np = sample[:, 0:1]
        t_np = sample[:, 1:2]

        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        t = torch.tensor(t_np, dtype=torch.float32, device=self.device)

        T_clean = self.pde.exact_temperature(x, t)
        if noise_std > 0.0:
            noise = torch.randn_like(T_clean) * noise_std
            T_noisy = T_clean + noise
        else:
            T_noisy = T_clean

        return x, t, T_noisy

    def evaluate_losses(
        self,
        model: nn.Module,
        n_pde: int = 2500,
        n_ic: int = 128,
        n_bc: int = 128,
        sensor_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluates individual loss components:
            - loss_pde: Residual MSE on interior domain
            - loss_ic: Initial condition MSE
            - loss_bc: Boundary conditions MSE (Dirichlet)
            - loss_data: Sensor observation MSE (for Inverse Mode)
        """
        # 1. PDE residual loss
        x_f, t_f = self.sample_interior(n_pde)
        res, _, _ = self.pde.residual(model, x_f, t_f)
        loss_pde = torch.mean(torch.square(res))

        # 2. Initial condition loss
        x_ic, t_ic, T_ic_target = self.sample_initial(n_ic)
        coords_ic = torch.cat([x_ic, t_ic], dim=1)
        T_ic_pred = model(coords_ic)
        loss_ic = torch.mean(torch.square(T_ic_pred - T_ic_target))

        # 3. Boundary condition loss
        x_left, t_left, T_left_tgt, x_right, t_right, T_right_tgt = self.sample_boundaries(n_bc)
        coords_left = torch.cat([x_left, t_left], dim=1)
        coords_right = torch.cat([x_right, t_right], dim=1)
        T_left_pred = model(coords_left)
        T_right_pred = model(coords_right)

        loss_bc = 0.5 * (
            torch.mean(torch.square(T_left_pred - T_left_tgt))
            + torch.mean(torch.square(T_right_pred - T_right_tgt))
        )

        losses = {
            "pde": loss_pde,
            "ic": loss_ic,
            "bc": loss_bc,
        }

        # 4. Sparse observation data loss
        if sensor_data is not None:
            x_meas, t_meas, T_noisy = sensor_data
            coords_meas = torch.cat([x_meas, t_meas], dim=1)
            T_meas_pred = model(coords_meas)
            loss_data = torch.mean(torch.square(T_meas_pred - T_noisy))
            losses["data"] = loss_data
        else:
            losses["data"] = torch.tensor(0.0, device=self.device)

        return losses
