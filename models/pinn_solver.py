"""
Physics-Informed Neural Networks for High-Temperature Turbomachinery & Thermodynamic Dynamics
Module: models/pinn_solver.py

Core PINN solver implementing:
- Dual-stage Hybrid Optimization:
    * Phase 1: Global convergence using the Adam optimizer.
    * Phase 2: High-precision convergence using L-BFGS with strong Wolfe line search.
- Dual Capability:
    * Forward Mode: Field prediction under aerothermal boundary conditions.
    * Inverse Mode: Dynamic calibration of unknown physical parameters (diffusivity alpha).
- Adaptive Loss Weighting and full telemetry logging.
"""

from typing import Dict, Any, Optional, Tuple, List, Callable
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from physics.heat_pde import HeatConductionPDE
from physics.boundary_conditions import BoundaryConditionEvaluator


class PINNSolver:
    r"""
    Comprehensive PINN solver managing forward and inverse aerothermal simulations.

    Loss Formulation:
        .. math::
            \mathcal{L}_{total} = w_{pde} \mathcal{L}_{pde} + w_{bc} \mathcal{L}_{bc} + w_{ic} \mathcal{L}_{ic} + w_{data} \mathcal{L}_{data}

    Optimization Strategy:
        1. Adam: Wide basin exploration, robust against noisy initial landscapes.
        2. L-BFGS: Quasi-Newton second-order curvature approximation for machine precision.
    """

    def __init__(
        self,
        model: nn.Module,
        pde: HeatConductionPDE,
        bc_evaluator: BoundaryConditionEvaluator,
        learning_rate_adam: float = 1e-3,
        learning_rate_lbfgs: float = 1.0,
        weight_decay: float = 1e-5,
        w_pde: float = 1.0,
        w_bc: float = 10.0,
        w_ic: float = 10.0,
        w_data: float = 20.0,
        warmup_epochs: int = 300,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.pde = pde.to(device)
        self.bc_evaluator = bc_evaluator
        self.device = device

        self.lr_adam = learning_rate_adam
        self.lr_lbfgs = learning_rate_lbfgs
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # Loss weights
        self.weights = {
            "pde": w_pde,
            "bc": w_bc,
            "ic": w_ic,
            "data": w_data,
        }

        # Cached sensor measurements for inverse mode
        self.sensor_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

        # Setup Adam optimizer
        self._init_adam_optimizer()

        # Telemetry history
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "loss_total": [],
            "loss_pde": [],
            "loss_bc": [],
            "loss_ic": [],
            "loss_data": [],
            "alpha_learned": [],
            "alpha_error_pct": [],
            "relative_l2": [],
            "optimizer_phase": [],
        }

    def _init_adam_optimizer(self):
        """Initializes the Adam optimizer, respecting inverse warmup freezing."""
        if self.pde.is_inverse and self.warmup_epochs > 0:
            self.pde.alpha_param.requires_grad = False
            self.optimizer_adam: Optimizer = Adam(
                self.model.parameters(), lr=self.lr_adam, weight_decay=self.weight_decay
            )
        elif self.pde.is_inverse:
            param_groups = [
                {"params": self.model.parameters(), "lr": self.lr_adam, "weight_decay": self.weight_decay},
                {"params": [self.pde.alpha_param], "lr": self.lr_adam * 2.0, "weight_decay": 0.0},
            ]
            self.optimizer_adam: Optimizer = Adam(param_groups)
        else:
            self.optimizer_adam: Optimizer = Adam(
                self.model.parameters(), lr=self.lr_adam, weight_decay=self.weight_decay
            )

    def _init_lbfgs_optimizer(self, max_iter: int = 20, history_size: int = 50) -> LBFGS:
        """Initializes the L-BFGS second-order optimizer."""
        params = list(self.model.parameters())
        if self.pde.is_inverse:
            self.pde.alpha_param.requires_grad = True
            params.append(self.pde.alpha_param)

        return LBFGS(
            params,
            lr=self.lr_lbfgs,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

    def set_sensor_data(
        self, x_meas: torch.Tensor, t_meas: torch.Tensor, T_meas: torch.Tensor
    ):
        """Sets fixed measurement data for inverse parameter calibration."""
        self.sensor_data = (
            x_meas.to(self.device),
            t_meas.to(self.device),
            T_meas.to(self.device),
        )

    def compute_composite_loss(
        self,
        n_pde: int = 2500,
        n_ic: int = 128,
        n_bc: int = 128,
        n_sensors: int = 150,
        sensor_noise: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the weighted composite loss."""
        # Initialize fixed sensors if needed for inverse mode
        if self.pde.is_inverse and self.sensor_data is None:
            self.sensor_data = self.bc_evaluator.sample_sensor_data(
                n_sensors=n_sensors, noise_std=sensor_noise
            )

        losses = self.bc_evaluator.evaluate_losses(
            model=self.model,
            n_pde=n_pde,
            n_ic=n_ic,
            n_bc=n_bc,
            sensor_data=self.sensor_data if self.pde.is_inverse else None,
        )

        w = self.weights
        w_data_eff = w["data"] if self.pde.is_inverse else 0.0

        loss_total = (
            w["pde"] * losses["pde"]
            + w["bc"] * losses["bc"]
            + w["ic"] * losses["ic"]
            + w_data_eff * losses["data"]
        )

        return loss_total, losses

    def train_step_adam(
        self,
        epoch: int,
        n_pde: int = 2500,
        n_ic: int = 128,
        n_bc: int = 128,
        n_sensors: int = 150,
        sensor_noise: float = 0.01,
    ) -> Dict[str, float]:
        """Performs a single Adam optimization iteration."""
        # Unfreeze alpha after warmup
        if self.pde.is_inverse and self.warmup_epochs > 0 and epoch == self.warmup_epochs:
            self.pde.alpha_param.requires_grad = True
            param_groups = [
                {"params": self.model.parameters(), "lr": self.lr_adam, "weight_decay": self.weight_decay},
                {"params": [self.pde.alpha_param], "lr": self.lr_adam * 2.0, "weight_decay": 0.0},
            ]
            self.optimizer_adam = Adam(param_groups)

        self.model.train()
        self.optimizer_adam.zero_grad()

        loss_total, losses = self.compute_composite_loss(
            n_pde=n_pde,
            n_ic=n_ic,
            n_bc=n_bc,
            n_sensors=n_sensors,
            sensor_noise=sensor_noise,
        )

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer_adam.step()

        # Telemetry
        alpha_val = float(self.pde.alpha.item())
        true_alpha = self.pde.alpha_true
        alpha_err = abs(alpha_val - true_alpha) / true_alpha * 100.0

        metrics = {
            "epoch": epoch,
            "loss_total": float(loss_total.item()),
            "loss_pde": float(losses["pde"].item()),
            "loss_bc": float(losses["bc"].item()),
            "loss_ic": float(losses["ic"].item()),
            "loss_data": float(losses["data"].item()),
            "alpha_learned": alpha_val,
            "alpha_error_pct": alpha_err,
            "optimizer_phase": 1,  # 1 = Adam
        }

        self._record_telemetry(metrics)
        return metrics

    def train_step_lbfgs(
        self,
        optimizer_lbfgs: LBFGS,
        step_idx: int,
        n_pde: int = 2500,
        n_ic: int = 128,
        n_bc: int = 128,
        n_sensors: int = 150,
        sensor_noise: float = 0.01,
    ) -> Dict[str, float]:
        """Performs an L-BFGS second-order quasi-Newton step using closure."""
        self.model.train()
        latest_losses: Dict[str, float] = {}

        def closure():
            optimizer_lbfgs.zero_grad()
            loss_total, losses = self.compute_composite_loss(
                n_pde=n_pde,
                n_ic=n_ic,
                n_bc=n_bc,
                n_sensors=n_sensors,
                sensor_noise=sensor_noise,
            )
            loss_total.backward()

            latest_losses["total"] = float(loss_total.item())
            latest_losses["pde"] = float(losses["pde"].item())
            latest_losses["bc"] = float(losses["bc"].item())
            latest_losses["ic"] = float(losses["ic"].item())
            latest_losses["data"] = float(losses["data"].item())
            return loss_total

        optimizer_lbfgs.step(closure)

        alpha_val = float(self.pde.alpha.item())
        true_alpha = self.pde.alpha_true
        alpha_err = abs(alpha_val - true_alpha) / true_alpha * 100.0

        metrics = {
            "epoch": step_idx,
            "loss_total": latest_losses.get("total", 0.0),
            "loss_pde": latest_losses.get("pde", 0.0),
            "loss_bc": latest_losses.get("bc", 0.0),
            "loss_ic": latest_losses.get("ic", 0.0),
            "loss_data": latest_losses.get("data", 0.0),
            "alpha_learned": alpha_val,
            "alpha_error_pct": alpha_err,
            "optimizer_phase": 2,  # 2 = L-BFGS
        }

        self._record_telemetry(metrics)
        return metrics

    def _record_telemetry(self, metrics: Dict[str, Any]):
        """Appends metrics to solver history."""
        for key in [
            "epoch",
            "loss_total",
            "loss_pde",
            "loss_bc",
            "loss_ic",
            "loss_data",
            "alpha_learned",
            "alpha_error_pct",
            "optimizer_phase",
        ]:
            if key in metrics:
                self.history[key].append(metrics[key])

    def evaluate_relative_l2(self, nx: int = 128, nt: int = 128) -> float:
        """Evaluates relative L2 error against exact analytical solution."""
        self.model.eval()
        with torch.no_grad():
            x = np.linspace(0.0, 1.0, nx)
            t = np.linspace(0.0, 1.0, nt)
            X, T_grid = np.meshgrid(x, t)

            coords = torch.tensor(
                np.column_stack([X.ravel(), T_grid.ravel()]),
                dtype=torch.float32,
                device=self.device,
            )
            T_pred = self.model(coords).cpu().numpy().reshape(X.shape)
            T_true = self.pde.exact_temperature(X, T_grid)

            rel_l2 = np.linalg.norm(T_pred - T_true) / np.linalg.norm(T_true)
            return float(rel_l2)
