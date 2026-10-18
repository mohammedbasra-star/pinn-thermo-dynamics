"""
Physics-Informed Neural Networks for High-Temperature Turbomachinery & Thermodynamic Dynamics
Module: physics/heat_pde.py

Pure PyTorch implementation of the 1D unsteady heat conduction PDE residual:
    ∂T/∂t - α ∂²T/∂x² - Q(x, t) = 0

Uses torch.autograd.grad with create_graph=True, retain_graph=True to compute
exact spatial and temporal derivatives without spatial discretization artifacts.
"""

from typing import Tuple, Dict, Union, Optional
import torch
import torch.nn as nn
import numpy as np


class HeatConductionPDE(nn.Module):
    r"""
    Governing unsteady aerothermal heat conduction PDE:
        .. math::
            \mathcal{R}(x, t) = \frac{\partial T}{\partial t} - \alpha \frac{\partial^2 T}{\partial x^2} - Q(x, t) = 0

    Supports both:
    1. Forward Mode: Known thermal diffusivity parameter :math:`\alpha \in \mathbb{R}^+`.
    2. Inverse Mode: Unknown parameter :math:`\alpha` formulated as a learnable `torch.nn.Parameter`
       calibrated dynamically from sparse sensor measurements.
    """

    def __init__(
        self,
        alpha_true: float = 0.05,
        is_inverse: bool = False,
        initial_guess_alpha: float = 0.01,
        decay_lambda: float = 0.5,
        T_hot: float = 1.0,
        T_cool: float = 0.2,
    ):
        super().__init__()
        self.alpha_true = float(alpha_true)
        self.is_inverse = is_inverse
        self.decay_lambda = float(decay_lambda)
        self.T_hot = float(T_hot)
        self.T_cool = float(T_cool)

        if is_inverse:
            # Learnable physical parameter initialized from an arbitrary initial guess
            self.alpha_param = nn.Parameter(
                torch.tensor(initial_guess_alpha, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "alpha_fixed", torch.tensor(alpha_true, dtype=torch.float32)
            )

    @property
    def alpha(self) -> torch.Tensor:
        """Returns the active thermal diffusivity, enforcing physical positivity."""
        if self.is_inverse:
            return torch.clamp(self.alpha_param, min=1e-5)
        else:
            return self.alpha_fixed

    def exact_temperature(
        self, x: Union[torch.Tensor, np.ndarray], t: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        r"""
        Exact manufactured analytical solution:
            .. math::
                T^*(x, t) = \sin(\pi x) e^{-\lambda t} + T_{hot}(1 - x) + T_{cool} x
        """
        if isinstance(x, torch.Tensor):
            sin_term = torch.sin(np.pi * x) * torch.exp(-self.decay_lambda * t)
            linear_profile = self.T_hot * (1.0 - x) + self.T_cool * x
            return sin_term + linear_profile
        else:
            sin_term = np.sin(np.pi * x) * np.exp(-self.decay_lambda * t)
            linear_profile = self.T_hot * (1.0 - x) + self.T_cool * x
            return sin_term + linear_profile

    def compute_source(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes the analytical heat source term satisfying the manufactured benchmark:
            .. math::
                Q(x, t) = (\alpha_{true} \pi^2 - \lambda) \sin(\pi x) e^{-\lambda t}
        """
        factor = (self.alpha_true * (np.pi ** 2)) - self.decay_lambda
        return factor * torch.sin(np.pi * x) * torch.exp(-self.decay_lambda * t)

    def residual(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Evaluates the differential residual of the PDE using PyTorch automatic differentiation:
            1. Forward pass: :math:`\hat{T} = \mathcal{N}_\theta(x, t)`
            2. Compute :math:`\frac{\partial \hat{T}}{\partial t}` and :math:`\frac{\partial \hat{T}}{\partial x}`
            3. Compute :math:`\frac{\partial^2 \hat{T}}{\partial x^2}`
            4. Compute :math:`\mathcal{R}(x, t) = \frac{\partial \hat{T}}{\partial t} - \alpha \frac{\partial^2 \hat{T}}{\partial x^2} - Q(x, t)`
        """
        coords = torch.cat([x, t], dim=1)
        T_pred = model(coords)

        # 1st-order derivatives
        grads = torch.autograd.grad(
            outputs=T_pred,
            inputs=[x, t],
            grad_outputs=torch.ones_like(T_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        T_x = grads[0]
        T_t = grads[1]

        # 2nd-order spatial derivative
        grads_2nd = torch.autograd.grad(
            outputs=T_x,
            inputs=x,
            grad_outputs=torch.ones_like(T_x),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        T_xx = grads_2nd[0]

        source = self.compute_source(x, t)
        alpha = self.alpha
        res = T_t - alpha * T_xx - source

        diagnostics = {
            "T_pred": T_pred,
            "T_t": T_t,
            "T_x": T_x,
            "T_xx": T_xx,
            "source": source,
            "alpha": alpha,
        }

        return res, T_pred, diagnostics
