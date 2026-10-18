"""
Physics-Informed Neural Networks (PINNs) Benchmark Suite - PyTorch 2.x
Module: physics/burgers.py

Mathematical physics formulation for the 1D viscous Burgers' equation:
    ∂u/∂t + u ∂u/∂x - ν ∂²u/∂x² = 0

Features:
- Dynamic reverse-mode Automatic Differentiation (torch.autograd) with full graph preservation.
- Vectorized differentiation support via torch.func (vmap / jacrev).
- Forward simulation mode (fixed ν = 0.01 / π) and inverse parameter discovery mode (learnable λ₁, λ₂).
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn


class BurgersPDE(nn.Module):
    r"""
    1D Viscous Burgers' Equation Residual:
        .. math::
            \mathcal{R}(u; \nu) = \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0

    Continuous Time Formulation:
    - Inputs: Spatiotemporal coordinates :math:`(x, t) \in [-1, 1] \times [0, 0.99]`.
    - Viscosity parameter: :math:`\nu = \frac{0.01}{\pi} \approx 0.00318309886`.

    In inverse discovery mode:
        .. math::
            f(x, t) = u_t + \lambda_1 u u_x - \lambda_2 u_{xx}
        where :math:`\lambda_1` and :math:`\lambda_2` are learnable parameters.
    """

    def __init__(
        self,
        nu: float = 0.01 / np.pi,
        inverse_mode: bool = False,
        init_lambda_1: float = 0.0,
        init_lambda_2: float = 0.0,
    ):
        super().__init__()
        self.inverse_mode = inverse_mode

        if inverse_mode:
            self.lambda_1 = nn.Parameter(torch.tensor([init_lambda_1], dtype=torch.float32))
            self.lambda_2 = nn.Parameter(torch.tensor([init_lambda_2], dtype=torch.float32))
        else:
            self.register_buffer("nu", torch.tensor(nu, dtype=torch.float32))
            self.lambda_1 = None
            self.lambda_2 = None

    def compute_derivatives_autograd(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Computes u(x, t) and partial derivatives using PyTorch dynamic autograd:
            .. math::
                u_x = \frac{\partial u}{\partial x}, \quad
                u_t = \frac{\partial u}{\partial t}, \quad
                u_{xx} = \frac{\partial^2 u}{\partial x^2}
        """
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)

        coords = torch.cat([x, t], dim=-1)
        u = model(coords)

        # First-order derivatives [u_x, u_t]
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=(x, t),
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )
        u_x = grad_u[0]
        u_t = grad_u[1]

        # Second-order spatial derivative u_xx
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
        )[0]

        return u, u_t, u_x, u_xx

    def compute_residual(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Evaluates the Burgers' equation PDE residual:
            .. math::
                f(x, t) = u_t + u u_x - \nu u_{xx}

        Returns:
            u: Field prediction (N, 1)
            f: Residual error (N, 1)
        """
        u, u_t, u_x, u_xx = self.compute_derivatives_autograd(model, x, t)

        if self.inverse_mode:
            f = u_t + self.lambda_1 * u * u_x - torch.exp(self.lambda_2) * u_xx
        else:
            f = u_t + u * u_x - self.nu * u_xx

        return u, f

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.compute_residual(model, x, t)

    def get_parameters(self) -> Tuple[Optional[float], Optional[float]]:
        """Returns the identified inverse parameters (lambda_1, lambda_2) if applicable."""
        if not self.inverse_mode:
            return 1.0, float(self.nu.item())
        l1 = float(self.lambda_1.item())
        l2 = float(torch.exp(self.lambda_2).item())
        return l1, l2
