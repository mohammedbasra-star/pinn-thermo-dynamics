"""
Physics-Informed Neural Networks (PINNs) Benchmark Suite - PyTorch 2.x
Module: physics/schrodinger.py

Mathematical physics formulation for the 1D Nonlinear Schrödinger (NLS) equation:
    i ∂ψ/∂t + 0.5 ∂²ψ/∂x² + |ψ|² ψ = 0,   x ∈ [-5, 5], t ∈ [0, π/2]
where ψ(x, t) = u(x, t) + i v(x, t).

Decomposed into real and imaginary coupled PDE residuals:
    f_u = ∂u/∂t + 0.5 ∂²v/∂x² + (u² + v²) v = 0
    f_v = ∂v/∂t - 0.5 ∂²u/∂x² - (u² + v²) u = 0
"""

from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn


class SchrodingerPDE(nn.Module):
    r"""
    1D Nonlinear Schrödinger (NLS) PDE Operator:
        .. math::
            i \psi_t + 0.5 \psi_{xx} + |\psi|^2 \psi = 0

    Coupled real-valued system for :math:`\psi = u + i v`:
        .. math::
            f_u = u_t + 0.5 v_{xx} + (u^2 + v^2) v = 0
        .. math::
            f_v = v_t - 0.5 u_{xx} - (u^2 + v^2) u = 0
    """

    def __init__(self):
        super().__init__()

    def compute_derivatives(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Computes (u, v) and their spatiotemporal derivatives using PyTorch autograd.
        Returns:
            u, v, u_t, v_t, u_xx, v_xx
        """
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)

        coords = torch.cat([x, t], dim=-1)
        uv = model(coords)
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        # First derivatives of u
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=(x, t),
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )
        u_x = grad_u[0]
        u_t = grad_u[1]

        # First derivatives of v
        grad_v = torch.autograd.grad(
            outputs=v,
            inputs=(x, t),
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
        )
        v_x = grad_v[0]
        v_t = grad_v[1]

        # Second spatial derivatives u_xx, v_xx
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
        )[0]

        v_xx = torch.autograd.grad(
            outputs=v_x,
            inputs=x,
            grad_outputs=torch.ones_like(v_x),
            create_graph=True,
            retain_graph=True,
        )[0]

        return u, v, u_t, v_t, u_xx, v_xx

    def compute_residual(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Evaluates coupled residuals f_u and f_v.
        """
        u, v, u_t, v_t, u_xx, v_xx = self.compute_derivatives(model, x, t)

        h_sq = u**2 + v**2
        f_u = u_t + 0.5 * v_xx + h_sq * v
        f_v = v_t - 0.5 * u_xx - h_sq * u

        return u, v, f_u, f_v

    def compute_boundary_derivatives(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Computes u, v, u_x, v_x at boundary coordinates for periodic boundary loss.
        """
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)

        coords = torch.cat([x, t], dim=-1)
        uv = model(coords)
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        v_x = torch.autograd.grad(
            outputs=v,
            inputs=x,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
        )[0]

        return u, v, u_x, v_x
