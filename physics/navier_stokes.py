"""
Physics-Informed Neural Networks (PINNs) Benchmark Suite - PyTorch 2.x
Module: physics/navier_stokes.py

Mathematical physics formulation for the 2D Incompressible Navier-Stokes Equations
(Vortex Shedding Behind a Cylinder benchmark, Raissi et al. 2019):
    u_t + λ₁ (u u_x + v u_y) = -p_x + λ₂ (u_xx + u_yy)
    v_t + λ₁ (u v_x + v v_y) = -p_y + λ₂ (v_xx + v_yy)

Using the stream-function formulation:
    u = ∂ψ/∂y,   v = -∂ψ/∂x
which identically satisfies the incompressibility constraint ∇ · u = 0.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class NavierStokes2DPDE(nn.Module):
    r"""
    2D Incompressible Navier-Stokes PDE Operator with stream-function parameterization:
        .. math::
            \psi, p = \mathcal{N}(x, y, t)
            u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}

    Momentum residuals:
        .. math::
            f_u = u_t + \lambda_1 (u u_x + v u_y) + p_x - \lambda_2 (u_{xx} + u_{yy}) = 0
            f_v = v_t + \lambda_1 (u v_x + v v_y) + p_y - \lambda_2 (v_{xx} + v_{yy}) = 0
    """

    def __init__(
        self,
        lambda_1: float = 1.0,
        lambda_2: float = 0.01,
        learnable_params: bool = False,
    ):
        super().__init__()
        self.learnable_params = learnable_params

        if learnable_params:
            self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            self.lambda_2 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        else:
            self.register_buffer("lambda_1", torch.tensor(lambda_1, dtype=torch.float32))
            self.register_buffer("lambda_2", torch.tensor(lambda_2, dtype=torch.float32))

    def compute_fields_and_residuals(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Computes velocity fields (u, v), pressure p, and momentum residuals (f_u, f_v).
        """
        for tensor in (x, y, t):
            if not tensor.requires_grad:
                tensor.requires_grad_(True)

        coords = torch.cat([x, y, t], dim=-1)
        out = model(coords)
        psi = out[:, 0:1]
        p = out[:, 1:2]

        # Velocity components from stream function
        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        # Derivatives of u
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        # Derivatives of v
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        # Derivatives of p
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        # Momentum residuals
        l1 = self.lambda_1
        l2 = self.lambda_2

        f_u = u_t + l1 * (u * u_x + v * u_y) + p_x - l2 * (u_xx + u_yy)
        f_v = v_t + l1 * (u * v_x + v * v_y) + p_y - l2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v
