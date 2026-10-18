"""
Physics-Informed Neural Networks for High-Temperature Turbomachinery & Thermodynamic Dynamics
Module: physics/pde_residuals.py

Formulates the governing aerothermal PDEs, high-temperature thermodynamic relationships,
and high-order automatic differentiation (Autograd) residual evaluation.
"""

from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import numpy as np


class PolytropicThermodynamics:
    r"""
    Thermodynamic relations for gas turbine compressor/turbine stages.

    Governs polytropic compression/expansion relations:
        .. math::
            \frac{T_{total, 2}}{T_{total, 1}} = \left( \frac{P_{total, 2}}{P_{total, 1}} \right)^{\frac{n-1}{n}}

    where:
        - :math:`n`: Polytropic expansion/compression exponent
        - :math:`\gamma`: Heat capacity ratio (:math:`c_p / c_v \approx 1.4` for dry air)
        - :math:`\eta_p`: Polytropic efficiency (:math:`\frac{n-1}{n} = \frac{\gamma - 1}{\gamma \cdot \eta_p}`)
    """

    def __init__(self, gamma: float = 1.4, eta_polytropic: float = 0.88):
        self.gamma = gamma
        self.eta_p = eta_polytropic
        # Exponent for polytropic compression
        self.exponent = (gamma - 1.0) / (gamma * eta_polytropic)

    def compute_stage_exit_temperature(
        self, T_inlet: torch.Tensor, pressure_ratio: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes stage total exit temperature given inlet total temperature and pressure ratio:
        .. math::
            T_{exit} = T_{inlet} \cdot (PR)^{\frac{n-1}{n}}
        """
        return T_inlet * torch.pow(pressure_ratio, self.exponent)

    def thermal_degradation_factor(
        self, operating_hours: torch.Tensor, degradation_rate: float = 1.2e-4
    ) -> torch.Tensor:
        r"""
        Models aerodynamic/thermal efficiency degradation over cumulative turbine firing hours:
        .. math::
            \eta(t) = \eta_0 \cdot \exp(-\kappa \cdot t)
        """
        return torch.exp(-degradation_rate * operating_hours)


class ManufacturedHeatConduction:
    r"""
    Manufactured analytical benchmark for high-temperature unsteady heat conduction in a turbine blade.

    The governing 1D aerothermal heat equation is:
        .. math::
            \frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2} + Q(x, t),
            \quad x \in [0, 1], \; t \in [0, 1]

    Analytical benchmark field:
        .. math::
            T^*(x, t) = \sin(\pi x) e^{-\lambda t} + T_{hot}(1 - x) + T_{cool} x

    where:
        - :math:`\alpha`: Thermal diffusivity (:math:`m^2/s`)
        - :math:`\lambda`: Thermal decay constant
        - :math:`T_{hot}`: Dimensionless exterior hot gas temperature (:math:`x=0`)
        - :math:`T_{cool}`: Dimensionless internal cooling air temperature (:math:`x=1`)

    Exact spatial and temporal derivatives:
        .. math::
            \frac{\partial T^*}{\partial t} = -\lambda \sin(\pi x) e^{-\lambda t}
        .. math::
            \frac{\partial^2 T^*}{\partial x^2} = -\pi^2 \sin(\pi x) e^{-\lambda t}

    The corresponding exact analytical heat source :math:`Q(x, t)` is:
        .. math::
            Q(x, t) = \frac{\partial T^*}{\partial t} - \alpha \frac{\partial^2 T^*}{\partial x^2}
                    = (\alpha \pi^2 - \lambda) \sin(\pi x) e^{-\lambda t}
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

    def exact_temperature(
        self, x: Union[torch.Tensor, np.ndarray], t: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Calculates exact analytical temperature field."""
        if isinstance(x, torch.Tensor):
            sin_term = torch.sin(np.pi * x) * torch.exp(-self.decay_lambda * t)
            linear_profile = self.T_hot * (1.0 - x) + self.T_cool * x
            return sin_term + linear_profile
        else:
            sin_term = np.sin(np.pi * x) * np.exp(-self.decay_lambda * t)
            linear_profile = self.T_hot * (1.0 - x) + self.T_cool * x
            return sin_term + linear_profile

    def exact_source(
        self, x: Union[torch.Tensor, np.ndarray], t: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Calculates exact analytical source term Q(x, t)."""
        factor = (self.alpha_true * (np.pi ** 2)) - self.decay_lambda
        if isinstance(x, torch.Tensor):
            return factor * torch.sin(np.pi * x) * torch.exp(-self.decay_lambda * t)
        else:
            return factor * np.sin(np.pi * x) * np.exp(-self.decay_lambda * t)


class HeatConductionPDE(nn.Module):
    r"""
    Physics module that formulates the unsteady heat conduction PDE residual:
        .. math::
            \mathcal{R}(x, t) = \frac{\partial T}{\partial t} - \alpha \frac{\partial^2 T}{\partial x^2} - Q(x, t)

    Supports both:
        1. Forward problem: Known diffusivity :math:`\alpha`
        2. Inverse problem: Learnable diffusivity parameter :math:`\alpha` initialized from guess.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        is_inverse: bool = False,
        decay_lambda: float = 0.5,
        initial_guess_alpha: float = 0.01,
    ):
        super().__init__()
        self.is_inverse = is_inverse
        self.benchmark = ManufacturedHeatConduction(
            alpha_true=alpha, decay_lambda=decay_lambda
        )

        if is_inverse:
            self.alpha_param = nn.Parameter(
                torch.tensor(initial_guess_alpha, dtype=torch.float32)
            )
            self._true_alpha = alpha
        else:
            self.register_buffer(
                "fixed_alpha", torch.tensor(alpha, dtype=torch.float32)
            )
            self._true_alpha = alpha

    @property
    def alpha(self) -> torch.Tensor:
        """Returns the effective thermal diffusivity parameter."""
        if self.is_inverse:
            return torch.clamp(self.alpha_param, min=1e-5)
        else:
            return self.fixed_alpha

    @property
    def true_alpha(self) -> float:
        """Returns the true ground truth thermal diffusivity."""
        return self._true_alpha

    def compute_source(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Computes the external thermal source/sink term."""
        return self.benchmark.exact_source(x, t)


def compute_pde_residuals(
    model: nn.Module,
    pde: HeatConductionPDE,
    x: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    r"""
    Computes PDE residuals using high-order Automatic Differentiation (Autograd).

    Given input coordinates :math:`(x, t)`:
    1. Forward pass: :math:`\hat{T} = \mathcal{N}_\theta(x, t)`
    2. First-order autograd:
       .. math::
           \frac{\partial \hat{T}}{\partial t}, \quad \frac{\partial \hat{T}}{\partial x}
    3. Second-order autograd:
       .. math::
           \frac{\partial^2 \hat{T}}{\partial x^2}
    4. Residual evaluation:
       .. math::
           \mathcal{R}(x, t) = \frac{\partial \hat{T}}{\partial t} - \alpha \frac{\partial^2 \hat{T}}{\partial x^2} - Q(x, t)

    Parameters
    ----------
    model : nn.Module
        Neural network mapping (x, t) -> T
    pde : HeatConductionPDE
        Governing PDE definition and parameters
    x : torch.Tensor
        Spatial coordinates (N, 1), requires_grad=True
    t : torch.Tensor
        Temporal coordinates (N, 1), requires_grad=True

    Returns
    -------
    residual : torch.Tensor
        PDE residual at collocation points (N, 1)
    T_pred : torch.Tensor
        Predicted temperature field (N, 1)
    gradients : dict
        Dictionary of intermediate derivatives (T_t, T_x, T_xx)
    """
    # Combine inputs into coordinates tensor
    coords = torch.cat([x, t], dim=1)
    T_pred = model(coords)

    # 1st-order temporal and spatial derivatives
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

    # Source term evaluation
    source = pde.compute_source(x, t)

    # PDE Residual: T_t - alpha * T_xx - source
    alpha = pde.alpha
    residual = T_t - alpha * T_xx - source

    derivatives = {
        "T_pred": T_pred,
        "T_t": T_t,
        "T_x": T_x,
        "T_xx": T_xx,
        "source": source,
        "alpha": alpha,
    }

    return residual, T_pred, derivatives
