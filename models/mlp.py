"""
Physics-Informed Neural Networks (PINNs) Benchmark Suite - PyTorch 2.x
Module: models/mlp.py

Configurable Multi-Layer Perceptron (MLP) with Xavier normal initialization,
flexible activation functions (Tanh, Sin, GELU, SiLU), and optional input normalization.
Faithful modernization of Raissi et al. (2019, JCP).
"""

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn


class SinActivation(nn.Module):
    r"""
    Periodic Sine activation function: :math:`\sigma(x) = \sin(\omega_0 x)`.
    Popularized in implicit neural representations (Sitzmann et al., SIREN, NeurIPS 2020)
    for capturing steep spatial gradients and high-frequency wave numbers.
    """

    def __init__(self, omega_0: float = 1.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class MLP(nn.Module):
    r"""
    Deep Multi-Layer Perceptron for Physics-Informed Neural Networks.

    Key Features:
    - Xavier normal (Glorot normal) weight initialization with activation gain matching.
    - Zero bias initialization.
    - Domain normalization layer mapping :math:`x \in [lb, ub]` to :math:`[-1, 1]`.
    - Compatible with PyTorch 2.x autograd and torch.compile.

    Args:
        in_dim: Number of input coordinates (e.g., 2 for (x, t)).
        out_dim: Number of output fields (e.g., 1 for u, 2 for (u, v)).
        hidden_layers: List of hidden layer widths (e.g., [20]*8 or [64]*4).
        activation: Activation function name ('tanh', 'sin', 'gelu', 'silu').
        lb: Optional lower bound array [x_min, t_min] for coordinate normalization.
        ub: Optional upper bound array [x_max, t_max] for coordinate normalization.
        init_type: Weight initialization scheme ('xavier_normal', 'xavier_uniform', 'kaiming').
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh",
        lb: Optional[Union[np.ndarray, List[float], torch.Tensor]] = None,
        ub: Optional[Union[np.ndarray, List[float], torch.Tensor]] = None,
        init_type: str = "xavier_normal",
    ):
        super().__init__()

        if hidden_layers is None:
            # Historic Raissi 2019 default: 8 hidden layers of width 20
            hidden_layers = [20] * 8

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layers = hidden_layers
        self.activation_name = activation.lower()
        self.init_type = init_type

        # Domain normalization buffers
        if lb is not None and ub is not None:
            lb_t = torch.as_tensor(lb, dtype=torch.float32).view(1, -1)
            ub_t = torch.as_tensor(ub, dtype=torch.float32).view(1, -1)
            self.register_buffer("lb", lb_t)
            self.register_buffer("ub", ub_t)
            self.normalize_input = True
        else:
            self.register_buffer("lb", torch.zeros(1, in_dim))
            self.register_buffer("ub", torch.ones(1, in_dim))
            self.normalize_input = False

        # Build activation factory
        def get_act():
            if self.activation_name == "tanh":
                return nn.Tanh()
            elif self.activation_name == "sin":
                return SinActivation(omega_0=1.0)
            elif self.activation_name == "gelu":
                return nn.GELU()
            elif self.activation_name in ("silu", "swish"):
                return nn.SiLU()
            else:
                raise ValueError(f"Unsupported activation '{self.activation_name}'")

        # Layer dimensions
        layer_sizes = [in_dim] + list(hidden_layers) + [out_dim]
        self.num_layers = len(layer_sizes) - 1

        layers = []
        for i in range(self.num_layers):
            linear = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self._init_weights(linear, is_output=(i == self.num_layers - 1))
            layers.append(linear)
            if i < self.num_layers - 1:
                layers.append(get_act())

        self.network = nn.Sequential(*layers)

    def _init_weights(self, layer: nn.Linear, is_output: bool = False):
        """Initializes weights according to scientific best practices."""
        if is_output:
            gain = 1.0
        elif self.activation_name == "tanh":
            gain = nn.init.calculate_gain("tanh")
        elif self.activation_name in ("silu", "swish", "gelu"):
            gain = 1.0
        elif self.activation_name == "sin":
            gain = 1.0
        else:
            gain = 1.0

        if self.init_type == "xavier_normal":
            nn.init.xavier_normal_(layer.weight, gain=gain)
        elif self.init_type == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight, gain=gain)
        elif self.init_type == "kaiming":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        else:
            nn.init.xavier_normal_(layer.weight, gain=gain)

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass through the network:
            .. math::
                \mathbf{y} = \mathcal{N}_{\theta}\left( 2 \frac{\mathbf{x} - lb}{ub - lb} - 1 \right)

        Args:
            coords: Tensor of shape (N, in_dim) e.g., concat [x, t].
        Returns:
            Output field predictions of shape (N, out_dim).
        """
        if self.normalize_input:
            # Map [lb, ub] -> [-1.0, 1.0]
            coords = 2.0 * (coords - self.lb) / (self.ub - self.lb) - 1.0
        return self.network(coords)

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"MLP(in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"layers={self.hidden_layers}, activation={self.activation_name}, "
            f"params={self.count_parameters():,})"
        )


class HardBoundaryAnsatzPINN(nn.Module):
    r"""
    Exact (Hard) Boundary Condition Enforcement via Distance Field Ansatz:
        .. math::
            u_{\theta}(x, t) = u_0(x) + D(x, t) \cdot \mathcal{N}_{\theta}(x, t)

    Where :math:`D(x, t) = (1 - x^2) \cdot t` vanishes exactly at boundaries :math:`x = \pm 1`
    and at initial condition :math:`t = 0`. This eliminates boundary loss penalty competition,
    guaranteeing :math:`\mathcal{L}_{\mathrm{bc}} \equiv 0` identically across all training iterations.
    """

    def __init__(self, backbone_mlp: MLP):
        super().__init__()
        self.net = backbone_mlp

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (N, 2) [x, t]
        x = coords[:, 0:1]
        t = coords[:, 1:2]

        # Initial condition: u_0(x) = -sin(pi * x)
        u0 = -torch.sin(np.pi * x)

        # Distance function vanishing at x = -1, x = 1, and t = 0
        dist = (1.0 - x ** 2) * t

        # Neural network output
        nn_out = self.net(coords)

        # Exact ansatz satisfying initial and boundary conditions
        return u0 + dist * nn_out
