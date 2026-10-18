"""
Physics-Informed Neural Networks for High-Temperature Turbomachinery & Thermodynamic Dynamics
Module: models/networks.py

Provides neural network architectures tailored for Scientific Machine Learning (SciML):
- Standard Fully-Connected Deep MLP with Xavier/He initialization.
- Random Fourier Feature (RFF) Embeddings (Tancik et al., NeurIPS 2020) to overcome spectral bias.
- Modified MLP with Transformer-inspired Gating (Wang, Yu, & Perdikaris, 2021) for stiff PDE gradients.
"""

from typing import List, Optional, Callable
import torch
import torch.nn as nn
import numpy as np


class FourierFeatureEmbedding(nn.Module):
    r"""
    Random Fourier Feature (RFF) positional embedding layer.

    Maps low-dimensional coordinates :math:`\mathbf{x} \in \mathbb{R}^d` to high-dimensional
    sinusoidal representations:
        .. math::
            \gamma(\mathbf{x}) = \left[ \cos(2\pi \mathbf{B}\mathbf{x}), \sin(2\pi \mathbf{B}\mathbf{x}) \right]^T

    where :math:`\mathbf{B} \sim \mathcal{N}(0, \sigma^2)` is a fixed random Gaussian projection matrix.
    Prevents the spectral bias (F-Principle) wherein standard MLPs struggle to learn
    sharp thermal boundary layers and high-frequency aerothermal transients.
    """

    def __init__(self, in_features: int, mapping_size: int = 32, scale: float = 2.0):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.out_features = mapping_size * 2

        # Fixed random Gaussian matrix B
        B = torch.randn(in_features, mapping_size) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, in_features)
        # B shape: (in_features, mapping_size)
        x_proj = 2.0 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class StandardMLP(nn.Module):
    r"""
    Deep Fully Connected Multi-Layer Perceptron (MLP) for PINN field approximation.

    Features:
    - Flexible activation functions (:math:`\tanh`, Sine, Swish, GELU)
    - Optional Fourier feature coordinate embedding
    - Xavier normal initialization with activation-specific gain
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 4,
        activation: str = "tanh",
        use_fourier: bool = False,
        fourier_scale: float = 2.0,
        fourier_dim: int = 32,
    ):
        super().__init__()
        self.use_fourier = use_fourier

        if use_fourier:
            self.embedding = FourierFeatureEmbedding(
                in_features=in_dim, mapping_size=fourier_dim, scale=fourier_scale
            )
            current_dim = self.embedding.out_features
        else:
            self.embedding = None
            current_dim = in_dim

        # Select activation
        act_dict = {
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "swish": nn.SiLU(),
            "sin": SinActivation(),
        }
        self.activation = act_dict.get(activation.lower(), nn.Tanh())

        layers: List[nn.Module] = []
        for _ in range(num_layers):
            linear = nn.Linear(current_dim, hidden_dim)
            self._init_weights(linear, activation)
            layers.append(linear)
            layers.append(self.activation)
            current_dim = hidden_dim

        # Final projection layer to scalar field T(x, t)
        final_layer = nn.Linear(current_dim, out_dim)
        nn.init.xavier_normal_(final_layer.weight, gain=1.0)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        self.network = nn.Sequential(*layers)

    def _init_weights(self, layer: nn.Linear, activation: str):
        if activation.lower() == "tanh":
            nn.init.xavier_normal_(layer.weight, gain=5.0 / 3.0)
        elif activation.lower() == "sin":
            nn.init.xavier_uniform_(layer.weight, gain=np.sqrt(6.0 / (layer.in_features + layer.out_features)))
        else:
            nn.init.xavier_normal_(layer.weight, gain=1.0)
        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fourier and self.embedding is not None:
            x = self.embedding(x)
        return self.network(x)


class SinActivation(nn.Module):
    """Sinusoidal activation function for SIREN-style implicit neural representations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class ModifiedMLP(nn.Module):
    r"""
    Modified Multi-Layer Perceptron (Wang, Yu, & Perdikaris, 2021).
    Reference: 'On the eigenvalue pathology of physics-informed neural networks',
    Journal of Computational Physics, 2021.

    Architecture introduces two transformer-like encoder projections :math:`U` and :math:`V`
    derived directly from the spatiotemporal coordinates :math:`X = (x, t)`:
        .. math::
            U = \phi(X W_u + b_u), \quad V = \phi(X W_v + b_v)
            H^{(1)} = \phi(X W_1 + b_1)
            H^{(k+1)} = (1 - H^{(k)}) \odot U + H^{(k)} \odot V, \quad k = 1, \dots, L-1
            \hat{T} = H^{(L)} W_{out} + b_{out}

    This bypasses gradient vanishing/exploding pathologies in stiff, multi-scale aerothermal PDEs.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 4,
        activation: str = "tanh",
        use_fourier: bool = False,
        fourier_scale: float = 2.0,
        fourier_dim: int = 32,
    ):
        super().__init__()
        self.use_fourier = use_fourier
        self.num_layers = num_layers

        if use_fourier:
            self.embedding = FourierFeatureEmbedding(
                in_features=in_dim, mapping_size=fourier_dim, scale=fourier_scale
            )
            input_dim = self.embedding.out_features
        else:
            self.embedding = None
            input_dim = in_dim

        # Activation
        act_dict = {
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "sin": SinActivation(),
        }
        self.phi = act_dict.get(activation.lower(), nn.Tanh())

        # Encoders U and V
        self.encoder_U = nn.Linear(input_dim, hidden_dim)
        self.encoder_V = nn.Linear(input_dim, hidden_dim)

        # Hidden transformation layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.first_layer = nn.Linear(input_dim, hidden_dim)

        # Output projection
        self.out_layer = nn.Linear(hidden_dim, out_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fourier and self.embedding is not None:
            x_emb = self.embedding(x)
        else:
            x_emb = x

        # Compute gating encoders
        u = self.phi(self.encoder_U(x_emb))
        v = self.phi(self.encoder_V(x_emb))

        # First hidden state
        h = self.phi(self.first_layer(x_emb))

        # Gated hidden transitions
        for layer in self.hidden_layers:
            z = self.phi(layer(h))
            h = (1.0 - z) * u + z * v

        # Final output
        return self.out_layer(h)


def build_network(
    arch_type: str = "modified_mlp",
    in_dim: int = 2,
    out_dim: int = 1,
    hidden_dim: int = 64,
    num_layers: int = 4,
    activation: str = "tanh",
    use_fourier: bool = False,
    fourier_scale: float = 2.0,
    fourier_dim: int = 32,
) -> nn.Module:
    """
    Factory function for building scientific neural network architectures.

    Parameters:
        arch_type: 'modified_mlp' or 'standard_mlp'
    """
    if arch_type.lower() == "modified_mlp":
        return ModifiedMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_fourier=use_fourier,
            fourier_scale=fourier_scale,
            fourier_dim=fourier_dim,
        )
    elif arch_type.lower() == "standard_mlp":
        return StandardMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_fourier=use_fourier,
            fourier_scale=fourier_scale,
            fourier_dim=fourier_dim,
        )
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}. Use 'modified_mlp' or 'standard_mlp'.")
