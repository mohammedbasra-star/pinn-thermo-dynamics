"""
Physics-Informed Neural Networks (PINNs) Benchmark Suite - PyTorch 2.x
Module: models
"""

from models.mlp import MLP, SinActivation
from models.compiled_solver import PINNCompiledSolver

__all__ = [
    "MLP",
    "SinActivation",
    "PINNCompiledSolver",
]
