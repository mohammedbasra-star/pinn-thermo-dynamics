"""
Physics-Informed Neural Networks (PINNs) Benchmark Suite - PyTorch 2.x
Module: physics
"""

from physics.burgers import BurgersPDE
from physics.schrodinger import SchrodingerPDE
from physics.navier_stokes import NavierStokes2DPDE

__all__ = [
    "BurgersPDE",
    "SchrodingerPDE",
    "NavierStokes2DPDE",
]
