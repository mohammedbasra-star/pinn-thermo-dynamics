"""
Physics-Informed Neural Networks (PINNs) Physics Suite.
"""

from physics.burgers import BurgersPDE
from physics.schrodinger import SchrodingerPDE
from physics.navier_stokes import NavierStokesPDE
from physics.thermal_conduction import TurbomachineryHeatConduction2D, solve_crank_nicolson_2d

__all__ = [
    "BurgersPDE",
    "SchrodingerPDE",
    "NavierStokesPDE",
    "TurbomachineryHeatConduction2D",
    "solve_crank_nicolson_2d",
]
