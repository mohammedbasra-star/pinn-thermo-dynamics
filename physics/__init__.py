"""
Physics module for High-Temperature Turbomachinery & Thermodynamic Dynamics.
Governing PDE formulations, boundary conditions, and automatic differentiation residuals.
"""

from .heat_pde import HeatConductionPDE
from .boundary_conditions import BoundaryConditionEvaluator

# Maintain backward compatibility
try:
    from .pde_residuals import (
        ManufacturedHeatConduction,
        PolytropicThermodynamics,
        compute_pde_residuals,
    )
    from .boundary_conds import (
        BoundaryConditionManager,
        CollocationSampler,
    )
except ImportError:
    pass

__all__ = [
    "HeatConductionPDE",
    "BoundaryConditionEvaluator",
]
