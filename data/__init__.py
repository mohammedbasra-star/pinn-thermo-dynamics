"""
Data module for ground truth generation and reference numerical solvers.
"""

from .generate_ground_truth import (
    AnalyticalHeatConduction,
    CrankNicolsonHeatSolver,
    generate_benchmark,
)

__all__ = [
    "AnalyticalHeatConduction",
    "CrankNicolsonHeatSolver",
    "generate_benchmark",
]
