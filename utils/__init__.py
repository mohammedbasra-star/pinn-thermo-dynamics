"""
Utilities module: scientific evaluation metrics and publication-ready visualization routines.
"""

from .metrics import (
    compute_metrics,
    relative_l2_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    max_absolute_error,
)
from .visualization import (
    set_academic_style,
    plot_solution_tripanel,
    plot_loss_history,
    plot_cross_sections,
    plot_parameter_convergence,
)

__all__ = [
    "compute_metrics",
    "relative_l2_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "max_absolute_error",
    "set_academic_style",
    "plot_solution_tripanel",
    "plot_loss_history",
    "plot_cross_sections",
    "plot_parameter_convergence",
]
