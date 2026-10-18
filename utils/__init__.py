"""
Physics-Informed Neural Networks (PINNs) Benchmark Suite - PyTorch 2.x
Module: utils
"""

from utils.metrics import (
    relative_l2_error,
    frobenius_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    max_absolute_error,
    compute_metrics,
)
from utils.visualization import (
    set_academic_style,
    plot_raissi_burgers_figure,
    plot_solution_tripanel,
    plot_loss_history,
)

__all__ = [
    "relative_l2_error",
    "frobenius_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "max_absolute_error",
    "compute_metrics",
    "set_academic_style",
    "plot_raissi_burgers_figure",
    "plot_solution_tripanel",
    "plot_loss_history",
]
