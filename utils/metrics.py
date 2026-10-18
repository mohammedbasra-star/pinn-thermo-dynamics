"""
Physics-Informed Neural Networks for High-Temperature Turbomachinery & Thermodynamic Dynamics
Module: utils/metrics.py

Rigorous scientific error metrics for benchmark quantification:
- Relative L2 error (Frobenius norm ratio)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Maximum Pointwise Error (L_inf norm)
- Inverse Parameter Discovery Error (%)
"""

from typing import Dict, Union
import numpy as np
import torch


def to_numpy(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Converts torch tensor or numpy array to a flattened 1D numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().ravel()
    return np.asarray(data).ravel()


def relative_l2_error(
    pred: Union[torch.Tensor, np.ndarray],
    true: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-12,
) -> float:
    r"""
    Computes Relative L2 Error (standard metric in SciML and PINN literature):
        .. math::
            \epsilon_{L_2} = \frac{\|\hat{T} - T^*\|_2}{\|T^*\|_2 + \epsilon}
    """
    p = to_numpy(pred)
    t = to_numpy(true)
    norm_diff = np.linalg.norm(p - t, 2)
    norm_true = np.linalg.norm(t, 2)
    return float(norm_diff / (norm_true + eps))


def mean_squared_error(
    pred: Union[torch.Tensor, np.ndarray],
    true: Union[torch.Tensor, np.ndarray],
) -> float:
    """Computes Mean Squared Error (MSE)."""
    p = to_numpy(pred)
    t = to_numpy(true)
    return float(np.mean((p - t) ** 2))


def root_mean_squared_error(
    pred: Union[torch.Tensor, np.ndarray],
    true: Union[torch.Tensor, np.ndarray],
) -> float:
    """Computes Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(pred, true)))


def mean_absolute_error(
    pred: Union[torch.Tensor, np.ndarray],
    true: Union[torch.Tensor, np.ndarray],
) -> float:
    """Computes Mean Absolute Error (MAE)."""
    p = to_numpy(pred)
    t = to_numpy(true)
    return float(np.mean(np.abs(p - t)))


def max_absolute_error(
    pred: Union[torch.Tensor, np.ndarray],
    true: Union[torch.Tensor, np.ndarray],
) -> float:
    r"""
    Computes Maximum Pointwise Absolute Error (:math:`L_\infty` norm):
        .. math::
            \epsilon_{\infty} = \max_i |\hat{T}_i - T^*_i|
    """
    p = to_numpy(pred)
    t = to_numpy(true)
    return float(np.max(np.abs(p - t)))


def compute_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    true: Union[torch.Tensor, np.ndarray],
    p_learned: float = None,
    p_true: float = None,
) -> Dict[str, float]:
    """Computes comprehensive scientific benchmark metrics dictionary."""
    metrics = {
        "relative_l2_error": relative_l2_error(pred, true),
        "mean_squared_error": mean_squared_error(pred, true),
        "rmse": root_mean_squared_error(pred, true),
        "mae": mean_absolute_error(pred, true),
        "max_error_linf": max_absolute_error(pred, true),
    }

    if p_learned is not None and p_true is not None:
        param_error_pct = abs(p_learned - p_true) / abs(p_true) * 100.0
        metrics["param_true"] = float(p_true)
        metrics["param_learned"] = float(p_learned)
        metrics["param_error_pct"] = float(param_error_pct)

    return metrics
