"""
Physics-Informed Neural Networks (PINNs) Benchmark Suite - PyTorch 2.x
Module: utils/metrics.py

Rigorous scientific error metrics for benchmark quantification (Raissi et al. 2019):
- Relative L2 error: ||u_pred - u_true||_2 / ||u_true||_2
- Frobenius norm error: ||u_pred - u_true||_F / ||u_true||_F
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Maximum Pointwise Error (L_inf norm)
- Parameter discovery percentage error (inverse problems)
"""

from typing import Dict, Union, Optional
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
            \epsilon_{L_2} = \frac{\|\hat{u} - u^*\|_2}{\|u^*\|_2 + \epsilon}
    """
    p = to_numpy(pred)
    t = to_numpy(true)
    norm_diff = np.linalg.norm(p - t, 2)
    norm_true = np.linalg.norm(t, 2)
    return float(norm_diff / (norm_true + eps))


def frobenius_error(
    pred: Union[torch.Tensor, np.ndarray],
    true: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-12,
) -> float:
    r"""
    Computes Relative Frobenius Norm Error for 2D fields:
        .. math::
            \epsilon_{F} = \frac{\|\hat{\mathbf{U}} - \mathbf{U}^*\|_F}{\|\mathbf{U}^*\|_F + \epsilon}
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    norm_diff = np.linalg.norm(pred - true, "fro")
    norm_true = np.linalg.norm(true, "fro")
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
            \epsilon_{\infty} = \max_i |\hat{u}_i - u^*_i|
    """
    p = to_numpy(pred)
    t = to_numpy(true)
    return float(np.max(np.abs(p - t)))


def compute_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    true: Union[torch.Tensor, np.ndarray],
    param_learned: Optional[float] = None,
    param_true: Optional[float] = None,
) -> Dict[str, float]:
    """Computes comprehensive scientific benchmark metrics dictionary."""
    metrics = {
        "relative_l2_error": relative_l2_error(pred, true),
        "frobenius_error": frobenius_error(pred, true),
        "mean_squared_error": mean_squared_error(pred, true),
        "rmse": root_mean_squared_error(pred, true),
        "mae": mean_absolute_error(pred, true),
        "max_error_linf": max_absolute_error(pred, true),
    }

    if param_learned is not None and param_true is not None:
        param_error_pct = abs(param_learned - param_true) / abs(param_true) * 100.0
        metrics["param_true"] = float(param_true)
        metrics["param_learned"] = float(param_learned)
        metrics["param_error_pct"] = float(param_error_pct)

    return metrics
