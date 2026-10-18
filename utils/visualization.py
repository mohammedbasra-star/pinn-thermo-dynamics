"""
Physics-Informed Neural Networks for High-Temperature Turbomachinery & Thermodynamic Dynamics
Module: utils/visualization.py

Publication-ready scientific plotting routines for journal articles and benchmark documentation.
Generates 300 DPI high-resolution figures:
- Tri-panel 2D heatmaps (Exact Solution, PINN Prediction, Absolute Pointwise Error)
- Dual-phase (Adam + L-BFGS) loss convergence trajectories
- 1D Spatiotemporal temperature profile slices
- Inverse parameter discovery convergence curves
"""

from typing import Dict, List, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_academic_style():
    """Configures matplotlib to publication-grade aesthetics."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "font.size": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_solution_tripanel(
    X: np.ndarray,
    T_grid: np.ndarray,
    T_true: np.ndarray,
    T_pred: np.ndarray,
    save_path: Optional[str] = None,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Plots a publication 3-panel 2D comparison:
    [Ground Truth] vs [PINN Prediction] vs [Absolute Pointwise Error]
    """
    set_academic_style()
    error = np.abs(T_pred - T_true)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)

    vmin = min(T_true.min(), T_pred.min())
    vmax = max(T_true.max(), T_pred.max())

    # 1. Ground Truth
    c0 = axes[0].contourf(T_grid, X, T_true, levels=60, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title(r"(a) Exact Ground Truth $T^*(x, t)$")
    axes[0].set_xlabel(r"Time $t$ [s]")
    axes[0].set_ylabel(r"Blade Wall Coordinate $x/L$")
    fig.colorbar(c0, ax=axes[0], fraction=0.046, pad=0.04, label=r"Temperature $T$")

    # 2. PINN Prediction
    c1 = axes[1].contourf(T_grid, X, T_pred, levels=60, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title(rf"(b) PINN Prediction $\hat{{T}}(x, t)$ {title_suffix}")
    axes[1].set_xlabel(r"Time $t$ [s]")
    axes[1].set_ylabel(r"Blade Wall Coordinate $x/L$")
    fig.colorbar(c1, ax=axes[1], fraction=0.046, pad=0.04, label=r"Predicted $T$")

    # 3. Absolute Pointwise Error
    c2 = axes[2].contourf(T_grid, X, error, levels=60, cmap="viridis")
    axes[2].set_title(r"(c) Absolute Error $|\hat{T} - T^*|$")
    axes[2].set_xlabel(r"Time $t$ [s]")
    axes[2].set_ylabel(r"Blade Wall Coordinate $x/L$")
    cb2 = fig.colorbar(c2, ax=axes[2], fraction=0.046, pad=0.04, label=r"Pointwise Error")
    cb2.formatter.set_powerlimits((0, 0))
    cb2.update_ticks()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_loss_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots multi-objective scientific loss convergence across Adam and L-BFGS phases."""
    set_academic_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    steps = list(range(len(history["loss_total"])))

    ax.semilogy(steps, history["loss_total"], label=r"Total Loss $\mathcal{L}_{total}$", color="#1f77b4", linewidth=2.0)
    if "loss_pde" in history and any(history["loss_pde"]):
        ax.semilogy(steps, history["loss_pde"], label=r"PDE Residual $\mathcal{L}_{pde}$", color="#d62728", linestyle="--", alpha=0.85)
    if "loss_bc" in history and any(history["loss_bc"]):
        ax.semilogy(steps, history["loss_bc"], label=r"Boundary $\mathcal{L}_{bc}$", color="#2ca02c", linestyle="-.", alpha=0.85)
    if "loss_ic" in history and any(history["loss_ic"]):
        ax.semilogy(steps, history["loss_ic"], label=r"Initial Cond. $\mathcal{L}_{ic}$", color="#ff7f0e", linestyle=":", alpha=0.85)
    if "loss_data" in history and any(v > 0 for v in history["loss_data"]):
        ax.semilogy(steps, history["loss_data"], label=r"Sensor Data $\mathcal{L}_{data}$", color="#9467bd", linestyle="-", alpha=0.85)

    # Highlight transition from Adam to L-BFGS if present
    if "optimizer_phase" in history and 2 in history["optimizer_phase"]:
        lbfgs_start = history["optimizer_phase"].index(2)
        ax.axvline(lbfgs_start, color="black", linestyle="--", alpha=0.7, label="L-BFGS Hand-off")

    ax.set_xlabel(r"Optimization Iteration")
    ax.set_ylabel(r"Loss Magnitude (Log Scale)")
    ax.set_title("Multi-Objective PINN Loss Convergence (Adam + L-BFGS)")
    ax.legend(frameon=True, loc="upper right")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_cross_sections(
    x_vals: np.ndarray,
    times: List[float],
    T_true_slices: List[np.ndarray],
    T_pred_slices: List[np.ndarray],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots 1D temperature profiles across blade wall thickness at discrete time steps."""
    set_academic_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    palette = sns.color_palette("rocket", len(times))

    for i, t_val in enumerate(times):
        c = palette[i]
        ax.plot(x_vals, T_true_slices[i], label=rf"$t={t_val:.2f}$ (Exact)", color=c, linewidth=2.0)
        ax.plot(x_vals[::4], T_pred_slices[i][::4], "o", markersize=4.5, markerfacecolor="white",
                markeredgecolor=c, markeredgewidth=1.2, label=rf"$t={t_val:.2f}$ (PINN)")

    ax.set_xlabel(r"Normalized Blade Coordinate $x/L$")
    ax.set_ylabel(r"Temperature $T(x, t)$")
    ax.set_title("Spatial Thermal Cross-Sections Over Time")
    ax.legend(ncol=2, frameon=True, fontsize=8, loc="upper right")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_parameter_convergence(
    history: Dict[str, List[float]],
    true_alpha: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots inverse parameter discovery convergence and relative error percentage."""
    set_academic_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True, constrained_layout=True)

    steps = list(range(len(history["alpha_learned"])))
    learned = history["alpha_learned"]
    errors = history["alpha_error_pct"]

    # Top panel: Parameter trajectory
    ax1.plot(steps, learned, color="#1f77b4", linewidth=2.0, label=r"Discovered $\hat{\alpha}$")
    ax1.axhline(true_alpha, color="#d62728", linestyle="--", linewidth=1.8, label=rf"True $\alpha = {true_alpha:.4f}$")
    ax1.set_ylabel(r"Diffusivity $\alpha$ [$m^2/s$]")
    ax1.set_title("Inverse System Identification: Thermal Diffusivity Discovery")
    ax1.legend(frameon=True, loc="lower right")

    # Bottom panel: Estimation error
    ax2.semilogy(steps, errors, color="#2ca02c", linewidth=2.0)
    ax2.axhline(0.8, color="gray", linestyle=":", label="0.8% Target Threshold")
    ax2.set_xlabel(r"Optimization Iteration")
    ax2.set_ylabel(r"Relative Error [%]")
    ax2.set_title("Parameter Calibration Error (%)")
    ax2.legend(frameon=True, loc="upper right")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig
