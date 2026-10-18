"""
Physics-Informed Neural Networks (PINNs) Benchmark Suite - PyTorch 2.x
Module: utils/visualization.py

Publication-ready scientific plotting routines reproducing the benchmark figures
from Maziar Raissi et al. (2019, Journal of Computational Physics):
1. Iconic Raissi 2019 Figure 1: 2D spatiotemporal heatmap of u(x, t) with collocation
   points overlay, paired with cross-sectional slices at t = 0.25, 0.50, 0.75.
2. Tri-panel 2D comparison: Exact Solution, PINN Prediction, and Absolute Error Map.
3. Dual-phase (Adam + L-BFGS) loss convergence trajectory.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


def plot_raissi_burgers_figure(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    u_exact: np.ndarray,
    u_pred: np.ndarray,
    x_train: Optional[np.ndarray] = None,
    t_train: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    r"""
    Faithful reproduction of Figure 1 from Raissi et al. (2019, JCP):
    Top panel: 2D spatiotemporal heatmap of :math:`u(x, t)` with training points overlaid.
    Bottom panel: Three temporal cross-sections at :math:`t = 0.25, 0.50, 0.75` comparing
    Exact vs Modern PINN prediction.
    """
    set_academic_style()

    # Ensure 1D coordinate vectors
    x = x_grid.flatten()
    t = t_grid.flatten()
    # u shapes should be (len(x), len(t))
    if u_exact.shape != (len(x), len(t)):
        u_exact = u_exact.reshape(len(x), len(t))
    if u_pred.shape != (len(x), len(t)):
        u_pred = u_pred.reshape(len(x), len(t))

    T, X = np.meshgrid(t, x)

    fig = plt.figure(figsize=(10, 7.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1.0], hspace=0.38, wspace=0.32)

    # Top Panel: 2D Spatiotemporal Profile
    ax_top = fig.add_subplot(gs[0, :])
    contour = ax_top.pcolor(T, X, u_pred, cmap="rainbow", shading="auto")
    cbar = fig.colorbar(contour, ax=ax_top, fraction=0.046, pad=0.03)
    cbar.set_label(r"$u(t, x)$")

    # Overlay training points if provided
    if x_train is not None and t_train is not None:
        ax_top.plot(
            t_train.flatten(),
            x_train.flatten(),
            "kx",
            markersize=3.5,
            alpha=0.6,
            label=f"Collocation/BC points (N={len(x_train)})",
        )
        ax_top.legend(loc="upper right", framealpha=0.85)

    # Draw vertical dashed lines for cross-section snapshot times
    target_times = [0.25, 0.50, 0.75]
    for snap_t in target_times:
        ax_top.axvline(snap_t, color="white", linestyle="--", linewidth=1.2, alpha=0.8)

    ax_top.set_title(r"Burgers' Equation: Modern PINN Prediction $\hat{u}(t, x)$", fontsize=12)
    ax_top.set_xlabel(r"Time $t$")
    ax_top.set_ylabel(r"Spatial coordinate $x$")
    ax_top.set_xlim([t.min(), t.max()])
    ax_top.set_ylim([x.min(), x.max()])

    # Bottom Panel: Cross-sections at t = 0.25, 0.50, 0.75
    for i, snap_t in enumerate(target_times):
        ax_snap = fig.add_subplot(gs[1, i])
        t_idx = int(np.argmin(np.abs(t - snap_t)))
        actual_t = t[t_idx]

        ax_snap.plot(
            x,
            u_exact[:, t_idx],
            "b-",
            linewidth=2.0,
            label="Exact (Chebfun)",
        )
        ax_snap.plot(
            x,
            u_pred[:, t_idx],
            "r--",
            linewidth=2.0,
            label="Modern PINN",
        )

        ax_snap.set_title(f"$t = {actual_t:.2f}$")
        ax_snap.set_xlabel(r"$x$")
        ax_snap.set_ylabel(r"$u(t, x)$")
        ax_snap.set_ylim([-1.1, 1.1])
        ax_snap.legend(loc="upper right", fontsize=8)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"[FIGURE SAVED] Raissi benchmark figure -> {save_path}")

    return fig


def plot_solution_tripanel(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    u_exact: np.ndarray,
    u_pred: np.ndarray,
    save_path: Optional[str] = None,
    title_prefix: str = "Burgers' Equation",
) -> plt.Figure:
    """
    Plots a 3-panel 2D comparison:
    [Exact Benchmark] vs [Modern PINN Prediction] vs [Absolute Pointwise Error Map]
    """
    set_academic_style()

    x = x_grid.flatten()
    t = t_grid.flatten()
    if u_exact.shape != (len(x), len(t)):
        u_exact = u_exact.reshape(len(x), len(t))
    if u_pred.shape != (len(x), len(t)):
        u_pred = u_pred.reshape(len(x), len(t))

    error = np.abs(u_pred - u_exact)
    T, X = np.meshgrid(t, x)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)

    vmin = min(u_exact.min(), u_pred.min())
    vmax = max(u_exact.max(), u_pred.max())

    # 1. Exact
    c0 = axes[0].pcolor(T, X, u_exact, cmap="rainbow", shading="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"(a) Exact Benchmark $u^*(t, x)$")
    axes[0].set_xlabel(r"Time $t$")
    axes[0].set_ylabel(r"Coordinate $x$")
    fig.colorbar(c0, ax=axes[0], fraction=0.046, pad=0.04, label=r"$u$")

    # 2. Modern PINN
    c1 = axes[1].pcolor(T, X, u_pred, cmap="rainbow", shading="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"(b) Modern PINN $\\hat{{u}}(t, x)$")
    axes[1].set_xlabel(r"Time $t$")
    axes[1].set_ylabel(r"Coordinate $x$")
    fig.colorbar(c1, ax=axes[1], fraction=0.046, pad=0.04, label=r"$\hat{u}$")

    # 3. Absolute Error
    c2 = axes[2].pcolor(T, X, error, cmap="viridis", shading="auto")
    axes[2].set_title(f"(c) Absolute Error $|u^* - \\hat{{u}}|$")
    axes[2].set_xlabel(r"Time $t$")
    axes[2].set_ylabel(r"Coordinate $x$")
    fig.colorbar(c2, ax=axes[2], fraction=0.046, pad=0.04, label=r"Absolute Error")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"[FIGURE SAVED] Tri-panel comparison -> {save_path}")

    return fig


def plot_loss_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots training loss convergence across Adam and L-BFGS optimization phases.
    """
    set_academic_style()
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)

    if "loss" in history:
        ax.semilogy(history["loss"], label=r"Total Loss $\mathcal{L}_{\mathrm{total}}$", color="#1f77b4", lw=1.8)
    if "loss_u" in history:
        ax.semilogy(history["loss_u"], label=r"Boundary / Data Loss $\mathcal{L}_u$", color="#2ca02c", lw=1.4, ls="--")
    if "loss_f" in history:
        ax.semilogy(history["loss_f"], label=r"PDE Residual Loss $\mathcal{L}_f$", color="#d62728", lw=1.4, ls=":")

    # Mark Adam to L-BFGS phase transition if logged
    if "phase_switch_step" in history and history["phase_switch_step"] > 0:
        switch = history["phase_switch_step"]
        ax.axvline(switch, color="gray", linestyle="-.", alpha=0.8, label=f"Phase 2 (L-BFGS) Start @ {switch}")

    ax.set_title("Dual-Optimizer Convergence History (Adam + L-BFGS)")
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel(r"Loss (Logarithmic Scale)")
    ax.legend(loc="upper right", framealpha=0.9)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"[FIGURE SAVED] Loss history -> {save_path}")

    return fig
