"""
Physics-Informed Neural Networks for High-Temperature Turbomachinery & Thermodynamic Dynamics
Script: evaluate.py

Evaluates trained PINN checkpoints, computes full error metrics table,
and reproduces publication-quality benchmark plots.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Ensure UTF-8 output on all operating systems
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from physics.heat_pde import HeatConductionPDE
from models.networks import build_network
from utils.metrics import compute_metrics
from utils.visualization import (
    plot_solution_tripanel,
    plot_cross_sections,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PINN model checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation figures")
    parser.add_argument("--nx", type=int, default=150, help="Evaluation spatial grid resolution")
    parser.add_argument("--nt", type=int, default=150, help="Evaluation temporal grid resolution")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint.get("args", {})

    arch = train_args.get("arch", "modified_mlp")
    hidden_dim = train_args.get("hidden_dim", 64)
    layers = train_args.get("layers", 4)
    activation = train_args.get("activation", "tanh")
    use_fourier = train_args.get("fourier_emb", False)
    fourier_scale = train_args.get("fourier_scale", 2.0)
    fourier_dim = train_args.get("fourier_dim", 32)
    alpha = train_args.get("alpha", 0.05)
    decay_lambda = train_args.get("decay_lambda", 0.5)

    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    # Reconstruct model
    model = build_network(
        arch_type=arch,
        in_dim=2,
        out_dim=1,
        hidden_dim=hidden_dim,
        num_layers=layers,
        activation=activation,
        use_fourier=use_fourier,
        fourier_scale=fourier_scale,
        fourier_dim=fourier_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Ground truth
    pde = HeatConductionPDE(alpha_true=alpha, is_inverse=False, decay_lambda=decay_lambda)

    # Evaluation Grid
    x = np.linspace(0.0, 1.0, args.nx)
    t = np.linspace(0.0, 1.0, args.nt)
    X, T_grid = np.meshgrid(x, t)

    with torch.no_grad():
        coords = torch.tensor(np.column_stack([X.ravel(), T_grid.ravel()]), dtype=torch.float32)
        T_pred = model(coords).numpy().reshape(X.shape)

    T_true = pde.exact_temperature(X, T_grid)

    alpha_learned = checkpoint.get("alpha_learned", None)
    metrics = compute_metrics(
        pred=T_pred,
        true=T_true,
        p_learned=alpha_learned if train_args.get("mode") == "inverse" else None,
        p_true=alpha if train_args.get("mode") == "inverse" else None,
    )

    print("=" * 65)
    print("  CHECKPOINT EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Checkpoint File        : {args.checkpoint}")
    print(f"  Trained Mode           : {train_args.get('mode', 'N/A').upper()}")
    print(f"  Architecture           : {arch} (Hidden: {hidden_dim}, Layers: {layers})")
    print(f"  Relative L2 Error      : {metrics['relative_l2_error']:.3e}")
    print(f"  RMSE                   : {metrics['rmse']:.3e}")
    print(f"  Mean Absolute Error    : {metrics['mae']:.3e}")
    print(f"  Max Pointwise Error    : {metrics['max_error_linf']:.3e}")
    if "param_error_pct" in metrics:
        print(f"  Learned Diffusivity    : {metrics['param_learned']:.5f} (True: {metrics['param_true']:.5f})")
        print(f"  Parameter Error        : {metrics['param_error_pct']:.3f} %")
    print("=" * 65)

    # Export Evaluation Plots
    tripanel_path = os.path.join(output_dir, "eval_tripanel.png")
    plot_solution_tripanel(X, T_grid, T_true, T_pred, save_path=tripanel_path)

    snapshots_path = os.path.join(output_dir, "eval_snapshots.png")
    eval_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    time_indices = [int(t_val * (args.nt - 1)) for t_val in eval_times]
    plot_cross_sections(
        x_vals=x,
        times=eval_times,
        T_true_slices=[T_true[i, :] for i in time_indices],
        T_pred_slices=[T_pred[i, :] for i in time_indices],
        save_path=snapshots_path,
    )
    print(f"[Done] Generated evaluation figures in {output_dir}")


if __name__ == "__main__":
    main()
