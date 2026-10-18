"""
Physics-Informed Neural Networks for High-Temperature Turbomachinery & Thermodynamic Dynamics
Script: train.py

Production-grade training CLI supporting:
1. Dual Modes: Forward aerothermal simulation & Inverse parameter identification.
2. Dual-Stage Optimization: Global Adam optimization followed by precision L-BFGS refinement.
3. Neural Architecture: Modified MLP (gated residual connections) and Standard MLP (Xavier / Tanh / Sin).
4. Full validation against 256x256 analytical and numerical ground truth.
5. Automated generation of publication figures.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Ensure UTF-8 output on all operating systems
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from physics.heat_pde import HeatConductionPDE
from physics.boundary_conditions import BoundaryConditionEvaluator
from models.networks import build_network
from models.pinn_solver import PINNSolver
from utils.metrics import compute_metrics
from utils.visualization import (
    plot_solution_tripanel,
    plot_loss_history,
    plot_cross_sections,
    plot_parameter_convergence,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PINN Solver for High-Temperature Turbomachinery Dynamics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Problem Formulation
    parser.add_argument("--mode", type=str, choices=["forward", "inverse"], default="forward",
                        help="Operating mode: 'forward' (PDE solution) or 'inverse' (diffusivity identification)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="True ground-truth thermal diffusivity alpha")
    parser.add_argument("--initial-alpha", type=float, default=0.01,
                        help="Initial guess for diffusivity in inverse mode")
    parser.add_argument("--decay-lambda", type=float, default=0.5,
                        help="Transient aerothermal decay rate lambda")

    # Neural Architecture
    parser.add_argument("--arch", type=str, choices=["modified_mlp", "standard_mlp"], default="modified_mlp",
                        help="Network architecture family")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Neurons per hidden layer")
    parser.add_argument("--layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "sin", "gelu", "silu"],
                        help="Activation function")
    parser.add_argument("--fourier-emb", action="store_true", help="Enable Random Fourier Feature coordinate embedding")

    # Dual-Stage Optimization
    parser.add_argument("--epochs-adam", type=int, default=2000, help="Phase 1: Adam optimization epochs")
    parser.add_argument("--epochs-lbfgs", type=int, default=100, help="Phase 2: L-BFGS optimization steps")
    parser.add_argument("--use-lbfgs", action="store_true", default=True, help="Enable Phase 2 L-BFGS refinement")
    parser.add_argument("--no-lbfgs", dest="use_lbfgs", action="store_false", help="Disable L-BFGS phase")
    parser.add_argument("--lr-adam", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--lr-lbfgs", type=float, default=1.0, help="L-BFGS learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=300,
                        help="Warmup epochs before unfreezing parameter in inverse mode")

    # Sampling & Noise
    parser.add_argument("--n-pde", type=int, default=2500, help="Interior collocation points per epoch")
    parser.add_argument("--n-bc", type=int, default=128, help="Boundary condition points")
    parser.add_argument("--n-ic", type=int, default=128, help="Initial condition points")
    parser.add_argument("--n-sensors", type=int, default=150, help="Sparse sensor observation points (inverse mode)")
    parser.add_argument("--sensor-noise", type=float, default=0.01, help="Gaussian noise std on sensor data")

    # Telemetry & Output
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random reproducibility seed")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu', or auto)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Hardware device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode_str = f"{args.mode}_{args.arch}"
    if args.save_dir is None:
        save_dir = os.path.join("results", f"{mode_str}_{timestamp}")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    fig_dir = os.path.join(save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 80)
    print("  SCIENTIFIC MACHINE LEARNING (SciML) - PINN ENGINE")
    print("  High-Temperature Turbomachinery & Thermodynamic Dynamics")
    print("=" * 80)
    print(f"  * Mode                  : {args.mode.upper()}")
    print(f"  * Architecture          : {args.arch} ({args.hidden_dim} width, {args.layers} layers)")
    print(f"  * Activation            : {args.activation}")
    print(f"  * True Diffusivity (a)  : {args.alpha}")
    if args.mode == "inverse":
        print(f"  * Initial Guess (a_0)   : {args.initial_alpha}")
        print(f"  * Sensor Noise Std (s)  : {args.sensor_noise}")
        print(f"  * Warmup Epochs         : {args.warmup_epochs}")
    print(f"  * Phase 1 (Adam) Epochs : {args.epochs_adam} (lr={args.lr_adam})")
    print(f"  * Phase 2 (L-BFGS) Steps: {args.epochs_lbfgs if args.use_lbfgs else 0} (lr={args.lr_lbfgs})")
    print(f"  * Hardware Target       : {device}")
    print(f"  * Output Directory      : {save_dir}")
    print("=" * 80)

    # 1. Physics & Boundary Condition Evaluator
    is_inverse = (args.mode == "inverse")
    pde = HeatConductionPDE(
        alpha_true=args.alpha,
        is_inverse=is_inverse,
        initial_guess_alpha=args.initial_alpha,
        decay_lambda=args.decay_lambda,
    ).to(device)

    bc_evaluator = BoundaryConditionEvaluator(pde=pde, device=device, seed=args.seed)

    # 2. Build Neural Network Architecture
    model = build_network(
        arch_type=args.arch,
        in_dim=2,
        out_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        activation=args.activation,
        use_fourier=args.fourier_emb,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Compiled] Trainable network parameters: {total_params:,}")

    # 3. Instantiate PINN Solver
    solver = PINNSolver(
        model=model,
        pde=pde,
        bc_evaluator=bc_evaluator,
        learning_rate_adam=args.lr_adam,
        learning_rate_lbfgs=args.lr_lbfgs,
        warmup_epochs=args.warmup_epochs if is_inverse else 0,
        device=device,
    )

    # 4. Phase 1: Adam Global Optimization
    print(f"\n[Phase 1] Launching Adam Global Convergence ({args.epochs_adam} epochs)...")
    start_time = time.time()
    pbar_adam = tqdm(range(args.epochs_adam), desc="Phase 1 (Adam)", ncols=100)

    best_rel_l2 = float("inf")
    for epoch in pbar_adam:
        metrics = solver.train_step_adam(
            epoch=epoch,
            n_pde=args.n_pde,
            n_ic=args.n_ic,
            n_bc=args.n_bc,
            n_sensors=args.n_sensors,
            sensor_noise=args.sensor_noise,
        )

        if epoch % 200 == 0 or epoch == args.epochs_adam - 1:
            rel_l2 = solver.evaluate_relative_l2(nx=128, nt=128)
            solver.history["relative_l2"].append(rel_l2)
            if rel_l2 < best_rel_l2:
                best_rel_l2 = rel_l2
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "alpha_learned": metrics["alpha_learned"],
                        "best_rel_l2": best_rel_l2,
                        "args": vars(args),
                    },
                    os.path.join(save_dir, "best_model.pt"),
                )

            postfix = {
                "Loss": f"{metrics['loss_total']:.2e}",
                "L2": f"{rel_l2:.2e}",
            }
            if is_inverse:
                postfix["Alpha"] = f"{metrics['alpha_learned']:.4f}"
                postfix["Err%"] = f"{metrics['alpha_error_pct']:.2f}%"
            pbar_adam.set_postfix(postfix)

    adam_time = time.time() - start_time
    print(f"[Phase 1 Finished] Adam elapsed time: {adam_time:.2f}s | Current L2: {solver.evaluate_relative_l2():.3e}")

    # 5. Phase 2: L-BFGS Precision Optimization (Optional)
    if args.use_lbfgs and args.epochs_lbfgs > 0:
        print(f"\n[Phase 2] Launching L-BFGS Precision Refinement ({args.epochs_lbfgs} steps)...")
        opt_lbfgs = solver._init_lbfgs_optimizer()
        pbar_lbfgs = tqdm(range(args.epochs_lbfgs), desc="Phase 2 (L-BFGS)", ncols=100)

        for step in pbar_lbfgs:
            global_step = args.epochs_adam + step
            metrics_lbfgs = solver.train_step_lbfgs(
                optimizer_lbfgs=opt_lbfgs,
                step_idx=global_step,
                n_pde=args.n_pde,
                n_ic=args.n_ic,
                n_bc=args.n_bc,
                n_sensors=args.n_sensors,
                sensor_noise=args.sensor_noise,
            )

            if step % 25 == 0 or step == args.epochs_lbfgs - 1:
                rel_l2 = solver.evaluate_relative_l2(nx=128, nt=128)
                solver.history["relative_l2"].append(rel_l2)
                if rel_l2 < best_rel_l2:
                    best_rel_l2 = rel_l2
                    torch.save(
                        {
                            "epoch": global_step,
                            "model_state_dict": model.state_dict(),
                            "alpha_learned": metrics_lbfgs["alpha_learned"],
                            "best_rel_l2": best_rel_l2,
                            "args": vars(args),
                        },
                        os.path.join(save_dir, "best_model.pt"),
                    )

                postfix = {
                    "Loss": f"{metrics_lbfgs['loss_total']:.2e}",
                    "L2": f"{rel_l2:.2e}",
                }
                if is_inverse:
                    postfix["Alpha"] = f"{metrics_lbfgs['alpha_learned']:.4f}"
                    postfix["Err%"] = f"{metrics_lbfgs['alpha_error_pct']:.2f}%"
                pbar_lbfgs.set_postfix(postfix)

    total_time = time.time() - start_time
    print(f"\n[All Training Phases Completed] Total wall-clock time: {total_time:.2f} seconds.")

    # 6. Dense 256x256 Spatiotemporal Evaluation
    print("\n[Evaluation] Conducting high-density 256x256 benchmark verification...")
    nx_eval, nt_eval = 256, 256
    x_test = np.linspace(0.0, 1.0, nx_eval)
    t_test = np.linspace(0.0, 1.0, nt_eval)
    X_mesh, T_mesh = np.meshgrid(x_test, t_test)

    model.eval()
    with torch.no_grad():
        coords_eval = torch.tensor(
            np.column_stack([X_mesh.ravel(), T_mesh.ravel()]),
            dtype=torch.float32,
            device=device,
        )
        T_pred = model(coords_eval).cpu().numpy().reshape(X_mesh.shape)

    T_true = pde.exact_temperature(X_mesh, T_mesh)
    alpha_final = float(pde.alpha.item())

    final_metrics = compute_metrics(
        pred=T_pred,
        true=T_true,
        p_learned=alpha_final if is_inverse else None,
        p_true=args.alpha if is_inverse else None,
    )
    final_metrics["wall_clock_time"] = total_time
    final_metrics["best_relative_l2"] = best_rel_l2

    # Save metrics JSON
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)

    # Save telemetry history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(solver.history, f, indent=2)

    # 7. Generate Publication Figures
    print("[Visualization] Rendering 300 DPI publication figures...")
    # Tri-panel solution heatmap
    tripanel_path = os.path.join(fig_dir, "tripanel_solution_comparison.png")
    plot_solution_tripanel(
        X=X_mesh,
        T_grid=T_mesh,
        T_true=T_true,
        T_pred=T_pred,
        save_path=tripanel_path,
        title_suffix=f"({args.arch})",
    )

    # Loss convergence history
    loss_path = os.path.join(fig_dir, "loss_history.png")
    plot_loss_history(solver.history, save_path=loss_path)

    # Cross-sectional profiles
    cross_path = os.path.join(fig_dir, "cross_sections.png")
    eval_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    time_indices = [int(t_val * (nt_eval - 1)) for t_val in eval_times]
    plot_cross_sections(
        x_vals=x_test,
        times=eval_times,
        T_true_slices=[T_true[i, :] for i in time_indices],
        T_pred_slices=[T_pred[i, :] for i in time_indices],
        save_path=cross_path,
    )

    # Inverse parameter convergence (if applicable)
    if is_inverse:
        param_path = os.path.join(fig_dir, "parameter_convergence.png")
        plot_parameter_convergence(solver.history, true_alpha=args.alpha, save_path=param_path)

    # Copy primary figures to top-level figures/ directory
    os.makedirs("figures", exist_ok=True)
    import shutil
    shutil.copy(tripanel_path, "figures/tripanel_solution_comparison.png")
    shutil.copy(loss_path, "figures/loss_history.png")
    shutil.copy(cross_path, "figures/cross_sections.png")
    if is_inverse:
        shutil.copy(param_path, "figures/parameter_convergence.png")

    # Final Academic Summary Report
    print("\n" + "=" * 80)
    print("  SCIENTIFIC VERIFICATION & BENCHMARK SUMMARY (256x256 Grid)")
    print("=" * 80)
    print(f"  * Relative L2 Error (eps_L2)   : {final_metrics['relative_l2_error']:.3e}")
    print(f"  * Mean Squared Error (MSE)     : {final_metrics['mean_squared_error']:.3e}")
    print(f"  * Root Mean Square Error (RMSE): {final_metrics['rmse']:.3e}")
    print(f"  * Mean Absolute Error (MAE)    : {final_metrics['mae']:.3e}")
    print(f"  * Max Pointwise Error (L_inf)  : {final_metrics['max_error_linf']:.3e}")
    if is_inverse:
        print(f"  * True Parameter (alpha)       : {args.alpha:.5f}")
        print(f"  * Discovered Parameter (alpha) : {alpha_final:.5f}")
        print(f"  * Parameter Estimation Error   : {final_metrics['param_error_pct']:.3f} %")
    print(f"  * Figures successfully saved to: {fig_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
