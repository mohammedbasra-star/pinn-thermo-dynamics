"""
Physics-Informed Neural Networks (PINNs) Benchmark Suite - PyTorch 2.x
Module: models/compiled_solver.py

High-performance PINN solver supporting:
1. Pure PyTorch 2.x dynamic reverse-mode automatic differentiation.
2. Optional graph compilation via torch.compile(mode="reduce-overhead").
3. Dual-Optimizer Convergence Protocol (faithful to Raissi et al. 2019):
   - Phase 1: Global basin exploration using Adam optimizer.
   - Phase 2: Second-order quasi-Newton refinement using L-BFGS with strong Wolfe line search.
4. Comprehensive loss telemetry and checkpoint serialization.
"""

from typing import Dict, List, Optional, Tuple, Callable, Union
import os
import sys
import time
import torch
import torch.nn as nn

from utils.metrics import relative_l2_error


class PINNCompiledSolver:
    r"""
    Modern Physics-Informed Neural Network Solver for continuous-time differential equations.

    Solves the multi-objective optimization problem:
        .. math::
            \min_{\theta} \mathcal{L}_{\mathrm{total}}(\theta) =
            w_u \mathcal{L}_u(\theta) + w_f \mathcal{L}_f(\theta)

    where:
        .. math::
            \mathcal{L}_u = \frac{1}{N_u} \sum_{i=1}^{N_u} |\hat{u}(x_u^i, t_u^i) - u^i|^2
        .. math::
            \mathcal{L}_f = \frac{1}{N_f} \sum_{j=1}^{N_f} |f(x_f^j, t_f^j)|^2
    """

    def __init__(
        self,
        model: nn.Module,
        pde_module: nn.Module,
        compile_model: bool = False,
        compile_mode: str = "reduce-overhead",
        device: Union[str, torch.device] = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.pde = pde_module.to(self.device)
        self.compile_model = compile_model
        self.compile_mode = compile_mode

        # Initialize execution target (compiled or eager)
        self.forward_target = self.model
        if self.compile_model:
            self._apply_torch_compile()

        # Telemetry
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "loss_u": [],
            "loss_f": [],
            "iter": [],
        }
        self.best_loss = float("inf")
        self.best_state_dict = None
        self.current_iter = 0

    def _apply_torch_compile(self):
        """Attempts torch.compile with graceful fallback if Inductor backend is unavailable."""
        try:
            print(f"[COMPILATION] Attempting torch.compile(mode='{self.compile_mode}') on {self.device}...")
            self.forward_target = torch.compile(self.model, mode=self.compile_mode)
            print("  [SUCCESS] torch.compile enabled.")
        except Exception as exc:
            print(f"  [FALLBACK] torch.compile unavailable or unsupported on this platform: {exc}")
            print("  [FALLBACK] Reverting to high-performance PyTorch 2.x eager autograd.")
            self.forward_target = self.model
            self.compile_model = False

    def parameters(self):
        """Returns all trainable parameters (network weights + optional PDE parameters)."""
        params = list(self.model.parameters())
        if hasattr(self.pde, "parameters"):
            params += list(self.pde.parameters())
        return params

    def compute_loss(
        self,
        x_u: torch.Tensor,
        t_u: torch.Tensor,
        u_train: torch.Tensor,
        x_f: torch.Tensor,
        t_f: torch.Tensor,
        w_u: float = 1.0,
        w_f: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Computes total PINN loss = w_u * L_u + w_f * L_f.
        """
        # 1. Boundary & Initial condition loss
        coords_u = torch.cat([x_u, t_u], dim=-1)
        u_pred = self.forward_target(coords_u)
        loss_u = torch.mean((u_pred - u_train) ** 2)

        # 2. PDE interior residual loss
        _, f_pred = self.pde(self.forward_target, x_f, t_f)
        loss_f = torch.mean(f_pred ** 2)

        total_loss = w_u * loss_u + w_f * loss_f
        return total_loss, loss_u, loss_f

    def train_adam(
        self,
        data: Dict[str, torch.Tensor],
        epochs: int = 2000,
        lr: float = 1e-3,
        w_u: float = 1.0,
        w_f: float = 1.0,
        log_interval: int = 200,
    ):
        """
        Phase 1: Adam optimization for robust global parameter basin localization.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        x_u = data["x_u"].to(self.device)
        t_u = data["t_u"].to(self.device)
        u_train = data["u_train"].to(self.device)
        x_f = data["x_f"].to(self.device)
        t_f = data["t_f"].to(self.device)

        print("=" * 70)
        print(f"Phase 1: Adam Optimization ({epochs} iterations, lr={lr})")
        print("=" * 70)
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            loss, loss_u, loss_f = self.compute_loss(x_u, t_u, u_train, x_f, t_f, w_u, w_f)
            loss.backward()
            optimizer.step()

            self.current_iter += 1
            loss_val = float(loss.item())
            loss_u_val = float(loss_u.item())
            loss_f_val = float(loss_f.item())

            self.history["loss"].append(loss_val)
            self.history["loss_u"].append(loss_u_val)
            self.history["loss_f"].append(loss_f_val)
            self.history["iter"].append(self.current_iter)

            if loss_val < self.best_loss:
                self.best_loss = loss_val
                self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if epoch % log_interval == 0 or epoch == epochs or epoch == 1:
                elapsed = time.time() - start_time
                ms_per_step = (elapsed / epoch) * 1000.0
                print(
                    f"  [Adam {epoch:5d}/{epochs}] "
                    f"Loss: {loss_val:.4e} | "
                    f"L_u: {loss_u_val:.4e} | "
                    f"L_f: {loss_f_val:.4e} | "
                    f"Speed: {ms_per_step:.1f} ms/step"
                )

        total_time = time.time() - start_time
        print(f"Phase 1 finished in {total_time:.2f} s. Best Adam Loss: {self.best_loss:.4e}\n")

    def train_lbfgs(
        self,
        data: Dict[str, torch.Tensor],
        max_iter: int = 50000,
        max_eval: Optional[int] = None,
        lr: float = 1.0,
        history_size: int = 50,
        w_u: float = 1.0,
        w_f: float = 1.0,
        log_interval: int = 200,
    ):
        """
        Phase 2: L-BFGS second-order quasi-Newton refinement with strong Wolfe line search.
        Faithful to the 2019 JCP benchmark execution protocol.
        """
        if max_eval is None:
            max_eval = max_iter * 2

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            history_size=history_size,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-16,
            tolerance_change=1e-16,
        )

        x_u = data["x_u"].to(self.device)
        t_u = data["t_u"].to(self.device)
        u_train = data["u_train"].to(self.device)
        x_f = data["x_f"].to(self.device)
        t_f = data["t_f"].to(self.device)

        print("=" * 70)
        print(f"Phase 2: L-BFGS Quasi-Newton Optimization (max_iter={max_iter}, Wolfe line-search)")
        print("=" * 70)
        start_time = time.time()
        lbfgs_step = [0]

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss, loss_u, loss_f = self.compute_loss(x_u, t_u, u_train, x_f, t_f, w_u, w_f)
            loss.backward()

            lbfgs_step[0] += 1
            self.current_iter += 1

            loss_val = float(loss.item())
            loss_u_val = float(loss_u.item())
            loss_f_val = float(loss_f.item())

            self.history["loss"].append(loss_val)
            self.history["loss_u"].append(loss_u_val)
            self.history["loss_f"].append(loss_f_val)
            self.history["iter"].append(self.current_iter)

            if loss_val < self.best_loss:
                self.best_loss = loss_val
                self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if lbfgs_step[0] % log_interval == 0 or lbfgs_step[0] == 1:
                elapsed = time.time() - start_time
                print(
                    f"  [L-BFGS step {lbfgs_step[0]:5d}] "
                    f"Loss: {loss_val:.4e} | "
                    f"L_u: {loss_u_val:.4e} | "
                    f"L_f: {loss_f_val:.4e} | "
                    f"Elapsed: {elapsed:.1f} s"
                )

            return loss

        optimizer.step(closure)
        total_time = time.time() - start_time
        print(f"Phase 2 finished in {total_time:.2f} s ({lbfgs_step[0]} evaluations). Best Loss: {self.best_loss:.4e}\n")

    def fit_dual_phase(
        self,
        data: Dict[str, torch.Tensor],
        adam_epochs: int = 2000,
        adam_lr: float = 1e-3,
        lbfgs_max_iter: int = 15000,
        lbfgs_lr: float = 1.0,
        w_u: float = 1.0,
        w_f: float = 1.0,
        log_interval: int = 200,
    ):
        """Executes complete 2-Phase convergence pipeline (Adam -> L-BFGS)."""
        if adam_epochs > 0:
            self.train_adam(data, epochs=adam_epochs, lr=adam_lr, w_u=w_u, w_f=w_f, log_interval=log_interval)
        self.history["phase_switch_step"] = self.current_iter
        if lbfgs_max_iter > 0:
            self.train_lbfgs(data, max_iter=lbfgs_max_iter, lr=lbfgs_lr, w_u=w_u, w_f=w_f, log_interval=log_interval)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluates model prediction u(x, t) on arbitrary spatiotemporal coordinates."""
        self.model.eval()
        x_in = x.to(self.device).view(-1, 1)
        t_in = t.to(self.device).view(-1, 1)
        coords = torch.cat([x_in, t_in], dim=-1)
        u_pred = self.model(coords)
        return u_pred

    def evaluate_benchmark(
        self,
        x_star: torch.Tensor,
        t_star: torch.Tensor,
        u_star: torch.Tensor,
    ) -> float:
        """Computes Relative L2 error against exact benchmark solution."""
        u_pred = self.predict(x_star, t_star)
        rel_l2 = relative_l2_error(u_pred, u_star)
        return rel_l2

    def save_checkpoint(self, filepath: str):
        """Serializes model weights, optimizer history, and best state."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "best_state_dict": self.best_state_dict or self.model.state_dict(),
            "best_loss": self.best_loss,
            "history": self.history,
            "current_iter": self.current_iter,
        }
        if hasattr(self.pde, "state_dict"):
            checkpoint["pde_state_dict"] = self.pde.state_dict()
        torch.save(checkpoint, filepath)
        print(f"[CHECKPOINT SAVED] -> {filepath}")

    def load_checkpoint(self, filepath: str):
        """Loads serialized checkpoint into model and solver."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["best_state_dict"])
        if "pde_state_dict" in checkpoint and hasattr(self.pde, "load_state_dict"):
            self.pde.load_state_dict(checkpoint["pde_state_dict"])
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)
        self.current_iter = checkpoint.get("current_iter", 0)
        print(f"[CHECKPOINT LOADED] <- {filepath}")
