"""PyTorch reference implementation of GenSDR via Conditional Stochastic Interpolation.

Reference:
    Xu, S., Yu, Z., & Huang, J. (2025).
    On Conditional Stochastic Interpolation for Generative Nonlinear Sufficient Dimension Reduction.
    arXiv preprint arXiv:2512.18971.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _make_mlp(in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int, act: str = "relu") -> nn.Sequential:
    act_cls = nn.ReLU if act == "relu" else nn.SiLU
    layers: list[nn.Module] = []
    cur = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(cur, h))
        layers.append(act_cls())
        cur = h
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


class GenSDR(nn.Module):
    """Joint representation-velocity learner for Conditional Stochastic Interpolation.

    Architecture matches Section 7.1 of Xu et al. (2025):
        R_phi: dx -> 64 -> 256 -> 128 -> d
        g_theta: (d + dy + 1) -> 64 -> 256 -> 128 -> dy
    """

    def __init__(self, dx: int, dy: int, d: int, act: str = "relu") -> None:
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.d = d
        self.rep_net = _make_mlp(dx, (64, 256, 128), d, act=act)
        self.vel_net = _make_mlp(d + dy + 1, (64, 256, 128), dy, act=act)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.rep_net(x)

    def velocity(self, z: torch.Tensor, y_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, y_t, t], dim=-1)
        return self.vel_net(inp)

    def csi_loss(self, x: torch.Tensor, y: torch.Tensor, tau_n: float = 0.001) -> torch.Tensor:
        """Compute empirical velocity-matching loss under the straight-line interpolant."""
        bsz = x.shape[0]
        eta = torch.randn_like(y)
        t = (1.0 - tau_n) * torch.rand(bsz, 1, device=x.device, dtype=x.dtype)
        y_t = (1.0 - t) * eta + t * y
        target_vel = y - eta

        z = self.encode(x)
        pred_vel = self.velocity(z, y_t, t)
        return ((pred_vel - target_vel) ** 2).mean()

    @torch.no_grad()
    def sample_ode(self, x: torch.Tensor, n_steps: int = 100, tau_n: float = 0.001) -> torch.Tensor:
        """Generate conditional samples Y | X=x via forward Euler ODE integration on [0, 1 - tau_n]."""
        self.eval()
        z = self.encode(x)
        y_t = torch.randn(x.shape[0], self.dy, device=x.device, dtype=x.dtype)
        dt = (1.0 - tau_n) / float(n_steps)
        for step in range(n_steps):
            t_val = torch.full((x.shape[0], 1), step * dt, device=x.device, dtype=x.dtype)
            y_t = y_t + dt * self.velocity(z, y_t, t_val)
        return y_t


def fit_gensdr(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    d: int,
    tau_n: float = 0.001,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    act: str = "relu",
    seed: int = 0,
    device: str = "cpu",
) -> Dict[str, object]:
    """Train GenSDR with Adam on standardized predictors and responses."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = np.where(X_train.std(axis=0, keepdims=True) > 1e-8, X_train.std(axis=0, keepdims=True), 1.0)
    y_mean = Y_train.mean(axis=0, keepdims=True)
    y_std = np.where(Y_train.std(axis=0, keepdims=True) > 1e-8, Y_train.std(axis=0, keepdims=True), 1.0)

    x_tr = torch.tensor((X_train - x_mean) / x_std, dtype=torch.float32, device=dev)
    y_tr = torch.tensor((Y_train - y_mean) / y_std, dtype=torch.float32, device=dev)

    model = GenSDR(dx=X_train.shape[1], dy=Y_train.shape[1], d=d, act=act).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)

    last_loss = 0.0
    model.train()
    for _ in range(n_epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = model.csi_loss(xb, yb, tau_n=tau_n)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * xb.shape[0]
        last_loss = total / len(X_train)

    return {
        "model": model,
        "train_loss": last_loss,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def predict_representation(fit_res: Dict[str, object], X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Evaluate learned representation R_hat(X) on arbitrary covariate arrays."""
    model: GenSDR = fit_res["model"]  # type: ignore[assignment]
    x_norm = (X - fit_res["x_mean"]) / fit_res["x_std"]  # type: ignore[operator]
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x_norm, dtype=torch.float32, device=torch.device(device))
        return model.encode(xt).cpu().numpy()
