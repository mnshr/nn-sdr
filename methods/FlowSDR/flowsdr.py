"""PyTorch reference implementation of FlowSDR and Neural Gaussian SDR.

Reference:
    Dong, Y., McAlinn, K., Airoldi, E., & Li, L. (2026).
    FlowSDR: Sufficient Dimension Reduction via Conditional Normalizing Flows.
    arXiv preprint arXiv:2606.01346.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def subspace_distances(
    B_hat: np.ndarray | torch.Tensor,
    B0: np.ndarray | torch.Tensor,
) -> Tuple[float, float]:
    """Compute operator and Frobenius projection distances on the Grassmannian.

    Uses Eq. (Section 4.2) of Dong et al. (2026):
        Delta_op(B_hat) = ||P_hat - P_0||_op,
        Delta_F(B_hat)  = ||P_hat - P_0||_F,
    where P = B B^T for orthonormal matrices B in V_{p,d}.
    """
    Bh = B_hat.detach().cpu().numpy() if isinstance(B_hat, torch.Tensor) else np.asarray(B_hat)
    B_true = B0.detach().cpu().numpy() if isinstance(B0, torch.Tensor) else np.asarray(B0)

    Qh, _ = np.linalg.qr(Bh, mode="reduced")
    Q0, _ = np.linalg.qr(B_true, mode="reduced")
    diff = Qh @ Qh.T - Q0 @ Q0.T
    delta_op = float(np.linalg.norm(diff, ord=2))
    delta_F = float(np.linalg.norm(diff, ord="fro"))
    return delta_op, delta_F


def _retract_stiefel_(B: torch.nn.Parameter) -> None:
    """Project B onto the Stiefel manifold V_{p,d} via reduced QR retraction."""
    with torch.no_grad():
        q, _ = torch.linalg.qr(B.data, mode="reduced")
        B.data.copy_(q)


def _rqs_1d(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    tail_bound: float = 5.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a 1D monotone rational-quadratic spline (Durkan et al., 2019).

    Args:
        inputs: 1D tensor of scalar targets with shape (batch,).
        unnormalized_widths: Shape (batch, K).
        unnormalized_heights: Shape (batch, K).
        unnormalized_derivatives: Interior knot derivatives of shape (batch, K - 1).
        tail_bound: Linear tail threshold [-tail_bound, tail_bound].

    Returns:
        Tuple (outputs, logabsdet), each with shape (batch,).
    """
    num_bins = unnormalized_widths.shape[-1]
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = 2.0 * tail_bound * cumwidths - tail_bound
    cumwidths[..., 0] = -tail_bound
    cumwidths[..., -1] = tail_bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = 2.0 * tail_bound * cumheights - tail_bound
    cumheights[..., 0] = -tail_bound
    cumheights[..., -1] = tail_bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # Bias softplus so raw zeros evaluate close to unit slope (identity init).
    deriv_shift = math.log(math.e - 1.0)
    interior_derivs = min_derivative + F.softplus(unnormalized_derivatives + deriv_shift)
    ones = torch.ones(inputs.shape[0], 1, device=inputs.device, dtype=inputs.dtype)
    derivatives = torch.cat([ones, interior_derivs, ones], dim=-1)

    inside_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outputs = inputs.clone()
    logabsdet = torch.zeros_like(inputs)

    if not torch.any(inside_mask):
        return outputs, logabsdet

    x_in = inputs[inside_mask]
    cw = cumwidths[inside_mask]
    ch = cumheights[inside_mask]
    w_in = widths[inside_mask]
    h_in = heights[inside_mask]
    d_in = derivatives[inside_mask]

    # Rightmost interior knot edges are excluding cw[:, -1], matching searchsorted bins [0, K-1].
    bin_idx = torch.searchsorted(cw[..., 1:-1].contiguous(), x_in.unsqueeze(-1), right=True).squeeze(-1)
    bin_idx = bin_idx.clamp(0, num_bins - 1)
    gather_idx = bin_idx.unsqueeze(-1)

    xk = cw.gather(-1, gather_idx).squeeze(-1)
    yk = ch.gather(-1, gather_idx).squeeze(-1)
    wk = w_in.gather(-1, gather_idx).squeeze(-1)
    hk = h_in.gather(-1, gather_idx).squeeze(-1)
    sk = d_in.gather(-1, gather_idx).squeeze(-1)
    sk1 = d_in.gather(-1, gather_idx + 1).squeeze(-1)

    rk = hk / wk
    xi = ((x_in - xk) / wk).clamp(0.0, 1.0)
    xi_one_minus_xi = xi * (1.0 - xi)

    denom = rk + (sk1 + sk - 2.0 * rk) * xi_one_minus_xi
    num = rk * (xi**2) + sk * xi_one_minus_xi
    outputs_in = yk + hk * (num / denom)

    deriv_num = (rk**2) * (sk1 * (xi**2) + 2.0 * rk * xi_one_minus_xi + sk * ((1.0 - xi) ** 2))
    logabsdet_in = torch.log(deriv_num.clamp_min(1e-12)) - 2.0 * torch.log(denom.clamp_min(1e-12))

    outputs[inside_mask] = outputs_in
    logabsdet[inside_mask] = logabsdet_in
    return outputs, logabsdet


class _BaseStiefelSDR(nn.Module):
    """Shared Stiefel projection and conditioner backbone for likelihood-based SDR."""

    def __init__(self, p: int, d: int, hidden_dim: int = 64, out_dim: int = 2) -> None:
        super().__init__()
        init_b = torch.randn(p, d)
        q, _ = torch.linalg.qr(init_b, mode="reduced")
        self.B = nn.Parameter(q)
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.B


class NeuralGaussianSDR(_BaseStiefelSDR):
    """Heteroscedastic Gaussian likelihood SDR (Section 3 of Dong et al., 2026)."""

    def __init__(self, p: int, d: int, hidden_dim: int = 64) -> None:
        super().__init__(p=p, d=d, hidden_dim=hidden_dim, out_dim=2)

    def nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w = self.project(x)
        params = self.net(w)
        mu = params[:, 0]
        log_sigma = params[:, 1].clamp(-6.0, 6.0)
        sigma = torch.exp(log_sigma)
        return (0.5 * math.log(2.0 * math.pi) + log_sigma + 0.5 * ((y - mu) / sigma) ** 2).mean()


class FlowSDR(_BaseStiefelSDR):
    """Conditional rational-quadratic spline normalizing flow SDR (Section 2 of Dong et al., 2026)."""

    def __init__(
        self,
        p: int,
        d: int,
        hidden_dim: int = 64,
        num_bins: int = 16,
        tail_bound: float = 5.0,
    ) -> None:
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        # Outputs K widths + K heights + (K - 1) interior derivatives per conditioning vector W.
        super().__init__(p=p, d=d, hidden_dim=hidden_dim, out_dim=3 * num_bins - 1)

    def nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w = self.project(x)
        raw = self.net(w)
        k = self.num_bins
        uw = raw[:, :k]
        uh = raw[:, k : 2 * k]
        ud = raw[:, 2 * k :]
        z, logabsdet = _rqs_1d(y, uw, uh, ud, tail_bound=self.tail_bound)
        log_f0 = -0.5 * math.log(2.0 * math.pi) - 0.5 * (z**2)
        return -(log_f0 + logabsdet).mean()


def fit_stiefel_sdr(
    model_cls: type[_BaseStiefelSDR],
    X: np.ndarray,
    y: np.ndarray,
    d: int,
    n_epochs: int = 400,
    batch_size: int = 256,
    lr_B: float = 1e-2,
    lr_net: float = 1e-3,
    n_restarts: int = 1,
    seed: int = 0,
    device: str = "cpu",
    **model_kwargs,
) -> Dict[str, object]:
    """Fit NeuralGaussianSDR or FlowSDR using split Adam learning rates and QR retraction.

    Centers X (preserving subspace invariance of span(B_0)) and standardizes scalar Y
    so spline knots and Gaussian scales operate on a well-conditioned numerical range.
    """
    dev = torch.device(device)
    x_centered = X - X.mean(axis=0, keepdims=True)
    y_mean = float(y.mean())
    y_std = float(y.std()) if float(y.std()) > 1e-8 else 1.0
    y_scaled = (y - y_mean) / y_std

    if model_cls is FlowSDR and "tail_bound" not in model_kwargs:
        model_kwargs["tail_bound"] = float(max(5.0, np.max(np.abs(y_scaled)) + 0.5))

    x_tensor = torch.tensor(x_centered, dtype=torch.float32, device=dev)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=dev)
    dataset = TensorDataset(x_tensor, y_tensor)

    best_loss = float("inf")
    best_B = None
    best_model = None

    for restart in range(n_restarts):
        torch.manual_seed(seed + 1000 * restart)
        model = model_cls(p=X.shape[1], d=d, **model_kwargs).to(dev)
        optimizer = torch.optim.Adam(
            [
                {"params": [model.B], "lr": lr_B},
                {"params": model.net.parameters(), "lr": lr_net},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = model.nll(xb, yb)
                loss.backward()
                optimizer.step()
                _retract_stiefel_(model.B)
            scheduler.step()

        model.eval()
        with torch.no_grad():
            full_loss = float(model.nll(x_tensor, y_tensor).item())

        if full_loss < best_loss:
            best_loss = full_loss
            best_B = model.B.detach().cpu().numpy().copy()
            best_model = model

    return {"B_hat": best_B, "train_nll": best_loss, "model": best_model}
