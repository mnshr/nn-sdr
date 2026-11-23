"""Simulation reproduction of FlowSDR and Neural Gaussian SDR on Model I.

Follows Section 4.2 (Model I, Table 1) of Dong et al. (2026):
    X_1, ..., X_p ~ i.i.d. N(2, 1), eps ~ N(0, 1) independent of X,
    Y = 2 sin((X_1 + X_2) * pi / 20) + 3 sin^2((X_3 + 3 X_4) * pi / 30) * eps.
True central subspace basis spans beta_1 = (1, 1, 0, ...) and beta_2 = (0, 0, 1, 3, 0, ...).
"""

from __future__ import annotations

import argparse
import numpy as np

from flowsdr import FlowSDR, NeuralGaussianSDR, fit_stiefel_sdr, subspace_distances


def generate_model_1(n: int, p: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate covariates, response, and orthonormal true basis B0 for Model I."""
    if p < 4:
        raise ValueError(f"Model I requires p >= 4, got p={p}.")
    X = rng.normal(loc=2.0, scale=1.0, size=(n, p))
    eps = rng.normal(loc=0.0, scale=1.0, size=n)
    y = 2.0 * np.sin((X[:, 0] + X[:, 1]) * np.pi / 20.0) + 3.0 * (
        np.sin((X[:, 2] + 3.0 * X[:, 3]) * np.pi / 30.0) ** 2
    ) * eps

    b1 = np.zeros(p)
    b1[0], b1[1] = 1.0, 1.0
    b2 = np.zeros(p)
    b2[2], b2[3] = 1.0, 3.0
    B0, _ = np.linalg.qr(np.column_stack([b1, b2]), mode="reduced")
    return X, y, B0


def run_benchmark(n: int, p_list: list[int], reps: int, n_epochs: int, batch_size: int, n_restarts: int) -> None:
    print(f"FlowSDR Model I Benchmark (Dong et al., 2026, Sec. 4.2): n={n}, d=2, reps={reps}, epochs={n_epochs}")
    print("-" * 76)
    print(f"{'p':>3} | {'Method':<20} | {'Delta_op (mean +/- SD)':<24} | {'Delta_F (mean +/- SD)':<24}")
    print("-" * 76)

    for p in p_list:
        for method_name, cls in [("Neural Gaussian SDR", NeuralGaussianSDR), ("FlowSDR", FlowSDR)]:
            op_errs, f_errs = [], []
            for rep in range(reps):
                rng = np.random.default_rng(202600 + 100 * p + rep)
                X, y, B0 = generate_model_1(n=n, p=p, rng=rng)
                res = fit_stiefel_sdr(
                    cls,
                    X,
                    y,
                    d=2,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    lr_B=1e-2,
                    lr_net=1e-3,
                    n_restarts=n_restarts,
                    seed=1000 * p + rep,
                )
                d_op, d_f = subspace_distances(res["B_hat"], B0)
                op_errs.append(d_op)
                f_errs.append(d_f)

            op_str = f"{np.mean(op_errs):.4f} +/- {np.std(op_errs):.4f}"
            f_str = f"{np.mean(f_errs):.4f} +/- {np.std(f_errs):.4f}"
            print(f"{p:>3} | {method_name:<20} | {op_str:<24} | {f_str:<24}")
    print("-" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate FlowSDR Model I (Dong et al., 2026).")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p-list", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--restarts", type=int, default=1)
    args = parser.parse_args()
    run_benchmark(
        n=args.n,
        p_list=args.p_list,
        reps=args.reps,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        n_restarts=args.restarts,
    )


if __name__ == "__main__":
    main()
