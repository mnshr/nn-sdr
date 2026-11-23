"""Simulation reproduction of GenSDR on Setting B (Section 7.1 of Xu et al., 2025).

Data-generating mechanism (Setting B):
    X ~ Unif([0, 1]^{dx}), dx = 50, n = 1000 (train) and n_test = 1000 (test)
    h0(X) = [X_(1)^2 * exp(2 X_(4)^2), (1 + 3 X_(2) - X_(3))^{4/3} * 1(X_(1)^{1/2} > 0.8)]^T
    Y = h0(X) + gamma * W,  W = (W1, W2)^T with W1 ~ chi^2_3, W2 ~ chi^2_5 independent of X.
Evaluates out-of-sample distance correlation dCor(R_hat(X_test), h0(X_test)).
"""

from __future__ import annotations

import argparse
import dcor
import numpy as np

from gensdr import fit_gensdr, predict_representation


def generate_setting_b(
    n: int,
    dx: int,
    gamma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate covariates X, response Y, and true sufficient representation h0(X) for Setting B."""
    X = rng.uniform(0.0, 1.0, size=(n, dx))
    h1 = (X[:, 0] ** 2) * np.exp(2.0 * (X[:, 3] ** 2))
    h2 = ((1.0 + 3.0 * X[:, 1] - X[:, 2]) ** (4.0 / 3.0)) * (np.sqrt(X[:, 0]) > 0.8).astype(float)
    h0 = np.column_stack([h1, h2])

    w1 = rng.chisquare(df=3.0, size=n)
    w2 = rng.chisquare(df=5.0, size=n)
    W = np.column_stack([w1, w2])
    Y = h0 + gamma * W
    return X, Y, h0


def run_benchmark(
    gammas: list[float],
    dx: int,
    n_train: int,
    n_test: int,
    reps: int,
    n_epochs: int,
    batch_size: int,
) -> None:
    print(f"GenSDR Setting B Benchmark (Xu et al., 2025, Sec. 7.1): dx={dx}, n={n_train}, n_test={n_test}, reps={reps}, epochs={n_epochs}")
    print("-" * 64)
    print(f"{'gamma':>6} | {'rho = dCor(R_hat, h0) mean +/- SD':<34} | {'median':>8}")
    print("-" * 64)

    for gamma in gammas:
        rhos = []
        for rep in range(reps):
            rng = np.random.default_rng(202500 + int(round(gamma * 100)) * 100 + rep)
            X_tr, Y_tr, _ = generate_setting_b(n=n_train, dx=dx, gamma=gamma, rng=rng)
            X_te, _, h0_te = generate_setting_b(n=n_test, dx=dx, gamma=gamma, rng=rng)

            fit_res = fit_gensdr(
                X_tr,
                Y_tr,
                d=2,
                tau_n=0.001,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=1e-3,
                seed=3000 + rep,
            )
            r_hat_te = predict_representation(fit_res, X_te)
            rho = float(dcor.distance_correlation(r_hat_te.astype(np.float64), h0_te.astype(np.float64)))
            rhos.append(rho)

        print(f"{gamma:>6.1f} | {np.mean(rhos):>14.4f} +/- {np.std(rhos):<15.4f} | {np.median(rhos):>8.4f}")
    print("-" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate GenSDR Setting B (Xu et al., 2025).")
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--dx", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    run_benchmark(
        gammas=args.gammas,
        dx=args.dx,
        n_train=args.n_train,
        n_test=args.n_test,
        reps=args.reps,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
