import time
import random
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm
# from scipy.optimize import minimize


np.random.seed(42)
random.seed(42)

DEFAULT_LR = 0.5  # fallback only when Lipschitz bound is non-positive


def _logistic_grad_lipschitz_bound(X: np.ndarray) -> float:
    n = int(X.shape[0])
    if n <= 0:
        return 0.0
    xtx = X.T @ X
    return float(np.linalg.eigvalsh(xtx / (4.0 * n))[-1])


def _logistic_gd_lr(
    X: np.ndarray,
    *,
    scale: float = 1.0,
    extra_lamb: float = 0.0,
    lipschitz_bound: Optional[float] = None,
) -> float:
    """Step size 1 / (scale * L + 2 * extra_lamb) from a one-time Lipschitz estimate."""
    L = float(lipschitz_bound) if lipschitz_bound is not None else _logistic_grad_lipschitz_bound(X)
    denom = float(scale) * L + 2.0 * float(extra_lamb)
    if denom <= 0.0:
        return DEFAULT_LR
    return 1.0 / denom


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _statistical_tolerance(p: int, n: int, tol_scale: float = 1e-3) -> float:
    n_safe = max(int(n), 1)
    return float(tol_scale) * float(np.sqrt(p / n_safe))


def _tl_statistical_tolerance(
    p: int,
    n: int,
    tol_scale: float,
    N: int,
    w_f: float,
    delta: float,
) -> float:
    n_safe = max(int(n), 1)
    N_safe = max(int(N), 1)
    return float(tol_scale) * min(
        float(np.sqrt(p / n_safe)),
        np.sqrt(p / N_safe) + float(w_f) * float(delta),
    )


def _logistic_grad(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    z = np.clip(X @ theta, -500.0, 500.0)
    p = _sigmoid(z)
    return (1.0 / n) * (X.T @ (p - y))


def _ce_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    z = np.clip(X @ theta, -500.0, 500.0)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def generate_linear_features(n: int, p: int, cov: np.ndarray) -> np.ndarray:
    return np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)


def generate_binary_labels(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    probs = _sigmoid(X @ theta)
    return (np.random.rand(X.shape[0]) < probs).astype(float)


def generate_retain_data(
    n: int,
    p: int,
    theta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retain data with identity covariance (uses global np.random state)."""
    X = generate_linear_features(n, p, np.eye(p))
    y = generate_binary_labels(X, theta)
    return X, y


def generate_forget_data(
    n: int,
    p: int,
    theta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forget data with AR(1)-like covariance (uses global np.random state)."""
    idx = np.arange(p)
    cov = 0.3 ** np.abs(idx[:, None] - idx[None, :])
    X = generate_linear_features(n, p, cov)
    y = generate_binary_labels(X, theta)
    return X, y


def ce_estimator(
    X: np.ndarray,
    y: np.ndarray,
    theta_init: Optional[np.ndarray] = None,
    tol_scale: float = 1e-3,
    max_iter: int = 200000,
    tol: Optional[float] = None,
    return_info: bool = False,
    lr: Optional[float] = None,
    lipschitz_bound: Optional[float] = None,
) -> np.ndarray:
    n, p = X.shape
    theta = np.zeros(p) if theta_init is None else theta_init.copy()

    if lr is None:
        lr = _logistic_gd_lr(X, lipschitz_bound=lipschitz_bound)

    if tol is None:
        tol = _statistical_tolerance(p=p, n=n, tol_scale=tol_scale)
    else:
        tol = float(tol)

    grad_init = _logistic_grad(X, y, theta)
    grad_init_norm = max(float(np.linalg.norm(grad_init)), np.finfo(float).tiny)

    iters = 0
    for _ in range(int(max_iter)):
        grad = _logistic_grad(X, y, theta)
        if (float(np.linalg.norm(grad)) / grad_init_norm) < tol:
            break
        theta = theta - lr * grad
        iters += 1

    if return_info:
        return theta, iters
    return theta



def uce_estimator(
    w_r: float,
    w_f: float,
    X_r_sub: np.ndarray,
    y_r_sub: np.ndarray,
    X_f: np.ndarray,
    y_f: np.ndarray,
    theta_ptr: np.ndarray,
    tol_scale: float = 1e-3,
    max_iter: int = 200000,
    tol: Optional[float] = None,
    return_info: bool = False,
    lr: Optional[float] = None,
    lipschitz_bound: Optional[float] = None,
) -> np.ndarray:
    """
    UCE via GD (Algorithm 1):
    θ_t = θ_{t-1} - α Ĝ(θ_{t-1}),
    Ĝ(θ) = ω_r g_r(θ) - ω_r g_r(θ_p) - ω_f g_f(θ_p),
    with g the average CE gradient.
    """
    n_r_sub = X_r_sub.shape[0]
    p = X_r_sub.shape[1]

    g_r_ptr = _logistic_grad(X_r_sub, y_r_sub, theta_ptr)
    g_f_ptr = _logistic_grad(X_f, y_f, theta_ptr)

    if lr is None:
        lr = _logistic_gd_lr(X_r_sub, lipschitz_bound=lipschitz_bound)

    if tol is None:
        tol = _statistical_tolerance(p=p, n=n_r_sub, tol_scale=tol_scale)
    else:
        tol = float(tol)

    # G at θ=θ_ptr collapses to −w_f · g_f_ptr (the w_r terms cancel by construction)
    G_init_norm = max(float(w_f) * float(np.linalg.norm(g_f_ptr)), np.finfo(float).tiny)

    theta = theta_ptr.copy()
    iters = 0
    for _ in range(int(max_iter)):
        g_r = _logistic_grad(X_r_sub, y_r_sub, theta)
        G = w_r * g_r - w_r * g_r_ptr - w_f * g_f_ptr
        if (float(np.linalg.norm(G)) / G_init_norm) < tol:
            break
        theta = theta - lr * G
        iters += 1

    if return_info:
        return theta, iters
    return theta


def tl_estimator(
    X_r_sub: np.ndarray,
    y_r_sub: np.ndarray,
    theta_ptr: np.ndarray,
    lamb: float,
    tol_scale: float = 1e-3,
    max_iter: int = 200000,
    tol: Optional[float] = None,
    return_info: bool = False,
    N: Optional[int] = None,
    w_f: Optional[float] = None,
    delta: Optional[float] = None,
    lr: Optional[float] = None,
    lipschitz_bound: Optional[float] = None,
    theta_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    TL for logistic regression via GD on
    (1/n) CE_r(θ) + λ||θ - θ_ptr||².
    """
    n_r_sub = X_r_sub.shape[0]
    p = X_r_sub.shape[1]
    theta0 = theta_ptr.copy() if theta_init is None else theta_init.copy()

    if tol is None:
        if N is not None and w_f is not None and delta is not None:
            tol = _tl_statistical_tolerance(
                p=p,
                n=n_r_sub,
                tol_scale=tol_scale,
                N=N,
                w_f=w_f,
                delta=delta,
            )
        else:
            tol = _statistical_tolerance(p=p, n=n_r_sub, tol_scale=tol_scale)
    else:
        tol = float(tol)

    if lr is None:
        lr = _logistic_gd_lr(
            X_r_sub,
            extra_lamb=float(lamb),
            lipschitz_bound=lipschitz_bound,
        )

    theta = theta0.copy()
    grad_init_ce = _logistic_grad(X_r_sub, y_r_sub, theta)
    grad_init = grad_init_ce + 2.0 * float(lamb) * (theta - theta_ptr)
    grad_init_norm = max(float(np.linalg.norm(grad_init)), np.finfo(float).tiny)

    iters = 0
    for _ in range(int(max_iter)):
        grad_ce = _logistic_grad(X_r_sub, y_r_sub, theta)
        grad = grad_ce + 2.0 * float(lamb) * (theta - theta_ptr)
        if not np.all(np.isfinite(grad)):
            theta = theta0.copy()
            break
        if (float(np.linalg.norm(grad)) / grad_init_norm) < tol:
            break
        theta = theta - lr * grad
        if not np.all(np.isfinite(theta)):
            theta = theta0.copy()
            break
        iters += 1

    if return_info:
        return theta, iters
    return theta


def _cross_validate_tl_lambda(
    X_r_sub: np.ndarray,
    y_r_sub: np.ndarray,
    lambda_candidates: np.ndarray,
    theta_ptr: np.ndarray,
    *,
    N: int,
    w_f: float,
    delta: float,
    tol_scale: float = 1e-3,
    k_folds: int = 5,
    random_state: int = 42,
) -> float:
    n_r_sub = X_r_sub.shape[0]
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_r_sub)
    rng.shuffle(indices)

    fold_size = max(1, n_r_sub // k_folds)
    folds = [indices[i * fold_size : (i + 1) * fold_size] for i in range(k_folds)]
    if len(indices) > k_folds * fold_size:
        folds[-1] = np.concatenate([folds[-1], indices[k_folds * fold_size :]])

    cv_scores = {float(lamb): [] for lamb in lambda_candidates}
    for fold_idx in range(k_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(k_folds) if i != fold_idx])
        X_tr, y_tr = X_r_sub[train_idx], y_r_sub[train_idx]
        X_val, y_val = X_r_sub[val_idx], y_r_sub[val_idx]

        # Per-fold Lipschitz: L only depends on X_tr (not on λ), so compute once per fold
        # and reuse across all λ trials in this fold.
        L_tr = _logistic_grad_lipschitz_bound(X_tr)

        for lamb in lambda_candidates:
            try:
                theta = tl_estimator(
                    X_tr,
                    y_tr,
                    theta_ptr,
                    float(lamb),
                    tol_scale=tol_scale,
                    N=N,
                    w_f=w_f,
                    delta=delta,
                    lipschitz_bound=L_tr,
                )
                cv_scores[float(lamb)].append(_ce_loss(X_val, y_val, theta))
            except Exception:
                continue

    mean_scores = {lamb: float(np.mean(s)) for lamb, s in cv_scores.items() if s}
    if not mean_scores:
        return float(lambda_candidates[0])
    return float(min(mean_scores, key=mean_scores.get))


def tl_estimator_with_lambda_selection(
    X_r_sub: np.ndarray,
    y_r_sub: np.ndarray,
    theta_ptr: np.ndarray,
    *,
    use_cv: bool = True,
    k_folds: int = 5,
    n_lambda_candidates: int = 50,
    cv_random_state: int = 42,
    tol_scale: float = 1e-3,
    max_iter: int = 200000,
    tol: Optional[float] = None,
    return_info: bool = False,
    N: Optional[int] = None,
    w_f: Optional[float] = None,
    delta: Optional[float] = None,
    lipschitz_bound: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    if lipschitz_bound is None:
        lipschitz_bound = _logistic_grad_lipschitz_bound(X_r_sub)
    if use_cv:
        lambda_candidates = np.logspace(
            np.log10(1e-2),
            np.log10(10.0),
            int(n_lambda_candidates),
        )
        lamb = _cross_validate_tl_lambda(
            X_r_sub,
            y_r_sub,
            lambda_candidates,
            theta_ptr,
            N=int(N),
            w_f=float(w_f),
            delta=float(delta),
            tol_scale=tol_scale,
            k_folds=k_folds,
            random_state=cv_random_state,
        )
    else:
        lamb = 1.0

    if return_info:
        theta, iters = tl_estimator(
            X_r_sub,
            y_r_sub,
            theta_ptr,
            lamb,
            tol_scale=tol_scale,
            max_iter=max_iter,
            tol=tol,
            return_info=True,
            N=N,
            w_f=w_f,
            delta=delta,
            lipschitz_bound=lipschitz_bound,
        )
        return theta, float(lamb), int(iters)

    theta = tl_estimator(
        X_r_sub,
        y_r_sub,
        theta_ptr,
        lamb,
        tol_scale=tol_scale,
        max_iter=max_iter,
        tol=tol,
        N=N,
        w_f=w_f,
        delta=delta,
        lipschitz_bound=lipschitz_bound,
    )
    return theta, float(lamb)


CV_RANDOM_STATE = 42  # base seed; per-repeat seed = CV_RANDOM_STATE + rep_idx

def prepare_shared_state(
    theta_r: np.ndarray,
    theta_f: np.ndarray,
    n_retain: int,
    n_forget: int,
    p: int,
    delta: float,
) -> Dict:

    X_r, y_r = generate_retain_data(n_retain, p, theta_r)
    X_f, y_f = generate_forget_data(n_forget, p, theta_f)
    X_ptr = np.vstack([X_r, X_f])
    y_ptr = np.concatenate([y_r, y_f])

    L_ptr = _logistic_grad_lipschitz_bound(X_ptr)
    L_retain = _logistic_grad_lipschitz_bound(X_r)
    lr_ptr = _logistic_gd_lr(X_ptr, lipschitz_bound=L_ptr)
    lr_retrain = _logistic_gd_lr(X_r, lipschitz_bound=L_retain)
    print(
        "[shared] Lipschitz / lr (full-data, one-shot per (p,delta,n_r,n_f,repeat)): "
        f"L_ptr={L_ptr:.6g}, lr_ptr={lr_ptr:.6g}; "
        f"L_retain={L_retain:.6g}, lr_retrain={lr_retrain:.6g}"
    )

    t0 = time.time()
    theta_ptr, ptr_iters = ce_estimator(
        X_ptr, y_ptr, return_info=True, lr=lr_ptr, lipschitz_bound=L_ptr,
    )
    ptr_time = time.time() - t0

    t0 = time.time()
    theta_retrain, retrain_iters = ce_estimator(
        X_r, y_r, return_info=True, lr=lr_retrain, lipschitz_bound=L_retain,
    )
    retrain_time = time.time() - t0

    return {
        "theta_r": theta_r,
        "theta_f": theta_f,
        "n_retain": int(n_retain),
        "n_forget": int(n_forget),
        "p": int(p),
        "delta": float(delta),
        "X_r": X_r,
        "y_r": y_r,
        "X_f": X_f,
        "y_f": y_f,
        "w_r": n_retain / (n_retain + n_forget),
        "w_f": n_forget / (n_retain + n_forget),
        "L_ptr": L_ptr,
        "L_retain": L_retain,
        "lr_ptr": lr_ptr,
        "lr_retrain": lr_retrain,
        "theta_ptr": theta_ptr,
        "ptr_iters": int(ptr_iters),
        "ptr_time": float(ptr_time),
        "theta_retrain": theta_retrain,
        "retrain_iters": int(retrain_iters),
        "retrain_time": float(retrain_time),
    }


def run_subsample_experiment(
    shared: Dict,
    n_retain_sub: int,
    rep_idx: int,
) -> Dict:
    """Inner-loop work: depends on n_retain_sub. Reuses shared pretrain/retrain."""
    n_retain = shared["n_retain"]
    n_forget = shared["n_forget"]
    p = shared["p"]
    delta = shared["delta"]
    theta_r = shared["theta_r"]
    w_r = shared["w_r"]
    w_f = shared["w_f"]
    X_r = shared["X_r"]
    y_r = shared["y_r"]
    X_f = shared["X_f"]
    y_f = shared["y_f"]
    theta_ptr = shared["theta_ptr"]
    theta_retrain = shared["theta_retrain"]

    X_r_sub, y_r_sub = X_r[:n_retain_sub], y_r[:n_retain_sub]
    L_retain_sub = _logistic_grad_lipschitz_bound(X_r_sub)
    lr_retain_sub = _logistic_gd_lr(X_r_sub, lipschitz_bound=L_retain_sub)
    print(
        f"[sub n_r_sub={n_retain_sub}] L_retain_sub={L_retain_sub:.6g}, "
        f"lr_retain_sub={lr_retain_sub:.6g} (UCE/TL share L_retain_sub)"
    )

    t0 = time.time()
    theta_retain_sub, retain_sub_iters = ce_estimator(
        X_r_sub, y_r_sub, return_info=True, lr=lr_retain_sub, lipschitz_bound=L_retain_sub,
    )
    retain_sub_time = time.time() - t0

    t0 = time.time()
    theta_uce, uce_iters = uce_estimator(
        w_r, w_f, X_r_sub, y_r_sub, X_f, y_f, theta_ptr,
        return_info=True, lr=lr_retain_sub, lipschitz_bound=L_retain_sub,
    )
    uce_time = time.time() - t0

    t0 = time.time()
    theta_tl, lambda_tl, tl_iters = tl_estimator_with_lambda_selection(
        X_r_sub, y_r_sub, theta_ptr,
        return_info=True, N=n_retain, w_f=w_f, delta=delta,
        lipschitz_bound=L_retain_sub, cv_random_state=CV_RANDOM_STATE + int(rep_idx),
    )
    tl_time = time.time() - t0

    ptr_error = float(norm(theta_ptr - theta_r))
    retain_error = float(norm(theta_retrain - theta_r))
    retain_sub_error = float(norm(theta_retain_sub - theta_r))
    uce_error = float(norm(theta_uce - theta_r))
    tl_error = float(norm(theta_tl - theta_r))

    ptr_ce = _ce_loss(X_r, y_r, theta_ptr)
    retain_ce = _ce_loss(X_r, y_r, theta_retrain)
    retain_sub_ce = _ce_loss(X_r, y_r, theta_retain_sub)
    uce_ce = _ce_loss(X_r, y_r, theta_uce)
    tl_ce = _ce_loss(X_r, y_r, theta_tl)

    return {
        "loss_type": "cross_entropy",
        "n_retain": n_retain,
        "n_retain_sub": int(n_retain_sub),
        "n_forget": n_forget,
        "p": p,
        "delta": delta,
        "repeat_idx": int(rep_idx),
        "ptr_time": shared["ptr_time"],
        "ptr_iters": shared["ptr_iters"],
        "retrain_time": shared["retrain_time"],
        "retrain_iters": shared["retrain_iters"],
        "retain_sub_time": float(retain_sub_time),
        "retain_sub_iters": int(retain_sub_iters),
        "uce_time": float(uce_time),
        "uce_iters": int(uce_iters),
        "tl_time": float(tl_time),
        "tl_iters": int(tl_iters),
        "tl_lambda": float(lambda_tl),
        "ptr_error": ptr_error,
        "retain_error": retain_error,
        "retain_sub_error": retain_sub_error,
        "uce_error": uce_error,
        "tl_error": tl_error,
        "ptr_ce": ptr_ce,
        "retain_ce": retain_ce,
        "retain_sub_ce": retain_sub_ce,
        "uce_ce": uce_ce,
        "tl_ce": tl_ce,
    }


def run_experiment(
    theta_r: np.ndarray,
    theta_f: np.ndarray,
    n_retain: int,
    n_forget: int,
    n_retain_sub: int,
    p: int,
    delta: float,
    rep_idx: int,
    seed: int = 42,  # unused; kept for backward compatibility
) -> Dict:
    shared = prepare_shared_state(theta_r, theta_f, n_retain, n_forget, p, delta)
    return run_subsample_experiment(shared, n_retain_sub, rep_idx)


if __name__ == "__main__":
    retain_sizes = [20000]
    retain_sub_sizes = [2000, 4000, 6000]
    forget_sizes = [1000, 2000]
    dims = [10, 50, 80]
    deltas = [1.0, 2.0, 3.0]

    repeats = 500
    results = []
    start_time = time.time()

    for p in dims:
        theta_r = np.random.normal(0, 1, p)
        direction = np.ones(p)
        direction = direction / norm(direction)
        for delta in deltas:
            theta_f = theta_r + delta * direction
            for n_r in retain_sizes:
                for n_f in forget_sizes:
                    for repeat in range(repeats):
                        print(
                            f"\n=== p={p} delta={delta} n_r={n_r} n_f={n_f} "
                            f"repeat={repeat} ==="
                        )
                        shared = prepare_shared_state(
                            theta_r, theta_f, n_r, n_f, p, delta,
                        )
                        for n_r_sub in retain_sub_sizes:
                            print(f"  -- n_r_sub={n_r_sub} --")
                            out = run_subsample_experiment(shared, n_r_sub, repeat)
                            print(out)
                            print(
                                "iters:",
                                {
                                    "ptr": out["ptr_iters"],
                                    "retrain": out["retrain_iters"],
                                    "retain_sub": out["retain_sub_iters"],
                                    "uce": out["uce_iters"],
                                    "tl": out["tl_iters"],
                                },
                            )
                            results.append(out)

    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(
        "/data/xiejingyi/WAGLE/plot/simulation_results/estimator_simulation_gd_ce.csv",
        index=False,
    )
    group_cols = ["n_retain", "n_retain_sub", "n_forget", "p", "delta"]
    non_numeric_cols = ["repeat_idx", "loss_type"]
    df_mean = (
        df.drop(columns=[c for c in non_numeric_cols if c in df.columns])
        .groupby(group_cols, as_index=False)
        .mean(numeric_only=True)
    )
    df_mean.to_csv(
        "/data/xiejingyi/WAGLE/plot/simulation_results/estimator_simulation_gd_ce_mean.csv",
        index=False,
    )

    end_time = time.time()
    print(f"time cost {end_time - start_time}s")
