import time
import random
from typing import Dict, Tuple, Optional

import numpy as np
from numpy.linalg import inv, norm
from sklearn.datasets import make_spd_matrix


np.random.seed(42)
random.seed(42)


def generate_retain_data(n: int, p: int, theta: np.ndarray, noise_sigma: float = 1.0):
    """Generate retain data with identity covariance."""
    cov = np.eye(p)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    eps = np.random.normal(0, noise_sigma, n)
    y = X @ theta + eps
    return X, y


def generate_forget_data(n: int, p: int, theta: np.ndarray, noise_sigma: float = 1.0):
    """Generate forget data with AR(1)-like covariance."""
    idx = np.arange(p)
    cov = 0.3 ** np.abs(idx[:, None] - idx[None, :])
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    eps = np.random.normal(0, noise_sigma, n)
    y = X @ theta + eps
    return X, y


def _statistical_tolerance(p: int, n: int, tol_scale: float = 1e-3) -> float:
    """stopping tolerance scale ~ sqrt(p/n)."""
    n_safe = max(int(n), 1)
    return float(tol_scale) * float(np.sqrt(p / n_safe))


def ols_estimator_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS closed-form for ptr theta."""
    XTX = X.T @ X
    XTy = X.T @ y
    return np.linalg.solve(XTX, XTy)

def ols_estimator(
    X: np.ndarray,
    y: np.ndarray,
    tol_scale: float = 1e-3,
    max_iter: int = 200000,
    tol: Optional[float] = None,
    return_info: bool = False,
) -> np.ndarray:
    """
    OLS via GD Stop when relative residual < tol.
    """
    n, d = X.shape
    cov = (1 / n) * (X.T @ X)
    M = (1 / n) * (X.T @ y)
    lr = 1e-2

    m_norm = max(np.linalg.norm(M), np.finfo(float).tiny)
    tol = _statistical_tolerance(p=d, n=n, tol_scale=tol_scale) if tol is None else float(tol)

    theta = np.zeros(d)
    iters = 0
    for _ in range(int(max_iter)):
        res = cov @ theta - M
        if (np.linalg.norm(res) / m_norm) < tol:
            break
        theta = theta - lr * res
        iters += 1
    if return_info:
        return theta, iters
    return theta


def uls_estimator(
    w_r: float,
    w_f: float,
    X_r_sub: np.ndarray,
    X_f: np.ndarray,
    y_f: np.ndarray,
    theta_ptr: np.ndarray,
    tol_scale: float = 1e-3,
    max_iter: int = 200000,
    tol: Optional[float] = None,
    return_info: bool = False,
) -> np.ndarray:
    n_r_sub = X_r_sub.shape[0]
    p = X_r_sub.shape[1]

    cov_r_sub = (1 / n_r_sub) * (X_r_sub.T @ X_r_sub)
    cov_f = (1 / X_f.shape[0]) * (X_f.T @ X_f)
    cov_ptr_sub = w_r * cov_r_sub + w_f * cov_f
    M_f = (1 / X_f.shape[0]) * (X_f.T @ y_f)

    rhs = cov_ptr_sub @ theta_ptr - w_f * M_f
    rhs_norm = max(np.linalg.norm(rhs), np.finfo(float).tiny)
    lr = 1e-2

    tol = _statistical_tolerance(p=p, n=n_r_sub, tol_scale=tol_scale) if tol is None else float(tol)

    theta = theta_ptr.copy()
    iters = 0
    for _ in range(int(max_iter)):
        res = w_r * (cov_r_sub @ theta) - rhs
        if (np.linalg.norm(res) / rhs_norm) < tol:
            break
        theta = theta - lr * res
        iters += 1
    if return_info:
        return theta, iters
    return theta


# def tl_estimator_closed_form(
#     X_r_sub: np.ndarray,
#     y_r_sub: np.ndarray,
#     theta_ptr: np.ndarray,
#     lamb: float,
# ) -> np.ndarray:
#     """
#     TL closed-form for
#     argmin (1/n)||y-Xθ||² + λ||θ-θ_ptr||²  =>  (Σ + λI)θ = M + λθ_ptr.
#     """
#     n_r_sub = X_r_sub.shape[0]
#     cov = (1 / n_r_sub) * (X_r_sub.T @ X_r_sub)
#     M = (1 / n_r_sub) * (X_r_sub.T @ y_r_sub)
#     A = cov + float(lamb) * np.eye(cov.shape[0])
#     b = M + float(lamb) * theta_ptr
#     try:
#         return np.linalg.solve(A, b)
#     except np.linalg.LinAlgError:
#         return np.linalg.lstsq(A, b, rcond=None)[0]


def tl_estimator(
    X_r_sub: np.ndarray,
    y_r_sub: np.ndarray,
    theta_ptr: np.ndarray,
    lamb: float,
    tol_scale: float = 1e-3,
    max_iter: int = 200000,
    tol: Optional[float] = None,
    return_info: bool = False,
    theta_init: Optional[np.ndarray] = None,
) -> np.ndarray:

    n_r_sub = X_r_sub.shape[0]
    p = X_r_sub.shape[1]

    cov = (1 / n_r_sub) * (X_r_sub.T @ X_r_sub)
    M = (1 / n_r_sub) * (X_r_sub.T @ y_r_sub)
    A = cov + float(lamb) * np.eye(p)
    b = M + float(lamb) * theta_ptr

    rhs_norm = max(np.linalg.norm(b), np.finfo(float).tiny)
    lr = 1e-2

    tol = _statistical_tolerance(p=p, n=n_r_sub, tol_scale=tol_scale) if tol is None else float(tol)

    theta0 = theta_ptr.copy() if theta_init is None else theta_init.copy()
    theta = theta0.copy()
    iters = 0
    for _ in range(int(max_iter)):
        res = A @ theta - b
        if (np.linalg.norm(res) / rhs_norm) < tol:
            break
        theta = theta - lr * res
        iters += 1
    if return_info:
        return theta, iters
    return theta


def _cross_validate_tl_lambda(
    X_r_sub: np.ndarray,
    y_r_sub: np.ndarray,
    lambda_candidates: np.ndarray,
    theta_ptr: np.ndarray,
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

        for lamb in lambda_candidates:
            try:
                theta = tl_estimator(X_tr, y_tr, theta_ptr, float(lamb))
                cv_scores[float(lamb)].append(_compute_retain_loss(X_val, y_val, theta))
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
    use_cv: bool = True,
    k_folds: int = 5,
    n_lambda_candidates: int = 50,
    cv_random_state: int = 42,
    tol_scale: float = 1e-3,
    max_iter: int = 200000,
    tol: Optional[float] = None,
    return_info: bool = False,
) -> Tuple[np.ndarray, float]:
    """TL with optional CV for λ; final θ solved via GD."""
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
    )
    return theta, float(lamb)


# def _compute_theta_for_lambda(
#     X_r_sub: np.ndarray,
#     y_r_sub: np.ndarray,
#     X_f: np.ndarray,
#     y_f: np.ndarray,
#     lamb: float,
#     *,
#     w_r: float,
#     w_f: float,
#     theta_ptr: np.ndarray,
# ) -> np.ndarray:
#     """Closed-form theta for ULS+ (same structure as yelp_estimate.py)."""
#     cov_r_sub = (1 / X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
#     cov_f = (1 / X_f.shape[0]) * (X_f.T @ X_f)
#     cov_ptr = w_f * cov_f + w_r * cov_r_sub

#     A = inv((w_r + lamb) * cov_r_sub)
#     b = (
#         -(w_f / X_f.shape[0]) * (X_f.T @ y_f)
#         + (lamb / X_r_sub.shape[0]) * (X_r_sub.T @ y_r_sub)
#         + cov_ptr @ theta_ptr
#     )
#     return A @ b


def _compute_retain_loss(X_r_sub: np.ndarray, y_r_sub: np.ndarray, theta: np.ndarray) -> float:
    residuals = y_r_sub - X_r_sub @ theta
    return float(np.mean(residuals**2))


# def _cross_validate_lambda(
#     X_r_sub: np.ndarray,
#     y_r_sub: np.ndarray,
#     X_f: np.ndarray,
#     y_f: np.ndarray,
#     lambda_candidates: np.ndarray,
#     *,
#     w_r: float,
#     w_f: float,
#     theta_ptr: np.ndarray,
#     k_folds: int = 5,
#     random_state: int = 42,
# ) -> float:
#     n_r_sub = X_r_sub.shape[0]
#     rng = np.random.RandomState(random_state)
#     indices = np.arange(n_r_sub)
#     rng.shuffle(indices)

#     fold_size = max(1, n_r_sub // k_folds)
#     folds = [indices[i * fold_size : (i + 1) * fold_size] for i in range(k_folds)]
#     if len(indices) > k_folds * fold_size:
#         folds[-1] = np.concatenate([folds[-1], indices[k_folds * fold_size :]])

#     best_lambda = float(lambda_candidates[0])
#     best_score = float("inf")
#     for fold_idx in range(k_folds):
#         val_idx = folds[fold_idx]
#         train_idx = np.concatenate([folds[i] for i in range(k_folds) if i != fold_idx])
#         X_tr, y_tr = X_r_sub[train_idx], y_r_sub[train_idx]
#         X_val, y_val = X_r_sub[val_idx], y_r_sub[val_idx]

#         for lamb in lambda_candidates:
#             try:
#                 theta = _compute_theta_for_lambda(
#                     X_tr,
#                     y_tr,
#                     X_f,
#                     y_f,
#                     float(lamb),
#                     w_r=w_r,
#                     w_f=w_f,
#                     theta_ptr=theta_ptr,
#                 )
#                 score = _compute_retain_loss(X_val, y_val, theta)
#                 if score < best_score:
#                     best_score = score
#                     best_lambda = float(lamb)
#             except Exception:
#                 continue

#     return best_lambda


# def uls_plus_estimator(
#     w_r: float,
#     w_f: float,
#     X_r_sub: np.ndarray,
#     y_r_sub: np.ndarray,
#     X_f: np.ndarray,
#     y_f: np.ndarray,
#     theta_ptr: np.ndarray,
#     *,
#     use_cv: bool = True,
#     k_folds: int = 5,
#     n_lambda_candidates: int = 50,
#     cv_random_state: int = 42,
# ) -> Tuple[np.ndarray, float]:
#     """ULS+ with optional CV for lambda; closed form for theta given lambda."""
#     if use_cv:
#         lambda_candidates = np.logspace(
#             np.log10(1e-2),
#             np.log10(10.0),
#             int(n_lambda_candidates),
#         )
#         lamb = _cross_validate_lambda(
#             X_r_sub,
#             y_r_sub,
#             X_f,
#             y_f,
#             lambda_candidates,
#             w_r=w_r,
#             w_f=w_f,
#             theta_ptr=theta_ptr,
#             k_folds=k_folds,
#             random_state=cv_random_state,
#         )
#     else:
#         # A simple default when CV is disabled.
#         lamb = 1.0

#     theta = _compute_theta_for_lambda(
#         X_r_sub,
#         y_r_sub,
#         X_f,
#         y_f,
#         lamb,
#         w_r=w_r,
#         w_f=w_f,
#         theta_ptr=theta_ptr,
#     )
#     return theta, float(lamb)


# def uls_inference(
#     p: int,
#     theta_uls: np.ndarray,
#     n_retain: int,
#     n_retain_sub: int,
#     X_r_sub: np.ndarray,
#     y_r_sub: np.ndarray,
#     theta_ptr: np.ndarray,
#     theta_r: np.ndarray,
#     z_alpha: float = 1.96,
# ):
#     v = np.zeros(p)
#     v[0] = 1
#     cov_r_sub = (1 / X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
#     inv_cov_r_sub = inv(cov_r_sub)
#     f_i = 0.0
#     g_i = 0.0
#     w_1 = 1 / (n_retain**2)
#     w_2 = (n_retain_sub - n_retain) / n_retain_sub
#     w_3 = (n_retain - n_retain_sub) / ((n_retain**2) * n_retain_sub)
#     for i in range(X_r_sub.shape[0]):
#         x_i = X_r_sub[i, :].reshape(-1, 1)
#         a_i = v.T @ inv_cov_r_sub @ X_r_sub[i, :] * (y_r_sub[i] - X_r_sub[i, :].T @ theta_uls)
#         mat = x_i @ x_i.T - cov_r_sub
#         b_i = v.T @ inv_cov_r_sub @ mat @ (theta_uls - theta_ptr)
#         f_i += float((a_i + w_2 * b_i) ** 2)
#         g_i += float((a_i + b_i) ** 2)
#     V_r = w_1 * f_i + (w_3 * g_i)
#     se_r = float(np.sqrt(V_r))
#     psi_r = float(v @ theta_uls)
#     true_val = float(v @ theta_r)
#     L = psi_r - z_alpha * se_r
#     U = psi_r + z_alpha * se_r

#     covered = int(L <= true_val <= U)
#     width = U - L
#     return covered, width, se_r


# def retain_sub_inference(
#     p: int,
#     theta_retain_sub: np.ndarray,
#     X_r_sub: np.ndarray,
#     y_r_sub: np.ndarray,
#     theta_r: np.ndarray,
#     z_alpha: float = 1.96,
# ):
#     v = np.zeros(p)
#     v[0] = 1

#     n_retain_sub = X_r_sub.shape[0]
#     residuals = y_r_sub - X_r_sub @ theta_retain_sub
#     sigma_sq_hat = np.sum(residuals**2) / max(n_retain_sub - p, 1)
#     inv_XTX = inv(X_r_sub.T @ X_r_sub)
#     cov_theta = sigma_sq_hat * inv_XTX
#     se = float(np.sqrt(v.T @ cov_theta @ v))

#     psi_retain_sub = float(v.T @ theta_retain_sub)
#     true_val = float(v.T @ theta_r)

#     L = psi_retain_sub - z_alpha * se
#     U = psi_retain_sub + z_alpha * se

#     covered = int(L <= true_val <= U)
#     width = U - L
#     return covered, width, se


def run_experiment(
    theta_r: np.ndarray,
    theta_f: np.ndarray,
    n_retain: int,
    n_forget: int,
    n_retain_sub: int,
    p: int,
    delta: float,
    rep_idx: int,
    noise_sigma: float = 1.0,
    seed: int = 42,
) -> Dict:
    X_r, y_r = generate_retain_data(n_retain, p, theta_r, noise_sigma)
    X_f, y_f = generate_forget_data(n_forget, p, theta_f, noise_sigma)
    X_r_sub, y_r_sub = X_r[:n_retain_sub], y_r[:n_retain_sub]

    X_ptr = np.vstack([X_r, X_f])
    y_ptr = np.concatenate([y_r, y_f])
    t0 = time.time()
    theta_ptr = ols_estimator_closed_form(X_ptr, y_ptr)
    ptr_time = time.time() - t0

    t0 = time.time()
    theta_retrain, retrain_iters = ols_estimator(X_r, y_r, return_info=True)
    retrain_time = time.time() - t0

    t0 = time.time()
    theta_retain_sub, retain_sub_iters = ols_estimator(X_r_sub, y_r_sub, return_info=True)
    retain_sub_time = time.time() - t0

    w_r = n_retain / (n_retain + n_forget)
    w_f = n_forget / (n_retain + n_forget)

    t0 = time.time()
    theta_uls, uls_iters = uls_estimator(w_r, w_f, X_r_sub, X_f, y_f, theta_ptr, return_info=True)
    uls_time = time.time() - t0

    t0 = time.time()
    theta_tl, lambda_tl, tl_iters = tl_estimator_with_lambda_selection(
        X_r_sub,
        y_r_sub,
        theta_ptr,
        return_info=True,
        cv_random_state=seed,
    )
    tl_time = time.time() - t0

    ptr_error = norm(theta_ptr - theta_r)
    retain_error = norm(theta_retrain - theta_r)
    retain_sub_error = norm(theta_retain_sub - theta_r)
    uls_error = norm(theta_uls - theta_r)
    tl_error = norm(theta_tl - theta_r)

    return {
        "n_retain": n_retain,
        "n_retain_sub": int(n_retain_sub),
        "n_forget": n_forget,
        "p": p,
        "delta": delta,
        "repeat_idx": int(rep_idx),
        "ptr_time": float(ptr_time),
        "retrain_time": float(retrain_time),
        "retrain_iters": int(retrain_iters),
        "retain_sub_time": float(retain_sub_time),
        "retain_sub_iters": int(retain_sub_iters),
        "uls_time": float(uls_time),
        "uls_iters": int(uls_iters),
        "tl_time": float(tl_time),
        "tl_iters": int(tl_iters),
        "tl_lambda": float(lambda_tl),
        "ptr_error": float(ptr_error),
        "retain_error": float(retain_error),
        "retain_sub_error": float(retain_sub_error),
        "uls_error": float(uls_error),
        "tl_error": float(tl_error),
    }


if __name__ == "__main__":
    retain_sizes = [20000]
    retain_sub_sizes = [1000, 2000, 4000, 6000]
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
                for n_r_sub in retain_sub_sizes:
                    for n_f in forget_sizes:
                        for repeat in range(repeats):
                            print(
                                "current experiment has reached: ",
                                n_r, n_f, n_r_sub, p, delta, repeat,
                            )
                            out = run_experiment(
                                theta_r, theta_f, n_r, n_f, n_r_sub, p, delta, repeat,
                                seed=42 + repeat,
                            )
                            results.append(out)

    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(
        "/data/xiejingyi/WAGLE/plot/simulation_results/estimator_simulation_gd.csv",
        index=False,
    )
    group_cols = ["n_retain", "n_retain_sub", "n_forget", "p", "delta"]
    df_mean = df.drop(columns=["repeat_idx"]).groupby(group_cols, as_index=False).mean()
    df_mean.to_csv(
        "/data/xiejingyi/WAGLE/plot/simulation_results/estimator_simulation_gd_mean.csv",
        index=False,
    )

    end_time = time.time()
    print(f"time cost {end_time - start_time}s")

