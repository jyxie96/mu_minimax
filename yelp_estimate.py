import json
import os
import random
import string
import time

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_env] = "8"


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _ridge_penalty_matrix(d, penalize_intercept=False):
    P = np.eye(d)
    if not penalize_intercept:
        P[0, 0] = 0.0
    return P


def _regularized_gd_lr(default_lr, ridge_lamb=0.0, extra_lamb=0.0):
    return float(default_lr) / max(1.0, 2.0 * float(ridge_lamb), 2.0 * float(extra_lamb))


def _statistical_tolerance(
    p,
    n,
    tol_scale=1e-3,
):
    """tol = tol_scale · sqrt(p/n)."""
    n_safe = max(int(n), 1)
    eps = np.sqrt(p / n_safe)
    return tol_scale * eps


def _compute_retain_loss(X, y, theta, *_unused):
    residuals = y - X @ theta
    return float((residuals ** 2).mean())


# ---------------------------------------------------------------------------
# Data loading & featurization
# ---------------------------------------------------------------------------

def extract_words(text):
    text = text.lower()
    exclude = set(string.punctuation + string.digits)
    text = "".join(ch for ch in text if ch not in exclude)
    return [w for w in text.split() if len(w) > 1]


def load_dataset(file_path):
    """Load JSONL into a list of dicts (skips malformed lines)."""
    dataset = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {file_path.split('/')[-1]}", unit="lines"):
                line = line.strip()
                if not line:
                    continue
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    return dataset


def build_vocabulary(dataset, vocab_size=5000, sample_size=None):
    texts = [r["text"] for r in dataset if "text" in r]
    if sample_size is not None and len(texts) > sample_size:
        texts = random.sample(texts, int(sample_size))
    tokenized = [extract_words(t) for t in tqdm(texts, desc="Extracting words", unit="reviews")]

    counts = {}
    for review in tqdm(tokenized, desc="Counting frequencies", unit="reviews"):
        for word in set(review):
            counts[word] = counts.get(word, 0) + 1
    return sorted(counts, key=counts.get, reverse=True)[:vocab_size]


def load_or_build_vocabulary(vocab_dataset, *, vocab_size, cache_path, sample_size=None):
    """Build vocabulary or read it from a JSON cache."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            if (
                isinstance(vocab, list)
                and len(vocab) == vocab_size
                and all(isinstance(x, str) for x in vocab)
            ):
                print(f"Loaded cached vocabulary ({len(vocab)}) from {cache_path}")
                return vocab
            print(f"Vocabulary cache invalid, rebuilding: {cache_path}")
        except Exception:
            print(f"Failed to load vocabulary cache, rebuilding: {cache_path}")

    vocab = build_vocabulary(vocab_dataset, vocab_size=vocab_size, sample_size=sample_size)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Saved vocabulary cache ({len(vocab)}) to {cache_path}")
    return vocab


def txt_to_vec(text, vocabulary):
    d = len(vocabulary)
    vocab_dict = {word: i for i, word in enumerate(vocabulary)}
    v = np.zeros(d)
    for word in extract_words(text):
        idx = vocab_dict.get(word)
        if idx is not None:
            v[idx] += 1
    return v


def dataset_to_matrix(dataset, vocabulary):
    """Featurize dataset to (X with intercept, y, user_ids)."""
    n = len(dataset)
    d = len(vocabulary)
    vocab_dict = {word: i for i, word in enumerate(vocabulary)}
    X = np.zeros((n, d))
    y = np.zeros(n)
    user_ids = []

    for idx in tqdm(range(n), desc="Converting features", unit="records"):
        review = dataset[idx]
        if "user_id" in review:
            user_ids.append(review["user_id"])
        if "text" in review:
            for word in extract_words(review["text"]):
                j = vocab_dict.get(word)
                if j is not None:
                    X[idx, j] += 1
        if "stars" in review:
            y[idx] = review["stars"]

    X = np.hstack([np.ones((n, 1)), X])
    return X, y, user_ids


# ---------------------------------------------------------------------------
# OLS / ridge via GD + ridge CV
# ---------------------------------------------------------------------------

def ols_estimator(
    X,
    y,
    ridge_lamb=0.0,
    tol_scale=1e-3,
    max_iter=800000,
    tol=None,
    penalize_intercept=False,
    verbose=False,
    return_iters=False,
    tol_n=None,
    cov=None,
    M=None,
):
    """Ridge regression via GD: (cov + λP) θ = M (initialized at θ = 0)."""
    n, d = X.shape
    if cov is None:
        cov = (X.T @ X) / n
    if M is None:
        M = (X.T @ y) / n
    A = cov + float(ridge_lamb) * _ridge_penalty_matrix(d, penalize_intercept=penalize_intercept)

    lr = _regularized_gd_lr(1e-2, ridge_lamb)
    m_norm = max(np.linalg.norm(M), np.finfo(float).tiny)
    n_for_tol = int(tol_n) if tol_n is not None else n
    tol = float(tol) if tol is not None else _statistical_tolerance(p=d, n=n_for_tol, tol_scale=tol_scale)

    theta = np.zeros(d)
    iters = 0
    for _ in range(max_iter):
        res = A @ theta - M
        if np.linalg.norm(res) / m_norm < tol:
            break
        theta = theta - lr * res
        iters += 1

    if verbose:
        print(f"[OLS] ridge_lamb={float(ridge_lamb):.4g} iters={iters}")
    return (theta, iters) if return_iters else theta


def cross_validate_ridge_lambda(
    X,
    y,
    lambda_candidates,
    k_folds=5,
    random_state=42,
    penalize_intercept=False,
    cv_label=None,
    tol_n=None,
):
    """K-fold MSE-CV over `lambda_candidates`; refits OLS via GD per fold."""
    n = X.shape[0]
    if tol_n is None:
        tol_n = n
    rng = np.random.RandomState(random_state)
    indices = np.arange(n)
    rng.shuffle(indices)

    fold_size = max(1, n // k_folds)
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k_folds)]
    if len(indices) > k_folds * fold_size:
        folds[-1] = np.concatenate([folds[-1], indices[k_folds * fold_size:]])

    cv_scores = {lamb: [] for lamb in lambda_candidates}
    for fold_idx in range(k_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(k_folds) if i != fold_idx])
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        for lamb in lambda_candidates:
            try:
                theta = ols_estimator(
                    X_tr, y_tr,
                    ridge_lamb=float(lamb),
                    penalize_intercept=penalize_intercept,
                    verbose=False,
                    tol_n=tol_n,
                )
                cv_scores[lamb].append(_compute_retain_loss(X_val, y_val, theta))
            except Exception:
                continue

    mean_scores = {lamb: float(np.mean(s)) for lamb, s in cv_scores.items() if s}
    if not mean_scores:
        return float(lambda_candidates[0]), cv_scores
    return float(min(mean_scores, key=mean_scores.get)), mean_scores


def select_ridge_lambda_cv(
    X,
    y,
    *,
    use_ridge=True,
    k_folds=5,
    n_lambda_candidates=20,
    lambda_min=0.01,
    lambda_max=10.0,
    cv_random_state=42,
    label=None,
    tol_n=None,
):
    """Returns (ridge_lamb, cv_time). Skips CV when use_ridge is False."""
    if not use_ridge:
        return 0.0, 0.0
    candidates = np.logspace(np.log10(lambda_min), np.log10(lambda_max), int(n_lambda_candidates))
    t_cv = time.time()
    ridge_lamb, _ = cross_validate_ridge_lambda(
        X, y, candidates,
        k_folds=k_folds,
        random_state=cv_random_state,
        cv_label=label,
        tol_n=tol_n,
    )
    return float(ridge_lamb), time.time() - t_cv


def ols_estimator_with_ridge_cv(
    X,
    y,
    *,
    use_ridge=True,
    k_folds=5,
    n_lambda_candidates=20,
    lambda_min=0.01,
    lambda_max=10.0,
    cv_random_state=42,
    label=None,
    verbose=True,
    tol_n=None,
    **ols_kwargs,
):
    """ridge_lambda CV + ols_estimator; returns (theta, ridge_lamb, comp_time)."""
    if tol_n is None:
        tol_n = X.shape[0]
    ridge_lamb, cv_time = select_ridge_lambda_cv(
        X, y,
        use_ridge=use_ridge,
        k_folds=k_folds,
        n_lambda_candidates=n_lambda_candidates,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        cv_random_state=cv_random_state,
        label=label,
        tol_n=tol_n,
    )
    t_fit = time.time()
    theta = ols_estimator(X, y, ridge_lamb=ridge_lamb, verbose=verbose, tol_n=tol_n, **ols_kwargs)
    comp_time = cv_time + (time.time() - t_fit)
    if verbose:
        tag = f"[{label}]" if label else "[OLS+CV]"
        print(f"{tag} ridge_lamb={ridge_lamb:.4g} comp_time={comp_time:.3f}s")
    return theta, ridge_lamb, comp_time


# ---------------------------------------------------------------------------
# ULS via GD (anchored ridge on θ - θ_ptr; ridge_lamb is NOT scaled by ω_r)
# ---------------------------------------------------------------------------

def uls_estimator(
    w_r,
    w_f,
    X_r_sub,
    X_f,
    y_f,
    theta_ptr,
    ridge_lamb=0.0,
    tol_scale=1e-3,
    max_iter=800000,
    tol=None,
    cov_r_sub=None,
    cov_f=None,
    M_f=None,
    penalize_intercept=False,  # ignored: anchored ridge uses full identity
    verbose=False,
    tol_n=None,
):

    n_r_sub, p = X_r_sub.shape
    if cov_r_sub is None:
        cov_r_sub = (X_r_sub.T @ X_r_sub) / n_r_sub
    if cov_f is None:
        cov_f = (X_f.T @ X_f) / X_f.shape[0]
    if M_f is None:
        M_f = (X_f.T @ y_f) / X_f.shape[0]

    lam = float(ridge_lamb)
    I_p = np.eye(p)
    A = w_r * cov_r_sub + lam * I_p
    rhs = (w_r * cov_r_sub + w_f * cov_f + lam * I_p) @ theta_ptr - w_f * M_f
    rhs_norm = max(np.linalg.norm(rhs), np.finfo(float).tiny)

    alpha = _regularized_gd_lr(1e-2, lam)
    n_for_tol = int(tol_n) if tol_n is not None else n_r_sub
    tol = float(tol) if tol is not None else _statistical_tolerance(p=p, n=n_for_tol, tol_scale=tol_scale)

    theta = theta_ptr.copy()
    iters = 0
    for _ in range(max_iter):
        res = A @ theta - rhs
        if np.linalg.norm(res) / rhs_norm < tol:
            break
        theta = theta - alpha * res
        iters += 1

    if verbose:
        print(f"[ULS] ridge_lamb={lam:.4g} iters={iters}")
    return theta


def cross_validate_uls_ridge_lambda(
    w_r,
    w_f,
    X_r_sub,
    y_r_sub,
    X_f,
    y_f,
    theta_ptr,
    lambda_candidates,
    *,
    k_folds=5,
    random_state=42,
    tol_n=None,
    cv_label=None,
):
    """K-fold MSE-CV that refits ULS itself per fold (forget data not split).

    For each (fold, λ) it solves the ULS fixed point
        (ω_r Σ_r_train + λ·I) θ = (ω_r Σ_r_train + ω_f Σ_f + λ·I) θ_ptr − ω_f M_f,
    then scores on the retain val fold via MSE. Each ULS call recomputes Σ_r,
    Σ_f, M_f from scratch (no fold-level caching, no outer precompute).
    """
    n_r_sub = X_r_sub.shape[0]

    rng = np.random.RandomState(random_state)
    indices = np.arange(n_r_sub)
    rng.shuffle(indices)

    fold_size = max(1, n_r_sub // k_folds)
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k_folds)]
    if len(indices) > k_folds * fold_size:
        folds[-1] = np.concatenate([folds[-1], indices[k_folds * fold_size:]])

    cv_scores = {lamb: [] for lamb in lambda_candidates}
    for fold_idx in range(k_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(k_folds) if i != fold_idx])
        X_tr, y_tr = X_r_sub[train_idx], y_r_sub[train_idx]
        X_val, y_val = X_r_sub[val_idx], y_r_sub[val_idx]
        tol_n_fold = int(tol_n) if tol_n is not None else int(X_tr.shape[0])
        for lamb in lambda_candidates:
            try:
                theta = uls_estimator(
                    w_r, w_f,
                    X_tr, X_f, y_f, theta_ptr,
                    ridge_lamb=float(lamb),
                    verbose=False,
                    tol_n=tol_n_fold,
                )
                cv_scores[lamb].append(_compute_retain_loss(X_val, y_val, theta))
            except Exception:
                continue

    mean_scores = {lamb: float(np.mean(s)) for lamb, s in cv_scores.items() if s}
    if not mean_scores:
        return float(lambda_candidates[0]), cv_scores
    return float(min(mean_scores, key=mean_scores.get)), mean_scores


def select_uls_ridge_lambda_cv(
    w_r,
    w_f,
    X_r_sub,
    y_r_sub,
    X_f,
    y_f,
    theta_ptr,
    *,
    use_ridge=True,
    k_folds=5,
    n_lambda_candidates=20,
    lambda_min=0.01,
    lambda_max=10.0,
    cv_random_state=42,
    label=None,
    tol_n=None,
):
    """Returns (ridge_lamb, cv_time) selected by ULS-native K-fold MSE-CV."""
    if not use_ridge:
        return 0.0, 0.0
    candidates = np.logspace(np.log10(lambda_min), np.log10(lambda_max), int(n_lambda_candidates))
    t_cv = time.time()
    ridge_lamb, _ = cross_validate_uls_ridge_lambda(
        w_r, w_f,
        X_r_sub, y_r_sub,
        X_f, y_f,
        theta_ptr,
        candidates,
        k_folds=k_folds,
        random_state=cv_random_state,
        cv_label=label,
        tol_n=tol_n,
    )
    return float(ridge_lamb), time.time() - t_cv


def uls_estimator_with_ridge_cv(
    w_r,
    w_f,
    X_r_sub,
    y_r_sub,
    X_f,
    y_f,
    theta_ptr,
    *,
    use_ridge=True,
    k_folds=5,
    n_lambda_candidates=20,
    lambda_min=0.01,
    lambda_max=10.0,
    cv_random_state=42,
    label=None,
    verbose=True,
    tol_n=None,
    **uls_kwargs,
):
    """ULS-native ridge_lambda CV + final ULS fit; returns (theta, ridge_lamb, comp_time).

    comp_time accounts for ULS's full work: ridge-lambda CV (K-fold ULS GD
    refits — every fold × λ recomputes Σ_r / Σ_f / M_f from scratch) + final
    ULS GD fit.
    """
    if tol_n is None:
        tol_n = X_r_sub.shape[0]

    ridge_lamb, cv_time = select_uls_ridge_lambda_cv(
        w_r, w_f,
        X_r_sub, y_r_sub,
        X_f, y_f,
        theta_ptr,
        use_ridge=use_ridge,
        k_folds=k_folds,
        n_lambda_candidates=n_lambda_candidates,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        cv_random_state=cv_random_state,
        label=label,
        tol_n=tol_n,
    )
    t_fit = time.time()
    theta = uls_estimator(
        w_r, w_f, X_r_sub, X_f, y_f, theta_ptr,
        ridge_lamb=ridge_lamb,
        verbose=verbose,
        tol_n=tol_n,
        **uls_kwargs,
    )
    comp_time = cv_time + (time.time() - t_fit)
    if verbose:
        tag = f"[{label}]" if label else "[ULS+CV]"
        print(f"{tag} ridge_lamb={ridge_lamb:.4g} comp_time={comp_time:.3f}s")
    return theta, ridge_lamb, comp_time


# ---------------------------------------------------------------------------
# TL via GD (anchored tl_lamb on θ - θ_ptr; optional ridge ‖Pθ‖²)
# ---------------------------------------------------------------------------

def tl_estimator(
    X_r_sub,
    y_r_sub,
    theta_ptr,
    tl_lamb,
    ridge_lamb=0.0,
    tol_scale=1e-3,
    max_iter=200000,
    tol=None,
    Nr=None,
    N_f=None,
    w_f=None,
    delta=None,
    penalize_intercept=False,
    verbose=False,
    return_iters=False,
    tol_n=None,
    cov=None,
    M=None,
):
    """TL via GD on (cov + ridge_lamb·P + tl_lamb·I) θ = M + tl_lamb·θ_ptr."""
    n_r_sub, p = X_r_sub.shape
    if cov is None:
        cov = (X_r_sub.T @ X_r_sub) / n_r_sub
    if M is None:
        M = (X_r_sub.T @ y_r_sub) / n_r_sub
    P = _ridge_penalty_matrix(p, penalize_intercept=penalize_intercept)
    A = cov + float(ridge_lamb) * P + float(tl_lamb) * np.eye(p)
    b = M + float(tl_lamb) * theta_ptr

    rhs_norm = max(np.linalg.norm(b), np.finfo(float).tiny)
    if float(np.linalg.eigvalsh(A)[-1]) <= 0:
        return (theta_ptr.copy(), 0) if return_iters else theta_ptr.copy()

    lr = _regularized_gd_lr(1e-2, ridge_lamb, tl_lamb)

    if tol is None:
        n_for_tol = int(tol_n) if tol_n is not None else n_r_sub
        if Nr is not None and N_f is not None and w_f is not None and delta is not None:
            N_safe = max(int(Nr) + int(N_f), 1)
            tol = float(tol_scale) * min(
                float(np.sqrt(p / max(n_for_tol, 1))),
                np.sqrt(p / N_safe) + float(w_f) * float(delta),
            )
        else:
            tol = _statistical_tolerance(p=p, n=n_r_sub, tol_scale=tol_scale)
    else:
        tol = float(tol)

    theta = theta_ptr.copy()
    iters = 0
    for _ in range(int(max_iter)):
        res = A @ theta - b
        if np.linalg.norm(res) / rhs_norm < tol:
            break
        theta = theta - lr * res
        iters += 1

    if verbose:
        print(f"[TL] ridge_lamb={float(ridge_lamb):.4g} tl_lamb={float(tl_lamb):.4g} iters={iters}")
    return (theta, iters) if return_iters else theta


def cross_validate_tl_lambda(
    X_r_sub,
    y_r_sub,
    lambda_candidates,
    theta_ptr,
    ridge_lamb=0.0,
    k_folds=5,
    random_state=42,
    penalize_intercept=False,
    cv_label=None,
    tol_n=None,
):
    """K-fold MSE-CV picking the tl_lamb with the lowest mean retain-val MSE across folds."""
    n_r_sub = X_r_sub.shape[0]
    if tol_n is None:
        tol_n = n_r_sub
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_r_sub)
    rng.shuffle(indices)

    fold_size = max(1, n_r_sub // k_folds)
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k_folds)]
    if len(indices) > k_folds * fold_size:
        folds[-1] = np.concatenate([folds[-1], indices[k_folds * fold_size:]])

    cv_scores = {lamb: [] for lamb in lambda_candidates}
    for fold_idx in range(k_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(k_folds) if i != fold_idx])
        X_tr, y_tr = X_r_sub[train_idx], y_r_sub[train_idx]
        X_val, y_val = X_r_sub[val_idx], y_r_sub[val_idx]
        for tl_lamb in lambda_candidates:
            try:
                theta = tl_estimator(
                    X_tr, y_tr, theta_ptr, float(tl_lamb),
                    ridge_lamb=ridge_lamb,
                    penalize_intercept=penalize_intercept,
                    verbose=False,
                    tol_n=tol_n,
                )
                cv_scores[tl_lamb].append(_compute_retain_loss(X_val, y_val, theta))
            except Exception:
                continue

    mean_scores = {lamb: float(np.mean(s)) for lamb, s in cv_scores.items() if s}
    if not mean_scores:
        return float(lambda_candidates[0])
    return float(min(mean_scores, key=mean_scores.get))


def tl_estimator_with_lambda_selection(
    X_r_sub,
    y_r_sub,
    theta_ptr,
    ridge_lamb=0.0,
    use_cv=True,
    k_folds=5,
    n_lambda_candidates=20,
    tl_lambda_min=0.01,
    tl_lambda_max=10.0,
    cv_random_state=42,
    tol_scale=1e-3,
    max_iter=200000,
    tol=None,
    Nr=None,
    N_f=None,
    w_f=None,
    delta=None,
    penalize_intercept=False,
    cv_label=None,
    tol_n=None,
    verbose=False,
):
    """TL with optional CV for tl_lamb; returns (theta, tl_lamb)."""
    if tol_n is None:
        tol_n = int(X_r_sub.shape[0])
    if use_cv:
        candidates = np.logspace(
            np.log10(float(tl_lambda_min)),
            np.log10(float(tl_lambda_max)),
            int(n_lambda_candidates),
        )
        tl_lamb = cross_validate_tl_lambda(
            X_r_sub, y_r_sub, candidates, theta_ptr,
            ridge_lamb=ridge_lamb,
            k_folds=k_folds,
            random_state=cv_random_state,
            penalize_intercept=penalize_intercept,
            cv_label=cv_label,
            tol_n=tol_n,
        )
    else:
        tl_lamb = 1.0

    theta = tl_estimator(
        X_r_sub, y_r_sub, theta_ptr, tl_lamb,
        ridge_lamb=ridge_lamb,
        tol_scale=tol_scale,
        max_iter=max_iter,
        tol=tol,
        Nr=Nr, N_f=N_f, w_f=w_f, delta=delta,
        penalize_intercept=penalize_intercept,
        tol_n=tol_n,
        verbose=verbose,
    )
    return theta, float(tl_lamb)


def tl_estimator_with_ridge_and_tl_cv(
    X_r_sub,
    y_r_sub,
    theta_ptr,
    *,
    use_ridge=True,
    ridge_k_folds=5,
    ridge_n_lambda_candidates=20,
    ridge_lambda_min=0.01,
    ridge_lambda_max=10.0,
    ridge_cv_random_state=42,
    label=None,
    verbose=True,
    **tl_kwargs,
):
    """ridge_lamb CV (optional) + tl_lamb CV + final TL fit; returns (theta, ridge_lamb, tl_lamb, comp_time)."""
    tol_n = tl_kwargs.pop("tol_n", None)
    if tol_n is None:
        tol_n = int(X_r_sub.shape[0])
    ridge_lamb, ridge_cv_time = select_ridge_lambda_cv(
        X_r_sub, y_r_sub,
        use_ridge=use_ridge,
        k_folds=ridge_k_folds,
        n_lambda_candidates=ridge_n_lambda_candidates,
        lambda_min=ridge_lambda_min,
        lambda_max=ridge_lambda_max,
        cv_random_state=ridge_cv_random_state,
        label=f"{label} ridge" if label else None,
        tol_n=tol_n,
    )
    t_fit = time.time()
    theta, tl_lamb = tl_estimator_with_lambda_selection(
        X_r_sub, y_r_sub, theta_ptr,
        ridge_lamb=ridge_lamb,
        cv_label=label,
        tol_n=tol_n,
        verbose=verbose,
        **tl_kwargs,
    )
    comp_time = ridge_cv_time + (time.time() - t_fit)
    if verbose:
        tag = f"[{label}]" if label else "[TL+CV]"
        print(f"{tag} ridge_lamb={ridge_lamb:.4g} tl_lamb={tl_lamb:.4g} comp_time={comp_time:.3f}s")
    return theta, ridge_lamb, tl_lamb, comp_time


# ---------------------------------------------------------------------------
# Optional baselines: ULS+ and PRU (toggled in main via RUN_ULS_PLUS / RUN_PRU)
# ---------------------------------------------------------------------------

def _compute_full_loss(X_r_sub, y_r_sub, X_f, y_f, theta, lamb, n_r_sub, n_f):
    residuals_f = y_f - X_f @ theta
    return -(residuals_f ** 2).sum() / n_f


def _uls_plus_theta_for_lambda(
    X_r_sub, y_r_sub, X_f, y_f, lamb,
    w_r=None, w_f=None, theta_ptr=None,
    ridge_lamb=0.0, penalize_intercept=False,
):
    """Closed-form-style GD for ULS+ at a single λ."""
    n_r_sub, p = X_r_sub.shape
    n_f = X_f.shape[0]
    cov_r_sub = (X_r_sub.T @ X_r_sub) / n_r_sub
    cov_f = (X_f.T @ X_f) / n_f
    cov_ptr = w_f * cov_f + w_r * cov_r_sub
    M_f = (X_f.T @ y_f) / n_f
    M_r = (X_r_sub.T @ y_r_sub) / n_r_sub
    P = _ridge_penalty_matrix(p, penalize_intercept=penalize_intercept)
    cov_r_reg = cov_r_sub + float(ridge_lamb) * P

    A = (w_r + lamb) * cov_r_reg
    b = (
        -w_f * M_f
        + lamb * M_r
        + cov_ptr @ theta_ptr
        - (w_r + lamb) * float(ridge_lamb) * P @ theta_ptr
    )
    b_norm = max(np.linalg.norm(b), np.finfo(float).tiny)
    tol = _statistical_tolerance(p=p, n=n_r_sub, tol_scale=1e-3)

    eigs = np.linalg.eigvalsh(cov_r_reg)
    L = (w_r + lamb) * float(eigs[-1])
    if L <= 0:
        return theta_ptr.copy()
    alpha = 1.0 / L

    theta = theta_ptr.copy()
    for _ in range(50000):
        res = A @ theta - b
        if np.linalg.norm(res) / b_norm < tol:
            break
        theta = theta - alpha * res
    return theta


def _cv_uls_plus_lambda(
    X_r_sub, y_r_sub, X_f, y_f, lambda_candidates,
    k_folds=5, use_full_loss=False,
    w_r=None, w_f=None, theta_ptr=None,
    ridge_lamb=0.0, random_state=42, penalize_intercept=False,
):
    """CV for ULS+ λ; loss is MSE on retain val (or full -forget-loss if use_full_loss)."""
    n_r_sub = X_r_sub.shape[0]
    n_f = X_f.shape[0]
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_r_sub)
    rng.shuffle(indices)
    fold_size = n_r_sub // k_folds
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k_folds)]
    if len(indices) > k_folds * fold_size:
        folds[-1] = np.concatenate([folds[-1], indices[k_folds * fold_size:]])

    cv_scores = {lamb: [] for lamb in lambda_candidates}
    for fold_idx in range(k_folds):
        val_indices = folds[fold_idx]
        train_indices = np.concatenate([folds[i] for i in range(k_folds) if i != fold_idx])
        X_r_train, y_r_train = X_r_sub[train_indices], y_r_sub[train_indices]
        X_r_val, y_r_val = X_r_sub[val_indices], y_r_sub[val_indices]

        for lamb in lambda_candidates:
            try:
                theta = _uls_plus_theta_for_lambda(
                    X_r_train, y_r_train, X_f, y_f, lamb,
                    w_r=w_r, w_f=w_f, theta_ptr=theta_ptr,
                    ridge_lamb=ridge_lamb, penalize_intercept=penalize_intercept,
                )
                n_r_val = len(X_r_val)
                score = (
                    _compute_full_loss(X_r_val, y_r_val, X_f, y_f, theta, lamb, n_r_val, n_f)
                    if use_full_loss
                    else _compute_retain_loss(X_r_val, y_r_val, theta)
                )
                cv_scores[lamb].append(score)
            except Exception:
                continue

    mean_scores = {lamb: float(np.mean(s)) for lamb, s in cv_scores.items() if s}
    if not mean_scores:
        return lambda_candidates[0], cv_scores
    return min(mean_scores, key=mean_scores.get), mean_scores


def uls_plus_estimator(
    w_r,
    w_f,
    X_r_sub,
    y_r_sub,
    X_f,
    y_f,
    theta_ptr,
    ridge_lamb=0.0,
    delta=1,
    c=1,
    use_cv=True,
    k_folds=5,
    n_lambda_candidates=10,
    use_full_loss=False,
    cv_random_state=42,
    tol_scale=1e-3,
    max_iter=200000,
    tol=None,
    penalize_intercept=False,
):
    """ULS+ via GD under ridge-regularized retain loss; CV-selects ULS+ λ when use_cv."""
    n_r_sub, p = X_r_sub.shape
    n_f = X_f.shape[0]

    if use_cv:
        u_candidates = np.logspace(np.log10(1e-4), np.log10(1e4), n_lambda_candidates)
        lambda_candidates = 1.0 / u_candidates
        lamb, _ = _cv_uls_plus_lambda(
            X_r_sub, y_r_sub, X_f, y_f, lambda_candidates,
            k_folds=k_folds, use_full_loss=use_full_loss,
            w_r=w_r, w_f=w_f, theta_ptr=theta_ptr,
            ridge_lamb=ridge_lamb, random_state=cv_random_state,
            penalize_intercept=penalize_intercept,
        )
    else:
        lamb = c * w_r * w_f * delta

    cov_r_sub = (X_r_sub.T @ X_r_sub) / n_r_sub
    cov_f = (X_f.T @ X_f) / n_f
    cov_ptr = w_f * cov_f + w_r * cov_r_sub
    M_f = (X_f.T @ y_f) / n_f
    M_r = (X_r_sub.T @ y_r_sub) / n_r_sub
    P = _ridge_penalty_matrix(p, penalize_intercept=penalize_intercept)
    cov_r_reg = cov_r_sub + float(ridge_lamb) * P

    A = (w_r + lamb) * cov_r_reg
    b = (
        -w_f * M_f
        + lamb * M_r
        + cov_ptr @ theta_ptr
        - (w_r + lamb) * float(ridge_lamb) * P @ theta_ptr
    )
    b_norm = max(np.linalg.norm(b), np.finfo(float).tiny)
    if tol is None:
        tol = _statistical_tolerance(p=p, n=n_r_sub, tol_scale=tol_scale)
    else:
        tol = float(tol)

    eigs = np.linalg.eigvalsh(cov_r_reg)
    L = (w_r + lamb) * float(eigs[-1])
    if L <= 0:
        return theta_ptr.copy(), lamb
    alpha = 1.0 / L

    theta = theta_ptr.copy()
    iters = 0
    for _ in range(max_iter):
        res = A @ theta - b
        if np.linalg.norm(res) / b_norm < tol:
            break
        theta = theta - alpha * res
        iters += 1

    print(f"[ULS+] ridge_lamb={float(ridge_lamb):.4g} uls_lamb={float(lamb):.4g} iters={iters}")
    return theta, lamb


def gram_schmidt(X):
    k, d = X.shape
    mode = "reduced" if k <= d else "complete"
    q, r = np.linalg.qr(X.T, mode=mode)
    return q.T, r.T


def compute_lko_predictions(X, Y, ind, H=None, ridge_lamb=0.0, penalize_intercept=False):
    """Leave-k-out predictions for rows `ind` without forming the full hat matrix."""
    n = len(Y)
    k = len(ind)
    d = X.shape[1]

    if H is None:
        P = _ridge_penalty_matrix(d, penalize_intercept=penalize_intercept)
        A = X.T @ X + float(ridge_lamb) * P
        v = np.linalg.solve(A, X.T @ Y)
        X_ind = X[ind, :]
        HY_ind = X_ind @ v
        W = np.linalg.solve(A, X_ind.T)
        H_block = X_ind @ W
    else:
        H_sub = H[ind, :]
        HY_ind = H_sub @ Y
        H_block = H[np.ix_(ind, ind)]

    H_diag = np.diag(H_block)
    LOO = (Y[ind] - HY_ind) / (1 - H_diag)
    S = -H_block / (1 - H_diag)[:, None]
    np.fill_diagonal(S, 1.0)
    LKO = np.linalg.solve(S, LOO)
    return Y[ind] - LKO


def pru_estimator(
    X_r,
    y_r,
    X_f,
    y_f,
    theta,
    H=None,
    ridge_lamb=0.0,
    penalize_intercept=False,
):
    """Projective residual update; avoids stacking the n×n hat matrix when H is None."""
    n_r = len(y_r)
    k = X_f.shape[0]
    d = X_r.shape[1]
    P = _ridge_penalty_matrix(d, penalize_intercept=penalize_intercept)

    if H is None:
        A = X_r.T @ X_r + X_f.T @ X_f + float(ridge_lamb) * P
        XTy = X_r.T @ y_r + X_f.T @ y_f
        v = np.linalg.solve(A, XTy)
        HY_f = X_f @ v
        W = np.linalg.solve(A, X_f.T)
        H_ff = X_f @ W

        H_diag = np.diag(H_ff)
        LOO = (y_f - HY_f) / (1 - H_diag)
        S = -H_ff / (1 - H_diag)[:, None]
        np.fill_diagonal(S, 1.0)
        LKO = y_f - np.linalg.solve(S, LOO)
    else:
        X = np.vstack([X_r, X_f])
        Y = np.concatenate([y_r, y_f])
        ind = list(range(n_r, n_r + k))
        LKO = compute_lko_predictions(X, Y, ind, H, ridge_lamb=ridge_lamb,
                                      penalize_intercept=penalize_intercept)

    U, C = gram_schmidt(X_f)
    eigenval, a = np.linalg.eigh(C.T @ C)
    V = a.T @ U
    grad = X_f.T @ (X_f @ theta - LKO)
    factors = np.where(eigenval > 1e-10, 1 / eigenval, 0)
    step = V.T @ (factors * (V @ grad))
    return theta - step


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_prediction_error(X_test, y_test, theta):
    return float(np.mean((X_test @ theta - y_test) ** 2))


def calculate_parameter_error(theta_est, theta_true):
    return float(norm(theta_est - theta_true))


# ---------------------------------------------------------------------------
# Main: pretrain / retrain / retain-sub / ULS / TL over subsample ratios × repeats
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()

    RUN_PRU = False
    RUN_ULS_PLUS = False
    RUN_RETAIN_SUB = True
    RUN_TL = True
    USE_RIDGE = True
    RIDGE_CV_K_FOLDS = 5
    RIDGE_N_LAMBDA = 20
    RIDGE_LAMBDA_MIN = 0.001
    RIDGE_LAMBDA_MAX = 10.0

    base_dir = "/data/xiejingyi/dataset/yelp_longest_5pct"
    retain_file = os.path.join(base_dir, "yelp_retain.json")
    forget_file = os.path.join(base_dir, "yelp_forget.json")

    retain_dataset = load_dataset(retain_file)
    forget_dataset = load_dataset(forget_file)
    if retain_dataset is None or forget_dataset is None:
        print("Failed to load required datasets.")
        return

    vocab_size = 1500
    vocab_dataset = retain_dataset + forget_dataset
    vocab_cache_file = os.path.join(base_dir, f"vocabulary_{vocab_size}_pool200k.json")
    vocabulary = load_or_build_vocabulary(
        vocab_dataset, vocab_size=vocab_size, cache_path=vocab_cache_file, sample_size=None,
    )
    print(
        f"Built vocabulary ({len(vocabulary)} tokens) from retain+forget pool "
        f"({len(vocab_dataset):,} records)"
    )

    X_forget, y_forget, _ = dataset_to_matrix(forget_dataset, vocabulary)
    X_retain_full, y_retain_full, _ = dataset_to_matrix(retain_dataset, vocabulary)
    print(f"Loaded retain dataset: {len(retain_dataset):,} records from {retain_file}")

    subsample_ratios = [0.05, 0.1, 0.2, 0.3]
    N_REPEATS = 20

    repeat_results = []
    for repeat_idx in range(1, N_REPEATS + 1):
        repeat_start = time.time()

        X_retain, X_test, y_retain, y_test = train_test_split(
            X_retain_full, y_retain_full,
            test_size=0.2,
            random_state=RANDOM_SEED + repeat_idx,
            shuffle=True,
        )

        n_retain = len(X_retain)
        n_forget = len(X_forget)
        w_r = n_retain / (n_retain + n_forget)
        w_f = n_forget / (n_retain + n_forget)

        ridge_kw = dict(
            use_ridge=USE_RIDGE,
            k_folds=RIDGE_CV_K_FOLDS,
            n_lambda_candidates=RIDGE_N_LAMBDA,
            lambda_min=RIDGE_LAMBDA_MIN,
            lambda_max=RIDGE_LAMBDA_MAX,
            verbose=True,
        )

        computation_times = {}
        ridge_lambdas = {}

        theta_ptr, ridge_lambdas["Pre-train"], computation_times["Pre-train"] = (
            ols_estimator_with_ridge_cv(
                np.vstack([X_retain, X_forget]),
                np.concatenate([y_retain, y_forget]),
                cv_random_state=RANDOM_SEED + repeat_idx,
                label=f"Repeat {repeat_idx} Pre-train",
                **ridge_kw,
            )
        )

        theta_retrain, ridge_lambdas["Retrain"], computation_times["Retrain"] = (
            ols_estimator_with_ridge_cv(
                X_retain, y_retain,
                cv_random_state=RANDOM_SEED + repeat_idx + 1,
                label=f"Repeat {repeat_idx} Retrain",
                **ridge_kw,
            )
        )

        theta_retain_subs = {}
        theta_uls_dict = {}
        theta_tl_dict = {}

        if RUN_RETAIN_SUB or RUN_TL:
            for idx, ratio in enumerate(subsample_ratios):
                sub_size = max(1, int(n_retain * ratio))
                sub_indices = np.array(random.sample(range(n_retain), sub_size))
                X_retain_sub = X_retain[sub_indices]
                y_retain_sub = y_retain[sub_indices]
                n_r_sub = int(X_retain_sub.shape[0])

                if RUN_RETAIN_SUB:
                    name = f"Retain-Sub-{idx}"
                    theta_sub, ridge_lambdas[name], computation_times[name] = (
                        ols_estimator_with_ridge_cv(
                            X_retain_sub, y_retain_sub,
                            cv_random_state=RANDOM_SEED + repeat_idx + 10 * (idx + 1),
                            label=f"Repeat {repeat_idx} {name}",
                            tol_n=n_r_sub,
                            **ridge_kw,
                        )
                    )
                    theta_retain_subs[idx] = theta_sub

                    name = f"ULS-{idx}"
                    theta_uls, ridge_lambdas[name], computation_times[name] = (
                        uls_estimator_with_ridge_cv(
                            w_r, w_f,
                            X_retain_sub, y_retain_sub,
                            X_forget, y_forget,
                            theta_ptr,
                            cv_random_state=RANDOM_SEED + repeat_idx + 100 * (idx + 1),
                            label=f"Repeat {repeat_idx} {name}",
                            tol_n=n_r_sub,
                            **ridge_kw,
                        )
                    )
                    theta_uls_dict[idx] = theta_uls

                if RUN_TL:
                    # TL: anchored tl_lamb · ‖θ - θ_ptr‖² only (no separate ‖Pθ‖² ridge).
                    name = f"TL-{idx}"
                    theta_tl, tl_ridge_lamb, tl_lamb, computation_times[name] = (
                        tl_estimator_with_ridge_and_tl_cv(
                            X_retain_sub, y_retain_sub, theta_ptr,
                            ridge_cv_random_state=RANDOM_SEED + repeat_idx + 1000 * (idx + 1),
                            cv_random_state=RANDOM_SEED + repeat_idx + 10000 * (idx + 1),
                            label=f"Repeat {repeat_idx} {name}",
                            tol_n=n_r_sub,
                            Nr=n_r_sub, N_f=n_forget, w_f=w_f, delta=1,
                            ridge_k_folds=RIDGE_CV_K_FOLDS,
                            ridge_n_lambda_candidates=RIDGE_N_LAMBDA,
                            ridge_lambda_min=RIDGE_LAMBDA_MIN,
                            ridge_lambda_max=RIDGE_LAMBDA_MAX,
                            use_ridge=False,
                            n_lambda_candidates=RIDGE_N_LAMBDA,
                            tl_lambda_min=RIDGE_LAMBDA_MIN,
                            tl_lambda_max=RIDGE_LAMBDA_MAX,
                        )
                    )
                    ridge_lambdas[name] = tl_ridge_lamb
                    theta_tl_dict[idx] = (theta_tl, tl_lamb)

        theta_true = theta_retrain
        repeat_result = {
            "Pre-train": {
                "pred_error": calculate_prediction_error(X_test, y_test, theta_ptr),
                "param_error": calculate_parameter_error(theta_ptr, theta_true),
                "comp_time": computation_times["Pre-train"],
                "ridge_lambda": ridge_lambdas["Pre-train"],
            },
            "Retrain": {
                "pred_error": calculate_prediction_error(X_test, y_test, theta_retrain),
                "param_error": calculate_parameter_error(theta_retrain, theta_true),
                "comp_time": computation_times["Retrain"],
                "ridge_lambda": ridge_lambdas["Retrain"],
            },
        }
        for idx, theta in theta_retain_subs.items():
            name = f"Retain-Sub-{idx}"
            repeat_result[name] = {
                "pred_error": calculate_prediction_error(X_test, y_test, theta),
                "param_error": calculate_parameter_error(theta, theta_true),
                "comp_time": computation_times[name],
                "ridge_lambda": ridge_lambdas[name],
            }
        for idx, theta in theta_uls_dict.items():
            name = f"ULS-{idx}"
            repeat_result[name] = {
                "pred_error": calculate_prediction_error(X_test, y_test, theta),
                "param_error": calculate_parameter_error(theta, theta_true),
                "comp_time": computation_times[name],
                "ridge_lambda": ridge_lambdas[name],
            }
        if RUN_TL:
            for idx, (theta, tl_lamb) in theta_tl_dict.items():
                name = f"TL-{idx}"
                repeat_result[name] = {
                    "pred_error": calculate_prediction_error(X_test, y_test, theta),
                    "param_error": calculate_parameter_error(theta, theta_true),
                    "comp_time": computation_times[name],
                    "ridge_lambda": ridge_lambdas[name],
                    "tl_lambda": tl_lamb,
                }

        print(
            f"\n  {'Estimator':<20} {'Pred Error (MSE)':<20} "
            f"{'Param Error (L2)':<20} {'Comp Time (s)':<15}"
        )
        for name, metrics in repeat_result.items():
            print(
                f"  {name:<20} {metrics['pred_error']:<20.6f} "
                f"{metrics['param_error']:<20.6f} {metrics['comp_time']:<15.6f}"
            )

        repeat_duration = time.time() - repeat_start
        print(f"Repeat {repeat_idx} completed in {repeat_duration:.2f} seconds")

        repeat_results.append({
            "repeat": repeat_idx,
            "results": repeat_result,
            "w_r": w_r,
            "w_f": w_f,
            "duration": repeat_duration,
        })

        df_repeat = pd.DataFrame.from_dict(repeat_result, orient="index").reset_index()
        df_repeat.rename(columns={"index": "estimator"}, inplace=True)
        df_repeat["repeat"] = repeat_idx
        output_csv = os.path.join(base_dir, f"yelp_estimation_results_repeat_{repeat_idx}.csv")
        df_repeat.to_csv(output_csv, index=False)
        print(f"Repeat {repeat_idx} results saved to: {output_csv}")

    if repeat_results:
        combined_rows = []
        for entry in repeat_results:
            for estimator, metrics in entry["results"].items():
                combined_rows.append({
                    "repeat": entry["repeat"],
                    "estimator": estimator,
                    "pred_error": metrics["pred_error"],
                    "param_error": metrics["param_error"],
                    "comp_time": metrics["comp_time"],
                    "w_r": entry["w_r"],
                    "w_f": entry["w_f"],
                    "repeat_duration": entry["duration"],
                    "ridge_lambda": metrics.get("ridge_lambda", 0.0),
                    "uls_plus_lambda": metrics.get("uls_plus_lambda", 0.0),
                    "tl_lambda": metrics.get("tl_lambda", 0.0),
                })
        all_csv = os.path.join(base_dir, "yelp_estimation_results_all_repeats.csv")
        pd.DataFrame(combined_rows).to_csv(all_csv, index=False)
        print(f"\nCombined results saved to: {all_csv}")

    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
