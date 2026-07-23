import pandas as pd
import numpy as np
import random
import time
import os
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def _statistical_tolerance(
    p,
    n,
    tol_scale=1e-3,
    Nr=None,
    N_f=None,
    w_f=None,
    *,
    from_uls=False,
):
    n_safe = max(int(n), 1)
    eps_sub = np.sqrt(p / n_safe)
    if from_uls and Nr is not None and N_f is not None:
        m = max(min(int(Nr), int(N_f)), 1)
        eps_bottleneck = np.sqrt(p / m)
        denom = np.sqrt(max(n_safe * m, 1))
        if w_f is not None:
            eps_omega = float(w_f) * float(p) / denom
            eps_stat = max(eps_sub, eps_omega)
        else:
            eps_stat = eps_sub
    else:
        eps_stat = eps_sub
    return tol_scale * eps_stat


def clean_data(df, feature_cols, target_col):
    available_feature_cols = [col for col in feature_cols if col in df.columns]

    if len(available_feature_cols) > 0:
        mask = df[target_col].notna() & df[available_feature_cols].notna().all(axis=1)
        df_clean = df[mask].copy()
    else:
        df_clean = df.copy()

    nonzero_df = df_clean[df_clean[target_col] != 0].copy()
    df_dedup, unique_indices, n_original, n_duplicates = deduplicate_raw_data(
        nonzero_df, feature_cols, target_col
    )
    return df_dedup, unique_indices, n_original, n_duplicates


def deduplicate_raw_data(df, feature_cols, target_col):
    n_original = len(df)
    cols_to_check = [col for col in feature_cols + [target_col] if col in df.columns]

    if len(cols_to_check) == 0:
        return df, np.arange(len(df)), n_original, 0

    df_with_index = df.reset_index(drop=True)
    df_with_index['_original_index'] = np.arange(len(df_with_index))
    df_dedup_with_index = df_with_index.drop_duplicates(subset=cols_to_check, keep='first')

    unique_indices = np.sort(df_dedup_with_index['_original_index'].values)
    df_dedup = df_dedup_with_index.drop(columns=['_original_index']).reset_index(drop=True)
    n_duplicates = n_original - len(df_dedup)
    return df_dedup, unique_indices, n_original, n_duplicates


def fit_target_encoder(df_train, col_name, target_col):
    col_data = df_train[col_name].astype(str).values.reshape(-1, 1)
    y_target = np.log10(df_train[target_col].values)
    target_encoder = TargetEncoder(random_state=RANDOM_SEED)
    target_encoder.fit(col_data, y_target)
    return target_encoder


def apply_target_encoding(df, col_name, target_encoder):
    col_data = df[col_name].astype(str).values.reshape(-1, 1)
    encoded_values = target_encoder.transform(col_data)
    return encoded_values.flatten()


def _process_features_for_encoding(df, feature_cols, target_encoding_cols, target_encoders,
                                   admimeth_top_classes=None, merge_admimeth_to_other=False):
    df_proc = df.copy()
    categorical_cols = [col for col in feature_cols if col not in target_encoding_cols]

    if (merge_admimeth_to_other and 'admimeth_uni' in categorical_cols
            and 'admimeth_uni' in df_proc.columns and admimeth_top_classes is not None):
        df_proc['admimeth_uni'] = df_proc['admimeth_uni'].astype(str).apply(
            lambda x: x if x in admimeth_top_classes else 'Other'
        )

    for col in categorical_cols:
        if col != 'admimeth_uni':
            df_proc[col] = df_proc[col].astype(str)

    for col in target_encoding_cols:
        if col in df_proc.columns and col in target_encoders:
            encoded_values = apply_target_encoding(df_proc, col, target_encoders[col])
            df_proc[col] = pd.Series(encoded_values, index=df_proc.index, dtype=float)

    return df_proc[feature_cols], df_proc


def prepare_features(df, feature_cols, target_encoding_cols=None, merge_admimeth_to_other=False):
    if target_encoding_cols is None:
        target_encoding_cols = []

    df_processed = df.copy()
    categorical_cols = [col for col in feature_cols if col not in target_encoding_cols]

    if 'admimeth_uni' in categorical_cols and 'admimeth_uni' in df_processed.columns:
        df_processed['admimeth_uni'] = df_processed['admimeth_uni'].astype(str)
        if merge_admimeth_to_other:
            top_classes = df_processed['admimeth_uni'].value_counts().head(10).index.tolist()
            df_processed['admimeth_uni'] = df_processed['admimeth_uni'].apply(
                lambda x: x if x in top_classes else 'Other'
            )

    for col in categorical_cols:
        if col != 'admimeth_uni':
            df_processed[col] = df_processed[col].astype(str)

    for col in target_encoding_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)

    return df_processed[feature_cols]


def fit_encoder(train_features, encoding_method='onehot',
                full_df=None, retain_df=None, forget_df=None, feature_cols=None, target_col=None,
                target_encoding_cols=None, merge_admimeth_to_other=False):
    if target_encoding_cols is None:
        target_encoding_cols = []

    categorical_cols = [col for col in feature_cols if col not in target_encoding_cols]

    target_encoders = {}
    if len(target_encoding_cols) > 0 and retain_df is not None:
        for col in target_encoding_cols:
            if col in retain_df.columns and target_col in retain_df.columns:
                target_encoders[col] = fit_target_encoder(full_df, col, target_col)

    if encoding_method != 'onehot':
        raise ValueError(f"Unsupported encoding method: {encoding_method}")

    train_features_categorical = train_features[categorical_cols] if categorical_cols else pd.DataFrame()

    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    if len(categorical_cols) > 0:
        onehot_encoder.fit(train_features_categorical)

    if full_df is None or retain_df is None or forget_df is None:
        raise ValueError("onehot method requires full_df, retain_df, forget_df to identify all-zero columns")

    admimeth_top_classes = None
    if merge_admimeth_to_other and 'admimeth_uni' in categorical_cols and 'admimeth_uni' in full_df.columns:
        admimeth_top_classes = full_df['admimeth_uni'].astype(str).value_counts().head(10).index.tolist()

    full_features, full_df_valid = _process_features_for_encoding(
        full_df, feature_cols, target_encoding_cols, target_encoders,
        admimeth_top_classes, merge_admimeth_to_other
    )
    retain_features, _ = _process_features_for_encoding(
        retain_df, feature_cols, target_encoding_cols, target_encoders,
        admimeth_top_classes, merge_admimeth_to_other
    )
    forget_features, _ = _process_features_for_encoding(
        forget_df, feature_cols, target_encoding_cols, target_encoders,
        admimeth_top_classes, merge_admimeth_to_other
    )

    if len(categorical_cols) > 0:
        X_full_onehot = onehot_encoder.transform(full_features[categorical_cols])
        X_retain_onehot = onehot_encoder.transform(retain_features[categorical_cols])
        X_forget_onehot = onehot_encoder.transform(forget_features[categorical_cols])
    else:
        X_full_onehot = np.zeros((len(full_features), 0))
        X_retain_onehot = np.zeros((len(retain_features), 0))
        X_forget_onehot = np.zeros((len(forget_features), 0))

    col_nonzero_full = (X_full_onehot != 0).any(axis=0)
    col_nonzero_retain = (X_retain_onehot != 0).any(axis=0)
    col_nonzero_forget = (X_forget_onehot != 0).any(axis=0)

    nonzero_cols = col_nonzero_full & col_nonzero_retain & col_nonzero_forget
    nonzero_col_indices = np.where(nonzero_cols)[0]

    X_full_nonzero = X_full_onehot[:, nonzero_col_indices]
    y_full_for_lasso = full_df_valid[target_col].values

    alphas = np.logspace(-2, 2, 50)
    lasso_cv = LassoCV(alphas=alphas, cv=50, random_state=RANDOM_SEED, n_jobs=-1, max_iter=10000)
    lasso_cv.fit(X_full_nonzero, y_full_for_lasso)
    lasso_selected_mask = (lasso_cv.coef_ != 0)
    n_selected = lasso_selected_mask.sum()

    if n_selected > 0:
        lasso_selected_indices = np.where(lasso_selected_mask)[0]
        final_col_indices = nonzero_col_indices[lasso_selected_indices]
    else:
        final_col_indices = nonzero_col_indices

    selected_onehot_feature_names = []
    if len(categorical_cols) > 0:
        all_onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
        selected_onehot_feature_names = all_onehot_feature_names[final_col_indices].tolist()

    all_selected_feature_names = selected_onehot_feature_names + target_encoding_cols
    print(all_selected_feature_names)
    if 'operstat_One or more operative procedures performed' not in all_selected_feature_names:
        raise ValueError(
            f"operstat_One or more operative procedures performed is not in Lasso selected features. "
            f"Selected features: {all_selected_feature_names}"
        )
    operstat_idx = all_selected_feature_names.index('operstat_One or more operative procedures performed')

    encoder = {
        'encoder': onehot_encoder,
        'final_cols': final_col_indices,
        'categorical_cols': categorical_cols,
        'target_encoding_cols': target_encoding_cols,
        'selected_target_encoding_cols': target_encoding_cols,
        'target_encoders': target_encoders,
        'admimeth_top_classes': admimeth_top_classes,
        'merge_admimeth_to_other': merge_admimeth_to_other,
        'selected_feature_names': all_selected_feature_names,
        'operstat_idx': operstat_idx,
    }
    return encoder, encoding_method


def prepare_X_y(df, feature_cols, target_col, encoder, encoding_method='onehot'):
    target_encoding_cols = encoder.get('target_encoding_cols', [])
    selected_target_encoding_cols = encoder.get('selected_target_encoding_cols', target_encoding_cols)
    target_encoders = encoder.get('target_encoders', {})
    admimeth_top_classes = encoder.get('admimeth_top_classes', None)
    merge_admimeth_to_other = encoder.get('merge_admimeth_to_other', False)
    categorical_cols = encoder.get('categorical_cols',
                                   [col for col in feature_cols if col not in target_encoding_cols])

    for col in target_encoding_cols:
        if col in df.columns and col not in target_encoders:
            raise ValueError(
                f"Target encoder not found for column '{col}'. "
                f"Available encoders: {list(target_encoders.keys())}"
            )

    features_processed, _ = _process_features_for_encoding(
        df, feature_cols, target_encoding_cols, target_encoders,
        admimeth_top_classes, merge_admimeth_to_other
    )

    if encoding_method != 'onehot':
        raise ValueError(f"Unsupported encoding method: {encoding_method}")

    if len(categorical_cols) > 0:
        X_onehot_full = encoder['encoder'].transform(features_processed[categorical_cols])
        X_onehot = X_onehot_full[:, encoder['final_cols']]
    else:
        X_onehot = np.zeros((len(features_processed), 0))

    if len(selected_target_encoding_cols) > 0:
        X_target_enc = features_processed[selected_target_encoding_cols].values.astype(float)
    else:
        X_target_enc = np.zeros((len(features_processed), 0))

    X = np.hstack([X_onehot, X_target_enc]) if len(selected_target_encoding_cols) > 0 else X_onehot
    y = df[target_col].values

    intercept = np.ones((X.shape[0], 1))
    X = np.hstack([intercept, X])

    return X, np.log10(y), np.ones(len(df), dtype=bool)


DEFAULT_LR = 1e-3  # fallback only when Lipschitz bound is non-positive


def _lipschitz_gd_lr(A, default_lr=DEFAULT_LR):
    """Step size 1 / λ_max(A) for GD on Aθ = b (A ⪰ 0)."""
    L = float(np.linalg.eigvalsh(A)[-1])
    return float(default_lr)
    


def ols_estimator(
    X,
    y,
    tol_scale=1e-3,
    max_iter=800000,
    tol=None,
    verbose=True,
    return_iters=False,
    tol_n=None,
):
    """OLS via GD on cov θ = M with cov = XᵀX/n, M = Xᵀy/n."""
    n, d = X.shape
    cov = (1 / n) * (X.T @ X)
    M = (1 / n) * (X.T @ y)
    A = cov

    lr = _lipschitz_gd_lr(A)
    theta = np.zeros(d)
    m_norm = max(np.linalg.norm(M), np.finfo(float).tiny)
    if tol is None:
        n_for_tol = int(tol_n) if tol_n is not None else n
        tol = _statistical_tolerance(p=d, n=n_for_tol, tol_scale=tol_scale)
    else:
        tol = float(tol)

    iters = 0
    for _ in range(max_iter):
        res = A @ theta - M
        rel_resid = np.linalg.norm(res) / m_norm
        if rel_resid < tol:
            break
        theta = theta - lr * res
        iters += 1

    if verbose:
        print(f"[OLS] lr={lr:.4g} iters={iters}")
    if return_iters:
        return theta, iters
    return theta


def _compute_retain_loss(X_r_sub, y_r_sub, theta):
    residuals = y_r_sub - X_r_sub @ theta
    return float(np.mean(residuals ** 2))


def tl_estimator(
    X_r_sub,
    y_r_sub,
    theta_ptr,
    tl_lamb,
    tol_scale=1e-3,
    max_iter=200000,
    tol=None,
    Nr=None,
    N_f=None,
    w_f=None,
    delta=None,
    verbose=True,
    return_iters=False,
    tol_n=None,
):
    """TL via GD on (cov + tl_lamb I)θ = M + tl_lamb θ_ptr."""
    n_r_sub, p = X_r_sub.shape
    cov = (1 / n_r_sub) * (X_r_sub.T @ X_r_sub)
    M = (1 / n_r_sub) * (X_r_sub.T @ y_r_sub)
    A = cov + float(tl_lamb) * np.eye(p)
    b = M + float(tl_lamb) * theta_ptr

    rhs_norm = max(np.linalg.norm(b), np.finfo(float).tiny)
    L = float(np.linalg.eigvalsh(A)[-1])
    if L <= 0:
        if return_iters:
            return theta_ptr.copy(), 0
        return theta_ptr.copy()
    lr = 1.0 / L

    if tol is None:
        n_for_tol = int(tol_n) if tol_n is not None else n_r_sub
        n_safe = max(n_for_tol, 1)
        if Nr is not None and N_f is not None and w_f is not None and delta is not None:
            N_safe = max(int(Nr) + int(N_f), 1)
            tol = float(tol_scale) * min(
                float(np.sqrt(p / n_safe)),
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
        rel_resid = np.linalg.norm(res) / rhs_norm
        if rel_resid < tol:
            break
        theta = theta - lr * res
        iters += 1

    if verbose:
        print(f"[TL] tl_lamb={float(tl_lamb):.4g} lr={lr:.4g} iters={iters}")
    if return_iters:
        return theta, iters
    return theta


def cross_validate_tl_lambda(
    X_r_sub,
    y_r_sub,
    lambda_candidates,
    theta_ptr,
    k_folds=5,
    random_state=42,
    tol_n=None,
):
    n_r_sub = X_r_sub.shape[0]
    if tol_n is None:
        tol_n = n_r_sub
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

        for tl_lamb in lambda_candidates:
            try:
                theta = tl_estimator(
                    X_tr, y_tr, theta_ptr, float(tl_lamb),
                    verbose=False, tol_n=tol_n,
                )
                cv_scores[float(tl_lamb)].append(_compute_retain_loss(X_val, y_val, theta))
            except Exception:
                continue

    mean_scores = {lamb: float(np.mean(s)) for lamb, s in cv_scores.items() if s}
    if mean_scores:
        return float(min(mean_scores, key=mean_scores.get))
    return float(lambda_candidates[0])


def tl_estimator_with_lambda_selection(
    X_r_sub,
    y_r_sub,
    theta_ptr,
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
    tol_n=None,
):
    if tol_n is None:
        tol_n = int(X_r_sub.shape[0])
    if use_cv:
        lambda_candidates = np.logspace(
            np.log10(float(tl_lambda_min)),
            np.log10(float(tl_lambda_max)),
            int(n_lambda_candidates),
        )
        tl_lamb = cross_validate_tl_lambda(
            X_r_sub, y_r_sub, lambda_candidates, theta_ptr,
            k_folds=k_folds, random_state=cv_random_state, tol_n=tol_n,
        )
    else:
        tl_lamb = 1.0

    theta = tl_estimator(
        X_r_sub, y_r_sub, theta_ptr, tl_lamb,
        tol_scale=tol_scale, max_iter=max_iter, tol=tol,
        Nr=Nr, N_f=N_f, w_f=w_f, delta=delta, tol_n=tol_n,
    )
    return theta, float(tl_lamb)


def uls_estimator(
    w_r,
    w_f,
    X_r_sub,
    X_f,
    y_f,
    theta_ptr,
    tol_scale=1e-3,
    max_iter=800000,
    tol=None,
    cov_f=None,
    M_f=None,
    verbose=True,
    tol_n=None,
):
    """ULS via GD (unregularized fixed-point)."""
    n_r_sub = X_r_sub.shape[0]
    p = X_r_sub.shape[1]

    cov_r_sub = (1 / n_r_sub) * (X_r_sub.T @ X_r_sub)

    if cov_f is None:
        cov_f = (1 / X_f.shape[0]) * (X_f.T @ X_f)
    if M_f is None:
        M_f = (1 / X_f.shape[0]) * (X_f.T @ y_f)

    rhs = (w_r * cov_r_sub + w_f * cov_f) @ theta_ptr - w_f * M_f
    rhs_norm = max(np.linalg.norm(rhs), np.finfo(float).tiny)
    # GD on (ω_r Σ_r) θ = rhs  ⇒  safe step 1 / λ_max(ω_r Σ_r)
    A = w_r * cov_r_sub
    alpha = _lipschitz_gd_lr(A)

    if tol is None:
        n_for_tol = int(tol_n) if tol_n is not None else n_r_sub
        tol = _statistical_tolerance(p=p, n=n_for_tol, tol_scale=tol_scale)
    else:
        tol = float(tol)

    theta = theta_ptr.copy()
    iters = 0
    for _ in range(max_iter):
        res = A @ theta - rhs
        rel_resid = np.linalg.norm(res) / rhs_norm
        if rel_resid < tol:
            break
        theta = theta - alpha * res
        iters += 1

    if verbose:
        print(f"[ULS] lr={alpha:.4g} iters={iters}")
    return theta


def calculate_prediction_error(X_test, y_test, theta):
    y_pred = X_test @ theta
    mse = np.mean((y_pred - y_test) ** 2)
    return mse


def main():
    start_time = time.time()
    RUN_RETAIN_SUB = True
    RUN_TL = True
    TL_CV_K_FOLDS = 5
    TL_N_LAMBDA = 20
    TL_LAMBDA_MIN = 0.01
    TL_LAMBDA_MAX = 10.0

    data_file_2022 = '/data/xiejingyi/dataset/hesin_2022.csv'
    feature_cols = [
        'admimeth_uni', 'classpat_uni', 'intmanag_uni',
        'operstat', 'epitype', 'tretspef_uni',
    ]
    target_col = 'epidur'
    target_encoding_cols = ['tretspef_uni']

    df = pd.read_csv(data_file_2022)
    df_dedup_2022, _, _, _ = clean_data(df, feature_cols, target_col)
    dedup_data_file = '/data/xiejingyi/dataset/hesin_2022_deduplicated.csv'
    df_dedup_2022.to_csv(dedup_data_file, index=False)

    target_data_dedup = df_dedup_2022[target_col].values
    q1 = np.quantile(target_data_dedup, 0.25)
    q3 = np.quantile(target_data_dedup, 0.75)
    outlier_threshold = q3 + 1.5 * (q3 - q1)

    forget_mask = target_data_dedup > outlier_threshold
    df_forget = df_dedup_2022[forget_mask].reset_index(drop=True)
    df_retain = df_dedup_2022[~forget_mask].reset_index(drop=True)

    encoding_method = 'onehot'
    merge_admimeth_to_other = True
    N_REPEATS = 20
    subsample_ratios = [0.1, 0.2, 0.3]

    results_dir = '/data/xiejingyi/dataset/ukb_fold_results'
    os.makedirs(results_dir, exist_ok=True)

    combined_rows = []
    successful_repeats = 0
    total_attempts = 0

    while successful_repeats < N_REPEATS:
        total_attempts += 1
        repeat_idx = successful_repeats + 1
        repeat_start = time.time()

        print(f"\n=== Repeat {repeat_idx} / {N_REPEATS} (attempt {total_attempts}) ===")

        try:
            train_idx, test_idx = train_test_split(
                np.arange(len(df_retain)),
                test_size=0.2,
                random_state=RANDOM_SEED + repeat_idx,
                shuffle=True,
            )

            df_retain_train = df_retain.iloc[train_idx].reset_index(drop=True)
            df_retain_test = df_retain.iloc[test_idx].reset_index(drop=True)

            df_full_for_encoder = pd.concat([df_retain_train, df_forget], ignore_index=True)
            train_features_combined = prepare_features(
                df_full_for_encoder,
                feature_cols,
                target_encoding_cols=target_encoding_cols,
                merge_admimeth_to_other=merge_admimeth_to_other,
            )

            encoder, encoding_method = fit_encoder(
                train_features_combined,
                encoding_method,
                full_df=df_full_for_encoder,
                retain_df=df_retain_train,
                forget_df=df_forget,
                feature_cols=feature_cols,
                target_col=target_col,
                target_encoding_cols=target_encoding_cols,
                merge_admimeth_to_other=merge_admimeth_to_other,
            )

            X_retain_train, y_retain_train, _ = prepare_X_y(
                df_retain_train, feature_cols, target_col, encoder, encoding_method,
            )
            X_test, y_test, _ = prepare_X_y(
                df_retain_test, feature_cols, target_col, encoder, encoding_method,
            )
            X_forget, y_forget, _ = prepare_X_y(
                df_forget, feature_cols, target_col, encoder, encoding_method,
            )

            computation_times = {}

            t0 = time.time()
            theta_ptr = ols_estimator(
                np.vstack([X_retain_train, X_forget]),
                np.concatenate([y_retain_train, y_forget]),
                verbose=False,
            )
            computation_times['Pre-train'] = time.time() - t0

            t0 = time.time()
            theta_retrain = ols_estimator(X_retain_train, y_retain_train, verbose=False)
            computation_times['Retrain'] = time.time() - t0

            theta_retain_subs = {}
            theta_uls_dict = {}
            theta_tl_dict = {}

            n_retain_train = len(X_retain_train)
            n_forget = len(X_forget)
            w_r = n_retain_train / (n_retain_train + n_forget)
            w_f = n_forget / (n_retain_train + n_forget)
            train_indices_for_subsampling = np.arange(len(X_retain_train)).tolist()

            cov_f = (1 / X_forget.shape[0]) * (X_forget.T @ X_forget)
            M_f = (1 / X_forget.shape[0]) * (X_forget.T @ y_forget)

            if RUN_RETAIN_SUB or RUN_TL:
                for idx, ratio in enumerate(subsample_ratios):
                    sub_sample_size = max(1, int(n_retain_train * ratio))
                    sampled_indices = random.sample(train_indices_for_subsampling, sub_sample_size)
                    X_retain_sub = X_retain_train[sampled_indices]
                    y_retain_sub = y_retain_train[sampled_indices]
                    n_r_sub = int(X_retain_sub.shape[0])

                    if RUN_RETAIN_SUB:
                        name = f'Retain-Sub-{idx}'
                        t0 = time.time()
                        theta_sub = ols_estimator(
                            X_retain_sub, y_retain_sub, verbose=False, tol_n=n_r_sub,
                        )
                        computation_times[name] = time.time() - t0
                        theta_retain_subs[idx] = theta_sub

                        name = f'ULS-{idx}'
                        t0 = time.time()
                        theta_uls = uls_estimator(
                            w_r, w_f,
                            X_retain_sub, X_forget, y_forget, theta_ptr,
                            verbose=False, tol_n=n_r_sub,
                            cov_f=cov_f, M_f=M_f,
                        )
                        computation_times[name] = time.time() - t0
                        theta_uls_dict[idx] = theta_uls

                    if RUN_TL:
                        name = f'TL-{idx}'
                        t0 = time.time()
                        theta_tl, tl_lamb = tl_estimator_with_lambda_selection(
                            X_retain_sub, y_retain_sub, theta_ptr,
                            k_folds=TL_CV_K_FOLDS,
                            n_lambda_candidates=TL_N_LAMBDA,
                            tl_lambda_min=TL_LAMBDA_MIN,
                            tl_lambda_max=TL_LAMBDA_MAX,
                            cv_random_state=RANDOM_SEED + repeat_idx + 10000 * (idx + 1),
                            tol_n=n_r_sub,
                            Nr=n_r_sub, N_f=n_forget, w_f=w_f, delta=1,
                        )
                        computation_times[name] = time.time() - t0
                        theta_tl_dict[idx] = (theta_tl, tl_lamb)

            repeat_result = {}

            repeat_result['Pre-train'] = {
                'pred_error': calculate_prediction_error(X_test, y_test, theta_ptr),
                'comp_time': computation_times['Pre-train'],
            }
            repeat_result['Retrain'] = {
                'pred_error': calculate_prediction_error(X_test, y_test, theta_retrain),
                'comp_time': computation_times['Retrain'],
            }
            for idx, theta in theta_retain_subs.items():
                name = f'Retain-Sub-{idx}'
                repeat_result[name] = {
                    'pred_error': calculate_prediction_error(X_test, y_test, theta),
                    'comp_time': computation_times[name],
                }
            for idx, theta in theta_uls_dict.items():
                name = f'ULS-{idx}'
                repeat_result[name] = {
                    'pred_error': calculate_prediction_error(X_test, y_test, theta),
                    'comp_time': computation_times[name],
                }
            for idx, (theta, tl_lamb) in theta_tl_dict.items():
                name = f'TL-{idx}'
                repeat_result[name] = {
                    'pred_error': calculate_prediction_error(X_test, y_test, theta),
                    'comp_time': computation_times[name],
                    'tl_lambda': tl_lamb,
                }

            print(f"\n{'Estimator':<20} {'Pred Error (MSE)':<20} {'Comp Time (s)':<15}")
            for name, metrics in repeat_result.items():
                print(f"{name:<20} {metrics['pred_error']:<20.6f} {metrics['comp_time']:<15.6f}")

            repeat_duration = time.time() - repeat_start
            print(f"Repeat {repeat_idx} completed in {repeat_duration:.2f} seconds")

            for name, metrics in repeat_result.items():
                combined_rows.append({
                    'repeat': repeat_idx,
                    'estimator': name,
                    'pred_error': metrics['pred_error'],
                    'comp_time': metrics['comp_time'],
                    'tl_lambda': metrics.get('tl_lambda', 0),
                })

            df_repeat = pd.DataFrame([
                {
                    'repeat': repeat_idx,
                    'estimator': name,
                    'pred_error': metrics['pred_error'],
                    'comp_time': metrics['comp_time'],
                    'tl_lambda': metrics.get('tl_lambda', 0),
                }
                for name, metrics in repeat_result.items()
            ])
            repeat_csv = os.path.join(
                results_dir, f'ukb_estimation_results_clean_repeat_{repeat_idx}.csv'
            )
            df_repeat.to_csv(repeat_csv, index=False)
            print(f"Repeat {repeat_idx} results saved to: {repeat_csv}")

            successful_repeats += 1

        except Exception:
            import traceback
            traceback.print_exc()
            continue

    if combined_rows:
        df_all = pd.DataFrame(combined_rows)
        combined_csv = os.path.join(results_dir, 'ukb_estimation_results_clean_all_repeats.csv')
        df_all.to_csv(combined_csv, index=False)
        print(f"Combined results saved to: {combined_csv}")

        grouped = df_all.groupby('estimator')
        print(f"\n{'Estimator':<20} {'Pred Mean':<15} {'Pred Std':<15} {'Comp Mean':<15}")
        for estimator, group in grouped:
            pred_mean = group['pred_error'].mean()
            pred_std = group['pred_error'].std(ddof=0)
            comp_mean = group['comp_time'].mean()
            print(f"{estimator:<20} {pred_mean:<15.6f} {pred_std:<15.6f} {comp_mean:<15.6f}")

    print(f"\nTotal wall time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
