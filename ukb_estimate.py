import pandas as pd
import numpy as np
import random
import time
import os
from numpy.linalg import inv
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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

    if len(target_encoding_cols) > 0:
        X_full_target_enc = full_features[target_encoding_cols].values.astype(float)
        X_retain_target_enc = retain_features[target_encoding_cols].values.astype(float)
        X_forget_target_enc = forget_features[target_encoding_cols].values.astype(float)
    else:
        X_full_target_enc = np.zeros((len(full_features), 0))
        X_retain_target_enc = np.zeros((len(retain_features), 0))
        X_forget_target_enc = np.zeros((len(forget_features), 0))

    col_nonzero_full = (X_full_onehot != 0).any(axis=0)
    col_nonzero_retain = (X_retain_onehot != 0).any(axis=0)
    col_nonzero_forget = (X_forget_onehot != 0).any(axis=0)

    nonzero_cols = col_nonzero_full & col_nonzero_retain & col_nonzero_forget
    nonzero_col_indices = np.where(nonzero_cols)[0]

    X_full_nonzero = X_full_onehot[:, nonzero_col_indices]
    y_full_for_lasso = full_df_valid[target_col].values

    alphas = np.logspace(-2, 2, 50)
    lasso_cv = LassoCV(alphas=alphas, cv=10, random_state=RANDOM_SEED, n_jobs=-1, max_iter=10000)
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


def ols_estimator(X, y):
    XTX = X.T @ X
    theta = np.linalg.inv(XTX) @ (X.T @ y)
    return theta


def uls_estimator(w_r, w_f, X_r_sub, X_f, y_f, theta_ptr):
    cov_r_sub = (1 / X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
    cov_f = (1 / X_f.shape[0]) * (X_f.T @ X_f)
    cov_ptr_sub = w_r * cov_r_sub + w_f * cov_f
    M_f = (1 / X_f.shape[0]) * (X_f.T @ y_f)

    matrix_to_invert = w_r * cov_r_sub
    theta = np.linalg.inv(matrix_to_invert) @ (cov_ptr_sub @ theta_ptr - w_f * M_f)
    return theta


def _compute_theta_for_lambda(X_r_sub, y_r_sub, X_f, y_f, lamb, estimator_type='gd',
                              w_r=None, w_f=None, theta_ptr=None):
    if estimator_type == 'gd':
        cov_r_sub = (1 / X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
        cov_f = (1 / X_f.shape[0]) * (X_f.T @ X_f)
        A = inv(-cov_f + lamb * cov_r_sub)
        b = -(1 / X_f.shape[0]) * (X_f.T @ y_f) + (lamb / X_r_sub.shape[0]) * (X_r_sub.T @ y_r_sub)
        return A @ b
    elif estimator_type == 'uls_plus':
        cov_r_sub = (1 / X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
        cov_f = (1 / X_f.shape[0]) * (X_f.T @ X_f)
        cov_ptr = w_f * cov_f + w_r * cov_r_sub
        A = inv((w_r + lamb) * cov_r_sub)
        b = (-(w_f / X_f.shape[0]) * (X_f.T @ y_f)
             + (lamb / X_r_sub.shape[0]) * (X_r_sub.T @ y_r_sub)
             + cov_ptr @ theta_ptr)
        return A @ b
    else:
        raise ValueError(f"Unknown estimator_type: {estimator_type}")


def _compute_retain_loss(X_r_sub, y_r_sub, theta, lamb, n_r_sub):
    residuals = y_r_sub - X_r_sub @ theta
    return (1 / n_r_sub) * (residuals ** 2).sum()


def _compute_full_loss(X_r_sub, y_r_sub, X_f, y_f, theta, lamb, n_r_sub, n_f):
    residuals_f = y_f - X_f @ theta
    forget_loss = -(1 / n_f) * (residuals_f ** 2).sum()
    return forget_loss


def _cross_validate_lambda(X_r_sub, y_r_sub, X_f, y_f, lambda_candidates, k_folds=5,
                           use_full_loss=False, estimator_type='gd', w_r=None, w_f=None,
                           theta_ptr=None, random_state=42):
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

        X_r_train = X_r_sub[train_indices]
        y_r_train = y_r_sub[train_indices]
        X_r_val = X_r_sub[val_indices]
        y_r_val = y_r_sub[val_indices]

        for lamb in lambda_candidates:
            try:
                if estimator_type == 'uls_plus':
                    theta = _compute_theta_for_lambda(
                        X_r_train, y_r_train, X_f, y_f, lamb,
                        estimator_type='uls_plus', w_r=w_r, w_f=w_f, theta_ptr=theta_ptr,
                    )
                else:
                    theta = _compute_theta_for_lambda(
                        X_r_train, y_r_train, X_f, y_f, lamb,
                        estimator_type=estimator_type,
                    )
                n_r_val = len(X_r_val)
                if use_full_loss:
                    score = _compute_full_loss(X_r_val, y_r_val, X_f, y_f, theta, lamb, n_r_val, n_f)
                else:
                    score = _compute_retain_loss(X_r_val, y_r_val, theta, lamb, n_r_val)
                cv_scores[lamb].append(score)
            except Exception:
                pass

    mean_scores = {lamb: np.mean(scores) for lamb, scores in cv_scores.items() if len(scores) > 0}

    if len(mean_scores) == 0:
        return lambda_candidates[0], cv_scores

    best_lambda = min(mean_scores, key=mean_scores.get)
    return best_lambda, mean_scores


def graddiff_estimator(X_r_sub, y_r_sub, X_f, y_f, w_f, w_r_sub, p, delta=1,
                       use_cv=True, k_folds=5, n_lambda_candidates=50,
                       use_full_loss=False, cv_random_state=42):
    """Returns (theta, lambda, cv_time) where cv_time is the time spent on CV only."""
    cov_r_sub = (1 / X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
    cov_f = (1 / X_f.shape[0]) * (X_f.T @ X_f)
    n_r_sub = X_r_sub.shape[0]
    n_f = X_f.shape[0]

    eig_f = np.linalg.eigvalsh(cov_f)
    max_eig_f = np.max(eig_f)
    eig_r_sub = np.linalg.eigvalsh(cov_r_sub)
    min_eig_r_sub = np.min(eig_r_sub)

    c1 = np.sqrt(w_r_sub / w_f) + np.sqrt(n_r_sub / p) * delta
    c2 = 2 * max_eig_f / min_eig_r_sub
    lambda_min = c2

    cv_time = 0.0
    if use_cv:
        u_max = 1.0 / lambda_min
        u_min = 1e-4
        u_candidates = np.logspace(np.log10(u_min), np.log10(u_max), n_lambda_candidates)
        lambda_candidates = 1.0 / u_candidates

        cv_start = time.time()
        best_lambda, _ = _cross_validate_lambda(
            X_r_sub, y_r_sub, X_f, y_f, lambda_candidates,
            k_folds=k_folds, use_full_loss=use_full_loss,
            random_state=cv_random_state,
        )
        cv_time = time.time() - cv_start
        lamb = best_lambda
    else:
        lamb = max(c1, c2)

    A = inv(-cov_f + lamb * cov_r_sub)
    b = -(1 / n_f) * (X_f.T @ y_f) + (lamb / n_r_sub) * (X_r_sub.T @ y_r_sub)
    theta = A @ b
    return theta, lamb, cv_time


def uls_plus_estimator(w_r, w_f, X_r_sub, y_r_sub, X_f, y_f, theta_ptr, delta=1, c=1,
                       use_cv=True, k_folds=5, n_lambda_candidates=50,
                       use_full_loss=False, cv_random_state=42):
    """Returns (theta, lambda, cv_time) where cv_time is the time spent on CV only."""
    n_r_sub = X_r_sub.shape[0]
    n_f = X_f.shape[0]

    cv_time = 0.0
    if use_cv:
        u_max = 1e4
        u_min = 1e-4
        u_candidates = np.logspace(np.log10(u_min), np.log10(u_max), n_lambda_candidates)
        lambda_candidates = 1.0 / u_candidates

        cv_start = time.time()
        best_lambda, _ = _cross_validate_lambda(
            X_r_sub, y_r_sub, X_f, y_f, lambda_candidates,
            k_folds=k_folds, use_full_loss=use_full_loss,
            estimator_type='uls_plus', w_r=w_r, w_f=w_f, theta_ptr=theta_ptr,
            random_state=cv_random_state,
        )
        cv_time = time.time() - cv_start
        lamb = best_lambda
    else:
        lamb = c * w_r * w_f * delta

    cov_r_sub = (1 / n_r_sub) * (X_r_sub.T @ X_r_sub)
    cov_f = (1 / n_f) * (X_f.T @ X_f)
    cov_ptr = w_f * cov_f + w_r * cov_r_sub
    A = inv((w_r + lamb) * cov_r_sub)
    b = (-(w_f / n_f) * (X_f.T @ y_f)
         + (lamb / n_r_sub) * (X_r_sub.T @ y_r_sub)
         + cov_ptr @ theta_ptr)
    theta = A @ b
    return theta, lamb, cv_time


def uls_inference(v, theta_uls, n_retain, n_retain_sub, X_r_sub, y_r_sub, theta_ptr, theta_r,
                  z_alpha=1.96):
    cov_r_sub = (1 / X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
    inv_cov_r_sub = inv(cov_r_sub)
    f_i = 0
    g_i = 0
    w_1 = 1 / (n_retain ** 2)
    w_2 = (n_retain_sub - n_retain) / n_retain_sub
    w_3 = (n_retain - n_retain_sub) / ((n_retain ** 2) * n_retain_sub)
    for i in range(X_r_sub.shape[0]):
        x_i = X_r_sub[i, :].reshape(-1, 1)
        a_i = v.T @ inv_cov_r_sub @ X_r_sub[i, :] * (y_r_sub[i] - X_r_sub[i, :].T @ theta_uls)
        mat = x_i @ x_i.T - cov_r_sub
        b_i = v.T @ inv_cov_r_sub @ mat @ (theta_uls - theta_ptr)
        f_i += (a_i + w_2 * b_i) ** 2
        g_i += (a_i + b_i) ** 2
    V_r = w_1 * f_i + (w_3 * g_i)
    se_r = np.sqrt(V_r)
    psi_r = float(v @ theta_uls)
    return psi_r, z_alpha * se_r


def retain_sub_inference(v, p, theta_retain_sub, X_r_sub, y_r_sub, theta_r, z_alpha=1.96):
    n_retain_sub = X_r_sub.shape[0]

    residuals = y_r_sub - X_r_sub @ theta_retain_sub
    sigma_sq_hat = np.sum(residuals ** 2) / (n_retain_sub - p)

    inv_XTX = inv(X_r_sub.T @ X_r_sub)
    cov_theta = sigma_sq_hat * inv_XTX

    se = np.sqrt(v.T @ cov_theta @ v)
    psi_retain_sub = float(v.T @ theta_retain_sub)
    return psi_retain_sub, z_alpha * se


def calculate_prediction_error(X_test, y_test, theta):
    y_pred = X_test @ theta
    mse = np.mean((y_pred - y_test) ** 2)
    return mse


def main():
    start_time = time.time()
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
    fold_count = 20
    subsample_ratios = [0.1, 0.2, 0.3]

    results_dir = '/data/xiejingyi/dataset/ukb_fold_results'
    os.makedirs(results_dir, exist_ok=True)

    combined_rows = []
    successful_folds = 0
    total_attempts = 0

    while successful_folds < fold_count:
        total_attempts += 1
        fold_idx = successful_folds + 1
        fold_start = time.time()

        print(f"\n=== Fold {fold_idx} / {fold_count} (attempt {total_attempts}) ===")

        try:
            train_idx_fold, test_idx_fold = train_test_split(
                np.arange(len(df_retain)),
                test_size=0.2,
                random_state=RANDOM_SEED + fold_idx,
                shuffle=True,
            )

            df_retain_train_fold = df_retain.iloc[train_idx_fold].reset_index(drop=True)
            df_retain_test_fold = df_retain.iloc[test_idx_fold].reset_index(drop=True)

            df_full_for_encoder = pd.concat([df_retain_train_fold, df_forget], ignore_index=True)
            train_features_combined = prepare_features(
                df_full_for_encoder, feature_cols,
                target_encoding_cols=target_encoding_cols,
                merge_admimeth_to_other=merge_admimeth_to_other,
            )

            encoder, encoding_method = fit_encoder(
                train_features_combined, encoding_method,
                full_df=df_full_for_encoder, retain_df=df_retain_train_fold,
                forget_df=df_forget, feature_cols=feature_cols,
                target_col=target_col,
                target_encoding_cols=target_encoding_cols,
                merge_admimeth_to_other=merge_admimeth_to_other,
            )

            X_retain_train_fold, y_retain_train_fold, _ = prepare_X_y(
                df_retain_train_fold, feature_cols, target_col, encoder, encoding_method,
            )
            X_test_fold, y_test_fold, _ = prepare_X_y(
                df_retain_test_fold, feature_cols, target_col, encoder, encoding_method,
            )
            X_forget_fold, y_forget_fold, _ = prepare_X_y(
                df_forget, feature_cols, target_col, encoder, encoding_method,
            )

            operstat_idx = encoder.get('operstat_idx')
            if operstat_idx is None:
                raise ValueError("operstat covariate index not found in encoder")

            p = X_retain_train_fold.shape[1]
            selected_feature_names = encoder.get('selected_feature_names', [])
            if operstat_idx >= len(selected_feature_names):
                raise ValueError(
                    f"operstat covariate index ({operstat_idx}) is out of range "
                    f"for selected features (len={len(selected_feature_names)})"
                )

            tretspef_uni_name = selected_feature_names[operstat_idx]
            if tretspef_uni_name != 'operstat_One or more operative procedures performed':
                raise ValueError(
                    f"Expected operstat at index {operstat_idx}, but found {tretspef_uni_name}"
                )

            operstat_idx_in_X = operstat_idx + 1
            if operstat_idx_in_X >= p:
                raise ValueError(
                    f"operstat covariate index in X ({operstat_idx_in_X}) is out of range (p={p})"
                )

            v = np.zeros(p)
            v[operstat_idx_in_X] = 1.0

            computation_times = {}

            # Pre-train (OLS on full = retain_train + forget)
            t_start = time.time()
            theta_ptr = ols_estimator(
                np.vstack([X_retain_train_fold, X_forget_fold]),
                np.concatenate([y_retain_train_fold, y_forget_fold]),
            )
            computation_times['Pre-train'] = time.time() - t_start

            # Retrain (OLS on retain_train only)
            t_start = time.time()
            theta_retrain = ols_estimator(X_retain_train_fold, y_retain_train_fold)
            computation_times['Retrain'] = time.time() - t_start

            theta_retain_subs = {}
            theta_uls_dict = {}
            theta_graddiff_dict = {}
            theta_uls_plus_dict = {}

            n_retain_train = len(X_retain_train_fold)
            n_forget = len(X_forget_fold)
            w_r = n_retain_train / (n_retain_train + n_forget)
            w_f = n_forget / (n_retain_train + n_forget)
            train_indices_for_subsampling = np.arange(len(X_retain_train_fold)).tolist()

            for idx, ratio in enumerate(subsample_ratios):
                sub_sample_size = max(1, int(n_retain_train * ratio))
                sampled_indices = random.sample(train_indices_for_subsampling, sub_sample_size)
                X_retain_sub = X_retain_train_fold[sampled_indices]
                y_retain_sub = y_retain_train_fold[sampled_indices]

                # Retain-Sub
                t_start = time.time()
                theta_sub = ols_estimator(X_retain_sub, y_retain_sub)
                computation_times[f'Retain-Sub-{idx}'] = time.time() - t_start
                theta_retain_subs[idx] = theta_sub

                # ULS
                t_start = time.time()
                theta_uls = uls_estimator(w_r, w_f, X_retain_sub, X_forget_fold, y_forget_fold, theta_ptr)
                computation_times[f'ULS-{idx}'] = time.time() - t_start
                theta_uls_dict[idx] = theta_uls

                # Graddiff
                w_r_sub = sub_sample_size / (n_retain_train + n_forget)
                t_start = time.time()
                theta_graddiff, lambda_graddiff, cv_time_gd = graddiff_estimator(
                    X_retain_sub, y_retain_sub, X_forget_fold, y_forget_fold, w_f, w_r_sub, p,
                )
                total_time_gd = time.time() - t_start
                computation_times[f'Graddiff-{idx}'] = total_time_gd - cv_time_gd
                theta_graddiff_dict[idx] = (theta_graddiff, lambda_graddiff)

                t_start = time.time()
                theta_uls_plus, lambda_uls_plus, cv_time_up = uls_plus_estimator(
                    w_r, w_f, X_retain_sub, y_retain_sub, X_forget_fold, y_forget_fold, theta_ptr,
                )
                total_time_up = time.time() - t_start
                computation_times[f'ULS-Plus-{idx}'] = total_time_up - cv_time_up
                theta_uls_plus_dict[idx] = (theta_uls_plus, lambda_uls_plus)

                # Inference
                n_retain_sub = len(X_retain_sub)
                try:
                    uls_psi, uls_se = uls_inference(
                        v, theta_uls, n_retain_train, n_retain_sub,
                        X_retain_sub, y_retain_sub, theta_ptr, theta_retrain, z_alpha=1.96,
                    )
                    ols_psi, ols_se = retain_sub_inference(
                        v, p, theta_sub, X_retain_sub, y_retain_sub, theta_retrain, z_alpha=1.96,
                    )

                    print(f"  Ratio {ratio:.2f} - ULS inference: CI = {uls_psi:.6f} +- {uls_se:.6f}")
                    print(f"  Ratio {ratio:.2f} - Retain-sub inference: CI = {ols_psi:.6f} +- {ols_se:.6f}")

                except Exception:
                    import traceback
                    traceback.print_exc()

            # Collect fold results
            fold_result = {}

            fold_result['Pre-train'] = {
                'pred_error': calculate_prediction_error(X_test_fold, y_test_fold, theta_ptr),
                'comp_time': computation_times['Pre-train'],
            }

            fold_result['Retrain'] = {
                'pred_error': calculate_prediction_error(X_test_fold, y_test_fold, theta_retrain),
                'comp_time': computation_times['Retrain'],
            }

            for idx, theta in theta_retain_subs.items():
                key = f'Retain-Sub-{idx}'
                fold_result[key] = {
                    'pred_error': calculate_prediction_error(X_test_fold, y_test_fold, theta),
                    'comp_time': computation_times[key],
                }

            for idx, theta in theta_uls_dict.items():
                key = f'ULS-{idx}'
                fold_result[key] = {
                    'pred_error': calculate_prediction_error(X_test_fold, y_test_fold, theta),
                    'comp_time': computation_times[key],
                }

            for idx, (theta, lamb) in theta_graddiff_dict.items():
                key = f'Graddiff-{idx}'
                fold_result[key] = {
                    'pred_error': calculate_prediction_error(X_test_fold, y_test_fold, theta),
                    'comp_time': computation_times[key],
                    'lambda': lamb,
                }

            for idx, (theta, lamb) in theta_uls_plus_dict.items():
                key = f'ULS-Plus-{idx}'
                fold_result[key] = {
                    'pred_error': calculate_prediction_error(X_test_fold, y_test_fold, theta),
                    'comp_time': computation_times[key],
                    'lambda': lamb,
                }

            print(f"\n{'Estimator':<20} {'Pred Error (MSE)':<20} {'Comp Time (s)':<15}")
            for name, metrics in fold_result.items():
                print(f"{name:<20} {metrics['pred_error']:<20.6f} {metrics['comp_time']:<15.6f}")

            for name, metrics in fold_result.items():
                combined_rows.append({
                    'fold': fold_idx,
                    'estimator': name,
                    'pred_error': metrics['pred_error'],
                    'comp_time': metrics['comp_time'],
                    'lambda': metrics.get('lambda', 0),
                })

            df_fold = pd.DataFrame([
                {
                    'fold': fold_idx,
                    'estimator': name,
                    'pred_error': metrics['pred_error'],
                    'comp_time': metrics['comp_time'],
                    'lambda': metrics.get('lambda', 0),
                }
                for name, metrics in fold_result.items()
            ])
            fold_csv = os.path.join(results_dir, f'ukb_estimation_results_fold_{fold_idx}_v4.csv')
            df_fold.to_csv(fold_csv, index=False)

            successful_folds += 1

        except Exception:
            continue

    if combined_rows:
        df_all = pd.DataFrame(combined_rows)
        combined_csv = os.path.join(results_dir, 'ukb_estimation_results_all_folds_v4.csv')
        df_all.to_csv(combined_csv, index=False)

        grouped = df_all.groupby('estimator')
        print(f"\n{'Estimator':<20} {'Pred Mean':<15} {'Pred Std':<15} {'Comp Mean':<15}")
        for estimator, group in grouped:
            pred_mean = group['pred_error'].mean()
            pred_std = group['pred_error'].std(ddof=0)
            comp_mean = group['comp_time'].mean()
            print(f"{estimator:<20} {pred_mean:<15.6f} {pred_std:<15.6f} {comp_mean:<15.6f}")


if __name__ == "__main__":
    main()