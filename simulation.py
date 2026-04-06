import numpy as np
from numpy.linalg import inv, norm, solve
from numpy.matlib import True_
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
import time

np.random.seed(42)

def generate_linear_data(n, p, theta, noise_sigma=1.0, seed=42):
    '''
    generate (X_f, y_f), (X_r, y_r)
    '''
    cov = make_spd_matrix(p, random_state=seed)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    eps = np.random.normal(0, noise_sigma, n)
    y = X @ theta + eps

    return X, y

def generate_retain_data(n, p, theta, noise_sigma=1.0, seed=42):
    '''
    generate retain data with identity covariance matrix
    '''
    cov = np.eye(p)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    eps = np.random.normal(0, noise_sigma, n)
    y = X @ theta + eps

    return X, y

def generate_forget_data(n, p, theta, noise_sigma=1.0, seed=42):
    '''
    generate forget data with covariance matrix: diagonal=1, off-diagonal=0.1
    '''
    # cov = np.eye(p) + (np.ones((p, p)) - np.eye(p)) * 0.1
    idx = np.arange(p)
    cov = 0.3 ** np.abs(idx[:, None] - idx[None, :])
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    eps = np.random.normal(0, noise_sigma, n)
    y = X @ theta + eps

    return X, y

def ols_estimator(X, y):
    '''
    OLS estimator of theta_ptr, theta_r, theta_subsampleing_r
    '''
    return inv(X.T @ X) @ (X.T @ y)
 
def transfer_learning_estimator():
    return 

def check_bias_term(w_r, w_f, X_r, X_f, theta_r, theta_f):
    print('check bias term of ptr, p = ', X_r.shape[1])
    cov_r = (1/X_r.shape[0]) * (X_r.T @ X_r)
    cov_f = (1/X_f.shape[0]) * (X_f.T @ X_f)
    cov_ptr = w_r * cov_r + w_f * cov_f
    C = norm(inv(cov_ptr) @ cov_f @ (theta_r - theta_f))
    print('bias constant: ', C)

def verify_gd_vs_closed(w_r, w_f, X_r_sub, X_f, y_f, theta_ptr, theta_true,
                        max_iter=500, alpha=0.05):
    cov_r_sub = (1/X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
    cov_f = (1/X_f.shape[0]) * (X_f.T @ X_f)
    cov_ptr_sub = w_r * cov_r_sub + w_f * cov_f
    M_f = (1/X_f.shape[0]) * (X_f.T @ y_f)
      
    theta_closed = np.linalg.inv(w_r * cov_r_sub) @ (cov_ptr_sub @ theta_ptr - w_f * M_f)
    err_closed = np.linalg.norm(theta_closed - theta_true)
    
    print(f"Closed-form parameter error = {err_closed:.6f}")
    
    theta_gd = theta_ptr.copy()
    errors = []
    
    for t in range(max_iter):
        G = cov_ptr_sub @ theta_ptr - w_f * M_f - w_r * cov_r_sub @ theta_gd
        theta_gd = theta_gd + alpha * G  
        
        err_t = np.linalg.norm(theta_gd - theta_true)
        errors.append(err_t)
    
    print(f"GD final error after {max_iter} steps = {errors[-1]:.6f}")


def uls_estimator(w_r, w_f, X_r_sub, X_f, y_f, theta_ptr):
    cov_r_sub = (1/X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
    cov_f = (1/X_f.shape[0]) * (X_f.T @ X_f)
    cov_ptr_sub = w_r * cov_r_sub + w_f * cov_f
    M_f = (1/X_f.shape[0]) * (X_f.T @ y_f)
    return inv(w_r * cov_r_sub) @ (cov_ptr_sub @ theta_ptr - w_f * M_f)


def gradient_descent_uls_estimator(w_r, w_f, X_r_sub, X_f, y_f, theta_ptr, n_iterations=500, alpha=0.05):
    '''Gradient descent algorithm for the ULS estimate (when p is large)'''

    cov_r_sub = (1/X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)  
    cov_f = (1/X_f.shape[0]) * (X_f.T @ X_f)     
    cov_ptr_sub = w_r * cov_r_sub + w_f * cov_f          
    M_f = (1/X_f.shape[0]) * (X_f.T @ y_f)                   
    
    theta_uls = theta_ptr.copy()
    
    for t in range(n_iterations):
        G = cov_ptr_sub @ theta_ptr - w_f * M_f - w_r * cov_r_sub @ theta_uls
        theta_uls = theta_uls + alpha * G
    
    return theta_uls

def _compute_theta_for_lambda(X_r_sub, y_r_sub, X_f, y_f, lamb, estimator_type='gd', 
                              w_r=None, w_f=None, theta_ptr=None):
    """
    Compute theta for a given lambda value.
    """
    if estimator_type == 'gd':
        cov_r_sub = (1/X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
        cov_f = (1/X_f.shape[0]) * (X_f.T @ X_f)
        A = inv(-cov_f + lamb * cov_r_sub)
        b = -(1/X_f.shape[0]) * (X_f.T @ y_f) + (lamb / X_r_sub.shape[0]) * (X_r_sub.T @ y_r_sub)
        return A @ b
    elif estimator_type == 'uls_plus':
        cov_r_sub = (1/X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
        cov_f = (1/X_f.shape[0]) * (X_f.T @ X_f)
        cov_ptr = w_f * cov_f + w_r * cov_r_sub
        A = inv((w_r + lamb) * cov_r_sub)
        b = -(w_f/X_f.shape[0]) * (X_f.T @ y_f) + (lamb / X_r_sub.shape[0]) * (X_r_sub.T @ y_r_sub) + cov_ptr @ theta_ptr
        return A @ b
    else:
        raise ValueError(f"Unknown estimator_type: {estimator_type}")


def _compute_retain_loss(X_r_sub, y_r_sub, theta, lamb, n_r_sub):
    residuals = y_r_sub - X_r_sub @ theta
    return (1 / n_r_sub) * (residuals ** 2).sum()


def _compute_full_loss(X_r_sub, y_r_sub, X_f, y_f, theta, lamb, n_r_sub, n_f):
    """
    Compute full loss: -(1/N_f) * ||X_f @ theta - y_f||^2 + (lambda / n_r_sub) * ||X_r_sub @ theta - y_r_sub||^2
    """
    residuals_r = y_r_sub - X_r_sub @ theta
    residuals_f = y_f - X_f @ theta
    retain_loss = (1 / n_r_sub) * (residuals_r ** 2).sum()
    forget_loss = -(1 / n_f) * (residuals_f ** 2).sum()
    return forget_loss


def _cross_validate_lambda(X_r_sub, y_r_sub, X_f, y_f, lambda_candidates, k_folds=5, 
                           use_full_loss=False, estimator_type='gd', w_r=None, w_f=None, theta_ptr=None,
                           random_state=42):

    n_r_sub = X_r_sub.shape[0]
    n_f = X_f.shape[0]
    
    rng = np.random.RandomState(random_state)
    
    indices = np.arange(n_r_sub)
    rng.shuffle(indices)  
    fold_size = n_r_sub // k_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(k_folds)]
    
    if len(indices) > k_folds * fold_size:
        folds[-1] = np.concatenate([folds[-1], indices[k_folds*fold_size:]])
    
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
                    theta = _compute_theta_for_lambda(X_r_train, y_r_train, X_f, y_f, lamb, 
                                                     estimator_type='uls_plus', w_r=w_r, w_f=w_f, theta_ptr=theta_ptr)
                else:
                    theta = _compute_theta_for_lambda(X_r_train, y_r_train, X_f, y_f, lamb, 
                                                     estimator_type=estimator_type)
                n_r_val = len(X_r_val)
                if use_full_loss:
                    score = _compute_full_loss(X_r_val, y_r_val, X_f, y_f, theta, lamb, 
                                                n_r_val, n_f)
                else:
                    score = _compute_retain_loss(X_r_val, y_r_val, theta, lamb, n_r_val)
                
                cv_scores[lamb].append(score)
            except Exception as e:
                print(e.message)

    mean_scores = {lamb: np.mean(scores) for lamb, scores in cv_scores.items() if len(scores) > 0}
    
    if len(mean_scores) == 0:
        # Fallback: use the first lambda candidate
        return lambda_candidates[0], cv_scores
    
    best_lambda = min(mean_scores, key=mean_scores.get)
    
    return best_lambda, mean_scores


def graddiff_estimator(X_r_sub, y_r_sub, X_f, y_f, w_f, w_r_sub, p, delta, 
                      use_cv=True, k_folds=5, n_lambda_candidates=50, 
                      use_full_loss=False, cv_random_state=42):
    cov_r_sub = (1/X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
    cov_f = (1/X_f.shape[0]) * (X_f.T @ X_f)
    n_r_sub = X_r_sub.shape[0]
    n_f = X_f.shape[0]
    
    eig_f = np.linalg.eigvalsh(cov_f)
    max_eig_f = np.max(eig_f)
    eig_r_sub = np.linalg.eigvalsh(cov_r_sub)
    min_eig_r_sub = np.min(eig_r_sub)
    
    c1 = np.sqrt(w_r_sub / w_f) + np.sqrt(n_r_sub / p) * delta
    c2 = 2 * max_eig_f / min_eig_r_sub
    lambda_min = c2
    
    if use_cv:
        u_max = 1.0 / lambda_min
        u_min = 1e-4  
        
        u_candidates = np.logspace(np.log10(u_min), np.log10(u_max), n_lambda_candidates)
        lambda_candidates = 1.0 / u_candidates
        
        best_lambda, cv_scores = _cross_validate_lambda(
            X_r_sub, y_r_sub, X_f, y_f, lambda_candidates, 
            k_folds=k_folds, use_full_loss=use_full_loss,
            random_state=cv_random_state
        )
        
        # print(f"Best lambda from CV: {best_lambda:.2e}"
        #       f"(score: {cv_scores[best_lambda]:.6f})")
        
        lamb = best_lambda
    else:
        lamb = max(c1, c2)
    
    A = inv(-cov_f + lamb * cov_r_sub)
    b = -(1/n_f) * (X_f.T @ y_f) + (lamb / n_r_sub) * (X_r_sub.T @ y_r_sub)
    theta = A @ b
    return theta, lamb


def uls_plus_estimator(w_r, w_f, X_r_sub, y_r_sub, X_f, y_f, theta_ptr, delta, c=1,
                       use_cv=True, k_folds=5, n_lambda_candidates=50, 
                       use_full_loss=False, cv_random_state=42):
    """
    ULS Plus estimator with optional cross-validation for lambda selection.
    """
    n_r_sub = X_r_sub.shape[0]
    n_f = X_f.shape[0]
    
    if use_cv:
        u_max = 1e4
        u_min = 1e-4  
        u_candidates = np.logspace(np.log10(u_min), np.log10(u_max), n_lambda_candidates)
        lambda_candidates = 1.0 / u_candidates
        
        best_lambda, cv_scores = _cross_validate_lambda(
            X_r_sub, y_r_sub, X_f, y_f, lambda_candidates, 
            k_folds=k_folds, use_full_loss=use_full_loss, 
            estimator_type='uls_plus', w_r=w_r, w_f=w_f, theta_ptr=theta_ptr,
            random_state=cv_random_state
        )
        
        # print(f"ULS Plus - Best lambda from CV: {best_lambda:.2e} "
        #       f"(score: {cv_scores[best_lambda]:.6f})")
        
        lamb = best_lambda
    else:
        lamb = c * w_r * w_f * delta
    
    cov_r_sub = (1/n_r_sub) * (X_r_sub.T @ X_r_sub)
    cov_f = (1/n_f) * (X_f.T @ X_f)
    cov_ptr = w_f * cov_f + w_r * cov_r_sub
    A = inv((w_r + lamb) * cov_r_sub)
    b = -(w_f/n_f) * (X_f.T @ y_f) + (lamb / n_r_sub) * (X_r_sub.T @ y_r_sub) + cov_ptr @ theta_ptr
    theta = A @ b
    return theta, lamb


def uls_inference(p, theta_uls, n_retain, n_retain_sub, X_r_sub, y_r_sub, theta_ptr, theta_r, z_alpha=1.96):
    v = np.zeros(p)
    v[0] = 1
    # v = np.ones(p)
    # v = v / norm(v)
    cov_r_sub = (1/X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
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
    true_val = float(v @ theta_r)
    L = psi_r - z_alpha * se_r
    U = psi_r + z_alpha * se_r

    covered = int(L <= true_val <= U)
    width = U - L
    return covered, width, se_r

def retain_sub_inference(p, theta_retain_sub, X_r_sub, y_r_sub, theta_r, z_alpha=1.96):
    v = np.zeros(p)
    v[0] = 1 
    
    n_retain_sub = X_r_sub.shape[0]
    
    residuals = y_r_sub - X_r_sub @ theta_retain_sub
    sigma_sq_hat = np.sum(residuals ** 2) / (n_retain_sub - p)  
    
    inv_XTX = inv(X_r_sub.T @ X_r_sub)
    cov_theta = sigma_sq_hat * inv_XTX
    
    se = np.sqrt(v.T @ cov_theta @ v)
    
    psi_retain_sub = float(v.T @ theta_retain_sub)
    true_val = float(v.T @ theta_r)
    
    L = psi_retain_sub - z_alpha * se
    U = psi_retain_sub + z_alpha * se
    
    covered = int(L <= true_val <= U)
    width = U - L
    return covered, width, se


def run_experiment(theta_r, theta_f, n_retain, n_forget, n_retain_sub, p, delta, rep_idx, noise_sigma=1.0, seed=42):
    # theta_r = np.random.normal(0, 1, p)
    # direction = np.random.normal(0, 1, p)
    # direction = direction / norm(direction)
    # theta_f = theta_r + delta * direction

    # X_r, y_r = generate_linear_data(n_retain, p, theta_r, noise_sigma, seed)
    # X_f, y_f = generate_linear_data(n_forget, p, theta_f, noise_sigma, seed)
    X_r, y_r = generate_retain_data(n_retain, p, theta_r, noise_sigma, seed)
    X_f, y_f = generate_forget_data(n_forget, p, theta_f, noise_sigma, seed)
    X_r_sub, y_r_sub = X_r[:n_retain_sub], y_r[:n_retain_sub]


    theta_ptr = ols_estimator(np.vstack([X_r, X_f]), np.concatenate([y_r, y_f]))
    theta_retrain = ols_estimator(X_r, y_r)
    theta_retain_sub = ols_estimator(X_r_sub, y_r_sub)

    # theta_tl = transfer_learning_estimator()
    w_r = n_retain / (n_retain + n_forget)
    w_f = n_forget / (n_retain + n_forget)
    w_r_sub = n_retain_sub / (n_retain + n_forget)
    theta_uls = uls_estimator(w_r, w_f, X_r_sub, X_f, y_f, theta_ptr)
    theta_uls_gd = gradient_descent_uls_estimator(w_r, w_f, X_r_sub, X_f, y_f, theta_ptr)
    # check_bias_term(w_r, w_f, X_r, X_f, theta_r, theta_f)
    # verify_gd_vs_closed(w_r, w_f, X_r_sub, X_f, y_f, theta_ptr, theta_r)
    theta_graddiff, lambda_graddiff = graddiff_estimator(X_r_sub, y_r_sub, X_f, y_f, w_f, w_r_sub, p, delta)
    theta_uls_plus, lambda_uls_plus = uls_plus_estimator(w_r, w_f, X_r_sub, y_r_sub, X_f, y_f, theta_ptr, delta)

    ptr_error = norm(theta_ptr - theta_r)
    retain_error = norm(theta_retrain - theta_r)
    retain_sub_error = norm(theta_retain_sub - theta_r)
    uls_error = norm(theta_uls - theta_r)
    uls_gd_error = norm(theta_uls_gd - theta_r)
    graddiff_error = norm(theta_graddiff - theta_r)
    uls_plus_error = norm(theta_uls_plus - theta_r)
    uls_ci_coverage, uls_ci_width, uls_ci_se = uls_inference(p, theta_uls, n_retain, n_retain_sub, X_r_sub, y_r_sub, theta_ptr, theta_r)
    retain_sub_ci_coverage, retain_sub_ci_width, retain_sub_ci_se = retain_sub_inference(p, theta_retain_sub, X_r_sub, y_r_sub, theta_r)

    return {
        "n_retain": n_retain,
        "n_retain_sub": n_retain_sub,
        "n_forget": n_forget,
        "p": p,
        "delta": delta,
        "repeat_idx": rep_idx,
        "ptr_error": ptr_error,
        "retain_error": retain_error,
        "retain_sub_error": retain_sub_error,
        "uls_error": uls_error,
        "uls_gd_error": uls_gd_error,
        "graddiff_error": graddiff_error,
        "uls_plus_error": uls_plus_error,
        "graddiff_lambda": lambda_graddiff,
        "uls_plus_lambda": lambda_uls_plus,
        "uls_ci_coverage": uls_ci_coverage,
        "uls_ci_width": uls_ci_width,
        "retain_sub_ci_coverage": retain_sub_ci_coverage,
        "retain_sub_ci_width": retain_sub_ci_width,
        "uls_ci_se": uls_ci_se,
        "retain_sub_ci_se": retain_sub_ci_se,
    }

if __name__ == "__main__":
    retain_sizes = [20000]
    retain_sub_sizes = [1000, 2000, 4000, 6000]
    forget_sizes = [1000, 2000]
    dims = [10, 50, 80, 100]
    deltas = [1.0, 2.0, 3.0]

    # retain_sizes = [20000]
    # retain_sub_sizes = [5000]
    # forget_sizes = [1000]
    # dims = [10]
    # deltas = [1.0]


    repeats= 1000
    results = []
    start_time = time.time()
    
    for p in dims:
        theta_r = np.random.normal(0, 1, p)
        direction = np.ones(p)
        direction = direction / norm(direction)
        # print("theta_r: ", theta_r)
        for delta in deltas:
            theta_f = theta_r + delta * direction
            # print("theta_f: ", theta_f)
            for n_r in retain_sizes:
                for n_r_sub in retain_sub_sizes:
                    for n_f in forget_sizes:
                        for repeat in range(repeats):
                            print("current experiment has reached: ", n_r, n_f, n_r_sub, p, delta, repeat)
                            out = run_experiment(theta_r, theta_f, n_r, n_f, n_r_sub, p, delta, repeat)
                            results.append(out)

    import pandas as pd
    df = pd.DataFrame(results)
    #print(df)
    df.to_csv("/data/xiejingyi/WAGLE/plot/simulation_results/estimator_simulation_top_inference_1000run_cv_v5.csv")
    group_cols = ["n_retain", "n_retain_sub", "n_forget", "p", "delta"]
    df_mean = (df.drop(columns=["repeat_idx"]).groupby(group_cols, as_index=False).mean())
    df_mean.to_csv("/data/xiejingyi/WAGLE/plot/simulation_results/estimator_simulation_top_inference_1000run_mean_cv_v5.csv")
    # print(df_mean['graddiff_error'])
    # print(df_mean['retain_sub_error'])
    # print(df_mean['uls_error'])
    # print(df_mean['uls_plus_error'])

    end_time = time.time()
    print(f"time cost{end_time-start_time}s")
