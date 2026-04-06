import json
import os
import numpy as np
import string
import random
import time
from numpy.linalg import inv, norm
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def extract_words(text):
    text = text.lower()
    exclude = set(string.punctuation + string.digits)
    text = ''.join(ch for ch in text if ch not in exclude)
    words = text.split()
    words = filter(lambda w: len(w) > 1, words)
    return list(words)

def load_dataset(file_path):
    """Load dataset from JSONL file"""
    dataset = []
    print(f"Loading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Loading {file_path.split('/')[-1]}", unit="lines"):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        dataset.append(data)
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    
    print(f"Successfully loaded {len(dataset)} records")
    return dataset

def build_vocabulary(dataset, vocab_size=5000, sample_size=5000):
    """Build vocabulary from dataset"""
    review_texts = [record['text'] for record in dataset if 'text' in record]
    if len(review_texts) > sample_size:
        review_sample = random.sample(review_texts, sample_size)
    else:
        review_sample = review_texts
    
    review_sample = [extract_words(r) for r in tqdm(review_sample, desc="Extracting words", unit="reviews")]
    
    vocabulary_set = set([])
    for review in tqdm(review_sample, desc="Collecting words", unit="reviews"):
        for word in review:
            vocabulary_set.add(word)
    
    counts = {word: 0 for word in vocabulary_set}
    for review in tqdm(review_sample, desc="Counting frequencies", unit="reviews"):
        review_set = set(review)
        for word in review_set:
            if word in counts:
                counts[word] += 1
    
    vocabulary = sorted(vocabulary_set, key=lambda w: counts[w], reverse=True)[:vocab_size]

    return vocabulary

def txt_to_vec(text, vocabulary):
    """Convert text to bag-of-words vector"""
    words = extract_words(text)
    d = len(vocabulary)
    v = np.zeros(d)
    
    vocab_dict = {word: i for i, word in enumerate(vocabulary)}
    
    for word in words:
        if word in vocab_dict:
            v[vocab_dict[word]] += 1
    
    return v

def dataset_to_matrix(dataset, vocabulary):
    """Convert dataset to feature matrix X and label vector y"""
    n = len(dataset)
    d = len(vocabulary)
    X = np.zeros((n, d))
    y = np.zeros(n)
    user_ids = []
    
    for idx in tqdm(range(n), desc="Converting features", unit="records"):
        review = dataset[idx]
        if 'user_id' in review:
            user_ids.append(review['user_id'])
        if 'text' in review:
            X[idx] = txt_to_vec(review['text'], vocabulary)
        if 'stars' in review:
            y[idx] = review['stars']  
    
    intercept = np.ones((n, 1))
    X = np.hstack([intercept, X])
    
    return X, y, user_ids

def ols_estimator(X, y):
    """OLS estimator with regularization"""
    return inv(X.T @ X) @ (X.T @ y)

def uls_estimator(w_r, w_f, X_r_sub, X_f, y_f, theta_ptr):
    """ULS (Unlearning via Linear Subsampling) estimator"""
    cov_r_sub = (1/X_r_sub.shape[0]) * (X_r_sub.T @ X_r_sub)
    cov_f = (1/X_f.shape[0]) * (X_f.T @ X_f)
    cov_ptr_sub = w_r * cov_r_sub + w_f * cov_f
    M_f = (1/X_f.shape[0]) * (X_f.T @ y_f)
    return inv(w_r * cov_r_sub) @ (cov_ptr_sub @ theta_ptr - w_f * M_f)

def _compute_theta_for_lambda(X_r_sub, y_r_sub, X_f, y_f, lamb, estimator_type='gd', 
                              w_r=None, w_f=None, theta_ptr=None):
    """Compute theta for a given lambda value."""
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
    """Compute full loss: -(1/N_f) * ||X_f @ theta - y_f||^2 + (lambda / n_r_sub) * ||X_r_sub @ theta - y_r_sub||^2"""
    residuals_f = y_f - X_f @ theta
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
        
        lamb = best_lambda
    else:
        lamb = max(c1, c2)
    
    A = inv(-cov_f + lamb * cov_r_sub)
    b = -(1/n_f) * (X_f.T @ y_f) + (lamb / n_r_sub) * (X_r_sub.T @ y_r_sub)
    theta = A @ b
    return theta, lamb


def uls_plus_estimator(w_r, w_f, X_r_sub, y_r_sub, X_f, y_f, theta_ptr, delta=1, c=1,
                       use_cv=True, k_folds=5, n_lambda_candidates=50, 
                       use_full_loss=False, cv_random_state=42):
    """ULS Plus estimator with optional cross-validation for lambda selection."""
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


def gram_schmidt(X):
    (k, d) = X.shape
    if k <= d:
        q, r = np.linalg.qr(np.transpose(X))
    else:
        q, r = np.linalg.qr(np.transpose(X), mode='complete')
    U = np.transpose(q)
    C = np.transpose(r)
    return U, C


def compute_lko_predictions(X, Y, ind, H=None, reg=1e-4):
    n = len(Y)
    k = len(ind)
    d = X.shape[1]
    
    if H is None:
        H = np.matmul(X, np.linalg.solve(np.matmul(X.T, X) + reg * np.eye(d), X.T))
    
    H_sub = H[ind, :]
    H_diag = np.diag(H[ind][:, ind])
    LOO = (Y[ind] - H_sub @ Y) / (1 - H_diag)
    
    H_block = H[np.ix_(ind, ind)]
    H_diag_denom = 1 - np.diag(H_block)
    S = -H_block / H_diag_denom[:, None]
    np.fill_diagonal(S, 1.0)
    
    LKO = np.linalg.solve(S, LOO)
    
    return Y[ind] - LKO


def pru_estimator(X_r, y_r, X_f, y_f, theta, H=None, reg=1e-4):
    """
    Approximate retraining via the projective residual update.
    """
    X = np.vstack([X_r, X_f])
    Y = np.concatenate([y_r, y_f])
    
    n_r = len(y_r)
    k = X_f.shape[0]
    d = X.shape[1]
    
    if H is None:
        H = X @ inv(X.T @ X + reg * np.eye(d)) @ X.T
    
    ind = list(range(n_r, n_r + k))
    LKO = compute_lko_predictions(X, Y, ind, H, reg)
    U, C = gram_schmidt(X_f)
    Cmatrix = np.matmul(C.T, C)
    eigenval, a = np.linalg.eigh(Cmatrix)
    V = np.matmul(a.T, U)

    grad = X_f.T @ (X_f @ theta - LKO)
    
    n_eigen = len(eigenval)
    factors = np.where(eigenval > 1e-10, 1/eigenval, 0)
    step = V.T @ (factors * (V @ grad))
    
    update = theta - step
    return update
    

def calculate_prediction_error(X_test, y_test, theta):
    y_pred = X_test @ theta
    mse = np.mean((y_pred - y_test) ** 2)
    return mse

def calculate_parameter_error(theta_est, theta_true):
    return norm(theta_est - theta_true)

def main():
    start_time = time.time()
    
    base_dir = '/data/xiejingyi/dataset/yelp_10_90%'
    train_file = os.path.join(base_dir, 'yelp_train_200k.json')
    retain_full_file = os.path.join(base_dir, 'yelp_retain_full.json')
    forget_file = os.path.join(base_dir, 'yelp_forget.json')
    vocab_file = os.path.join(base_dir, 'yelp_vocab_200k.json')
    
    print("Step 1: Loading base datasets")
    full_dataset = load_dataset(train_file)
    retain_full_dataset = load_dataset(retain_full_file)
    forget_dataset = load_dataset(forget_file)
    vocab_dataset = load_dataset(vocab_file)
    
    if full_dataset is None or retain_full_dataset is None or forget_dataset is None:
        print("Failed to load required datasets.")
        return
    
    if vocab_dataset is None:
        print("Vocabulary dataset not found. Using full dataset instead.")
        vocab_dataset = full_dataset
    
    print("Step 2: Building vocabulary from vocabulary dataset")
    vocabulary = build_vocabulary(vocab_dataset, vocab_size=1500, sample_size=200000)
    
    print("Step 3: Precomputing feature matrices for static datasets")
    print("  Converting full training set...")
    X_full, y_full, user_ids_full = dataset_to_matrix(full_dataset, vocabulary)
    
    print("  Converting forget dataset...")
    X_forget, y_forget, user_ids_forget = dataset_to_matrix(forget_dataset, vocabulary)
    
    print("  Converting retain full dataset...")
    X_retain_full, y_retain_full, user_ids_retain_full = dataset_to_matrix(retain_full_dataset, vocabulary)
    
    subsample_ratios = [0.05, 0.1, 0.2, 0.3]
    
    fold_results = []
    for fold_idx in range(1, 21):
        print(f"\n=== Fold {fold_idx} / 20 ===")
        fold_start = time.time()
        
        print(f"  Splitting retain dataset into train (80%) and test (20%)")
        X_retain, X_test, y_retain, y_test = train_test_split(
            X_retain_full, y_retain_full, 
            test_size=0.2, 
            random_state=RANDOM_SEED + fold_idx,
            shuffle=True
        )
        print(f"Train size: {X_retain.shape[0]}, Test size: {X_test.shape[0]}")
        print(f"Shapes -> X_retain: {X_retain.shape}, X_forget: {X_forget.shape}, X_test: {X_test.shape}")
        
        computation_times = {}
        
        print("Computing estimators")
        t_start = time.time()
        theta_ptr = ols_estimator(np.vstack([X_retain, X_forget]), np.concatenate([y_retain, y_forget]))
        computation_times['Pre-train'] = time.time() - t_start
        
        t_start = time.time()
        theta_retrain = ols_estimator(X_retain, y_retain)
        computation_times['Retrain'] = time.time() - t_start
        
        # Precompute H matrix for PRU (not included in PRU timing)
        X_combined = np.vstack([X_retain, X_forget])
        d = X_combined.shape[1]
        H_pru = X_combined @ inv(X_combined.T @ X_combined + 1e-4 * np.eye(d)) @ X_combined.T
        
        t_start = time.time()
        theta_pru = pru_estimator(X_retain, y_retain, X_forget, y_forget, theta_ptr, H=H_pru)
        computation_times['PRU'] = time.time() - t_start
        
        theta_retain_subs = {}
        theta_uls_dict = {}
        theta_graddiff_dict = {}
        theta_uls_plus_dict = {}
        n_retain = len(X_retain)
        n_forget = len(X_forget)
        p = X_retain.shape[1]
        w_r = n_retain / (n_retain + n_forget)
        w_f = n_forget / (n_retain + n_forget)
        
        for idx, ratio in enumerate(subsample_ratios):
            sub_sample_size = max(1, int(n_retain * ratio))
            sub_indices = random.sample(range(n_retain), sub_sample_size)
            sub_indices = np.array(sub_indices)
            
            X_retain_sub = X_retain[sub_indices]
            y_retain_sub = y_retain[sub_indices]
            
            t_start = time.time()
            theta_sub = ols_estimator(X_retain_sub, y_retain_sub)
            computation_times[f'Retain-Sub-{idx}'] = time.time() - t_start
            theta_retain_subs[idx] = theta_sub
            
            t_start = time.time()
            theta_uls = uls_estimator(w_r, w_f, X_retain_sub, X_forget, y_forget, theta_ptr)
            computation_times[f'ULS-{idx}'] = time.time() - t_start
            theta_uls_dict[idx] = theta_uls

            # ── GradDiff: CV outside timing, only count theta computation ──
            w_r_sub = sub_sample_size / (n_retain + n_forget)
            _, lambda_graddiff = graddiff_estimator(
                X_retain_sub, y_retain_sub, X_forget, y_forget, w_f, w_r_sub, p
            )
            t_start = time.time()
            theta_graddiff = _compute_theta_for_lambda(
                X_retain_sub, y_retain_sub, X_forget, y_forget,
                lambda_graddiff, estimator_type='gd'
            )
            computation_times[f'Graddiff-{idx}'] = time.time() - t_start
            theta_graddiff_dict[idx] = (theta_graddiff, lambda_graddiff)

            # ── ULS-Plus: CV outside timing, only count theta computation ──
            _, lambda_uls_plus = uls_plus_estimator(
                w_r, w_f, X_retain_sub, y_retain_sub, X_forget, y_forget, theta_ptr
            )
            t_start = time.time()
            theta_uls_plus = _compute_theta_for_lambda(
                X_retain_sub, y_retain_sub, X_forget, y_forget,
                lambda_uls_plus, estimator_type='uls_plus',
                w_r=w_r, w_f=w_f, theta_ptr=theta_ptr
            )
            computation_times[f'ULS-Plus-{idx}'] = time.time() - t_start
            theta_uls_plus_dict[idx] = (theta_uls_plus, lambda_uls_plus)

    
        print("  Evaluating estimators on test set")
        theta_true = theta_retrain
        fold_result = {}
        
        fold_result['Pre-train'] = {
            'pred_error': calculate_prediction_error(X_test, y_test, theta_ptr),
            'param_error': calculate_parameter_error(theta_ptr, theta_true),
            'comp_time': computation_times['Pre-train']
        }
        
        fold_result['Retrain'] = {
            'pred_error': calculate_prediction_error(X_test, y_test, theta_retrain),
            'param_error': calculate_parameter_error(theta_retrain, theta_true),
            'comp_time': computation_times['Retrain']
        }
        
        fold_result['PRU'] = {
            'pred_error': calculate_prediction_error(X_test, y_test, theta_pru),
            'param_error': calculate_parameter_error(theta_pru, theta_true),
            'comp_time': computation_times['PRU']
        }
        
        for idx, theta in theta_retain_subs.items():
            fold_result[f'Retain-Sub-{idx}'] = {
                'pred_error': calculate_prediction_error(X_test, y_test, theta),
                'param_error': calculate_parameter_error(theta, theta_true),
                'comp_time': computation_times[f'Retain-Sub-{idx}']
            }
        
        for idx, theta in theta_uls_dict.items():
            fold_result[f'ULS-{idx}'] = {
                'pred_error': calculate_prediction_error(X_test, y_test, theta),
                'param_error': calculate_parameter_error(theta, theta_true),
                'comp_time': computation_times[f'ULS-{idx}']
            }
        
        for idx, (theta, lamb) in theta_graddiff_dict.items():
            fold_result[f'Graddiff-{idx}'] = {
                'pred_error': calculate_prediction_error(X_test, y_test, theta),
                'param_error': calculate_parameter_error(theta, theta_true),
                'comp_time': computation_times[f'Graddiff-{idx}'],
                'lambda': lamb
            }
        
        for idx, (theta, lamb) in theta_uls_plus_dict.items():
            fold_result[f'ULS-Plus-{idx}'] = {
                'pred_error': calculate_prediction_error(X_test, y_test, theta),
                'param_error': calculate_parameter_error(theta, theta_true),
                'comp_time': computation_times[f'ULS-Plus-{idx}'],
                'lambda': lamb
            }
        
        print(f"\n  {'Estimator':<20} {'Pred Error (MSE)':<20} {'Param Error (L2)':<20} {'Comp Time (s)':<15}")
        for name, metrics in fold_result.items():
            print(f"  {name:<20} {metrics['pred_error']:<20.6f} {metrics['param_error']:<20.6f} {metrics['comp_time']:<15.6f}")
        
        fold_duration = time.time() - fold_start
        print(f"Fold {fold_idx} completed in {fold_duration:.2f} seconds")
        
        fold_results.append({
            'fold': fold_idx,
            'results': fold_result,
            'w_r': w_r,
            'w_f': w_f,
            'duration': fold_duration
        })
        
        df_fold = pd.DataFrame.from_dict(fold_result, orient='index')
        df_fold.reset_index(inplace=True)
        df_fold.rename(columns={'index': 'estimator'}, inplace=True)
        df_fold['fold'] = fold_idx
        
        output_csv = os.path.join(base_dir, f'yelp_estimation_results_fold_{fold_idx}.csv')
        df_fold.to_csv(output_csv, index=False)
        print(f"Fold {fold_idx} results saved to: {output_csv}")
    
    if fold_results:
        combined_rows = []
        for fold_entry in fold_results:
            fold_idx = fold_entry['fold']
            for estimator, metrics in fold_entry['results'].items():
                row = {
                    'fold': fold_idx,
                    'estimator': estimator,
                    'pred_error': metrics['pred_error'],
                    'param_error': metrics['param_error'],
                    'comp_time': metrics['comp_time'],
                    'w_r': fold_entry['w_r'],
                    'w_f': fold_entry['w_f'],
                    'fold_duration': fold_entry['duration'],
                    'lambda': metrics.get('lambda', 0)
                }
                combined_rows.append(row)
        
        df_all = pd.DataFrame(combined_rows)
        all_csv = os.path.join(base_dir, 'yelp_estimation_results_all_folds.csv')
        df_all.to_csv(all_csv, index=False)
        print(f"\nCombined results saved to: {all_csv}")
    
    end_time = time.time()
    print(f"\nTotal runtime: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()