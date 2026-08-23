"""
Imputation with a Gaussian PSD density model, fit directly on the partially-observed data via
alternating_minimization (K_S/H_NS already handle arbitrary per-observation missing patterns),
then, for each row, filling its missing entries with the mean of the conditional distribution
given its observed entries (see marginalize_condition.condition and sampling.mean).
"""

import numpy as np
from sklearn.impute import SimpleImputer

from alternating_minimization import alternating_minimization
from marginalize_condition import condition
from sampling import mean as model_mean, sample_bisection, check_normalized


def fit_psd_model(dataset, masks, m, eta_init, alpha, lbd, mu, l_rate_nodes, l_rate_param,
                   nbr_bounce, nbr_gradient_steps, nbr_newton_step_Q, seed=0, verbose=False,
                   verbose_newton=False, anchor_init='data_subset', eta_init_mode='fixed'):
    """
    :param dataset: (n, d) data; entries on the missing (masks == True) positions are never read
    :param masks: (n, d) boolean (or 0/1), True/1 = missing
    :param m: number of anchor nodes
    :param eta_init: initial LOG-precision of the Gaussians (same value on every dimension) --
        matches the log-space reparametrization the gradient step on precision uses (see
        alternating_minimization), so the actual starting precision is exp(eta_init), not
        eta_init itself. Ignored if eta_init_mode == 'empirical'
    :param alpha: parameter inside the log of the loss, i.e. the loss has a term
        -log(Tr(Q A_i) + alpha) for each observation i (see psd.loss) -- keeps the log well
        defined/well conditioned even where Tr(Q A_i) is close to 0
    :param lbd: hyperparameter weighting the trace regularizer Tr(Q A_0) in the loss
    :param mu: hyperparameter weighting the -log det(Q) barrier term in the loss
    :param l_rate_nodes: learning rate for the gradient steps on anchor_nodes during
        alternating_minimization
    :param l_rate_param: learning rate for the gradient steps on precision during
        alternating_minimization (analogous to l_rate_nodes)
    :param nbr_bounce: number of outer alternating_minimization iterations (Newton-for-Q, then
        gradient steps on anchor_nodes/precision -- except the last iteration, which only does
        the Newton-for-Q step, see alternating_minimization)
    :param nbr_gradient_steps: number of gradient steps taken on [anchor_nodes, precision] per
        outer iteration, with Q held fixed at that iteration's Q*
    :param nbr_newton_step_Q: maximum number of Newton iterations for each inner Q update
        (see optimization.newton_method)
    :param seed: seed fixing the randomness of the initial anchor_nodes/Q
    :param verbose: if True, print a per-bounce summary every 5 bounces (Newton decrement,
        gradient norms -- see alternating_minimization)
    :param verbose_newton: if True, print every Newton iteration (loss, decrement) inside every
        bounce's inner Q update -- see optimization.newton_method
    :param anchor_init: 'data_subset' (default) picks the anchor nodes as a random subset of the
        mean-imputed data; 'uniform_hypercube' samples them i.i.d. uniformly in the bounding box
        of the (mean-imputed) data instead
    :param eta_init_mode: 'fixed' (default) starts every dimension's precision at eta_init;
        'empirical' instead starts each dimension's precision at 1 / empirical_variance, the
        per-dimension variance of the observed (non-missing) entries
    :return: (Q, anchor_nodes, precision, history), see alternating_minimization
    """
    n, d = dataset.shape
    rng = np.random.default_rng(seed)

    masks_bool = np.asarray(masks, dtype=bool)
    dataset_nan = np.where(masks_bool, np.nan, dataset)
    initial_imputed = SimpleImputer(strategy='mean').fit_transform(dataset_nan)

    if anchor_init == 'uniform_hypercube':
        low, high = initial_imputed.min(axis=0), initial_imputed.max(axis=0)
        anchor_nodes = rng.uniform(low, high, size=(m, d))
    else:
        anchor_idx = rng.choice(n, size=m, replace=(m > n))
        anchor_nodes = initial_imputed[anchor_idx].copy()

    if eta_init_mode == 'empirical':
        empirical_variance = np.nanvar(dataset_nan, axis=0)
        precision = 1.0 / empirical_variance
    else:
        precision = np.exp(np.full(d, eta_init))  # eta_init is a log-precision -- see docstring

    # L = rng.standard_normal((m, m))
    # Q0 = L @ L.T + np.eye(m) * 2  # PD starting point, alternating_minimization renormalizes it
    Q0 = np.eye(m)  # identity starting point, alternating_minimization renormalizes it

    info = {
        'dataset': dataset, 'masks': masks, 'anchor_nodes': anchor_nodes, 'precision': precision,
        'Q': Q0, 'alpha': alpha, 'lbd': lbd, 'mu': mu,
        'l_rate_nodes': l_rate_nodes, 'l_rate_param': l_rate_param,
        'nbr_bounce': nbr_bounce, 'nbr_gradient_steps': nbr_gradient_steps,
        'max_iter': nbr_newton_step_Q, 'verbose': verbose, 'verbose_newton': verbose_newton,
    }
    Q, anchor_nodes, precision, history = alternating_minimization(info)
    check_normalized(anchor_nodes, precision, Q)
    return Q, anchor_nodes, precision, history


def impute_mean(dataset, mask, Q, anchor_nodes, precision):
    """
    Fill each row's missing entries with the mean of its conditional distribution (see
    marginalize_condition.condition and sampling.mean).

    :param dataset: (n, d) data; entries on the missing (mask == True) positions are never read
    :param mask: (n, d) boolean, True = missing
    :param Q, anchor_nodes, precision: a fitted (see fit_psd_model), normalized PSD model
    :return: (n, d) copy of dataset with the missing entries filled in
    """
    n = dataset.shape[0]
    X_imputed = dataset.copy()
    for i in range(n):
        row_mask = mask[i]
        if not row_mask.any():
            continue
        W_ns, eta_ns, Q_cond = condition(anchor_nodes, precision, Q, row_mask, dataset[i])
        X_imputed[i, row_mask] = model_mean(W_ns, eta_ns, Q_cond)
    return X_imputed


def impute_multiple(dataset, mask, Q, anchor_nodes, precision, n_imputations):
    """
    Draw n_imputations independent completions of the missing entries, by sampling (rather than
    taking the mean of) each row's conditional distribution (see marginalize_condition.condition
    and sampling.sample_bisection).

    :param dataset, mask, Q, anchor_nodes, precision: see impute_mean
    :param n_imputations: number of independent completions to draw
    :return: (n_imputations, n, d) array; observed entries identical across imputations, missing
        entries independently sampled each time
    """
    n = dataset.shape[0]
    imputations = np.repeat(dataset[np.newaxis], n_imputations, axis=0)
    for i in range(n):
        row_mask = mask[i]
        if not row_mask.any():
            continue
        W_ns, eta_ns, Q_cond = condition(anchor_nodes, precision, Q, row_mask, dataset[i])
        imputations[:, i, row_mask] = sample_bisection(W_ns, eta_ns, Q_cond, N=n_imputations)
    return imputations


def psd_impute(X_nas, mask, m=50, eta_init=2.0, alpha=1e-6, lbd=1e-4, mu=1e-4,
               l_rate_nodes=1e-1, l_rate_param=1e-2, nbr_bounce=30, nbr_gradient_steps=5,
               nbr_newton_step_Q=100, seed=0, verbose=False, verbose_newton=False,
               anchor_init='data_subset', eta_init_mode='fixed', n_imputations=0):
    """
    :param X_nas: (n, d) data, NaN on the missing entries
    :param mask: (n, d) boolean (or 0/1), True/1 = missing
    :param m, eta_init, alpha, lbd, mu, l_rate_nodes, l_rate_param, nbr_bounce,
        nbr_gradient_steps, nbr_newton_step_Q, seed, verbose, verbose_newton, anchor_init,
        eta_init_mode: see fit_psd_model
    :param n_imputations: if > 0, also draw this many independent completions per row by sampling
        (rather than taking the mean of) each row's conditional distribution, reusing the same fit
        (see impute_multiple)
    :return: (X_imputed, history, multiple_imp) -- X_imputed is (n, d) with the missing entries
        mean-filled, observed entries left untouched; multiple_imp is (n_imputations, n, d) if
        n_imputations > 0, else None
    """
    X_nas = np.asarray(X_nas, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    dataset = np.where(mask, 0.0, X_nas)  # dummy value on missing entries, ignored by K_S/H_NS
    Q, W, eta, history = fit_psd_model(
        dataset, mask.astype(float), m, eta_init, alpha, lbd, mu,
        l_rate_nodes, l_rate_param, nbr_bounce, nbr_gradient_steps, nbr_newton_step_Q, seed=seed,
        verbose=verbose, verbose_newton=verbose_newton, anchor_init=anchor_init,
        eta_init_mode=eta_init_mode,
    )

    X_imputed = impute_mean(dataset, mask, Q, W, eta)
    multiple_imp = impute_multiple(dataset, mask, Q, W, eta, n_imputations) if n_imputations > 0 else None

    return X_imputed, history, multiple_imp


def cross_validate_hyperparams(dataset_loaded, p, lr_nodes_grid, lr_param_grid, eta_grid,
                                m_cv=10, n_cv=200, nbr_bounce_cv=20, nbr_gradient_steps=5,
                                nbr_newton_step_Q=20, alpha=1e-6, lbd=1e-1, mu=1e-3, seed=0,
                                verbose=False):
    """
    Fast cross-validation for (l_rate_nodes, l_rate_param, eta_init): fit small, quick psd models
    on a small validation subsample and score each candidate by how well it recovers ADDITIONALLY
    hidden entries -- not the ones already missing -- via RMSE.

    Procedure: draw an n_cv-row subsample of dataset_loaded, apply the usual MCAR mask at
    probability p (the "already missing" entries), then hide an extra MCAR fraction (also at
    probability p) of what's left observed. Only that second, extra mask is scored: for each
    candidate, fit with m_cv anchor nodes / nbr_bounce_cv bounces (small and few, for speed) and
    compute RMSE between the imputed and true values at the extra-hidden positions.

    :param dataset_loaded: (N, d) full dataset to subsample from
    :param p: MCAR probability, used both for the base mask and the additional validation mask
        (same convention as the main experiment's --p)
    :param lr_nodes_grid, lr_param_grid, eta_grid: lists of candidate l_rate_nodes, l_rate_param,
        eta_init (LOG-precision, see fit_psd_model) values; every combination is tried
    :param m_cv: number of anchor nodes for the validation fits (default 10, small for speed)
    :param n_cv: validation subsample size (default 200, small for speed)
    :param nbr_bounce_cv: number of alternating_minimization bounces for the validation fits
        (default 20, few for speed)
    :param nbr_gradient_steps, nbr_newton_step_Q, alpha, lbd, mu, seed: see fit_psd_model (lbd/mu
        are divided by n_cv internally, matching the convention used for the full-scale fit)
    :param verbose: if True, print every candidate's validation RMSE as it's evaluated
    :return: (best_lr_nodes, best_lr_param, best_eta, best_rmse, results) -- results is the full
        list of (lr_nodes, lr_param, eta, rmse) tuples tried, in evaluation order
    """
    rng = np.random.default_rng(seed)
    n_total, d = dataset_loaded.shape
    n_cv = min(n_cv, n_total)
    idx = rng.choice(n_total, size=n_cv, replace=False)
    X_cv = dataset_loaded[idx]

    base_mask = rng.random((n_cv, d)) < p
    fully_missing = np.where(base_mask.sum(axis=1) == d)[0]
    if len(fully_missing) > 0:
        revealed = rng.integers(0, d, size=len(fully_missing))
        base_mask[fully_missing, revealed] = False

    # additionally hide some of what's still observed -- these are the entries we score against
    observed = ~base_mask
    extra_hide = (rng.random((n_cv, d)) < p) & observed
    # never extra-hide the last observed entry of a row (would leave nothing to condition on)
    would_be_fully_missing = np.where((base_mask | extra_hide).sum(axis=1) == d)[0]
    extra_hide[would_be_fully_missing] = False

    combined_mask = base_mask | extra_hide
    X_true_cv = X_cv.copy()
    X_nas_cv = np.where(combined_mask, np.nan, X_cv)

    results = []
    for lr_nodes in lr_nodes_grid:
        for lr_param in lr_param_grid:
            for eta_init in eta_grid:
                X_imputed, _, _ = psd_impute(
                    X_nas_cv, combined_mask,
                    m=m_cv, eta_init=eta_init, alpha=alpha, lbd=lbd / n_cv, mu=mu / n_cv,
                    l_rate_nodes=lr_nodes, l_rate_param=lr_param,
                    nbr_bounce=nbr_bounce_cv, nbr_gradient_steps=nbr_gradient_steps,
                    nbr_newton_step_Q=nbr_newton_step_Q, seed=seed,
                )
                rmse = float(np.sqrt(np.mean((X_imputed[extra_hide] - X_true_cv[extra_hide]) ** 2)))
                results.append((lr_nodes, lr_param, eta_init, rmse))
                if verbose:
                    print(f"  CV: lr_nodes={lr_nodes:.1e}  lr_param={lr_param:.1e}  "
                          f"eta_init={eta_init:.4f}  RMSE={rmse:.4f}")

    best_lr_nodes, best_lr_param, best_eta, best_rmse = min(results, key=lambda r: r[3])
    return best_lr_nodes, best_lr_param, best_eta, best_rmse, results

