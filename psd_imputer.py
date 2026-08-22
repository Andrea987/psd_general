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

