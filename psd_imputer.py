"""
Imputation with a Gaussian PSD density model, fit directly on the partially-observed data via
alternating_minimization (K_S/H_NS already handle arbitrary per-observation missing patterns),
then, for each row, filling its missing entries with the mean of the conditional distribution
given its observed entries (see marginalize_condition.condition and sampling.mean).
"""

import numpy as np

from alternating_minimization import alternating_minimization
from marginalize_condition import condition
from sampling import mean as model_mean, check_normalized


def fit_psd_model(dataset, masks, m, eta_init, alpha, lbd, mu, l_rate_nodes, l_rate_param,
                   nbr_bounce, nbr_gradient_steps, seed=0):
    """
    :param dataset: (n, d) data; entries on the missing (masks == True) positions are never read
    :param masks: (n, d) boolean (or 0/1), True/1 = missing
    :param m: number of anchor nodes
    :param eta_init: initial precision of the Gaussians (same value on every dimension, used to
        fill the (d,) 'precision' array alternating_minimization starts from)
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
    :param seed: seed fixing the randomness of the initial anchor_nodes/Q
    :return: (Q, anchor_nodes, precision, history), see alternating_minimization
    """
    n, d = dataset.shape
    rng = np.random.default_rng(seed)

    observed = ~np.asarray(masks, dtype=bool)
    low = np.array([dataset[observed[:, j], j].min() if observed[:, j].any() else -1.0 for j in range(d)])
    high = np.array([dataset[observed[:, j], j].max() if observed[:, j].any() else 1.0 for j in range(d)])

    anchor_nodes = rng.uniform(low, high, size=(m, d))
    precision = np.full(d, eta_init)

    L = rng.standard_normal((m, m))
    Q0 = L @ L.T + np.eye(m) * 2  # PD starting point, alternating_minimization renormalizes it

    info = {
        'dataset': dataset, 'masks': masks, 'anchor_nodes': anchor_nodes, 'precision': precision,
        'Q': Q0, 'alpha': alpha, 'lbd': lbd, 'mu': mu,
        'l_rate_nodes': l_rate_nodes, 'l_rate_param': l_rate_param,
        'nbr_bounce': nbr_bounce, 'nbr_gradient_steps': nbr_gradient_steps,
    }
    Q, anchor_nodes, precision, history = alternating_minimization(info)
    check_normalized(anchor_nodes, precision, Q)
    return Q, anchor_nodes, precision, history


def psd_impute(X_nas, mask, m=50, eta_init=2.0, alpha=1e-6, lbd=1e-4, mu=1e-4,
               l_rate_nodes=1e-1, l_rate_param=1e-2, nbr_bounce=30, nbr_gradient_steps=5, seed=0):
    """
    :param X_nas: (n, d) data, NaN on the missing entries
    :param mask: (n, d) boolean (or 0/1), True/1 = missing
    :param m, eta_init, alpha, lbd, mu, l_rate_nodes, l_rate_param, nbr_bounce,
        nbr_gradient_steps, seed: see fit_psd_model
    :return: (X_imputed, history) -- X_imputed is (n, d) with the missing entries filled in,
        observed entries left untouched
    """
    X_nas = np.asarray(X_nas, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n, d = X_nas.shape

    dataset = np.where(mask, 0.0, X_nas)  # dummy value on missing entries, ignored by K_S/H_NS
    Q, W, eta, history = fit_psd_model(
        dataset, mask.astype(float), m, eta_init, alpha, lbd, mu,
        l_rate_nodes, l_rate_param, nbr_bounce, nbr_gradient_steps, seed=seed,
    )

    X_imputed = dataset.copy()
    for i in range(n):
        row_mask = mask[i]
        if not row_mask.any():
            continue
        W_ns, eta_ns, Q_cond = condition(W, eta, Q, row_mask, dataset[i])
        X_imputed[i, row_mask] = model_mean(W_ns, eta_ns, Q_cond)

    return X_imputed, history
