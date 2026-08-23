"""
Imputation with a Gaussian PSD density model, fit directly on the partially-observed data via
alternating_minimization (K_S/H_NS already handle arbitrary per-observation missing patterns),
then, for each row, filling its missing entries with the mean of the conditional distribution
given its observed entries (see marginalize_condition.condition and sampling.mean).
"""

import numpy as np
import ot
from sklearn.impute import SimpleImputer

from alternating_minimization import alternating_minimization
from marginalize_condition import condition
from psd import energy_distance
from sampling import mean as model_mean, sample_bisection, check_normalized


def fit_psd_model(dataset, masks, m, eta_init, alpha, lbd, mu, l_rate_nodes, l_rate_param,
                   nbr_bounce, nbr_gradient_steps, nbr_newton_step_Q, seed=0, verbose=False,
                   verbose_newton=False, anchor_init='data_subset', eta_init_mode='fixed',
                   fixed_anchor_nodes=None, dataset_true=None):
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
    :param fixed_anchor_nodes: if given, an (m, d) array used directly as the anchor nodes,
        bypassing anchor_init entirely (e.g. for cross_validate_anchor_nodes, which needs to try
        specific candidate anchor sets rather than let fit_psd_model pick its own)
    :param dataset_true: if given (and verbose is True), the true (unmasked) (n, d) array used to
        print ORACLE MAE/RMSE/OT/ED at every 5th bounce (and the last bounce) -- oracle because it
        compares against the real values behind the entries masks hides, which the fit itself
        never sees; purely diagnostic, see alternating_minimization._impute_mean_and_score
    :return: (Q, anchor_nodes, precision, history), see alternating_minimization
    """
    n, d = dataset.shape
    rng = np.random.default_rng(seed)

    masks_bool = np.asarray(masks, dtype=bool)
    dataset_nan = np.where(masks_bool, np.nan, dataset)
    initial_imputed = SimpleImputer(strategy='mean').fit_transform(dataset_nan)

    if fixed_anchor_nodes is not None:
        anchor_nodes = np.asarray(fixed_anchor_nodes, dtype=float).copy()
    elif anchor_init == 'uniform_hypercube':
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
        'dataset_true': dataset_true,
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
               anchor_init='data_subset', eta_init_mode='fixed', fixed_anchor_nodes=None,
               dataset_true=None, n_imputations=0):
    """
    :param X_nas: (n, d) data, NaN on the missing entries
    :param mask: (n, d) boolean (or 0/1), True/1 = missing
    :param m, eta_init, alpha, lbd, mu, l_rate_nodes, l_rate_param, nbr_bounce,
        nbr_gradient_steps, nbr_newton_step_Q, seed, verbose, verbose_newton, anchor_init,
        eta_init_mode, fixed_anchor_nodes, dataset_true: see fit_psd_model
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
        eta_init_mode=eta_init_mode, fixed_anchor_nodes=fixed_anchor_nodes,
        dataset_true=dataset_true,
    )

    X_imputed = impute_mean(dataset, mask, Q, W, eta)
    multiple_imp = impute_multiple(dataset, mask, Q, W, eta, n_imputations) if n_imputations > 0 else None

    return X_imputed, history, multiple_imp


def cross_validate_hyperparams(dataset_loaded, p, lr_nodes_grid, lr_param_grid, eta_grid,
                                m_cv=10, n_cv=200, nbr_bounce_cv=20, nbr_gradient_steps=5,
                                nbr_newton_step_Q=20, alpha=1e-6, lbd=1e-1, mu=1e-3, seed=0,
                                verbose=False, cv_metric='rmse'):
    """
    Fast cross-validation for (l_rate_nodes, l_rate_param, eta_init): fit small, quick psd models
    on a small validation subsample and score each candidate by how well it recovers ADDITIONALLY
    hidden entries -- not the ones already missing -- via RMSE, ED, or OT (see cv_metric).

    Procedure: draw an n_cv-row subsample of dataset_loaded, apply the usual MCAR mask at
    probability p (the "already missing" entries), then hide an extra MCAR fraction (also at
    probability p) of what's left observed. Only that second, extra mask is scored: for each
    candidate, fit with m_cv anchor nodes / nbr_bounce_cv bounces (small and few, for speed) and
    compute RMSE/ED/OT between the imputed and true values at the extra-hidden positions (see
    _cv_score).

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
    :param verbose: if True, print every candidate's validation RMSE/ED/OT as it's evaluated
        (all three are always computed and printed, regardless of cv_metric, purely for visibility)
    :param cv_metric: 'rmse' (default), 'ed', or 'ot' -- which of the three printed scores is
        actually used to pick the winning combination; see _cv_score. 'ed'/'ot' compare whole
        rows, so the entries already missing before extra-hiding (base_mask) are zeroed out in
        both the imputed and true arrays first, so the score reflects only how well the
        extra-hidden entries were recovered
    :return: (best_lr_nodes, best_lr_param, best_eta, best_score, results) -- best_score is
        whichever of rmse/ed/ot cv_metric selected on; results is the full list of
        (lr_nodes, lr_param, eta, rmse, ed, ot) tuples tried, in evaluation order
    """
    metric_idx = {'rmse': 3, 'ed': 4, 'ot': 5}[cv_metric]
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

    if verbose:
        _print_cv_setup(base_mask, extra_hide)

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
                rmse = _cv_score(X_imputed, X_true_cv, extra_hide, base_mask, 'rmse')
                ed = _cv_score(X_imputed, X_true_cv, extra_hide, base_mask, 'ed')
                ot_dist = _cv_score(X_imputed, X_true_cv, extra_hide, base_mask, 'ot')
                results.append((lr_nodes, lr_param, eta_init, rmse, ed, ot_dist))
                if verbose:
                    print(f"  CV: lr_nodes={lr_nodes:.1e}  lr_param={lr_param:.1e}  "
                          f"eta_init={eta_init:.4f}  RMSE={rmse:.4f}  ED={ed:.4f}  OT={ot_dist:.4f}"
                          + ("" if cv_metric == 'rmse' else f"  (selecting by {cv_metric.upper()})"))

    best = min(results, key=lambda r: r[metric_idx])
    best_lr_nodes, best_lr_param, best_eta, best_score = best[0], best[1], best[2], best[metric_idx]
    return best_lr_nodes, best_lr_param, best_eta, best_score, results


def _print_cv_setup(base_mask, extra_hide):
    """Print, once per CV call, what the validation masking actually did and what each printed
    metric compares -- see _cv_score."""
    n_base = int(base_mask.sum())
    n_extra = int(extra_hide.sum())
    print(f"  CV setup: {n_base} entries were already missing beforehand (base mask); "
          f"{n_extra} additional entries were hidden on top of that purely for validation "
          f"(extra_hide) -- these are the ones being scored. "
          f"RMSE = imputed vs. true value, at the {n_extra} extra-hidden entries only. "
          f"ED/OT compare whole rows, so for those two the {n_base} entries that were already "
          f"missing beforehand are set to 0 in BOTH the imputed and true arrays first (so they "
          f"cancel out and contribute nothing); what's left is again effectively just the "
          f"extra-hidden entries' imputed values vs. their true values.")


def _cv_score(X_imputed, X_true_cv, extra_hide, base_mask, metric, otlim=5000):
    """
    Score a cross-validation candidate: lower is better for every metric option, so callers can
    always pick with min(..., key=...).

    :param X_imputed, X_true_cv: (n_cv, d) imputed / true validation arrays
    :param extra_hide: (n_cv, d) boolean, the entries hidden purely for validation scoring
        (see cross_validate_anchor_nodes/cross_validate_hyperparams) -- what every metric scores
        recovery of
    :param base_mask: (n_cv, d) boolean, the entries that were already missing before extra_hide
        was applied (mirrors the real --p missingness). Only used by 'ed'/'ot': RMSE indexes
        extra_hide directly, but ED/OT compare whole rows, so base_mask positions are zeroed out in
        both arrays first -- they're already imputed too, but their reconstruction quality isn't
        what's being validated here, and leaving them in would let it leak into the score
    :param metric: 'rmse', 'ed', or 'ot'
    :return: float score
    """
    if metric == 'rmse':
        return float(np.sqrt(np.mean((X_imputed[extra_hide] - X_true_cv[extra_hide]) ** 2)))

    X_imp_scored = np.where(base_mask, 0.0, X_imputed)
    X_true_scored = np.where(base_mask, 0.0, X_true_cv)

    if metric == 'ed':
        return float(energy_distance(X_imp_scored, X_true_scored))

    if metric == 'ot':
        M = extra_hide.sum(axis=1) > 0
        n_m = int(M.sum())
        if n_m == 0 or n_m >= otlim:
            return float('nan')
        dists = ((X_imp_scored[M][:, None] - X_true_scored[M]) ** 2).sum(axis=2) / 2.
        return float(ot.emd2(np.ones(n_m) / n_m, np.ones(n_m) / n_m, dists))

    raise ValueError(f"unknown metric: {metric!r} (expected 'rmse', 'ed', or 'ot')")


def cross_validate_anchor_nodes(dataset_loaded, p, m, lr_nodes, lr_param, eta_init, n_trials=5,
                                 n_cv=200, newton_steps=10, alpha=1e-6, lbd=1e-1, mu=1e-3, seed=0,
                                 verbose=False, cv_metric='rmse'):
    """
    Fast cross-validation for the CHOICE of anchor nodes, run after cross_validate_hyperparams (or
    with hand-picked hyperparameters): try n_trials random m-row subsamples of dataset_loaded as
    candidate anchor-node sets, score each by running just the Newton method (nbr_bounce=1, so
    anchor_nodes/precision are never moved -- there's no gradient-step phase with a single bounce)
    for up to newton_steps iterations on a small validation subsample (same hide-extra-entries
    methodology as cross_validate_hyperparams, drawn once and reused across every candidate so
    they're compared on the same data), and returns the anchor set with the lowest score.

    :param dataset_loaded: (N, d) fully-observed data to draw candidate anchor nodes AND the
        validation subsample from (e.g. the experiment's ground_truth training split)
    :param p: MCAR probability, used both for the base mask and the additional validation mask
        (same convention as the main experiment's --p)
    :param m: number of anchor nodes -- should match the m used for the real run
    :param lr_nodes, lr_param, eta_init: the hyperparameters already chosen (e.g. by
        cross_validate_hyperparams); held fixed throughout this stage since bounce=1 means they're
        never actually used for a gradient step anyway
    :param n_trials: number of candidate anchor-node subsamples to try (default 5)
    :param n_cv: validation subsample size (default 200)
    :param newton_steps: max Newton iterations per candidate (default 10)
    :param alpha, lbd, mu, seed: see fit_psd_model (lbd/mu divided by n_cv, matching the
        full-scale convention)
    :param verbose: if True, print every candidate's validation RMSE/ED/OT as it's evaluated
        (all three are always computed and printed, regardless of cv_metric, purely for visibility)
    :param cv_metric: 'rmse' (default), 'ed', or 'ot' -- which of the three printed scores is
        actually used to pick the winning candidate; see _cv_score. 'ed'/'ot' compare whole rows,
        so the entries already missing before extra-hiding (base_mask) are zeroed out in both the
        imputed and true arrays first, so the score reflects only how well the extra-hidden
        entries were recovered
    :return: (best_anchor_nodes, best_score, results) -- best_anchor_nodes is (m, d); results is
        the list of (trial_index, rmse, ed, ot) tuples tried, in order
    """
    rng = np.random.default_rng(seed)
    n_total, d = dataset_loaded.shape
    n_cv = min(n_cv, n_total)

    # validation subsample + extra-hiding, drawn once and reused for every candidate anchor set
    # (same construction as cross_validate_hyperparams)
    val_idx = rng.choice(n_total, size=n_cv, replace=False)
    X_cv = dataset_loaded[val_idx]

    base_mask = rng.random((n_cv, d)) < p
    fully_missing = np.where(base_mask.sum(axis=1) == d)[0]
    if len(fully_missing) > 0:
        revealed = rng.integers(0, d, size=len(fully_missing))
        base_mask[fully_missing, revealed] = False

    observed = ~base_mask
    extra_hide = (rng.random((n_cv, d)) < p) & observed
    would_be_fully_missing = np.where((base_mask | extra_hide).sum(axis=1) == d)[0]
    extra_hide[would_be_fully_missing] = False

    combined_mask = base_mask | extra_hide
    X_true_cv = X_cv.copy()
    X_nas_cv = np.where(combined_mask, np.nan, X_cv)

    if verbose:
        _print_cv_setup(base_mask, extra_hide)

    results = []
    best_anchor_nodes, best_score = None, np.inf

    for trial in range(n_trials):
        anchor_idx = rng.choice(n_total, size=m, replace=(m > n_total))
        candidate_anchors = dataset_loaded[anchor_idx].copy()

        X_imputed, _, _ = psd_impute(
            X_nas_cv, combined_mask,
            m=m, eta_init=eta_init, alpha=alpha, lbd=lbd / n_cv, mu=mu / n_cv,
            l_rate_nodes=lr_nodes, l_rate_param=lr_param,
            nbr_bounce=1, nbr_gradient_steps=1, nbr_newton_step_Q=newton_steps,
            seed=seed, fixed_anchor_nodes=candidate_anchors,
        )
        rmse = _cv_score(X_imputed, X_true_cv, extra_hide, base_mask, 'rmse')
        ed = _cv_score(X_imputed, X_true_cv, extra_hide, base_mask, 'ed')
        ot_dist = _cv_score(X_imputed, X_true_cv, extra_hide, base_mask, 'ot')
        score = {'rmse': rmse, 'ed': ed, 'ot': ot_dist}[cv_metric]
        results.append((trial, rmse, ed, ot_dist))
        if verbose:
            print(f"  Anchor CV trial {trial + 1}/{n_trials}: "
                  f"RMSE={rmse:.4f}  ED={ed:.4f}  OT={ot_dist:.4f}"
                  + ("" if cv_metric == 'rmse' else f"  (selecting by {cv_metric.upper()})"))

        if score < best_score:
            best_score = score
            best_anchor_nodes = candidate_anchors

    return best_anchor_nodes, best_score, results

