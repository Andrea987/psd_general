"""
Imputation with a Gaussian PSD density model, fit directly on the partially-observed data via
alternating_minimization (K_S/H_NS already handle arbitrary per-observation missing patterns),
then, for each row, filling its missing entries with the mean of the conditional distribution
given its observed entries (see marginalize_condition.condition and sampling.mean).
"""

import warnings
from collections import namedtuple

import numpy as np
import ot
from sklearn.impute import SimpleImputer

from alternating_minimization import alternating_minimization
from marginalize_condition import condition
from psd import energy_distance
from sampling import mean as model_mean, sample_bisection, check_normalized


def reveal_random_entry_if_fully_missing(mask, rng):
    """
    Guarantee no row of mask is fully missing: for any row where every entry is True (missing),
    reveal (set to False) one randomly-chosen entry in that row. A fully-missing row has nothing
    to condition on -- conditioning falls back to the model's unconditional marginal for that row,
    which is a real, unwanted edge case (a real dataset row, or a cross-validation validation row,
    contributing no genuine information) rather than an error, so it is silently possible unless
    explicitly excluded here.

    :param mask: (n, d) boolean array, True = missing. Modified in place.
    :param rng: numpy Generator used to pick which entry to reveal
    :return: mask, for chaining
    """
    d = mask.shape[1]
    fully_missing = np.where(mask.sum(axis=1) == d)[0]
    if len(fully_missing) > 0:
        revealed = rng.integers(0, d, size=len(fully_missing))
        mask[fully_missing, revealed] = False
    return mask


def hide_at_least_one_entry_if_none_hidden(extra_hide, observed, rng):
    """
    Guarantee cross-validation always has at least one entry to actually validate against: if the
    random extra_hide draw came up empty everywhere (possible with a small n_cv and/or small p --
    every per-entry MCAR(p) draw simply missed), force one randomly-chosen currently-observed
    entry to be hidden instead of silently leaving nothing to score. Without this, every scoring
    metric ends up computed over zero entries (RMSE: nan from an empty mean; OT: nan by
    _cv_score's own n_m == 0 guard), and since Python's min() with an all-nan key silently returns
    whichever candidate was tried first rather than comparing anything, cross-validation would
    appear to pick a "winner" that was never actually validated at all.

    Only ever picks from a row that has MORE THAN ONE observed entry, so the forced hide can never
    itself create a fully-missing row -- same rule already applied to the random extra_hide draw
    itself (see the would_be_fully_missing correction next to every call site). If literally every
    row has at most one observed entry, there is truly nothing safe to hide (any pick would leave
    some row fully missing), and extra_hide is left as-is.

    :param extra_hide: (n, d) boolean array, True = hidden for validation. Modified in place.
    :param observed: (n, d) boolean array, True = currently observed (not already missing) --
        the pool this can pick from
    :param rng: numpy Generator used to pick which entry to hide
    :return: extra_hide, for chaining
    """
    if not extra_hide.any():
        rows_with_spare_observed = np.where(observed.sum(axis=1) > 1)[0]
        if len(rows_with_spare_observed) > 0:
            row = rows_with_spare_observed[rng.integers(len(rows_with_spare_observed))]
            observed_cols = np.where(observed[row])[0]
            col = observed_cols[rng.integers(len(observed_cols))]
            extra_hide[row, col] = True
    return extra_hide


def fit_psd_model(dataset, masks, m, eta_init, alpha, lbd, mu, l_rate_nodes, l_rate_param,
                   nbr_bounce, nbr_gradient_steps, nbr_newton_step_Q, seed=0, verbose=False,
                   verbose_newton=False, anchor_init='data_subset', anchor_impute='mean',
                   eta_init_mode='fixed', fixed_anchor_nodes=None, dataset_true=None):
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
        initially-imputed data (see anchor_impute); 'uniform_hypercube' samples them i.i.d.
        uniformly in the bounding box of that same initially-imputed data instead
    :param anchor_impute: how the missing entries are filled to build that initial imputation --
        any sklearn SimpleImputer strategy: 'mean' (default), 'median', 'most_frequent', or
        'constant' (fills with 0). Only ever used to place the anchor nodes, never to produce the
        returned imputation. 'constant' is the leak-proof choice by construction: the filled value
        is 0 regardless of the data, so it cannot carry information about any entry
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
    # fill_value is ignored unless anchor_impute == 'constant'
    initial_imputed = SimpleImputer(strategy=anchor_impute,
                                    fill_value=0.0).fit_transform(dataset_nan)

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
               anchor_init='data_subset', anchor_impute='mean', eta_init_mode='fixed',
               fixed_anchor_nodes=None, dataset_true=None, n_imputations=0):
    """
    :param X_nas: (n, d) data, NaN on the missing entries
    :param mask: (n, d) boolean (or 0/1), True/1 = missing
    :param m, eta_init, alpha, lbd, mu, l_rate_nodes, l_rate_param, nbr_bounce,
        nbr_gradient_steps, nbr_newton_step_Q, seed, verbose, verbose_newton, anchor_init,
        anchor_impute, eta_init_mode, fixed_anchor_nodes, dataset_true: see fit_psd_model
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
        anchor_impute=anchor_impute, eta_init_mode=eta_init_mode,
        fixed_anchor_nodes=fixed_anchor_nodes, dataset_true=dataset_true,
    )

    X_imputed = impute_mean(dataset, mask, Q, W, eta)
    multiple_imp = impute_multiple(dataset, mask, Q, W, eta, n_imputations) if n_imputations > 0 else None

    return X_imputed, history, multiple_imp


_CvSplit = namedtuple('_CvSplit', [
    'training_dataset',       # (num_training_rows, d) rows to fit on, 0 on the missing entries
    'training_mask',          # (num_training_rows, d) True = missing
    'training_anchor_pool',   # (num_training_rows, d) initially-imputed, to draw anchor nodes from
    'validation_dataset',     # (num_validation_rows, d) rows to score on, 0 where hidden
    'hidden_from_predictor',  # (num_validation_rows, d) already_missing | extra_hide
    'validation_truth',       # (num_validation_rows, d) true values at the extra_hide positions
    'extra_hide',             # (num_validation_rows, d) the entries actually scored
    'already_missing',        # (num_validation_rows, d) missing before extra-hiding
])


def _split_sizes(total_training_rows, num_training_rows, num_validation_rows):
    """
    Validated sizes of one disjoint training/validation partition. num_training_rows is the knob
    (it drives the cost -- the Newton solve scales with the rows fit on); the validation part then
    takes EVERY remaining row unless num_validation_rows caps it, since scoring is much cheaper
    than fitting and more validation rows mean a lower-variance score.
    """
    num_training_rows = min(num_training_rows, total_training_rows - 1)
    rows_left_over = total_training_rows - num_training_rows
    num_validation_rows = (rows_left_over if num_validation_rows is None
                           else min(num_validation_rows, rows_left_over))
    assert num_training_rows >= 1 and num_validation_rows >= 1, (
        f"need at least 2 training rows to split into a training set and a validation set, got "
        f"{total_training_rows} (num_training_rows={num_training_rows}, "
        f"num_validation_rows={num_validation_rows})"
    )
    return num_training_rows, num_validation_rows


def _draw_cv_split(train_data_with_nan, train_missing_mask, extra_hide_probability,
                   num_training_rows, num_validation_rows, anchor_impute, rng):
    """
    Draw ONE disjoint training/validation partition of the training rows -- the single
    methodology every cross-validation function in this module uses, so they cannot drift apart.

    The training rows are what a candidate model is fit on (and the only rows its anchor nodes are
    ever drawn from); the validation rows are held out, have an extra MCAR fraction of their still
    observed entries hidden, and are then imputed by the fitted model and scored on the recovery
    of exactly those extra-hidden entries.

    The two parts share no row. That is what makes a leak impossible: the extra-hidden entries are
    ordinary OBSERVED training data (that is the whole premise of being able to grade them), so
    anything built from a validation row -- an anchor node above all -- would carry the very
    values being scored. Drawing anchors only from the disjoint training rows removes that channel
    by construction rather than guarding against it.

    :return: _CvSplit
    """
    total_training_rows, num_columns = train_data_with_nan.shape
    permuted = rng.permutation(total_training_rows)
    training_idx = permuted[:num_training_rows]
    validation_idx = permuted[num_training_rows:num_training_rows + num_validation_rows]

    # ---- training side: never touched by the extra hiding below ----
    training_data = train_data_with_nan[training_idx]  # already NaN where the mask says so
    training_mask = train_missing_mask[training_idx]
    # fill_value is ignored unless the strategy is 'constant'
    training_anchor_pool = SimpleImputer(strategy=anchor_impute, fill_value=0.0).fit_transform(
        np.where(training_mask, np.nan, training_data))
    training_dataset = np.where(training_mask, 0.0, training_data)  # dummy, as psd_impute does

    # ---- validation side ----
    validation_data = train_data_with_nan[validation_idx]
    already_missing = train_missing_mask[validation_idx]
    # a fully-missing row can't be fixed here -- there is no real value left in
    # train_data_with_nan to reveal, it's already NaN. This must be prevented where the mask is
    # actually created, before any value is discarded (see main_real_dataset.py, which reveals one
    # entry per fully-missing row right after building the real mask). Fail loudly instead of
    # silently mishandling it if that invariant was ever violated.
    assert not (already_missing.sum(axis=1) == num_columns).any(), (
        "train_missing_mask has at least one fully-missing row in the validation subsample -- fix "
        "this where the mask is constructed (see reveal_random_entry_if_fully_missing), not here"
    )

    still_observed = ~already_missing
    extra_hide = (rng.random((num_validation_rows, num_columns))
                  < extra_hide_probability) & still_observed
    # never extra-hide the last observed entry of a row (would leave nothing to condition on)
    extra_hide[np.where((already_missing | extra_hide).sum(axis=1) == num_columns)[0]] = False
    # guarantee there is always at least one entry to actually validate against
    hide_at_least_one_entry_if_none_hidden(extra_hide, still_observed, rng)

    # the "true" value at every extra_hide position is simply what validation_data already holds
    # there (it's observed -- not NaN -- since extra_hide is a subset of ~already_missing);
    # capture it BEFORE additionally hiding it below
    validation_truth = validation_data.copy()

    hidden_from_predictor = already_missing | extra_hide
    assert np.array_equal(hidden_from_predictor | already_missing, hidden_from_predictor), (
        "hidden_from_predictor must be a superset of already_missing -- an entry already missing "
        "beforehand must never come back as observed"
    )
    validation_dataset = np.where(hidden_from_predictor, 0.0, validation_data)

    return _CvSplit(training_dataset, training_mask, training_anchor_pool, validation_dataset,
                    hidden_from_predictor, validation_truth, extra_hide, already_missing)


def _score_all_metrics(imputed, split):
    """(rmse, ed, ot) for an imputation of split's validation rows -- see _cv_score."""
    return (_cv_score(imputed, split.validation_truth, split.extra_hide, split.already_missing, m)
            for m in ('rmse', 'ed', 'ot'))


def cross_validate_hyperparams(train_data_with_nan, train_missing_mask, extra_hide_probability,
                                lr_nodes_grid, lr_param_grid, eta_grid, num_anchor_nodes=10,
                                num_splits=1, num_training_rows=200, num_validation_rows=None,
                                num_bounces=20, num_gradient_steps=5, newton_iterations=20,
                                alpha=1e-6, lbd=1e-1, mu=1e-3, seed=0, verbose=False,
                                cv_metric='rmse', anchor_impute='mean'):
    """
    Cross-validation for (l_rate_nodes, l_rate_param, eta_init), on the same DISJOINT
    TRAINING/VALIDATION ROW SPLIT that cross_validate_anchor_nodes uses (see _draw_cv_split):

      OUTER, num_splits times -- draw a fresh, independent split of the training rows.
        INNER, once per (lr_nodes, lr_param, eta_init) combination -- fit on that split's TRAINING
        rows, impute its held-out VALIDATION rows, and score the recovery of just the entries
        extra-hidden there (see _cv_score).

      Return the combination with the best single score over all splits * combinations.

    num_splits=1 is one split, every combination sitting one common exam. Raising it makes the
    winner less dependent on one split coming up easy or hard, at proportional cost; combinations
    from different splits are scored on different validation rows, so their scores are not
    perfectly comparable.

    (train_data_with_nan, train_missing_mask) is the ONLY data source anywhere in this function:
    the split and its already_missing mask come from the REAL, already-missing training data, not
    a freshly synthesized MCAR mask over a separate fully-observed array. Beyond avoiding leaks,
    that matters because a freshly-synthesized mask is always plain MCAR, so under
    --MAR/--MNAR_log/--MNAR_quant the hyperparameters would be tuned against a missingness pattern
    that doesn't match what the real fit faces.

    :param train_data_with_nan: (total_training_rows, num_columns) the REAL training data, NaN at
        the entries train_missing_mask marks missing -- typically main_real_dataset.py's data_nas
    :param train_missing_mask: (total_training_rows, num_columns) boolean (or 0/1), True/1 =
        missing -- typically main_real_dataset.py's mask
    :param extra_hide_probability: MCAR probability for the additional validation-only hide, same
        convention as the main experiment's --p
    :param lr_nodes_grid, lr_param_grid, eta_grid: lists of candidate l_rate_nodes, l_rate_param,
        eta_init (LOG-precision, see fit_psd_model) values; every combination is tried
    :param num_anchor_nodes: anchor nodes for the validation fits (default 10, small for speed).
        These are picked by fit_psd_model itself, from the split's training rows
    :param num_splits, num_training_rows, num_validation_rows: the split geometry -- see
        _split_sizes and cross_validate_anchor_nodes, which take the same three
    :param num_bounces: alternating_minimization bounces per validation fit (default 20, few for
        speed)
    :param num_gradient_steps, newton_iterations, alpha, lbd, mu, seed: see fit_psd_model (lbd/mu
        divided by num_training_rows, matching the full-scale convention)
    :param verbose: if True, print every combination's validation RMSE/ED/OT as it's evaluated
        (all three are always computed and printed, regardless of cv_metric, purely for visibility)
    :param cv_metric: 'rmse' (default), 'ed', or 'ot' -- which of the three printed scores is
        actually used to pick the winning combination; see _cv_score
    :param anchor_impute: how missing entries are filled to place each validation fit's anchor
        nodes -- see fit_psd_model. Should match the value used for the real fit
    :return: (best_lr_nodes, best_lr_param, best_eta, best_score, results) -- best_score is
        whichever of rmse/ed/ot cv_metric selected on; results is the full list of
        (split_index, lr_nodes, lr_param, eta, rmse, ed, ot) tuples tried, in evaluation order
    """
    metric_idx = {'rmse': 4, 'ed': 5, 'ot': 6}[cv_metric]
    rng = np.random.default_rng(seed)
    train_missing_mask = np.asarray(train_missing_mask, dtype=bool)
    total_training_rows = train_data_with_nan.shape[0]
    num_training_rows, num_validation_rows = _split_sizes(
        total_training_rows, num_training_rows, num_validation_rows)

    results = []
    for split_index in range(num_splits):
        split = _draw_cv_split(train_data_with_nan, train_missing_mask, extra_hide_probability,
                               num_training_rows, num_validation_rows, anchor_impute, rng)
        if verbose:
            print(f"  Split {split_index + 1}/{num_splits}: {num_training_rows} training rows, "
                  f"{num_validation_rows} validation rows, disjoint (of {total_training_rows} "
                  f"training rows).")
            if split_index == 0:
                _print_cv_setup(split.already_missing, split.extra_hide)

        for lr_nodes in lr_nodes_grid:
            for lr_param in lr_param_grid:
                for eta_init in eta_grid:
                    # fit on the TRAINING rows (anchor nodes picked by fit_psd_model from those
                    # rows), then apply the fitted model to the held-out VALIDATION rows
                    Q, anchors, precision = fit_psd_model(
                        split.training_dataset, split.training_mask.astype(float),
                        num_anchor_nodes, eta_init, alpha, lbd / num_training_rows,
                        mu / num_training_rows, lr_nodes, lr_param, num_bounces,
                        num_gradient_steps, newton_iterations, seed=seed,
                        anchor_impute=anchor_impute,
                    )[:3]
                    imputed = impute_mean(split.validation_dataset, split.hidden_from_predictor,
                                          Q, anchors, precision)

                    rmse, ed, ot_dist = _score_all_metrics(imputed, split)
                    results.append((split_index, lr_nodes, lr_param, eta_init, rmse, ed, ot_dist))
                    if verbose:
                        print(f"    Split {split_index + 1}/{num_splits} "
                              f"lr_nodes={lr_nodes:.1e}  lr_param={lr_param:.1e}  "
                              f"eta_init={eta_init:.4f}  RMSE={rmse:.4f}  ED={ed:.4f}  "
                              f"OT={ot_dist:.4f}"
                              + ("" if cv_metric == 'rmse'
                                 else f"  (selecting by {cv_metric.upper()})"))

    best = min(results, key=lambda r: r[metric_idx])
    return best[1], best[2], best[3], best[metric_idx], results


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


def cross_validate_anchor_nodes(train_data_with_nan, train_missing_mask, extra_hide_probability,
                                 num_anchor_nodes, lr_nodes, lr_param, eta_init,
                                 num_candidates_per_split=5, num_splits=1,
                                 num_training_rows=200, num_validation_rows=None,
                                 newton_iterations=10, alpha=1e-6, lbd=1e-1, mu=1e-3, seed=0,
                                 verbose=False, cv_metric='rmse', anchor_impute='mean'):
    """
    Fast cross-validation for the CHOICE of anchor nodes, run after cross_validate_hyperparams (or
    with hand-picked hyperparameters). Two nested loops:

      OUTER, num_splits times -- draw a fresh, independent split of the training rows into a
      training set (num_training_rows rows) and a validation set (every remaining row, unless
      num_validation_rows caps it), the two sharing no row:
        1. Build this split's candidate anchor pool by initially-imputing (see anchor_impute) its
           FIT rows only.
        2. Hide an extra MCAR fraction (extra_hide_probability) of what is still observed in this
           split's VALIDATION rows. Those entries are the exam; their true values are known but
           withheld from every predictor below.

        INNER, num_candidates_per_split times -- one candidate anchor-node set:
        3. Take a random num_anchor_nodes-row subsample of this split's pool and fit on this
           split's fit rows -- just the Newton method (nbr_bounce=1, so anchor nodes and precision
           are never moved; there is no gradient-step phase with a single bounce) for up to
           newton_iterations iterations. That yields a predictor (Q, anchor_nodes, precision).
        4. Impute this split's held-out validation rows with that predictor and score the recovery
           of just the extra-hidden entries (see _cv_score).

      Finally, return the single best-scoring candidate over all num_splits *
      num_candidates_per_split of them.

    num_splits=1 reproduces the single-split procedure exactly: one split, every candidate sitting
    one common exam. Raising it helps in two distinct ways: no row is permanently barred from
    being anchor material (a row excluded from the fit set of one split lands in another's), and
    the winner is less dependent on one particular split coming up easy or hard. The cost is
    proportional -- num_splits * num_candidates_per_split fits in total.

    Caveat when num_splits > 1: candidates from different splits are scored on DIFFERENT
    validation rows, so their scores are not perfectly comparable, and a candidate can win partly
    because its split's exam was easier. Within a split the comparison is exact. num_splits=1
    avoids the issue entirely at the cost of the two benefits above.

    Because each split's fit set and validation set share no rows, no candidate anchor can contain
    a value that the same split then scores: the anchors come from fit rows, the scored entries
    live in validation rows, and the two are disjoint by construction. Steps 1 and 3 also never
    see the extra hiding at all -- it is applied only to the validation rows, after fitting.

    (train_data_with_nan, train_missing_mask) remains the ONLY data source anywhere in this
    function; there is no separate ground-truth array. The "true" value at every extra-hidden
    position is simply whatever train_data_with_nan already holds there -- those entries are
    observed (train_missing_mask == False) -- captured before additionally hiding them.

    (Two earlier versions leaked here. The first drew candidates straight from a separate,
    fully-observed ground_truth array, so a candidate could coincide with a training row and leak
    that row's true value into its own imputation -- a row's error collapsed 4-130x when its true
    vector was planted as an anchor. The second built candidates from the whole training array
    before the extra hiding: the extra-hidden entries are observed training data at that point, so
    SimpleImputer passed their true values through verbatim and a candidate drawn from a
    validation row carried the very answers it was scored on -- candidates overlapping the
    validation rows scored ~13.5% better for that reason alone. The row split removes the shared
    rows that both leaks travelled through.)

    :param train_data_with_nan: (total_training_rows, num_columns) the REAL training data, NaN at
        the entries train_missing_mask marks missing -- typically main_real_dataset.py's data_nas
    :param train_missing_mask: (total_training_rows, num_columns) boolean (or 0/1), True/1 =
        missing -- typically main_real_dataset.py's mask
    :param extra_hide_probability: MCAR probability for the additional validation-only hide, same
        convention as the main experiment's --p
    :param num_anchor_nodes: how many anchor nodes each candidate set contains -- should match the
        number used for the real run
    :param lr_nodes, lr_param, eta_init: the hyperparameters already chosen (e.g. by
        cross_validate_hyperparams); held fixed throughout this stage since bounce=1 means they're
        never actually used for a gradient step anyway
    :param num_candidates_per_split: how many candidate anchor-node sets to try within each split
        (default 5)
    :param num_splits: how many independent train/validation splits to draw (default 1, which
        reproduces the single-split procedure exactly). Total fits = num_splits *
        num_candidates_per_split
    :param num_training_rows: rows each candidate is trained on, per split (default 200). This is
        the knob that sets the cost -- the Newton solve scales with the number of rows it fits on
    :param num_validation_rows: rows held out to score on, per split. None (default) uses EVERY
        row not taken by the training set, which is free variance reduction: scoring is much
        cheaper than fitting, so there is no reason to leave rows idle. Pass an integer only to
        cap it deliberately
    :param newton_iterations: max Newton iterations per candidate (default 10)
    :param alpha, lbd, mu, seed: see fit_psd_model (lbd/mu divided by num_training_rows, the size
        of the set actually being fit, matching the full-scale convention)
    :param verbose: if True, print each split's geometry and every candidate's validation
        RMSE/ED/OT as it's evaluated (all three are always computed and printed, regardless of
        cv_metric, purely for visibility)
    :param cv_metric: 'rmse' (default), 'ed', or 'ot' -- which of the three printed scores is
        actually used to pick the winning candidate; see _cv_score. 'ed'/'ot' compare whole rows,
        so the entries already missing before extra-hiding are zeroed out in both the imputed and
        true arrays first, so the score reflects only how well the extra-hidden entries were
        recovered
    :param anchor_impute: how missing entries are filled to build the candidate anchors -- see
        fit_psd_model. Should match the value used for the real fit, since the candidates returned
        here are used there directly
    :return: (best_anchor_nodes, best_score, results) -- best_anchor_nodes is
        (num_anchor_nodes, num_columns), ready to use directly as fixed_anchor_nodes for the real
        fit on (train_data_with_nan, train_missing_mask); results is the list of
        (split_index, candidate_index, rmse, ed, ot) tuples tried, in evaluation order
    """
    rng = np.random.default_rng(seed)
    train_missing_mask = np.asarray(train_missing_mask, dtype=bool)
    total_training_rows, num_columns = train_data_with_nan.shape

    num_training_rows, num_validation_rows = _split_sizes(
        total_training_rows, num_training_rows, num_validation_rows)
    if num_anchor_nodes > num_training_rows:
        warnings.warn(
            f"num_anchor_nodes={num_anchor_nodes} exceeds num_training_rows={num_training_rows}, "
            f"so the anchor nodes must be drawn WITH replacement and some candidates will contain "
            f"duplicate rows (a degenerate basis). Either lower num_anchor_nodes, raise "
            f"num_training_rows, or use a larger training set.",
            stacklevel=2,
        )

    results = []
    best_anchor_nodes, best_score = None, np.inf

    # OUTER LOOP: a fresh, independently drawn training/validation split each time. With
    # num_splits=1 this collapses to the single-split procedure exactly
    for split_index in range(num_splits):
        split = _draw_cv_split(train_data_with_nan, train_missing_mask, extra_hide_probability,
                               num_training_rows, num_validation_rows, anchor_impute, rng)
        if verbose:
            print(f"  Split {split_index + 1}/{num_splits}: {num_training_rows} training rows, "
                  f"{num_validation_rows} validation rows, disjoint (of {total_training_rows} "
                  f"training rows).")
            if split_index == 0:
                _print_cv_setup(split.already_missing, split.extra_hide)

        # INNER LOOP: candidate anchor-node sets, all drawn from THIS split's training rows and
        # all scored on THIS split's validation rows, so they face an identical exam
        for candidate in range(num_candidates_per_split):
            anchor_idx = rng.choice(num_training_rows, size=num_anchor_nodes,
                                    replace=(num_anchor_nodes > num_training_rows))
            candidate_anchors = split.training_anchor_pool[anchor_idx].copy()

            # fit on the TRAINING rows (Newton only -- nbr_bounce=1 leaves the anchor nodes and
            # precision exactly as given), then apply it to the held-out validation rows
            Q, anchors, precision = fit_psd_model(
                split.training_dataset, split.training_mask.astype(float), num_anchor_nodes,
                eta_init, alpha, lbd / num_training_rows, mu / num_training_rows,
                lr_nodes, lr_param, nbr_bounce=1, nbr_gradient_steps=1,
                nbr_newton_step_Q=newton_iterations, seed=seed,
                fixed_anchor_nodes=candidate_anchors, anchor_impute=anchor_impute,
            )[:3]
            imputed = impute_mean(split.validation_dataset, split.hidden_from_predictor,
                                  Q, anchors, precision)

            rmse, ed, ot_dist = _score_all_metrics(imputed, split)
            score = {'rmse': rmse, 'ed': ed, 'ot': ot_dist}[cv_metric]
            results.append((split_index, candidate, rmse, ed, ot_dist))
            if verbose:
                print(f"    Split {split_index + 1}/{num_splits} candidate "
                      f"{candidate + 1}/{num_candidates_per_split}: "
                      f"RMSE={rmse:.4f}  ED={ed:.4f}  OT={ot_dist:.4f}"
                      + ("" if cv_metric == 'rmse'
                         else f"  (selecting by {cv_metric.upper()})"))

            if score < best_score:
                best_score = score
                best_anchor_nodes = candidate_anchors

    return best_anchor_nodes, best_score, results


def cross_validate_hyperparams_and_anchors(
        train_data_with_nan, train_missing_mask, extra_hide_probability, num_anchor_nodes,
        lr_nodes_grid, lr_param_grid, eta_grid, num_candidates_per_split=5, num_splits=1,
        num_training_rows=200, num_validation_rows=None, num_bounces=5, num_gradient_steps=5,
        newton_iterations=20, alpha=1e-6, lbd=1e-1, mu=1e-3, seed=0, verbose=False,
        cv_metric='rmse', anchor_impute='mean'):
    """
    JOINT cross-validation of the hyperparameters AND the anchor nodes, as one search, on the same
    disjoint training/validation row split the other two use (see _draw_cv_split).

    cross_validate_hyperparams followed by cross_validate_anchor_nodes is a GREEDY search: the
    hyperparameters are chosen first, against whatever anchor nodes fit_psd_model happened to pick,
    and are then frozen while the anchors are chosen. That misses any interaction between the two
    -- an anchor set that is poor under the winning learning rates but excellent under some other
    combination is never seen. This function searches the product instead:

      OUTER, num_splits times -- draw a fresh split, and draw num_candidates_per_split candidate
      anchor-node sets from its TRAINING rows. The same candidates are reused across every
      hyperparameter combination in this split, so the comparison between combinations is exact.
        INNER, for every (lr_nodes, lr_param, eta_init) x candidate pair -- fit on the training
        rows with that combination AND that anchor set, impute the held-out validation rows, score.

      Return the single best (hyperparameters, anchor nodes) pair over everything tried.

    Unlike cross_validate_anchor_nodes, the fit here runs the full num_bounces (not a single
    Newton-only bounce), so the anchor nodes are moved by gradient steps exactly as they will be in
    the real fit. A candidate is therefore judged as a STARTING POINT that the fit refines, which
    is how the real run actually uses it -- at the cost of being much more expensive per candidate.

    COST: num_splits * |lr_nodes_grid| * |lr_param_grid| * |eta_grid| * num_candidates_per_split
    fits, i.e. the anchor stage's cost multiplied by the whole hyperparameter grid. With
    run_experiment_1.sbatch's 625-combination grid and 10 candidates that is 6250 fits per split
    per experiment -- shrink the grids before enabling this.

    :param num_anchor_nodes: anchor nodes per candidate set -- should match the real run's
    :param lr_nodes_grid, lr_param_grid, eta_grid: as in cross_validate_hyperparams
    :param num_candidates_per_split: candidate anchor-node sets drawn per split
    :param num_splits, num_training_rows, num_validation_rows: split geometry, see _split_sizes
    :param num_bounces: alternating minimization bounces per fit (default 5, deliberately small).
        This is the main cost lever: the search runs |grid| * candidates FULL fits, so every extra
        bounce is multiplied by that whole product
    :param num_gradient_steps, newton_iterations, alpha, lbd, mu, seed: see fit_psd_model (lbd/mu
        divided by num_training_rows)
    :param verbose, cv_metric, anchor_impute: as in the other two CV functions
    :return: (best_lr_nodes, best_lr_param, best_eta, best_anchor_nodes, best_score, results) --
        best_anchor_nodes is (num_anchor_nodes, num_columns), ready to pass as fixed_anchor_nodes;
        results is the list of (split_index, candidate_index, lr_nodes, lr_param, eta, rmse, ed,
        ot) tuples tried, in evaluation order
    """
    rng = np.random.default_rng(seed)
    train_missing_mask = np.asarray(train_missing_mask, dtype=bool)
    total_training_rows = train_data_with_nan.shape[0]
    num_training_rows, num_validation_rows = _split_sizes(
        total_training_rows, num_training_rows, num_validation_rows)
    if num_anchor_nodes > num_training_rows:
        warnings.warn(
            f"num_anchor_nodes={num_anchor_nodes} exceeds num_training_rows={num_training_rows}, "
            f"so the anchor nodes must be drawn WITH replacement and some candidates will contain "
            f"duplicate rows (a degenerate basis). Either lower num_anchor_nodes, raise "
            f"num_training_rows, or use a larger training set.",
            stacklevel=2,
        )

    total_fits = (num_splits * len(lr_nodes_grid) * len(lr_param_grid) * len(eta_grid)
                  * num_candidates_per_split)
    results = []
    best = {'score': np.inf, 'anchors': None, 'lr_nodes': None, 'lr_param': None, 'eta': None}

    for split_index in range(num_splits):
        split = _draw_cv_split(train_data_with_nan, train_missing_mask, extra_hide_probability,
                               num_training_rows, num_validation_rows, anchor_impute, rng)
        # candidate anchor sets for this split, drawn ONCE and reused across every hyperparameter
        # combination below, so combinations are compared on identical anchor sets
        candidates = [
            split.training_anchor_pool[
                rng.choice(num_training_rows, size=num_anchor_nodes,
                           replace=(num_anchor_nodes > num_training_rows))].copy()
            for _ in range(num_candidates_per_split)
        ]

        if verbose:
            print(f"  Split {split_index + 1}/{num_splits}: {num_training_rows} training rows, "
                  f"{num_validation_rows} validation rows, disjoint (of {total_training_rows} "
                  f"training rows); {len(candidates)} anchor candidates x "
                  f"{len(lr_nodes_grid) * len(lr_param_grid) * len(eta_grid)} hyperparameter "
                  f"combinations = {total_fits // num_splits} fits this split "
                  f"({total_fits} in total).")
            if split_index == 0:
                _print_cv_setup(split.already_missing, split.extra_hide)

        for lr_nodes in lr_nodes_grid:
            for lr_param in lr_param_grid:
                for eta_init in eta_grid:
                    for candidate_index, candidate_anchors in enumerate(candidates):
                        # full num_bounces: the anchors move, exactly as in the real fit
                        Q, anchors, precision = fit_psd_model(
                            split.training_dataset, split.training_mask.astype(float),
                            num_anchor_nodes, eta_init, alpha, lbd / num_training_rows,
                            mu / num_training_rows, lr_nodes, lr_param, num_bounces,
                            num_gradient_steps, newton_iterations, seed=seed,
                            fixed_anchor_nodes=candidate_anchors, anchor_impute=anchor_impute,
                        )[:3]
                        imputed = impute_mean(split.validation_dataset,
                                              split.hidden_from_predictor, Q, anchors, precision)

                        rmse, ed, ot_dist = _score_all_metrics(imputed, split)
                        score = {'rmse': rmse, 'ed': ed, 'ot': ot_dist}[cv_metric]
                        results.append((split_index, candidate_index, lr_nodes, lr_param,
                                        eta_init, rmse, ed, ot_dist))
                        if verbose:
                            print(f"    Split {split_index + 1}/{num_splits} candidate "
                                  f"{candidate_index + 1}/{num_candidates_per_split}  "
                                  f"lr_nodes={lr_nodes:.1e}  lr_param={lr_param:.1e}  "
                                  f"eta_init={eta_init:.4f}  RMSE={rmse:.4f}  ED={ed:.4f}  "
                                  f"OT={ot_dist:.4f}"
                                  + ("" if cv_metric == 'rmse'
                                     else f"  (selecting by {cv_metric.upper()})"))

                        if score < best['score']:
                            best = {'score': score, 'anchors': candidate_anchors,
                                    'lr_nodes': lr_nodes, 'lr_param': lr_param, 'eta': eta_init}

    return (best['lr_nodes'], best['lr_param'], best['eta'], best['anchors'], best['score'],
            results)
