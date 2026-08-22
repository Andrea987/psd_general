import numpy as np
import ot
from sklearn.datasets import make_moons

from alternating_minimization import alternating_minimization
from plot_toy_2d import plot_initial_vs_final_model
from psd import energy_distance
from sampling import sample_bisection
from psd_imputer import impute_multiple


def generate_toy_info(n=400, m=30, seed=0):
    rng = np.random.default_rng(seed)
    d = 2

    X, _ = make_moons(n_samples=n, noise=0.1, random_state=seed)

    p_missing = 0.1
    mask = (rng.random((n, d)) < p_missing).astype(float)  # 1 = missing, 0 = seen
    fully_missing_rows = np.where(mask.sum(axis=1) == d)[0]
    if len(fully_missing_rows) > 0:
        revealed_dims = rng.integers(0, d, size=len(fully_missing_rows))
        mask[fully_missing_rows, revealed_dims] = 0

    low, high = X.min(axis=0), X.max(axis=0)
    W = rng.uniform(low, high, size=(m, d))
    eta = np.array([5.0, 5.0])

    L = rng.standard_normal((m, m))
    Q0 = L @ L.T + np.eye(m) * 2  # PD starting point (alternating_minimization renormalizes it)

    return {
        'dataset': X, 'masks': mask, 'anchor_nodes': W, 'precision': eta, 'Q': Q0,
        'alpha': 1e-6, 'lbd': 0.0001, 'mu': 0.0001,
        'l_rate_nodes': 1e-1, 'l_rate_param': 1e-4, 'nbr_bounce': 1, 'nbr_gradient_steps': 5,
        'verbose': True,
    }


if __name__ == '__main__':
    info = generate_toy_info()
    info = dict(info)
    info['nbr_bounce'] = 150
    Q, W, eta, history = alternating_minimization(info)
    final_loss, final_lagrangian = history[-1]
    plot_initial_vs_final_model(info, Q, W, eta, final_loss, save_path='toy_test_1_final_model.png')

    # Sanity check: does energy_distance actually discriminate a good fit from a bad one?
    X = info['dataset']
    n = X.shape[0]
    model_samples = sample_bisection(W, eta, Q, N=n)

    rng = np.random.default_rng(1)
    fresh_moons, _ = make_moons(n_samples=n, noise=0.1, random_state=1)
    low, high = X.min(axis=0), X.max(axis=0)
    uniform_noise = rng.uniform(low, high, size=X.shape)

    print(f"energy_distance(X, model samples):  {energy_distance(X, model_samples):.4f}  "
          f"(learned model vs. original data -- should be small if the fit is good)")
    print(f"energy_distance(X, fresh moons):     {energy_distance(X, fresh_moons):.4f}  "
          f"(two i.i.d. draws from the true distribution -- noise floor)")
    print(f"energy_distance(X, uniform noise):   {energy_distance(X, uniform_noise):.4f}  "
          f"(clearly different distribution -- should be clearly larger)")

    # Impute the missing components with the final learned model via multiple imputation: draw
    # n_imputations independent completions per row (sampling each row's conditional distribution,
    # not just its mean), then check whether OT/ED on the stacked point cloud agree with what the
    # plots show visually.
    mask = info['masks'].astype(bool)
    M = mask.sum(axis=1) > 0  # rows with at least one missing entry
    nimp = M.sum()

    n_imputations = 20
    imputations = impute_multiple(X, mask, Q, W, eta, n_imputations)  # (n_imputations, n, d)

    # OT, missing rows only: stack the n_imputations completions of just the affected rows
    stacked_M = imputations[:, M, :].reshape(-1, X.shape[1])  # (n_imputations * nimp, d)
    n_stacked_M = stacked_M.shape[0]
    dists = ((stacked_M[:, None] - X[M]) ** 2).sum(2) / 2.
    ot_distance = ot.emd2(np.ones(n_stacked_M) / n_stacked_M, np.ones(nimp) / nimp, dists)

    # ED, full dataset: stack the n_imputations completions of every row
    stacked_full = imputations.reshape(-1, X.shape[1])  # (n_imputations * n, d)
    ed_distance = energy_distance(stacked_full, X)

    # same two metrics, but between the (fully-observed) original data and a fresh same-distribution
    # sample -- a "should be small" baseline to judge the imputation numbers against
    dists_baseline = ((fresh_moons[M][:, None] - X[M]) ** 2).sum(2) / 2.
    ot_baseline = ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, dists_baseline)
    ed_baseline = energy_distance(fresh_moons, X)

    print(f"\nmultiple imputation ({n_imputations} draws) of {mask.sum()} missing entries "
          f"across {nimp}/{n} rows")
    print(f"OT(imputed, true), missing rows only:  {ot_distance:.4f}   "
          f"(baseline, fresh moons vs true: {ot_baseline:.4f})")
    print(f"ED(imputed, true), full dataset:       {ed_distance:.4f}   "
          f"(baseline, fresh moons vs true: {ed_baseline:.4f})")
