import numpy as np
from sklearn.datasets import make_moons
from sklearn.impute import SimpleImputer

from alternating_minimization import alternating_minimization
from plot_toy_2d import plot_initial_vs_final_model


def sample_mask(n, d, p_missing, rng):
    mask = (rng.random((n, d)) < p_missing).astype(float)  # 1 = missing, 0 = seen
    fully_missing_rows = np.where(mask.sum(axis=1) == d)[0]
    if len(fully_missing_rows) > 0:
        revealed_dims = rng.integers(0, d, size=len(fully_missing_rows))
        mask[fully_missing_rows, revealed_dims] = 0
    return mask


def sample_anchor_nodes_uniform_hypercube(X, m, rng):
    """Anchor nodes sampled i.i.d. uniformly on the hypercube bounding the observations."""
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    return rng.uniform(lo, hi, size=(m, X.shape[1]))


def sample_anchor_nodes_from_data(X, mask, m, rng):
    """Anchor nodes as a random subset of a mean-imputed X (see psd_imputer.fit_psd_model)."""
    n = X.shape[0]
    X_nan = np.where(mask == 0, X, np.nan)
    X_imputed = SimpleImputer(strategy='mean').fit_transform(X_nan)
    idx = rng.choice(n, size=m, replace=(m > n))
    return X_imputed[idx].copy()


def build_info(X, mask, W, nbr_bounce, seed):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    m = W.shape[0]

    eta = np.full(d, 5.0)
    L = rng.standard_normal((m, m))
    Q0 = L @ L.T + np.eye(m) * 2  # PD starting point (alternating_minimization renormalizes it)

    return {
        'dataset': X, 'masks': mask, 'anchor_nodes': W, 'precision': eta, 'Q': Q0,
        'alpha': 1e-6, 'lbd': 1e-4, 'mu': 1e-4,
        'l_rate_nodes': 1e-1, 'l_rate_param': 1e-4,
        'nbr_bounce': nbr_bounce, 'nbr_gradient_steps': 5,
        'verbose': True,
    }


def run_experiment(X, mask, W, nbr_bounce, seed, save_path):
    info = build_info(X, mask, W, nbr_bounce, seed)
    Q, W_final, eta, history = alternating_minimization(info)
    plot_initial_vs_final_model(info, Q, W_final, eta, history[-1][0], save_path=save_path)


if __name__ == '__main__':
    seed = 0
    n = 500
    p_missing = 0.1
    m_many = 100
    m_few = 20

    rng = np.random.default_rng(seed)
    X, _ = make_moons(n_samples=n, noise=0.1, random_state=seed)
    mask = sample_mask(n, 2, p_missing, rng)

    # Experiment 1: many anchor nodes, sampled i.i.d. uniformly on the hypercube containing the
    # observations from X (not reused from X itself), alternating minimization run just once --
    # the only bounce is also the last one, so it only optimizes Q; anchor_nodes/precision stay
    # at their sampled values.
    W_many = sample_anchor_nodes_uniform_hypercube(X, m_many, rng)
    run_experiment(X, mask, W_many, nbr_bounce=1, seed=seed,
                   save_path='toy_test_3_many_nodes.png')

    # Other experiments: fewer anchor nodes (sampled as a subset of the mean-imputed data, as in
    # psd_imputer.fit_psd_model), full alternating minimization (anchor_nodes/precision are
    # optimized too, across many bounces).
    W_few = sample_anchor_nodes_from_data(X, mask, m_few, rng)
    run_experiment(X, mask, W_few, nbr_bounce=30, seed=seed,
                   save_path='toy_test_3_few_nodes.png')
