import numpy as np
from sklearn.datasets import make_circles

from plot_toy_2d import run_and_collect_steps, plot_steps, plot_initial_vs_final_model


def generate_toy_info(n=400, m=30, seed=0):
    rng = np.random.default_rng(seed)
    d = 2

    X, _ = make_circles(n_samples=n, noise=0.1, factor=0.35, random_state=seed)

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
        'l_rate_nodes': 1e-1, 'l_rate_param': 1e-2, 'nbr_bounce': 1, 'nbr_gradient_steps': 5,
    }


if __name__ == '__main__':
    info = generate_toy_info()
    snapshots = run_and_collect_steps(info, num_steps=30)
    plot_steps(info, snapshots, save_path='toy_test_2_steps.png')
    plot_initial_vs_final_model(info, snapshots, save_path='toy_test_2_final_model.png')
