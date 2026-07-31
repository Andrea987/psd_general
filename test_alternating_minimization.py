import numpy as np
from alternating_minimization import alternating_minimization


def make_alternating_minimization_info(n=20, m=4, d=2, seed=0):
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, d))
    mask = (rng.random((n, d)) < 0.3).astype(float)  # 1 = missing, 0 = seen
    fully_missing_rows = np.where(mask.sum(axis=1) == d)[0]
    if len(fully_missing_rows) > 0:
        revealed_dims = rng.integers(0, d, size=len(fully_missing_rows))
        mask[fully_missing_rows, revealed_dims] = 0

    W = rng.standard_normal((m, d))
    eta = rng.uniform(0.5, 2.0, size=d)

    L = rng.standard_normal((m, m))
    Q0 = L @ L.T + np.eye(m) * 2  # PD starting point (not yet normalized: alternating_minimization does that)

    return {
        'dataset': X, 'masks': mask, 'anchor_nodes': W, 'precision': eta, 'Q': Q0,
        'alpha': 0.7, 'lbd': 0.3, 'mu': 0.5,
        'l_rate_nodes': 1e-3, 'l_rate_param': 1e-3, 'nbr_bounce': 5,
    }


def test_alternating_minimization_runs_end_to_end():
    info = make_alternating_minimization_info()
    Q, W, eta, history = alternating_minimization(info)

    m, d = info['anchor_nodes'].shape
    assert Q.shape == (m, m)
    assert W.shape == (m, d)
    assert eta.shape == (d,)
    assert len(history) == info['nbr_bounce']

    print('history (loss, lagrangian):')
    for loss_value, lagrangian_value in history:
        print(' ', loss_value, lagrangian_value)


if __name__ == '__main__':
    test_alternating_minimization_runs_end_to_end()
    print('all tests passed')
