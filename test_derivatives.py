import numpy as np
from psd import c_2eta, H_eta_half
from general_loss import general_loss, general_lagrangian
from derivatives import (general_loss_gradient_anchor_nodes, general_loss_gradient_precision,
                          general_lagrangian_gradient_anchor_nodes, general_lagrangian_gradient_precision)


def make_general_loss_info(n=10, m=5, d=3, seed=4):
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, d))
    mask = (rng.random((n, d)) < 0.4).astype(float)  # 1 = missing (NS), 0 = observed (S)
    fully_missing_rows = np.where(mask.sum(axis=1) == d)[0]
    if len(fully_missing_rows) > 0:
        revealed_dims = rng.integers(0, d, size=len(fully_missing_rows))
        mask[fully_missing_rows, revealed_dims] = 0

    W = rng.standard_normal((m, d))
    eta = rng.uniform(0.5, 2.0, size=d)  # must stay positive: it's a kernel precision

    L = rng.standard_normal((m, m))
    Q = L @ L.T + np.eye(m) * 2  # PD

    L_0 = rng.standard_normal((m, m))
    A_0 = L_0 @ L_0.T + np.eye(m)  # PD

    return {
        'Q': Q, 'anchor_nodes': W, 'precision': eta, 'dataset': X, 'masks': mask,
        'A_0': A_0, 'alpha': 0.7, 'lbd': 0.3, 'mu': 0.5,
    }


def test_general_loss_gradient_anchor_nodes_matches_finite_difference():
    info = make_general_loss_info()
    W0 = info['anchor_nodes'].copy()
    m, d = W0.shape
    eps = 1e-6

    grad_fd = np.zeros((m, d))
    for k in range(m):
        for e in range(d):
            Wp, Wm = W0.copy(), W0.copy()
            Wp[k, e] += eps
            Wm[k, e] -= eps
            loss_p = general_loss({**info, 'anchor_nodes': Wp})
            loss_m = general_loss({**info, 'anchor_nodes': Wm})
            grad_fd[k, e] = (loss_p - loss_m) / (2 * eps)

    grad_analytic = general_loss_gradient_anchor_nodes(info)
    max_diff = np.max(np.abs(grad_fd - grad_analytic))
    print('general_loss_gradient_anchor_nodes vs finite-difference max abs diff:', max_diff)
    assert max_diff < 1e-4


def test_general_loss_gradient_precision_matches_finite_difference():
    info = make_general_loss_info()
    eta0 = info['precision'].copy()
    d = eta0.shape[0]
    eps = 1e-6

    grad_fd = np.zeros(d)
    for e in range(d):
        eta_p, eta_m = eta0.copy(), eta0.copy()
        eta_p[e] += eps
        eta_m[e] -= eps
        loss_p = general_loss({**info, 'precision': eta_p})
        loss_m = general_loss({**info, 'precision': eta_m})
        grad_fd[e] = (loss_p - loss_m) / (2 * eps)

    grad_analytic = general_loss_gradient_precision(info)
    max_diff = np.max(np.abs(grad_fd - grad_analytic))
    print('general_loss_gradient_precision vs finite-difference max abs diff:', max_diff)
    assert max_diff < 1e-4


def make_feasible_general_loss_info(**kwargs):
    """
    Same as make_general_loss_info, but with Q rescaled so that Tr(Q H) == 1 exactly, where
    H = c_2eta(info) * H_eta_half(info) is the true constraint matrix general_lagrangian uses
    (Q must be the constrained optimum at the current anchor nodes/precision). Rescaling a PD
    matrix by a positive scalar keeps it PD.
    """
    info = make_general_loss_info(**kwargs)
    H = c_2eta(info) * H_eta_half(info)
    info['Q'] = info['Q'] / np.einsum('kl,kl->', info['Q'], H)
    return info


def test_general_lagrangian_gradient_anchor_nodes_matches_finite_difference():
    # At a Q that exactly satisfies Tr(Q H) = 1, the constraint-violation term is zero, so the
    # (deliberately omitted) derivative of omega^* gets multiplied by zero: the full finite
    # difference of general_lagrangian below matches our partial-derivative formula, which only
    # differentiates general_loss and the constraint-violation term, holding omega^* fixed.
    info = make_feasible_general_loss_info()
    W0 = info['anchor_nodes'].copy()
    m, d = W0.shape
    eps = 1e-6

    grad_fd = np.zeros((m, d))
    for k in range(m):
        for e in range(d):
            Wp, Wm = W0.copy(), W0.copy()
            Wp[k, e] += eps
            Wm[k, e] -= eps
            lagrangian_p = general_lagrangian({**info, 'anchor_nodes': Wp})
            lagrangian_m = general_lagrangian({**info, 'anchor_nodes': Wm})
            grad_fd[k, e] = (lagrangian_p - lagrangian_m) / (2 * eps)

    grad_analytic = general_lagrangian_gradient_anchor_nodes(info)
    max_diff = np.max(np.abs(grad_fd - grad_analytic))
    print('general_lagrangian_gradient_anchor_nodes vs finite-difference max abs diff:', max_diff)
    assert max_diff < 1e-4


def test_general_lagrangian_gradient_precision_matches_finite_difference():
    info = make_feasible_general_loss_info()
    eta0 = info['precision'].copy()
    d = eta0.shape[0]
    eps = 1e-6

    grad_fd = np.zeros(d)
    for e in range(d):
        eta_p, eta_m = eta0.copy(), eta0.copy()
        eta_p[e] += eps
        eta_m[e] -= eps
        lagrangian_p = general_lagrangian({**info, 'precision': eta_p})
        lagrangian_m = general_lagrangian({**info, 'precision': eta_m})
        grad_fd[e] = (lagrangian_p - lagrangian_m) / (2 * eps)

    grad_analytic = general_lagrangian_gradient_precision(info)
    max_diff = np.max(np.abs(grad_fd - grad_analytic))
    print('general_lagrangian_gradient_precision vs finite-difference max abs diff:', max_diff)
    assert max_diff < 1e-4


if __name__ == '__main__':
    test_general_loss_gradient_anchor_nodes_matches_finite_difference()
    test_general_loss_gradient_precision_matches_finite_difference()
    test_general_lagrangian_gradient_anchor_nodes_matches_finite_difference()
    test_general_lagrangian_gradient_precision_matches_finite_difference()
    print('all tests passed')
