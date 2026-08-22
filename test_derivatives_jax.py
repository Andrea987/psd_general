"""
Cross-check derivatives.py's hand-written d general_lagrangian / d anchor_nodes and
d general_lagrangian / d precision against JAX autodiff of the same formula:
    general_loss(W, eta) + omega_fixed * (Tr(Q @ general_H(W, eta)) - 1)
with Q and omega_fixed (= general_omega_star(info), evaluated once) held fixed -- exactly what
derivatives.general_lagrangian_gradient_anchor_nodes/precision compute (see their docstrings: Q
and omega^* are not differentiated through, only the explicit W/eta dependence is).
"""

import jax
jax.config.update("jax_enable_x64", True)  # match the rest of the codebase's float64

import jax.numpy as jnp
import numpy as np

from general_loss import general_omega_star
from derivatives import general_lagrangian_gradient_anchor_nodes, general_lagrangian_gradient_precision


def make_info(n=10, m=8, d=4, seed=0):
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, d))
    mask = (rng.random((n, d)) < 0.4).astype(float)
    fully_missing = np.where(mask.sum(axis=1) == d)[0]
    if len(fully_missing) > 0:
        revealed = rng.integers(0, d, size=len(fully_missing))
        mask[fully_missing, revealed] = 0

    W = rng.standard_normal((m, d))
    eta = rng.uniform(0.5, 2.0, size=d)

    L = rng.standard_normal((m, m))
    Q = L @ L.T + np.eye(m) * 2  # PD, invertible

    return {
        'dataset': X, 'masks': mask, 'anchor_nodes': W, 'precision': eta, 'Q': Q,
        'alpha': 1e-6, 'lbd': 0.3, 'mu': 0.2,
    }


def K_S_jax(X, W, eta, mask):
    observed = 1.0 - mask  # (n, d)
    eta_S = eta[None, :] * observed  # (n, d)
    diff2 = (X[:, None, :] - W[None, :, :]) ** 2  # (n, m, d)
    phi_S = jnp.exp(-jnp.sum(eta_S[:, None, :] * diff2, axis=-1))  # (n, m)
    return jnp.einsum('ki,kj->kij', phi_S, phi_S)


def H_NS_jax(W, eta, mask):
    eta_NS_half = (eta[None, :] * mask) / 2  # (n, d)
    diff2 = (W[:, None, :] - W[None, :, :]) ** 2  # (m, m, d)
    d2 = jnp.einsum('kd,ijd->kij', eta_NS_half, diff2)  # (n, m, m)
    return jnp.exp(-d2)


def H_eta_jax(W, eta):
    diff2 = (W[:, None, :] - W[None, :, :]) ** 2  # (m, m, d)
    d2 = jnp.einsum('d,ijd->ij', eta, diff2)
    return jnp.exp(-d2)


def H_eta_half_jax(W, eta):
    diff2 = (W[:, None, :] - W[None, :, :]) ** 2  # (m, m, d)
    d2 = jnp.einsum('d,ijd->ij', eta / 2, diff2)
    return jnp.exp(-d2)


def c_2eta_jax(eta):
    d = eta.shape[0]
    return jnp.exp((d / 2) * jnp.log(jnp.pi / 2) - 0.5 * jnp.sum(jnp.log(eta)))


def log_C_2eta_NS_jax(eta, mask):
    return jnp.sum(mask * (0.5 * jnp.log(jnp.pi / 2) - 0.5 * jnp.log(eta)[None, :]), axis=-1)


def general_loss_jax(W, eta, X, mask, Q, alpha, lbd, mu):
    A = K_S_jax(X, W, eta, mask) * H_NS_jax(W, eta, mask)  # (n, m, m)
    A_0 = H_eta_jax(W, eta)  # (m, m)

    trace_QA = jnp.einsum('jk,ikj->i', Q, A)  # (n,)
    trace_QA0 = jnp.einsum('jk,kj->', Q, A_0)
    _, logdet_Q = jnp.linalg.slogdet(Q)
    loss_value = -jnp.mean(jnp.log(trace_QA + alpha)) + lbd * trace_QA0 - mu * logdet_Q

    return loss_value + jnp.sum(log_C_2eta_NS_jax(eta, mask))


def general_H_jax(W, eta):
    return c_2eta_jax(eta) * H_eta_half_jax(W, eta)


def lagrangian_fixed_omega_jax(W, eta, X, mask, Q, alpha, lbd, mu, omega_fixed):
    constraint_violation = jnp.einsum('kl,kl->', Q, general_H_jax(W, eta)) - 1
    return general_loss_jax(W, eta, X, mask, Q, alpha, lbd, mu) + omega_fixed * constraint_violation


def test_lagrangian_gradient_anchor_nodes_matches_jax():
    info = make_info()
    omega_fixed = general_omega_star(info)  # evaluated once, held fixed -- see module docstring

    X = jnp.asarray(info['dataset'])
    mask = jnp.asarray(info['masks'])
    W = jnp.asarray(info['anchor_nodes'])
    eta = jnp.asarray(info['precision'])
    Q = jnp.asarray(info['Q'])

    grad_jax = jax.grad(lagrangian_fixed_omega_jax, argnums=0)(
        W, eta, X, mask, Q, info['alpha'], info['lbd'], info['mu'], omega_fixed
    )
    grad_analytic = general_lagrangian_gradient_anchor_nodes(info)

    max_diff = np.max(np.abs(np.asarray(grad_jax) - grad_analytic))
    print('d lagrangian / d anchor_nodes: analytic vs jax, max abs diff:', max_diff)
    assert max_diff < 1e-8


def test_lagrangian_gradient_precision_matches_jax():
    info = make_info()
    omega_fixed = general_omega_star(info)

    X = jnp.asarray(info['dataset'])
    mask = jnp.asarray(info['masks'])
    W = jnp.asarray(info['anchor_nodes'])
    eta = jnp.asarray(info['precision'])
    Q = jnp.asarray(info['Q'])

    grad_jax = jax.grad(lagrangian_fixed_omega_jax, argnums=1)(
        W, eta, X, mask, Q, info['alpha'], info['lbd'], info['mu'], omega_fixed
    )
    grad_analytic = general_lagrangian_gradient_precision(info)

    max_diff = np.max(np.abs(np.asarray(grad_jax) - grad_analytic))
    print('d lagrangian / d precision: analytic vs jax, max abs diff:', max_diff)
    assert max_diff < 1e-8


def lagrangian_fixed_omega_logeta_jax(W, log_eta, X, mask, Q, alpha, lbd, mu, omega_fixed):
    """Same as lagrangian_fixed_omega_jax, but parametrized by log(precision) instead of precision
    directly -- letting jax differentiate through the exp() gives the true d/d log(eta) gradient,
    to check against the chain-rule reparametrization used in alternating_minimization.py
    (precision = exp(log(precision) - l_rate_param * grad_eta * precision))."""
    eta = jnp.exp(log_eta)
    return lagrangian_fixed_omega_jax(W, eta, X, mask, Q, alpha, lbd, mu, omega_fixed)


def test_lagrangian_gradient_log_precision_matches_chain_rule():
    """
    Option A reparametrization check (see alternating_minimization.py): taking the gradient step
    on log(precision) instead of precision directly is done via the chain rule
    (d lagrangian / d log(eta) = d lagrangian / d eta * eta) rather than by re-deriving the
    gradient formulas from scratch. Verify that chain-rule-scaled analytic gradient against JAX
    autodiff of the Lagrangian reparametrized in terms of log(precision) directly.
    """
    info = make_info()
    omega_fixed = general_omega_star(info)

    X = jnp.asarray(info['dataset'])
    mask = jnp.asarray(info['masks'])
    W = jnp.asarray(info['anchor_nodes'])
    eta = info['precision']
    log_eta = jnp.asarray(np.log(eta))
    Q = jnp.asarray(info['Q'])

    grad_log_eta_jax = jax.grad(lagrangian_fixed_omega_logeta_jax, argnums=1)(
        W, log_eta, X, mask, Q, info['alpha'], info['lbd'], info['mu'], omega_fixed
    )

    grad_eta_analytic = general_lagrangian_gradient_precision(info)
    grad_log_eta_analytic = grad_eta_analytic * eta  # chain rule: d/d log(eta) = (d/d eta) * eta

    max_diff = np.max(np.abs(np.asarray(grad_log_eta_jax) - grad_log_eta_analytic))
    print('d lagrangian / d log(precision): chain-rule analytic vs jax, max abs diff:', max_diff)
    assert max_diff < 1e-8


if __name__ == '__main__':
    test_lagrangian_gradient_anchor_nodes_matches_jax()
    test_lagrangian_gradient_precision_matches_jax()
    test_lagrangian_gradient_log_precision_matches_chain_rule()
    print('all tests passed')
