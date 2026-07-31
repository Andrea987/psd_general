import numpy as np
from psd import K_S, H_NS, H_eta
from general_loss import general_omega_star, general_H


def _s_and_u(info):
    """
    Shared building blocks for the W/eta gradients of general_loss: A_i = K_S_i * H_NS_i,
    g[i] = Tr(Q A_i) + alpha, s[i, k] = sum_l Q[k, l] A_i[k, l] = (Q @ A_i)[k, k], and
    u[i, k, e] = sum_l Q[k, l] A_i[k, l] W[l, e]. Both K_S and H_NS depend on each anchor node W_k
    only through the k-th "slot" of A_i, which is why summing A_i against Q collapses cleanly into
    these two (n, m) / (n, m, d) objects instead of a full (n, m, m, m, d) derivative tensor.
    """
    Q = info['Q']
    W = info['anchor_nodes']
    alpha = info['alpha']

    A = K_S(info) * H_NS(info)  # (n, m, m)
    g = np.einsum('kl,ikl->i', Q, A) + alpha  # (n,): Tr(Q A_i) + alpha
    QA = Q[None, :, :] * A  # (n, m, m): QA[i, k, l] = Q[k, l] * A[i, k, l]
    s = np.sum(QA, axis=-1)  # (n, m): s[i, k] = sum_l Q[k, l] A[i, k, l]
    u = np.einsum('ikl,le->ike', QA, W)  # (n, m, d): u[i, k, e] = sum_l Q[k, l] A[i, k, l] W[l, e]
    return A, g, s, u


def _s0_and_u0(info):
    """
    Same idea as _s_and_u, but for A_0 = H_eta(info) (a single (m, m) matrix, not a per-observation
    stack): s0[k] = sum_l Q[k, l] A_0[k, l], u0[k, e] = sum_l Q[k, l] A_0[k, l] W[l, e].
    """
    Q = info['Q']
    W = info['anchor_nodes']

    A_0 = H_eta(info)  # (m, m)
    QA_0 = Q * A_0  # (m, m): QA_0[k, l] = Q[k, l] * A_0[k, l]
    s0 = np.sum(QA_0, axis=-1)  # (m,)
    u0 = np.einsum('kl,le->ke', QA_0, W)  # (m, d)
    return s0, u0


def _sc_and_uc(info):
    """
    Same idea as _s0_and_u0, but for C = general_H(info) = c_2eta(info) * H_eta_half(info) (the
    true constraint matrix used in general_lagrangian's constraint-violation term):
    sc[k] = sum_l Q[k, l] C[k, l], uc[k, e] = sum_l Q[k, l] C[k, l] W[l, e].
    """
    Q = info['Q']
    W = info['anchor_nodes']

    C = general_H(info)  # (m, m)
    QC = Q * C  # (m, m): QC[k, l] = Q[k, l] * C[k, l]
    sc = np.sum(QC, axis=-1)  # (m,)
    uc = np.einsum('kl,le->ke', QC, W)  # (m, d)
    return sc, uc


def general_loss_gradient_anchor_nodes(info):
    """
    info: dict with keys 'Q', 'dataset', 'anchor_nodes', 'precision', 'masks', 'alpha', 'lbd'
          (see K_S, H_NS, H_eta, loss). 'A_0' is not read from info: it is H_eta(info), so its
          dependence on the anchor nodes (through the lbd * Tr(Q A_0) term of loss) is included.
    returns: (m, d) matrix, d general_loss / d anchor_nodes
    """
    W = info['anchor_nodes']
    eta = info['precision']
    lbd = info['lbd']
    X = info['dataset']
    mask = np.asarray(info['masks'], dtype=bool)  # 1 = missing (NS), 0 = observed (S)
    observed = ~mask

    _, g, s, u = _s_and_u(info)

    diff_XW = X[:, None, :] - W[None, :, :]  # (n, m, d): X[i, e] - W[k, e]
    v = s[:, :, None] * W[None, :, :] - u  # (n, m, d): sum_l Q[k,l] A_i[k,l] (W[k,e]-W[l,e])

    # contribution from d K_S / d W (locally, per anchor node k) ...
    term_K_S = 4 * eta[None, None, :] * observed[:, None, :] * diff_XW * s[:, :, None]
    # ... and from d H_NS / d W
    term_H_NS = -2 * eta[None, None, :] * mask[:, None, :] * v
    d_sum = term_K_S + term_H_NS  # (n, m, d)
    grad_from_A = -np.mean(d_sum / g[:, None, None], axis=0)  # (m, d)

    # contribution from d (lbd * Tr(Q A_0)) / d W, A_0 = H_eta(info) (same shape as H_NS's
    # derivative, but with the full eta -- not eta/2 -- and no mask, since A_0 is not per-observation)
    s0, u0 = _s0_and_u0(info)
    v0 = s0[:, None] * W - u0  # (m, d)
    grad_from_A0 = -4 * lbd * eta[None, :] * v0  # (m, d)

    return grad_from_A + grad_from_A0


def general_loss_gradient_precision(info):
    """
    info: dict with keys 'Q', 'dataset', 'anchor_nodes', 'precision', 'masks', 'alpha', 'lbd'
          (see K_S, H_NS, H_eta, loss). 'A_0' is not read from info: it is H_eta(info), so its
          dependence on the precision (through the lbd * Tr(Q A_0) term of loss) is included.
    returns: (d,) vector, d general_loss / d precision
    """
    W = info['anchor_nodes']
    eta = info['precision']
    lbd = info['lbd']
    X = info['dataset']
    mask = np.asarray(info['masks'], dtype=bool)  # 1 = missing (NS), 0 = observed (S)
    observed = ~mask

    _, g, s, u = _s_and_u(info)

    diff_XW2 = (X[:, None, :] - W[None, :, :]) ** 2  # (n, m, d)
    T1 = np.einsum('ik,ike->ie', s, diff_XW2)  # (n, d)
    T2 = 2 * np.einsum('ik,ke->ie', s, W ** 2) - 2 * np.einsum('ke,ike->ie', W, u)  # (n, d)

    grad_from_A = np.mean((2 * observed * T1 + 0.5 * mask * T2) / g[:, None], axis=0)  # (d,)
    grad_from_log_C = -0.5 / eta * mask.sum(axis=0)  # (d,): d sum_i log_C_2eta_NS_i / d eta

    # contribution from d (lbd * Tr(Q A_0)) / d eta, A_0 = H_eta(info) (same shape as T2 above,
    # but built from s0/u0 instead of s/u, with no per-observation index or mask)
    s0, u0 = _s0_and_u0(info)
    T2_0 = 2 * np.sum(s0[:, None] * W ** 2, axis=0) - 2 * np.sum(W * u0, axis=0)  # (d,)
    grad_from_A0 = -lbd * T2_0  # (d,)

    return grad_from_A + grad_from_log_C + grad_from_A0


def general_lagrangian_gradient_anchor_nodes(info):
    """
    info: dict as expected by general_loss_gradient_anchor_nodes, plus 'Q' must be Q(X, eta), the
          optimal solution at the current anchor nodes/precision (see general_lagrangian).
    returns: (m, d) matrix, d general_lagrangian / d anchor_nodes, computed as
        d general_loss / d anchor_nodes + omega^* * d(Tr(Q H) - 1) / d anchor_nodes,
        H = general_H(info) = c_2eta(info) * H_eta_half(info)
    omega^* is evaluated at info and then held fixed (not differentiated through): its implicit
    dependence on the anchor nodes, via the optimal Q, doesn't need to be tracked here -- only
    general_loss and the constraint-violation term are differentiated. c_2eta(info) doesn't depend
    on the anchor nodes, so it contributes no extra product-rule term here (unlike in the
    precision gradient below).
    """
    W = info['anchor_nodes']
    eta = info['precision']
    omega = general_omega_star(info)

    grad_loss = general_loss_gradient_anchor_nodes(info)

    sc, uc = _sc_and_uc(info)
    vc = sc[:, None] * W - uc  # (m, d): sum_l Q[k,l] C[k,l] (W[k,e]-W[l,e])
    grad_constraint = -2 * eta[None, :] * vc  # (m, d): d Tr(Q H) / d W

    return grad_loss + omega * grad_constraint


def general_lagrangian_gradient_precision(info):
    """
    info: dict as expected by general_loss_gradient_precision, plus 'Q' must be Q(X, eta), the
          optimal solution at the current anchor nodes/precision (see general_lagrangian).
    returns: (d,) vector, d general_lagrangian / d precision, computed as
        d general_loss / d precision + omega^* * d(Tr(Q H) - 1) / d precision,
        H = general_H(info) = c_2eta(info) * H_eta_half(info)
    omega^* is evaluated at info and then held fixed (not differentiated through), same reasoning
    as general_lagrangian_gradient_anchor_nodes. Unlike the anchor-node gradient, c_2eta(info) does
    depend on the precision, so d Tr(Q H) / d eta needs the product rule:
        d Tr(Q H) / d eta_e = (d c_2eta / d eta_e) * Tr(Q H_eta_half) + c_2eta * d Tr(Q H_eta_half) / d eta_e
                             = -0.5 / eta_e * Tr(Q H) - 0.5 * Tc2[e]
    where the first term uses d log(c_2eta) / d eta_e = -0.5 / eta_e, and Tc2 is the same T2-style
    quantity as in general_loss_gradient_precision, but built from sc/uc (already scaled by
    c_2eta, since _sc_and_uc uses general_H) instead of s0/u0.
    """
    W = info['anchor_nodes']
    eta = info['precision']
    omega = general_omega_star(info)

    grad_loss = general_loss_gradient_precision(info)

    sc, uc = _sc_and_uc(info)
    trace_QH = np.sum(sc)  # Tr(Q H), H = general_H(info)
    Tc2 = 2 * np.sum(sc[:, None] * W ** 2, axis=0) - 2 * np.sum(W * uc, axis=0)  # (d,)
    grad_constraint = -0.5 / eta * trace_QH - 0.5 * Tc2  # (d,): d Tr(Q H) / d eta

    return grad_loss + omega * grad_constraint
