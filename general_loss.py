import numpy as np
from psd import K_S, H_NS, H_eta, H_eta_half, c_2eta, log_C_2eta_NS, loss, newton_step_and_decrement, omega_star


def general_H(info):
    """
    info: dict with keys 'anchor_nodes', 'precision' (see psd.H_eta_half, psd.c_2eta)
    returns: (m, m) matrix, the true constraint matrix H = c_2eta(info) * H_eta_half(info) (Tr(Q H)
             is meant to equal 1). H_eta_half alone is only the "matrix part": c_2eta is the
             normalizing constant from the Gaussian self-convolution identity (same construction
             as C_2eta_NS, but over all d dimensions since H_eta_half has no mask).
    """
    return c_2eta(info) * H_eta_half(info)


def general_loss(info):
    """
    info: dict with keys 'dataset', 'anchor_nodes', 'precision', 'masks' (see K_S, H_NS, H_eta,
          log_C_2eta_NS), plus 'Q', 'alpha', 'lbd', 'mu' (see loss). 'A_0' is not read from info:
          it is always set to H_eta(info), the anchor-node kernel matrix at the full bandwidth.
    returns: scalar, loss(info) with 'A' set to the Hadamard product K_S(info) * H_NS(info) and
             'A_0' set to H_eta(info), plus sum(log_C_2eta_NS(info)) (added on top rather than
             folded into A, since it doesn't depend on Q -- it only matters once derivatives wrt
             the kernel precision are taken)
    """
    A = K_S(info) * H_NS(info)  # (n, m, m): A_i = K_S(x_S)_i * H_NS_i, Hadamard product
    A_0 = H_eta(info)  # (m, m)
    return loss({**info, 'A': A, 'A_0': A_0}) + np.sum(log_C_2eta_NS(info))


def general_newton_step_and_decrement(info):
    """
    info: dict as expected by psd.newton_step_and_decrement, except 'A' and 'A_0' are built the
          same way as in general_loss, and 'H' is not read from info: it is always set to
          general_H(info) = c_2eta(info) * H_eta_half(info).
    returns: same as psd.newton_step_and_decrement: (DQ, nu, lambda_Q_squared)
    """
    A = K_S(info) * H_NS(info)  # (n, m, m)
    A_0 = H_eta(info)  # (m, m)
    H = general_H(info)  # (m, m)
    return newton_step_and_decrement({**info, 'A': A, 'A_0': A_0, 'H': H})


def general_omega_star(info):
    """
    info: dict as expected by psd.omega_star, except 'A' and 'A_0' are built the same way as in
          general_loss (A_0 = H_eta(info)). 'Q' must still be Q(X, eta), the optimal solution for
          the given anchor nodes and precision (see psd.omega_star).
    returns: same as psd.omega_star: the dual parameter omega^* = omega(X, eta)
    """
    A = K_S(info) * H_NS(info)  # (n, m, m)
    A_0 = H_eta(info)  # (m, m)
    return omega_star({**info, 'A': A, 'A_0': A_0})


def general_lagrangian(info):
    """
    info: dict as expected by general_loss / general_omega_star, plus 'Q' (must be Q(X, eta), the
          optimal solution of the convex problem for the current anchor nodes/precision -- see
          psd.omega_star -- since Tr(Q H) below is also evaluated at this Q).
    returns: scalar, the general Lagrangian
        lagrangian = general_loss(info) + omega^*(info) * (Tr(Q H) - 1),  H = general_H(info)
    omega^* = omega(anchor_nodes, precision) is a genuine function of the anchor nodes and
    precision -- it is not treated as a constant. It is Q-independent only in the sense that it
    has no free Q argument of its own: it depends on Q solely through Q = Q(anchor_nodes,
    precision), the optimal solution at those anchor nodes/precision, so once that Q is plugged
    in, omega^* is fully determined by the anchor nodes/precision alone.
    """
    Q = info['Q']
    H = general_H(info)
    constraint_violation = np.einsum('kl,kl->', Q, H) - 1  # Tr(Q H) - 1
    return general_loss(info) + general_omega_star(info) * constraint_violation
