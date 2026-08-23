import time

import numpy as np
from psd import K_S, H_NS, H_eta
from general_loss import general_H, general_loss, general_lagrangian
from derivatives import general_lagrangian_gradient_anchor_nodes, general_lagrangian_gradient_precision
from optimization import newton_method


def general_lagrangian_fixed(info):
    """
    info: dict as expected by general_lagrangian
    returns: a function (Q, anchor_nodes, precision) -> general_lagrangian(...), with everything
        in info other than 'Q', 'anchor_nodes', 'precision' (i.e. 'dataset', 'masks', 'alpha',
        'lbd', 'mu') held fixed at their values in info -- exactly the quantities
        alternating_minimization keeps fixed throughout a run, so the Lagrangian becomes a
        genuine function of only (Q, anchor_nodes, precision) during that run.
    """
    fixed = {k: v for k, v in info.items() if k not in ('Q', 'anchor_nodes', 'precision')}

    def lagrangian(Q, anchor_nodes, precision):
        return general_lagrangian({**fixed, 'Q': Q, 'anchor_nodes': anchor_nodes, 'precision': precision})

    return lagrangian


def alternating_minimization(info):
    """
    info: dict with keys
        'dataset': (n, d) data (see K_S, H_NS)
        'masks': (n, d) boolean (or 0/1), 1 = missing, 0 = seen (see K_S, H_NS)
        'anchor_nodes': (m, d) initial anchor nodes
        'precision': (d,) initial bandwidth/precision
        'Q': (m, m) starting point for the primal variable; renormalized internally so that
             Tr(Q H) = 1, H = general_H(info) = c_2eta(info) * H_eta_half(info)
        'alpha', 'lbd', 'mu': see loss / general_loss
        'l_rate_nodes': step size for the gradient step on anchor_nodes
        'l_rate_param': step size for the gradient step on precision -- the step is taken in
             log(precision) space (precision = exp(log(precision) - l_rate_param * grad_eta *
             precision), i.e. the chain-rule-scaled gradient), so precision stays positive
             regardless of step size; see test_derivatives_jax.py for a check that this matches
             autodiff of the Lagrangian reparametrized in terms of log(precision)
        'nbr_bounce': number of outer alternating iterations
        'nbr_gradient_steps': number of gradient steps taken on [anchor_nodes, precision] per
            outer iteration, all with Q held fixed at that iteration's Q* (default 1)
    'dataset', 'masks', 'alpha', 'lbd', 'mu', 'l_rate_nodes', 'l_rate_param', 'nbr_bounce',
    'nbr_gradient_steps' are fixed for the whole run; only 'Q', 'anchor_nodes', 'precision' change
    (see general_lagrangian_fixed, which makes this explicit for the Lagrangian).

    Each outer iteration: (1) fix anchor_nodes/precision, renormalize Q against the current H, and
    run Newton's method on general_loss to find the optimal Q*; (2) with Q held fixed at Q*, take
    nbr_gradient_steps gradient steps on anchor_nodes and precision on the Lagrangian (see
    general_lagrangian_gradient_anchor_nodes/precision -- omega^* is held fixed there, not
    differentiated through, by the envelope theorem). The last outer iteration only does step (1):
    anchor_nodes/precision are left untouched after the final Q* is found, so the function returns
    Q optimized against the anchor_nodes/precision of the second-to-last iteration.

    returns: (Q, anchor_nodes, precision, history), history a list of (loss, lagrangian) pairs,
        one per outer iteration, recorded right after each Newton-for-Q phase
    """
    l_rate_nodes = info['l_rate_nodes']
    l_rate_param = info['l_rate_param']
    nbr_bounce = info['nbr_bounce']
    nbr_gradient_steps = info.get('nbr_gradient_steps', 1)  # falls back to 1 is the key is not present
    verbose = info.get('verbose', False)
    verbose_newton = info.get('verbose_newton', False)
    lagrangian_fixed = general_lagrangian_fixed(info)

    info = dict(info)  # local copy: 'Q', 'anchor_nodes', 'precision' are updated in place below
    history = []

    for step in range(nbr_bounce):
        # fix anchor_nodes/precision, renormalize Q so it is feasible for the current H, then run
        # Newton's method to find the optimal Q* -- A, A_0, H don't change during this phase, so
        # they are computed once and plugged into the plain (non-"general_") Newton machinery
        t_kernel_start = time.perf_counter()
        H = general_H(info)
        info['Q'] = info['Q'] / np.einsum('kl,kl->', info['Q'], H)
        A = K_S(info) * H_NS(info)
        A_0 = H_eta(info)
        kernel_build_time = time.perf_counter() - t_kernel_start
        if verbose_newton:
            print(f"  kernel matrices (K_S, H_NS, H_eta) built in {kernel_build_time:.4f}s")

        Q_star, newton_history = newton_method({**info, 'A': A, 'A_0': A_0, 'H': H})
        newton_decrement = np.sqrt(newton_history[-1][1])

        trace_Q_star_H = np.einsum('kl,kl->', Q_star, H)
        assert np.isclose(trace_Q_star_H, 1), f"Tr(Q* H) = {trace_Q_star_H}, expected 1"
        # print("trace(Qˆ* H) in alteranating minimization ", trace_Q_star_H)
        info['Q'] = Q_star

        history.append((
            general_loss(info),
            lagrangian_fixed(info['Q'], info['anchor_nodes'], info['precision']),
        ))

        # gradient steps on [anchor_nodes, precision], with Q held fixed at Q* -- skipped on the
        # last bounce, which should optimize Q only and leave anchor_nodes/precision untouched
        grad_W_norm = grad_eta_norm = None
        if step < nbr_bounce - 1:
            for grad_step in range(nbr_gradient_steps):
                grad_W = general_lagrangian_gradient_anchor_nodes(info)
                grad_eta = general_lagrangian_gradient_precision(info)
                info['anchor_nodes'] = info['anchor_nodes'] - l_rate_nodes * grad_W
                # reparametrize precision = exp(log(precision)) and step in log-space (chain rule:
                # d loss / d log(eta) = grad_eta * eta) so precision can never go negative
                log_eta = np.log(info['precision']) - l_rate_param * (grad_eta * info['precision'])
                info['precision'] = np.exp(log_eta)
            grad_W_norm = np.linalg.norm(grad_W)
            grad_eta_norm = np.linalg.norm(grad_eta)

        if verbose and step % 5 == 0:
            msg = f"Bounce {step + 1}/{nbr_bounce}\tNewton decrement: {newton_decrement:.6f}"
            if grad_W_norm is not None:
                msg += (f"\t|grad anchor_nodes|: {grad_W_norm:.6f}\t"
                        f"|grad precision|: {grad_eta_norm:.6f}")
            msg += f"\tprecision: {np.array2string(info['precision'], precision=4)}"
            print(msg)

    return info['Q'], info['anchor_nodes'], info['precision'], history
