import time

import numpy as np
from psd import loss, newton_step_and_decrement


def newton_method(info):
    """
    Damped Newton's method for min_{Q > 0, Tr(Q H) = 1} loss(Q), using the Newton step and
    decrement from psd.newton_step_and_decrement (which already accounts for the Tr(Q H) = 1
    equality constraint via its Lagrange multiplier nu).

    info: dict as expected by psd.loss / psd.newton_step_and_decrement, i.e. with keys 'Q', 'A',
          'A_0', 'alpha', 'lbd', 'mu', 'H'. 'Q' is the starting point, and must be positive
          definite and feasible (Tr(Q H) = 1). Also accepts, all optional:
        'tol': stop once the squared Newton decrement lambda_Q^2 <= tol (default 0.68 ** 2)
        'max_iter': maximum number of Newton iterations (default 100)
        'alpha_backtracking': Armijo parameter of the backtracking line search, in (0, 0.5)
              (named alpha_backtracking, not alpha, to avoid clashing with info['alpha'];
              default 0.1)
        'beta_backtracking': shrink factor of the backtracking line search, in (0, 1)
              (default 0.8)
        'verbose_newton': if True, print progress every Newton iteration (loss, decrement) --
              deliberately a separate flag from 'verbose' (used by alternating_minimization for
              its own, coarser per-bounce summary), so enabling one doesn't also enable the other
    returns: (Q, history)
        Q: (m, m) matrix, the final iterate
        history: list of (loss(Q), lambda_Q_squared) pairs, one entry per iteration
    """
    tol = info.get('tol', 0.68 ** 2)
    max_iter = info.get('max_iter', 100)
    alpha_backtracking = info.get('alpha_backtracking', 0.1)
    beta_backtracking = info.get('beta_backtracking', 0.8)
    verbose = info.get('verbose_newton', False)

    Q = info['Q']
    history = []

    for i in range(max_iter):
        t_start = time.perf_counter()

        info = {**info, 'Q': Q}
        DQ, nu, lambda_Q_squared = newton_step_and_decrement(info)
        f_Q = loss(info)
        history.append((f_Q, lambda_Q_squared))
        converged = lambda_Q_squared <= tol

        if not converged:
            direction = DQ
            directional_derivative = -lambda_Q_squared  # <grad f(Q), direction> = -lambda_Q^2

            # backtracking line search: shrink the step until Q_candidate stays positive definite
            # (the domain of -log det(Q)) and satisfies the Armijo sufficient-decrease condition.
            # t shrinking below machine precision means neither condition is meaningfully
            # satisfiable (the direction is numerically unreliable, e.g. an ill-conditioned
            # Hessian solve) -- bail out with Q unchanged rather than spin for ~3000 more
            # iterations until t underflows to exact 0.0 (at which point the Armijo check would
            # trivially pass anyway, since f_Q <= f_Q).
            t = 1.0
            while True:
                if t < np.finfo(float).eps:
                    print(f"    Warning: backtracking line search step size underflowed below "
                          f"machine precision (t={t:.3e}); leaving Q unchanged for this Newton "
                          f"iteration.")
                    Q_candidate = Q
                    break
                Q_candidate = Q + t * direction
                try:
                    np.linalg.cholesky(Q_candidate)
                except np.linalg.LinAlgError:
                    t *= beta_backtracking
                    continue
                f_candidate = loss({**info, 'Q': Q_candidate})
                if f_candidate <= f_Q + alpha_backtracking * t * directional_derivative:
                    break
                t *= beta_backtracking

            Q = Q_candidate
            H = info['H']
            # in exact arithmetic Tr(Q @ H) = 1; re-orthogonalize against H to remove any
            # numerical drift accumulated in newton_step_and_decrement
            trace_direction_H = np.einsum('ij,ij->', direction, H)
            trace_HH = np.einsum('ij,ij->', H, H)
            Q = Q - (trace_direction_H / trace_HH) * H  # renorma

        if verbose:
            elapsed = time.perf_counter() - t_start
            print(f"    Newton iteration {i + 1}/{max_iter}\tloss: {f_Q:.6f}\t"
                  f"decrement: {np.sqrt(lambda_Q_squared):.6f}\ttime: {elapsed:.4f}s")

        if converged:
            break

    return Q, history
