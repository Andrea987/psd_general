import numpy as np
from scipy.spatial import distance_matrix


def Distance_Matrix_Vector_Matrices(x, y, eta):  # distance matrices with vectors of matrices
    norm_square_x = np.sum((x ** 2) * eta, axis=-1)
    norm_square_y = np.sum((y ** 2) * eta, axis=-1)
    norm_square_y = np.expand_dims(norm_square_y, axis=-1)
    norm_square_y = np.swapaxes(norm_square_y, -1, -2)
    norm_square_x = np.expand_dims(norm_square_x, axis=-1)
    y = np.swapaxes(y, -1, -2)
    dot_xy = np.matmul(x * eta, y)  # matrix of scalar products
    distances = norm_square_x + norm_square_y - 2 * dot_xy
    return distances


def K_S(info):
    """
    info: dict with keys
        'dataset': (n, d) data, n observations in d dimensions (values on missing entries ignored)
        'anchor_nodes': (m, d) anchor/inducing points
        'precision': (d,) per-dimension precision of the Gaussian kernel
        'masks': (n, d) boolean (or 0/1), 1 where that observation's dimension is missing, 0 where
                 observed (S); mask is observation-dependent, so each observation may have its own S
    returns: (n, m, m) stack of K_S(x_S) = phi_S(x_S) @ phi_S(x_S).T, one per observation
    """
    x = info['dataset']
    w = info['anchor_nodes']
    eta = info['precision']
    mask = np.asarray(info['masks'], dtype=bool)
    observed = ~mask  # (n, d): True on the observed (S) dimensions of each observation
    eta_S = eta[None, :] * observed  # (n, d)
    diff2 = (x[:, None, :] - w[None, :, :]) ** 2  # (n, m, d)
    phi_S = np.exp(-np.sum(eta_S[:, None, :] * diff2, axis=-1))  # (n, m)
    return np.einsum('ki,kj->kij', phi_S, phi_S)  # (n, m, m)


def C_2eta_NS(info):
    """
    info: dict with keys
        'precision': (d,) per-dimension precision of the Gaussian kernel
        'masks': (n, d) boolean (or 0/1), 1 where that observation's dimension is missing (NS), 0
                 where observed
    returns: (n,) array of c_{2 eta_NS} = (pi/2)^{|NS|/2} * prod_{d in NS} eta_d^{-1/2}, one per
             observation
    """
    eta = info['precision']
    mask = np.asarray(info['masks'], dtype=bool)  # True on missing (NS) dimensions
    log_c_2eta_NS = np.sum(mask * (0.5 * np.log(np.pi / 2) - 0.5 * np.log(eta)[None, :]), axis=-1)  # (n,)
    return np.exp(log_c_2eta_NS)


def H_NS(info):
    """
    info: dict with keys
        'anchor_nodes': (m, d) anchor/inducing points
        'precision': (d,) per-dimension precision of the Gaussian kernel
        'masks': (n, d) boolean (or 0/1), 1 where that observation's dimension is missing (NS), 0
                 where observed
    returns: (n, m, m) stack of k_{eta_NS / 2}(w_NS, w_NS), the matrix part of the closed form of
             integrating phi_i(x) phi_j(x) over the missing dimensions x_NS (Gaussian
             self-convolution); multiply by C_2eta_NS(info) to get the full H_NS
    """
    w = info['anchor_nodes']
    eta = info['precision']
    mask = np.asarray(info['masks'], dtype=bool)  # True on missing (NS) dimensions
    eta_NS_half = (eta[None, :] * mask) / 2  # (n, d)
    diff2 = (w[:, None, :] - w[None, :, :]) ** 2  # (m, m, d)
    d2 = np.einsum('kd,ijd->kij', eta_NS_half, diff2)  # (n, m, m)
    return np.exp(-d2)


def loss(info):
    """
    info: dict with keys
        'Q': (m, m) PSD matrix, the optimization variable
        'A': (N, m, m) stack of matrices A_i, i = 1, ..., N
        'A_0': (m, m) matrix
        'alpha': scalar added inside the log
        'lbd': scalar weight of the Tr(Q A_0) regularization term
        'mu': scalar weight of the -log det(Q) barrier term
    returns: scalar f(Q) = -(1/N) sum_i log(Tr(Q A_i) + alpha) + lbd * Tr(Q A_0) - mu * log det(Q)
    """
    Q = info['Q']
    A = info['A']
    A_0 = info['A_0']
    alpha = info['alpha']
    lbd = info['lbd']
    mu = info['mu']
    trace_QA = np.einsum('jk,ikj->i', Q, A)  # (N,): trace_QA[i] = Tr(Q @ A_i)
    trace_QA0 = np.einsum('jk,kj->', Q, A_0)  # Tr(Q @ A_0)
    _, logdet_Q = np.linalg.slogdet(Q)
    return -np.mean(np.log(trace_QA + alpha)) + lbd * trace_QA0 - mu * logdet_Q


def gradient(info):
    """
    info: dict with keys 'Q', 'A', 'A_0', 'alpha', 'lbd', 'mu' (see loss)
    returns: (m, m) matrix, grad f(Q) = -(1/N) sum_i A_i / (Tr(Q A_i) + alpha) + lbd * A_0 - mu * Q^-1
             (uses that A_i, A_0 are symmetric, so d Tr(Q A_i)/dQ = A_i; and d log det(Q)/dQ = Q^-1)
    """
    Q = info['Q']
    A = info['A']
    A_0 = info['A_0']
    alpha = info['alpha']
    lbd = info['lbd']
    mu = info['mu']
    g = np.einsum('jk,ikj->i', Q, A) + alpha  # (N,): g[i] = Tr(Q A_i) + alpha
    Q_inv = np.linalg.inv(Q)
    return -np.mean(A / g[:, None, None], axis=0) + lbd * A_0 - mu * Q_inv


def hessian(info):
    """
    info: dict with keys 'Q', 'A', 'alpha', 'mu' (A_0, lbd are not needed: the lbd * Tr(Q A_0)
          term of f is linear in Q, so it vanishes in the Hessian)
    returns: (m*m, m*m) matrix, the Hessian of f wrt Q flattened in row-major (C) order, i.e. the
             row/column index for entry Q[i, j] is I = i*m + j. So for row I = i*m+j and column
             J = t*m+s, entry [I, J] of the returned matrix equals the double derivative
             d^2f / (dQ_{ij} dQ_{ts})
                 = (1/N) sum_k A_k[i, j] * A_k[t, s] / (Tr(Q A_k) + alpha)^2
                   + mu * Q^-1[j, t] * Q^-1[s, i]
             (the log(Tr(QA_i)+alpha) part is (1/N) sum_k outer(vec(A_k), vec(A_k)) / g_k^2 with
             vec = row-major flatten; the -mu*log det(Q) part uses
             d^2 log det(Q) / (dQ_{ij} dQ_{ts}) = -Q^-1[j, t] * Q^-1[s, i])
    """
    Q = info['Q']
    A = info['A']
    alpha = info['alpha']
    mu = info['mu']
    n, m, _ = A.shape
    g = np.einsum('jk,ikj->i', Q, A) + alpha  # (N,): g[i] = Tr(Q A_i) + alpha
    vec_A = A.reshape(n, m * m)  # (N, m*m), row-major flatten, matches Q.reshape(-1)
    weighted_vec_A = vec_A / (g ** 2)[:, None]  # (N, m*m)
    hess_log_term = (vec_A.T @ weighted_vec_A) / n  # (m*m, m*m)
    Q_inv = np.linalg.inv(Q)
    hess_logdet_term = np.einsum('jt,si->ijts', Q_inv, Q_inv).reshape(m * m, m * m)  # (m*m, m*m)
    return hess_log_term + mu * hess_logdet_term


def hessian_vector_product(info):
    """
    info: dict with keys 'Q', 'A', 'alpha', 'mu' (see hessian), and 'dQ': (m, m) direction matrix
    returns: (m, m) matrix, the Hessian of f wrt Q applied to dQ:
             Hess[dQ] = (1/N) sum_i A_i * Tr(A_i @ dQ) / (Tr(Q A_i) + alpha)^2 + mu * Q^-1 @ dQ @ Q^-1
             (equivalent to reshape(hessian(info) @ dQ.reshape(-1), (m, m)), computed directly
             without forming the (m*m, m*m) matrix)
    """
    Q = info['Q']
    A = info['A']
    alpha = info['alpha']
    mu = info['mu']
    dQ = info['dQ']
    n = A.shape[0]
    g = np.einsum('jk,ikj->i', Q, A) + alpha  # (N,): g[i] = Tr(Q A_i) + alpha
    trace_AdQ = np.einsum('kjl,lj->k', A, dQ)  # (N,): trace_AdQ[i] = Tr(A_i @ dQ)
    coeff = trace_AdQ / (g ** 2)  # (N,)
    hvp_log_term = np.einsum('kjl,k->jl', A, coeff) / n  # (m, m)
    Q_inv = np.linalg.inv(Q)
    hvp_logdet_term = Q_inv @ dQ @ Q_inv  # (m, m)
    return hvp_log_term + mu * hvp_logdet_term


def hessian_inverse_vector_product(info):
    """
    Applies the inverse of the Hessian G_Q = A_{Q,alpha}^* A_{Q,alpha} + mu * Q^-1 (x) Q^-1 to a
    direction V, without forming the (m*m, m*m) Hessian matrix. Uses the Woodbury identity

        mu * G_Q^-1 = Q(x)Q - (Q(x)Q) A^* (mu * Id_N + A (Q(x)Q) A^*)^-1 A (Q(x)Q),   A = A_{Q,alpha}

    where, for W in S_p, A(W) in R^N has components A(W)_i = Tr(A_i W) / (sqrt(N)(Tr(A_i Q)+alpha)),
    and A^*(x) = (1/sqrt(N)) sum_i x_i A_i / (Tr(A_i Q) + alpha). Since (Q(x)Q) applied to a matrix
    M is just Q @ M @ Q, this reduces to a single N x N linear solve instead of an (m*m) x (m*m)
    one -- efficient whenever N (the number of A_i matrices) is smaller than m*m.

    info: dict with keys 'Q', 'A', 'alpha', 'mu' (see hessian), and 'V': (m, m) direction matrix
          (typically minus the gradient, to get a Newton step)
    returns: (m, m) matrix W solving G_Q(W) = V
    """
    Q = info['Q']
    A = info['A']
    alpha = info['alpha']
    mu = info['mu']
    V = info['V']
    N = A.shape[0]

    g = np.einsum('jk,ikj->i', Q, A) + alpha  # (N,): g[i] = Tr(Q A_i) + alpha
    C = A @ Q  # (N, m, m): C[i] = A_i @ Q

    trace_CiCj = np.einsum('ipq,jqp->ij', C, C)  # (N, N): Tr(C_i @ C_j) = Tr(A_i Q A_j Q)
    M = trace_CiCj / (N * np.outer(g, g))  # (N, N): entries of A (Q(x)Q) A^*
    S = mu * np.eye(N) + M  # (N, N)

    QVQ = Q @ V @ Q  # (Q(x)Q)(V)
    r = np.einsum('ipq,qp->i', A, QVQ) / (np.sqrt(N) * g)  # (N,): A(QVQ)
    y = np.linalg.solve(S, r)  # (N,): S^-1 A(QVQ)
    Z = np.einsum('i,ijk->jk', y / g, A) / np.sqrt(N)  # (m, m): A^*(y)

    return Q @ (V - Z) @ Q / mu



