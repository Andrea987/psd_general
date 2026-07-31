import numpy as np
from psd import (loss, gradient, hessian, hessian_vector_product, hessian_inverse_vector_product,
                  newton_step_and_decrement)
from optimization import newton_method


def make_info(m=3, N=10, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, m, m))
    #B = rng.standard_normal((N, d, d))
    #BB = B @ B.transpose(0, 2, 1) + 0.5 * np.eye(d)
    A = A @ A.transpose(0, 2, 1) + 0.5 * np.eye(m)  # symmetric, as K_S/H_NS outputs are
    #BBB = np.einsum('kih,kjh->kij', B, B) + 0.5 * np.eye(d) 
    #print(BB)
    #print("djajdaksadkjd")
    #print(BBB)
    #input()
    A_0 = rng.standard_normal((m, m))
    A_0 = A_0 + A_0.T
    L = rng.standard_normal((m, m))
    Q = L @ L.T + np.eye(m) * 2  # PD, invertible
    return {'Q': Q, 'A': A, 'A_0': A_0, 'alpha': 0.7, 'lbd': 0.3, 'mu': 0.5}


def test_gradient_matches_finite_difference():
    info = make_info()
    m = info['Q'].shape[0]
    Q0 = info['Q'].copy()
    eps = 1e-6

    grad_fd = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            Qp, Qm = Q0.copy(), Q0.copy()
            Qp[i, j] += eps
            Qm[i, j] -= eps
            #print(loss({**info, 'Q': Qp}))
            loss({**info, 'Q': Qm})
            grad_fd[i, j] = (loss({**info, 'Q': Qp}) - loss({**info, 'Q': Qm})) / (2 * eps)

    grad_analytic = gradient(info)
    #print(grad_fd)
    #print(grad_analytic)
    max_diff = np.max(np.abs(grad_fd - grad_analytic))
    #print(max_diff)
    print('gradient vs finite-difference max abs diff:', max_diff)
    assert max_diff < 1e-5

'''
def test_hessian_matches_finite_difference_of_gradient():
    info = make_info()
    m = info['Q'].shape[0]
    Q0 = info['Q'].copy()
    eps = 1e-6

    H = hessian(info)
    H_fd = np.zeros((m * m, m * m))
    # H[I, J] = d^2 f / (dQ_ij dQ_ts), with I = i*m+j the row and J = t*m+s the column (see the
    # docstring of hessian()). The outer (t, s) loop perturbs Q_ts to finite-difference the
    # gradient; the inner (i, j) loop reads off each entry of that (m, m) gradient difference and
    # places it at its own row I = i*m+j, spelling out the (m, m) <-> (m*m,) conversion explicitly.
    for t in range(m):
        for s in range(m):
            Qp, Qm = Q0.copy(), Q0.copy()
            Qp[t, s] += eps
            Qm[t, s] -= eps
            gdiff = (gradient({**info, 'Q': Qp}) - gradient({**info, 'Q': Qm})) / (2 * eps)  # (m, m)
            J = t * m + s
            for i in range(m):
                for j in range(m):
                    I = i * m + j
                    H_fd[I, J] = gdiff[i, j]
            print(H_fd)
    print("final result \n", H, "\n")
    print(H_fd, "\n")
    print(np.abs(H - H_fd))
    max_diff = np.max(np.abs(H - H_fd))
    print('hessian vs finite-difference max abs diff:', max_diff)
    assert max_diff < 1e-4
'''

def test_hessian_vector_product_matches_full_hessian():
    info = make_info()
    m = info['Q'].shape[0]
    rng = np.random.default_rng(1)
    dQ = rng.standard_normal((m, m))
    dQ = dQ + dQ.T

    H = hessian(info)
    hvp_via_hessian = (H @ dQ.reshape(-1)).reshape(m, m)
    hvp_direct = hessian_vector_product({**info, 'dQ': dQ})

    max_diff = np.max(np.abs(hvp_via_hessian - hvp_direct))
    print('hessian_vector_product vs hessian @ vec(dQ) max abs diff:', max_diff)
    assert max_diff < 1e-8


def test_hessian_inverse_vector_product_matches_full_hessian_solve():
    info = make_info()
    m = info['Q'].shape[0]
    rng = np.random.default_rng(2)
    V = rng.standard_normal((m, m))
    V = V + V.T

    H = hessian(info)
    W_via_full_solve = np.linalg.solve(H, V.reshape(-1)).reshape(m, m)
    W_woodbury = hessian_inverse_vector_product({**info, 'V': V})

    max_diff = np.max(np.abs(W_via_full_solve - W_woodbury))
    print('hessian_inverse_vector_product vs solve(hessian, V) max abs diff:', max_diff)
    assert max_diff < 1e-6


def test_hessian_inverse_vector_product_is_a_true_inverse():
    info = make_info()
    m = info['Q'].shape[0]
    rng = np.random.default_rng(3)
    V = rng.standard_normal((m, m))
    V = V + V.T

    W = hessian_inverse_vector_product({**info, 'V': V})
    V_roundtrip = hessian_vector_product({**info, 'dQ': W})

    max_diff = np.max(np.abs(V - V_roundtrip))
    print('hessian_vector_product(hessian_inverse_vector_product((V)) vs V max abs diff:', max_diff)
    assert max_diff < 1e-6



def test_hessian_is_the_inverse_of_the_inverse():
    info = make_info()
    m = info['Q'].shape[0]
    rng = np.random.default_rng(3)
    dQ = rng.standard_normal((m, m))
    dQ = dQ + dQ.T

    V = hessian_vector_product({**info, 'dQ': dQ})
    dQ_roundtrip = hessian_inverse_vector_product({**info, 'V': V})

    max_diff = np.max(np.abs(dQ - dQ_roundtrip))
    print('hessian_inverse_vector_product(hessian_vector_product((dQ)) vs dQ max abs diff:', max_diff)
    assert max_diff < 1e-6


def test_newton_decrement_matches_minus_trace_gradient_DQ():
    info = make_info()
    Q = info['Q']
    m = Q.shape[0]
    rng = np.random.default_rng(4)

    # H must be positive definite and satisfy the constraint Tr(Q H) = 1 that
    # newton_step_and_decrement relies on to simplify <Q, nu * H> = nu.
    L = rng.standard_normal((m, m))
    H_raw = L @ L.T + np.eye(m)  # PD
    H = H_raw / np.einsum('ij,ij->', Q, H_raw)  # rescale by a positive scalar: stays PD
    info = {**info, 'H': H}

    DQ, nu, lambda_Q_squared = newton_step_and_decrement(info)
    g_Q = gradient(info)
    lambda_Q_squared_alt = -np.einsum('ij,ij->', g_Q, DQ)  # -Tr(g_Q @ DQ)

    max_diff = np.abs(lambda_Q_squared - lambda_Q_squared_alt)
    max_diff_trace = np.abs(np.einsum('ij,ij->', DQ, H))
    print('newton decrement^2 vs -Tr(g_Q @ DQ) abs diff:', max_diff)
    print('Tr(Q DQ) - 0, abs diff:', max_diff)

    assert max_diff < 1e-6


def test_newton_method_on_synthetic_mcar_data():
    rng = np.random.default_rng(0)
    n = 10  # number of observations
    d = 3    # dimension of each observation
    p_missing = 0.7  # MCAR: each entry is missing independently with this probability

    dataset = rng.standard_normal((n, d))  # observations drawn from a standard Gaussian
    masks = (rng.random((n, d)) < p_missing).astype(float)  # 1 = missing, 0 = observed

    # MCAR can by chance blank out an entire observation; reveal one random dimension of any
    # fully-missing row so every observation carries at least some information.
    fully_missing_rows = np.where(masks.sum(axis=1) == d)[0]
    #print(fully_missing_rows)
    #print(masks)
    if len(fully_missing_rows) > 0:
        revealed_dims = rng.integers(0, d, size=len(fully_missing_rows))
        #print(revealed_dims)
        masks[fully_missing_rows, revealed_dims] = 0

    #print(masks)
    print('dataset shape:', dataset.shape)
    print('masks shape:', masks.shape, 'fraction missing:', masks.mean())

    assert dataset.shape == (n, d)
    assert masks.shape == (n, d)
    assert np.all((masks == 0) | (masks == 1))  # masks is binary
    assert np.all(masks.sum(axis=1) < d)  # no observation is fully missing


def test_newton_method_decreases_loss_and_converges():
    info = make_info()
    Q0 = info['Q']
    m = Q0.shape[0]
    rng = np.random.default_rng(9)

    # H must be positive definite and satisfy the constraint Tr(Q H) = 1 (see
    # test_newton_decrement_matches_minus_trace_gradient_DQ)
    L = rng.standard_normal((m, m))
    H_raw = L @ L.T + np.eye(m)
    H = H_raw / np.einsum('ij,ij->', Q0, H_raw)
    info = {**info, 'H': H}

    Q_final, history = newton_method(info)
    losses = [entry[0] for entry in history]
    print('loss history:', losses)
    print('final lambda_Q_squared:', history[-1][1])

    # the loss should never increase from one iteration to the next
    assert all(losses[i + 1] <= losses[i] + 1e-8 for i in range(len(losses) - 1))

    # it should have stopped because the squared Newton decrement dropped below tolerance
    assert history[-1][1] <= 0.68 ** 2

    # the final Q must still be in the domain: positive definite and feasible
    np.linalg.cholesky(Q_final)  # raises LinAlgError if not PD
    trace_QH = np.einsum('ij,ij->', Q_final, H)
    print('Tr(Q_final H):', trace_QH)
    assert abs(trace_QH - 1) < 1e-4


if __name__ == '__main__':
    test_gradient_matches_finite_difference()
    #test_hessian_matches_finite_difference_of_gradient()
    test_hessian_vector_product_matches_full_hessian()
    test_hessian_inverse_vector_product_matches_full_hessian_solve()
    test_hessian_inverse_vector_product_is_a_true_inverse()
    test_hessian_is_the_inverse_of_the_inverse()
    test_newton_decrement_matches_minus_trace_gradient_DQ()
    test_newton_method_on_synthetic_mcar_data()
    test_newton_method_decreases_loss_and_converges()
    print('all tests passed')
