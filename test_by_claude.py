import numpy as np
from psd import loss, gradient, hessian, hessian_vector_product, hessian_inverse_vector_product


def make_info(m=4, N=6, seed=0):
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
            print(loss({**info, 'Q': Qp}))
            loss({**info, 'Q': Qm})
            grad_fd[i, j] = (loss({**info, 'Q': Qp}) - loss({**info, 'Q': Qm})) / (2 * eps)

    grad_analytic = gradient(info)
    print(grad_fd)
    print(grad_analytic)
    max_diff = np.max(np.abs(grad_fd - grad_analytic))
    print(max_diff)
    print('gradient vs finite-difference max abs diff:', max_diff)
    assert max_diff < 1e-5


def test_hessian_matches_finite_difference_of_gradient():
    info = make_info()
    m = info['Q'].shape[0]
    Q0 = info['Q'].copy()
    eps = 1e-6

    H = hessian(info)
    H_fd = np.zeros((m * m, m * m))
    for t in range(m):
        for s in range(m):
            Qp, Qm = Q0.copy(), Q0.copy()
            Qp[t, s] += eps
            Qm[t, s] -= eps
            gdiff = (gradient({**info, 'Q': Qp}) - gradient({**info, 'Q': Qm})) / (2 * eps)
            H_fd[:, t * m + s] = gdiff.reshape(-1)

    max_diff = np.max(np.abs(H - H_fd))
    print('hessian vs finite-difference max abs diff:', max_diff)
    assert max_diff < 1e-4


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
    print('hessian_inverse_vector_product(hessian_vector_product(V)) vs V max abs diff:', max_diff)
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
    print('hessian_inverse_vector_product(hessian_vector_product((V)) vs V max abs diff:', max_diff)
    assert max_diff < 1e-6


if __name__ == '__main__':
    test_gradient_matches_finite_difference()
    #test_hessian_matches_finite_difference_of_gradient()
    test_hessian_vector_product_matches_full_hessian()
    test_hessian_inverse_vector_product_matches_full_hessian_solve()
    test_hessian_inverse_vector_product_is_a_true_inverse()
    test_hessian_is_the_inverse_of_the_inverse()
    print('all tests passed')
