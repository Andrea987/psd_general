"""
Standalone, torch-free re-implementation of GaussianPsdModel's bisection sampler
(psd_models/GaussianPsdModel.py:_sample_bisection).

Everything needed to run it is included here (kernel, closed-form Gaussian box
integral, mean, normalization constant, hypercube integration) so this module
has no dependency on torch or on the rest of psd_models -- only numpy and scipy.

A GaussianPsdModel is defined by:
  * W:     (N, d) anchor points
  * gamma: (1, d) or (d,) per-dimension bandwidth
  * Q:     (N, N) PSD weight matrix (called `B` on the model itself)

To sample from an existing (torch-based) model without modifying it, pull its
arrays out as numpy and call `sample_bisection`:

>>> W = model.X.to_bs_matrix().decompress().X.detach().cpu().numpy()
>>> gamma = model.gamma.to_bs_matrix().X.detach().cpu().numpy()
>>> Q = model.B.detach().cpu().numpy()
>>> samples = sample_bisection(W, gamma, Q, N=1000)
"""

import math

import numpy as np
from scipy.special import erf


def sqdist(X, Y):
    normX = (X ** 2).sum(-1)
    normY = (Y ** 2).sum(-1)
    K = X @ Y.T
    K *= -2
    K += normX[:, None]
    K += normY[None, :]
    return K


def gaussKern(X, Y, gamma):
    sg = np.sqrt(gamma)
    K = sqdist(X * sg, Y * sg)
    K *= -1
    return np.exp(K)


def gaussian_integrate(X, gamma, a, b):
    """
    :param X: (N, d)
    :param gamma: (1, d) or (d,)
    :param a: (m, d)
    :param b: (m, d)
    :return: (N, m) matrix M where M[i, j] = int_[a[j,:],b[j,:]] exp(-gamma*(x-X[i,:])**2) dx
    """
    sg = np.sqrt(gamma)
    X = X * sg
    a = a * sg
    b = b * sg

    c = math.sqrt(math.pi) / 2.0 / sg
    return (
        c * (erf(b[None, :, :] - X[:, None, :]) - erf(a[None, :, :] - X[:, None, :]))
    ).prod(2)


def _pairwise_weight(W, gamma, Q):
    """Q[i, j] * gaussKern(W[i], W[j], gamma / 2), flattened, plus the pairwise midpoints."""
    N, d = W.shape
    weight = Q * gaussKern(W, W, gamma / 2)
    midpoints = 0.5 * (W.reshape(-1, 1, d) + W.reshape(1, -1, d)).reshape(-1, d)
    return weight.reshape(-1, 1), midpoints


def normalization_constant(W, gamma, Q):
    """Tr(Q H_{gamma/2}) * c_{2gamma}, H_{gamma/2} = gaussKern(W, W, gamma / 2)."""
    weight = Q * gaussKern(W, W, gamma / 2)
    return weight.sum() * np.sqrt(math.pi / (2 * gamma)).prod()


def check_normalized(W, gamma, Q, tol=1e-6):
    """
    Sanity check that (W, gamma, Q) is a properly normalized probability density -- we can only
    sample from a base matrix that integrates to 1, not an arbitrary PSD one.

    :raises AssertionError: if Tr(Q H_{gamma/2}) * c_{2gamma} is not within `tol` of 1
    """
    Z = normalization_constant(W, gamma, Q)
    assert abs(Z - 1.0) < tol, (
        f"(W, gamma, Q) is not normalized: Tr(Q H_(gamma/2)) * c_2gamma = {Z}, expected 1. "
        "Normalize Q (e.g. divide by its normalization_constant) before sampling from it."
    )


def mean(W, gamma, Q):
    weight = Q * gaussKern(W, W, gamma / 2)
    return W.T @ weight.sum(1)


def integrate_hypercube(W, gamma, Q, a, b):
    """
    :param a: (m, d)
    :param b: (m, d)
    :return: (m,) mass of the model integrated over each of the m hyper-rectangles [a[i], b[i]]
    """
    weight, midpoints = _pairwise_weight(W, gamma, Q)
    G = gaussian_integrate(midpoints, 2.0 * gamma, a, b)
    return (G * weight).sum(axis=0)


def sample_bisection(W, gamma, Q, N, tol=1e-3):
    """
    Draw N i.i.d. samples from a GaussianPsdModel(Q, W, gamma) by repeatedly
    bisecting an axis-aligned hyper-rectangle according to the mass the model
    assigns to each half.

    :param W: (N_anchors, d) anchor points
    :param gamma: (1, d) or (d,) per-dimension bandwidth
    :param Q: (N_anchors, N_anchors) PSD weight matrix
    :param N: number of samples to draw
    :param tol: (default 1e-3) tolerance of the sampling strategy
    :return: (N, d) numpy array of i.i.d. samples
    """
    W = np.asarray(W, dtype=float)
    gamma = np.asarray(gamma, dtype=float).reshape(1, -1)
    Q = np.asarray(Q, dtype=float)
    check_normalized(W, gamma, Q)

    d = W.shape[1]
    x0 = mean(W, gamma, Q).reshape(1, d)
    qq = math.log(normalization_constant(W, gamma, Q))

    c = 1.0
    while True:
        c *= 10
        a = x0 - c
        b = x0 + c
        v = float(integrate_hypercube(W, gamma, Q, a, b)[0])
        if math.log(v) - qq >= math.log(1.0 - tol):
            break

    steps = math.ceil(d * (math.log(2 * c / tol) / math.log(2)))

    A = np.repeat(a, N, axis=0)
    B = np.repeat(b, N, axis=0)

    for i in range(steps):
        j = i % d
        A1, A2, B1, B2 = A.copy(), A.copy(), B.copy(), B.copy()
        B1[:, j] = (B[:, j] + A[:, j]) / 2.0
        A2[:, j] = B1[:, j]

        z = np.random.rand(N, 1) * v

        v1 = integrate_hypercube(W, gamma, Q, A1, B1).reshape(N, 1)

        eq, lt, gt = (z == v1), (z < v1), (z > v1)
        A = A * eq + A1 * lt + A2 * gt
        B = B * eq + B1 * lt + B2 * gt
        v = v * eq + v1 * lt + (v - v1) * gt

    V = (B - A) * np.random.rand(N, d) + A
    return V[np.random.permutation(N), :]
