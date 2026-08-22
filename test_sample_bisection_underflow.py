"""
Regression test for a math-domain-error crash in sample_bisection, discovered while running
main_real_dataset.py's multiple imputation on a real dataset (airfoil_self_noise) after the psd
model's precision grew large during optimization.

Root cause: _pairwise_weight's weight = Q * gaussKern(W, W, gamma / 2) can have negative entries
(Q is only PSD as a whole matrix, individual entries can be negative), so v = sum(G * weight) for
a still-too-small search window can come out slightly negative due to floating-point cancellation,
even though the true integral is mathematically non-negative. sample_bisection used to pass v
straight into math.log(v), which raises ValueError: math domain error for v <= 0.

test_fixtures/sample_bisection_negative_v_case.npz holds a synthetic (W, gamma, Q) that reproduces
v == 0.0 exactly at the very first search window, found by random search (see git history of this
file) rather than by re-running the full ~150-bounce real-dataset fit that originally triggered it.
"""

import math
import os

import numpy as np

from sampling import sample_bisection, integrate_hypercube, normalization_constant, mean, check_normalized

FIXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'test_fixtures', 'sample_bisection_negative_v_case.npz')


def load_fixture():
    data = np.load(FIXTURE_PATH)
    return data['W'], data['gamma'], data['Q']


def test_fixture_reproduces_the_original_crash_mechanism():
    """Confirms the fixture is still meaningful: without the v > 0 guard, this case really does
    hit math.log(v <= 0), i.e. this isn't testing an already-safe case."""
    W, gamma, Q = load_fixture()
    check_normalized(W, gamma, Q)

    d = W.shape[1]
    x0 = mean(W, gamma, Q).reshape(1, d)
    c = 10.0
    a, b = x0 - c, x0 + c
    v = float(integrate_hypercube(W, gamma, Q, a, b)[0])

    print('fixture v at first window:', v)
    assert v <= 0


def test_sample_bisection_does_not_crash_on_the_fixture():
    W, gamma, Q = load_fixture()
    samples = sample_bisection(W, gamma, Q, N=20)
    print('samples:', samples.ravel())
    assert samples.shape == (20, W.shape[1])
    assert np.all(np.isfinite(samples))


if __name__ == '__main__':
    test_fixture_reproduces_the_original_crash_mechanism()
    test_sample_bisection_does_not_crash_on_the_fixture()
    print('all tests passed')
