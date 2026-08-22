"""
One-off script: fit the psd model on a real dataset with a long enough run (high precision,
large enough anchor count) that it lands in the same numerically extreme regime that used to
crash marginalize_condition.condition()/sampling.sample_bisection() with a division by zero /
math domain error (every anchor node's kernel value underflowing to exactly 0.0 for some row).

Saves the fitted (dataset, mask, Q, anchor_nodes, precision) to test_fixtures/, so
test_marginalize_condition_underflow.py can reproduce the exact real-world crash scenario without
re-running the full ~150-bounce optimization every time.
"""

import os
import sys

import numpy as np
from sklearn.preprocessing import scale

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MissingDataOT_master'))
from MissingDataOT_master.data_loaders import dataset_loader

from psd_imputer import fit_psd_model

if __name__ == '__main__':
    seed = 0
    rng = np.random.default_rng(seed)

    X_full = scale(dataset_loader('airfoil_self_noise'))
    idx = rng.permutation(X_full.shape[0])
    X_train = X_full[idx[:400]]

    n, d = X_train.shape
    mask = (rng.random((n, d)) < 0.3).astype(float)
    fully_missing = np.where(mask.sum(axis=1) == d)[0]
    if len(fully_missing) > 0:
        revealed = rng.integers(0, d, size=len(fully_missing))
        mask[fully_missing, revealed] = 0

    Q, W, eta, history = fit_psd_model(
        X_train, mask, m=65, eta_init=np.log(5), alpha=1e-6, lbd=1e-1 / n, mu=1e-3 / n,
        l_rate_nodes=1e-4, l_rate_param=1e-4, nbr_bounce=150, nbr_gradient_steps=5,
        nbr_newton_step_Q=50, seed=seed, verbose=True,
    )

    print(f'\nfinal precision: {eta}')

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'test_fixtures', 'airfoil_underflow_case.npz')
    np.savez(out_path, dataset=X_train, mask=mask, Q=Q, anchor_nodes=W, precision=eta)
    print(f'saved fixture to {out_path}')
