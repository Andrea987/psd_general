import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from psd import Distance_Matrix_Vector_Matrices
from alternating_minimization import alternating_minimization


def run_and_collect_steps(info, num_steps):
    """
    Runs alternating_minimization one full [Q, anchor_nodes, precision] step at a time (a "step"
    is one bounce: Newton for Q, then the gradient steps on anchor_nodes/precision), recording a
    snapshot after each step for plotting.
    """
    info = dict(info)
    info['nbr_bounce'] = 1  # one bounce per call, so we can snapshot in between calls

    snapshots = []
    for _ in range(num_steps):
        Q, W, eta, history = alternating_minimization(info)
        loss_value, lagrangian_value = history[0]
        snapshots.append({
            'Q': Q, 'anchor_nodes': W.copy(), 'precision': eta.copy(),
            'loss': loss_value, 'lagrangian': lagrangian_value,
        })
        info['Q'], info['anchor_nodes'], info['precision'] = Q, W, eta

    return snapshots


def plot_steps(info, snapshots, save_path='toy_2d_alternating_minimization.png'):
    X = info['dataset']
    mask = info['masks']
    fully_observed = mask.sum(axis=1) == 0

    num_steps = len(snapshots)
    ncols = min(4, num_steps)
    nrows = int(np.ceil(num_steps / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for step, snap in enumerate(snapshots):
        ax = axes[step // ncols][step % ncols]
        ax.scatter(X[fully_observed, 0], X[fully_observed, 1], s=15, c='tab:blue',
                   label='fully observed')
        ax.scatter(X[~fully_observed, 0], X[~fully_observed, 1], s=15, c='tab:orange', alpha=0.6,
                   label='partially missing')
        W = snap['anchor_nodes']
        ax.scatter(W[:, 0], W[:, 1], marker='x', s=90, c='black', linewidths=2, label='anchor nodes')
        ax.set_title(f"step {step + 1}\nloss={snap['loss']:.3f}")
        if step == 0:
            ax.legend(fontsize=7, loc='best')

    for k in range(num_steps, nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f'saved figure to {save_path}')


def evaluate_model(Q, W, eta, X_grid):
    """
    Q: (m, m), W: (m, d) anchor nodes, eta: (d,) precision, X_grid: (N, d) points to evaluate
    returns: (N,) array, f(x) = Tr(Q phi(x) phi(x)^T) = phi(x)^T Q phi(x), where
             phi(x)_k = k_eta(x, w_k) = exp(-eta . (x - w_k)^2) is the plain (unmasked) feature map
    """
    phi = np.exp(-Distance_Matrix_Vector_Matrices(X_grid, W, eta))  # (N, m)
    return np.einsum('gi,ij,gj->g', phi, Q, phi)


def _plot_model_panel(ax, X, fully_observed, Q, W, eta, title, xx, yy):
    f_values = evaluate_model(Q, W, eta, np.stack([xx.ravel(), yy.ravel()], axis=1)).reshape(xx.shape)

    contour = ax.contourf(xx, yy, f_values, levels=30, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Tr(Q phi(x) phi(x)^T)')
    ax.scatter(X[fully_observed, 0], X[fully_observed, 1], s=15, c='white', edgecolors='black',
               label='fully observed')
    ax.scatter(X[~fully_observed, 0], X[~fully_observed, 1], s=15, c='tab:orange',
               edgecolors='black', label='partially missing')
    ax.scatter(W[:, 0], W[:, 1], marker='x', s=90, c='red', linewidths=2, label='anchor nodes')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='best')


def plot_initial_vs_final_model(info, snapshots, save_path='toy_2d_final_model.png'):
    X = info['dataset']
    mask = info['masks']
    fully_observed = mask.sum(axis=1) == 0

    initial_Q, initial_W, initial_eta = info['Q'], info['anchor_nodes'], info['precision']
    final = snapshots[-1]
    final_Q, final_W, final_eta = final['Q'], final['anchor_nodes'], final['precision']

    low = X.min(axis=0) - 0.3
    high = X.max(axis=0) + 0.3
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 150), np.linspace(low[1], high[1], 150))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _plot_model_panel(axes[0], X, fully_observed, initial_Q, initial_W, initial_eta,
                       'initial conditions', xx, yy)
    _plot_model_panel(axes[1], X, fully_observed, final_Q, final_W, final_eta,
                       f"final model, loss={final['loss']:.3f}", xx, yy)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f'saved figure to {save_path}')
