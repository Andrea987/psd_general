import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from psd import Distance_Matrix_Vector_Matrices


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


def plot_initial_vs_final_model(info, final_Q, final_W, final_eta, final_loss,
                                 save_path='toy_2d_final_model.png'):
    """
    info: dict with keys 'dataset', 'masks', 'Q', 'anchor_nodes', 'precision' -- the initial
        conditions (see alternating_minimization)
    final_Q, final_W, final_eta: the model returned by alternating_minimization after a full run
    final_loss: general_loss at (final_Q, final_W, final_eta), e.g. history[-1][0]
    """
    X = info['dataset']
    mask = info['masks']
    fully_observed = mask.sum(axis=1) == 0

    initial_Q, initial_W, initial_eta = info['Q'], info['anchor_nodes'], info['precision']

    low = X.min(axis=0) - 0.3
    high = X.max(axis=0) + 0.3
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 150), np.linspace(low[1], high[1], 150))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _plot_model_panel(axes[0], X, fully_observed, initial_Q, initial_W, initial_eta,
                       'initial conditions', xx, yy)
    _plot_model_panel(axes[1], X, fully_observed, final_Q, final_W, final_eta,
                       f"final model, loss={final_loss:.3f}", xx, yy)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f'saved figure to {save_path}')
