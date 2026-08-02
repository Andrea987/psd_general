import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from toy_test_2 import generate_toy_info
from toy_test_1 import generate_toy_info
from alternating_minimization import alternating_minimization
from plot_toy_2d import evaluate_model
from sampling import sample_bisection, check_normalized
from marginalize_condition import condition


def fit_model(num_steps=30):
    info = generate_toy_info()
    info = dict(info)
    info['nbr_bounce'] = num_steps
    Q, W, eta, history = alternating_minimization(info)
    loss_value, lagrangian_value = history[-1]
    print(f'final loss={loss_value:.4f}, lagrangian={lagrangian_value:.4f}')
    check_normalized(W, eta, Q)
    return info, Q, W, eta


def plot_heatmap(ax, info, Q, W, eta, title):
    X = info['dataset']
    mask = info['masks']
    fully_observed = mask.sum(axis=1) == 0

    low = X.min(axis=0) - 0.3
    high = X.max(axis=0) + 0.3
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 150), np.linspace(low[1], high[1], 150))
    f_values = evaluate_model(Q, W, eta, np.stack([xx.ravel(), yy.ravel()], axis=1)).reshape(xx.shape)

    contour = ax.contourf(xx, yy, f_values, levels=30, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Tr(Q phi(x) phi(x)^T)')
    ax.scatter(X[fully_observed, 0], X[fully_observed, 1], s=15, c='white', edgecolors='black',
               label='fully observed')
    ax.scatter(X[~fully_observed, 0], X[~fully_observed, 1], s=15, c='tab:orange',
               edgecolors='black', label='partially missing')
    ax.scatter(W[:, 0], W[:, 1], marker='x', s=90, c='red', linewidths=2, label='anchor nodes')
    ax.set_title(title)
    ax.legend(fontsize=7, loc='best')


def plot_joint_samples(ax, info, Q, W, eta, num_samples=1000):
    X = info['dataset']
    joint_samples = sample_bisection(W, eta, Q, N=num_samples, tol=1e-3)

    ax.scatter(X[:, 0], X[:, 1], s=10, c='lightgray', label='training data')
    ax.scatter(joint_samples[:, 0], joint_samples[:, 1], s=10, c='tab:blue', alpha=0.5,
               label='joint samples')
    ax.scatter(W[:, 0], W[:, 1], marker='x', s=90, c='red', linewidths=2, label='anchor nodes')
    ax.set_title('sampling without missing components')
    ax.legend(fontsize=7, loc='best')


def plot_conditional_samples(ax, info, Q, W, eta, condition_on=0, num_conditioning_points=14,
                              samples_per_point=150):
    """
    Condition on dimension `condition_on` (observed) and sample the other dimension (missing)
    with `condition` + `sample_bisection`, mirroring what happens for a partially-missing
    observation at test time.
    """
    X = info['dataset']
    missing_dim = 1 - condition_on
    mask = np.zeros(2, dtype=bool)
    mask[missing_dim] = True  # True on the dimension being marginalized out/sampled

    given_values = np.linspace(X[:, condition_on].min(), X[:, condition_on].max(), num_conditioning_points)
    conditional_points = []
    for given in given_values:
        x = np.zeros(2)
        x[condition_on] = given
        W_ns, eta_ns, Q_cond = condition(W, eta, Q, mask, x)
        missing_samples = sample_bisection(W_ns, eta_ns, Q_cond, N=samples_per_point, tol=1e-2)
        for missing in missing_samples[:, 0]:
            point = [0.0, 0.0]
            point[condition_on] = given
            point[missing_dim] = missing
            conditional_points.append(point)
    conditional_points = np.array(conditional_points)

    given_name, missing_name = ('x', 'y') if condition_on == 0 else ('y', 'x')

    ax.scatter(X[:, 0], X[:, 1], s=10, c='lightgray', label='training data')
    ax.scatter(conditional_points[:, 0], conditional_points[:, 1], s=6, c='tab:green', alpha=0.35,
               label=f'samples of {missing_name} | {given_name}')
    ax.scatter(W[:, 0], W[:, 1], marker='x', s=90, c='red', linewidths=2, label='anchor nodes')
    ax.set_title(f'sampling with missing components\n(condition on {given_name}, sample {missing_name})')
    ax.legend(fontsize=7, loc='best')


def plot_sampling_demo(info, Q, W, eta, save_path='sampling_missing_components.png'):
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    axes = axes.ravel()

    plot_heatmap(axes[0], info, Q, W, eta, 'fitted joint model (heat map)')
    plot_joint_samples(axes[1], info, Q, W, eta)
    plot_conditional_samples(axes[2], info, Q, W, eta, condition_on=0)  # sample y | x
    plot_conditional_samples(axes[3], info, Q, W, eta, condition_on=1)  # sample x | y

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f'saved figure to {save_path}')


if __name__ == '__main__':
    info, Q, W, eta = fit_model(num_steps=30)
    plot_sampling_demo(info, Q, W, eta)
