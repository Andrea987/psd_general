import numpy as np
import argparse
import ast
from generate_masks import generate_masks_2d
from sklearn.datasets import make_moons, make_circles

parser = argparse.ArgumentParser()
parser.add_argument('-mi_nbr', '--mult_imp_nbr', type=int, default=30, help='number of multiple imputation')
parser.add_argument('--c_mu', default=0.5, type=float, help='constant log det')
parser.add_argument('--eta', default=None, help='inverse variance, parameter Gaussian cross val psd model')
parser.add_argument('--nbr_nodes', type=int, default=7, help='set a max nbr of nodes that our model can bear')
parser.add_argument('--nbr_bounce', type=int, default=120, help='how many times bounce between optimizing the matrix and the nodes')
parser.add_argument('--lbd_constraint', type=float, default=1e-9, help='regularizer for constraint matrix')
parser.add_argument('--lbd_kernel', type=float, default=1e-9, help='regularizer for kernel matrices')
parser.add_argument('--lbd_kernel_after', type=float, default=1e-9, help='regularizer for kernel matrices after change of variable')
parser.add_argument('--first_nwt', type=int, default=100, help='first iteration newton method')
parser.add_argument('--intermediate_nwt', type=int, default=10, help='intermediate iteration newton method')
parser.add_argument('--last_nwt', type=int, default=100, help='last iteration newton method')
parser.add_argument('--hit_opt_algo', type=int, default=5, help='nbr of hit with opt algo')
parser.add_argument('--l_rate_nodes', type=float, default=1e-3, help='learning rate nodes')
parser.add_argument('--l_rate_param', type=float, default=1e-3, help='learning rate parameter Gaussian')
parser.add_argument('--tolerance', type=float, default=0.68**2, help='tolerance Newton method')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha backtracking line search (if condition)')
parser.add_argument('--beta', type=float, default=0.8, help='beta backtracking line search (damping term)')
parser.add_argument('--prior', default=None, help='define the prior')
parser.add_argument('--p', default='[0.3, 0.35, 0.35]', help='probability of missing of one component')
parser.add_argument('--max_iter_newton', type=int, default=20,
                    help='max iter newton method for psd model')
parser.add_argument('--percentage_out', type=float, default=0.3,
                    help='percentage of ground_truth that we see for sure. This will be the nodes of the psd model')

parser.add_argument('--nbr_exp', type=int, default=1)
parser.add_argument('--seed_random', type=int, default=176)
parser.add_argument('--thickness_grid', type=int, default=800)
parser.add_argument('--nbr_training_point', type=int, default=350)
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--dataset_chosen', type=str, default='circles')

args = parser.parse_args()
np.random.seed(args.seed_random)

if __name__ == "__main__":
    number_experiment = args.nbr_exp
    
    thickness_grid = args.thickness_grid
    total_training_points = args.nbr_training_point
    d = args.dim 
    c_mu = args.c_mu  # c_mu = (mu - r - 1) / 2
    if args.eta is None:
        eta = np.array([5, 5]) if d == 2 else np.array([5, 5, 5])
    else:
        eta = np.array(ast.literal_eval(args.eta))
    nb_nodes = args.nbr_nodes  # nbr anchor points
    alpha = args.alpha  # parameter newton method
    beta = args.beta  # parameter newthon method
    nbr_bounce = args.nbr_bounce
    p = args.p

    if d == 2:
        dataset_chosen = args.dataset_chosen
        if dataset_chosen == 'half_moon':
            observations, _ = make_moons(n_samples=total_training_points, shuffle=True, noise=0.1)  # random_state=42)
        elif dataset_chosen == 'circles':
            total_training_points_circle = total_training_points
            observations, _ = make_circles(n_samples=total_training_points_circle, shuffle=True, noise=0.1, factor=0.35)
        masks = generate_masks_2d(observations.shape[0], [0.3, 0.35, 0.35])
        low = np.min(observations * masks, axis=0)
        high = np.max(observations * masks, axis=0)
        a_n = np.random.uniform(low=low, high=high, size=(nb_nodes, d))
    M = generate_masks_2d(nbr_of_sample=10, p_missing=[0.5, 0.25, 0.25])

        
    info = {'dataset': observations, 'masks': M, 'anchor_nodes': a_n, 'model_precision': eta}

    


















