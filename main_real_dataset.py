#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn

from geomloss import SamplesLoss

import ot

import os
import sys
import pickle as pkl
import copy
import time
import resource

from sklearn.preprocessing import scale
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

# MissingDataOT_master/imputers.py does a bare "from utils import ..." internally, assuming it is
# run from inside that folder rather than imported as a subpackage -- add the folder itself to
# sys.path so that bare import resolves too, alongside the package-style imports below.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MissingDataOT_master'))

from MissingDataOT_master.utils import *
from MissingDataOT_master.softimpute import softimpute, cv_softimpute
from MissingDataOT_master.data_loaders import dataset_loader
from MissingDataOT_master.imputers import OTimputer, RRimputer

from psd_imputer import psd_impute, cross_validate_hyperparams, cross_validate_anchor_nodes
from psd import energy_distance

import argparse
import logging

parser = argparse.ArgumentParser()

# reproducibility / output
parser.add_argument('--seed', type=int, default=42,
                    help='seed for reproducibility')
parser.add_argument('--out_path', type=str, default=None,
                    help='filename for the results')
parser.add_argument('--out_data', type=str, default=None,
                    help='filename for the data')
parser.add_argument('--out_dir', type=str, default='exps',
                    help='directory name for results')
parser.add_argument('--nexp', type=int, default=1,
                    help='number of experiences per parameter setting')

# dataset and how missingness is introduced into it
parser.add_argument('--dataset', type=str, default="iris",
                    help='dataset on which to run the experiments')
parser.add_argument('--perc_test_set', type=float, default=0,
                    help='percentage of dataset held out as a test set')
parser.add_argument('--large_dataset_threshold', type=int, default=1000,
                    help='datasets with more observations than this use --max_train_obs for '
                         'training and put everything else in the test set, ignoring '
                         '--perc_test_set; datasets at or below this threshold use the usual '
                         '--perc_test_set split')
parser.add_argument('--max_train_obs', type=int, default=500,
                    help='training-set size used for datasets over --large_dataset_threshold '
                         '(see above)')
parser.add_argument('--p', type=float, default=0.3, help='Proportion of imps')
parser.add_argument('--MAR', action='store_true')
parser.add_argument('--p_obs', type=float, default=0.3,
                    help='Proportion of variables that are fully observed (MAR & MNAR model)')
parser.add_argument('--MNAR_log', action='store_true')
parser.add_argument('--MNAR_quant', action='store_true')
parser.add_argument('--q_mnar', type=float, default=0.75,
                    help='quantile that will have imps (MNAR quantiles model)')

# Sinkhorn / round-robin imputation baselines
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--decay', type=float, default=1e-5,
                    help='weight decay (round robin)')
parser.add_argument('--scaling', type=float, default=.9,
                    help='sinkhorn scaling parameter (speed/precision tradeoff)')
parser.add_argument('-b', '--batchsize', type=int, default=128,
                    help='batchsize(s) for the experiments')
parser.add_argument('--sinkhorn_niter', type=int, default=3000,
                    help='number of GD iterations (Sinkhorn imputation)')
parser.add_argument('--max_iter', type=int, default=15,
                    help='maximum number of cycles (round robin)')
parser.add_argument('--rr_niter', type=int, default=15,
                    help='number of GD iterations (round robin)')
parser.add_argument('--n_pairs', type=int, default=10,
                    help='number of pairs batches to sample (round robin)')
parser.add_argument('-e', '--epsilon', type=float, default=None,
                    help='Sinkhorn regularization parameter. '
                         'Automatically select using median distance by default')
parser.add_argument('--quantile', type=float, default=.5,
                    help='distance quantile to select epsilon')
parser.add_argument('-qm', '--quantile_multiplier', type=float, default=0.05,
                    help='distance quantile x multiplier =  epsilon')

# logging
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--verbose_newton', action='store_true',
                    help='print every Newton iteration (loss, decrement) inside each psd bounce, '
                         'separately from --verbose (which only summarizes every 5th bounce)')
parser.add_argument('--psd_verbose_metrics', action='store_true',
                    help='alongside --verbose, also print MAE/RMSE/OT/ED (mean-imputed against '
                         'the true training data) every 5th bounce -- more expensive than the '
                         'other --verbose output, since it mean-imputes the whole training set '
                         'each time it prints, so opt-in separately')
parser.add_argument('--report_interval', type=int, default=500)

# our method (psd_impute, see psd_imputer.py / fit_psd_model for what each of these controls)
parser.add_argument('--psd_m', type=int, default=80,
                    help='number of anchor nodes')
parser.add_argument('--psd_eta', type=float, default=np.log(2),
                    help='initial LOG-precision of the Gaussians -- actual starting precision '
                         'is exp(this value), matching the log-space reparametrization used for '
                         'the precision gradient step (see alternating_minimization). Ignored if '
                         '--psd_eta_init_mode=empirical')
parser.add_argument('--psd_alpha', type=float, default=1e-6,
                    help='parameter inside the log of the loss, log(Tr(Q.) + alpha)')
parser.add_argument('--psd_lbd', type=float, default=1e-1,
                    help='hyperparameter of the trace regularizer')
parser.add_argument('--psd_mu', type=float, default=1e-1,
                    help='hyperparameter of the logdet regularizer')
parser.add_argument('--psd_lr_nodes', type=float, default=1e-4,
                    help='learning rate for the anchor nodes in the alternating minimization')
parser.add_argument('--psd_lr_param', type=float, default=1e-4,
                    help='learning rate for the precision in the alternating minimization')
parser.add_argument('--psd_bounce', type=int, default=75,
                    help='number of alternating minimization steps')
parser.add_argument('--psd_gradient_steps', type=int, default=5,
                    help='number of gradient steps to optimize anchor nodes and precision, per bounce')
parser.add_argument('--psd_newton_step_Q', type=int, default=50,
                    help='maximum number of Newton iterations for each inner Q update')
parser.add_argument('--psd_n_imputations', type=int, default=10,
                    help='number of independent completions drawn for the multiple-imputation '
                         'OT/ED-vs-test-set metric (0 to disable)')
parser.add_argument('--psd_anchor_init', type=str, default='data_subset',
                    choices=['data_subset', 'uniform_hypercube'],
                    help='data_subset: anchor nodes are a random subset of the mean-imputed data. '
                         'uniform_hypercube: anchor nodes are sampled i.i.d. uniformly in the '
                         'bounding box of the (mean-imputed) data')
parser.add_argument('--psd_eta_init_mode', type=str, default='fixed',
                    choices=['fixed', 'empirical'],
                    help='fixed: every dimension starts at --psd_eta. empirical: every dimension '
                         'starts at 1 / empirical_variance, computed from the observed entries')

# fast cross-validation of --psd_lr_nodes / --psd_lr_param / --psd_eta on a small subsample,
# before the main fit (see psd_imputer.cross_validate_hyperparams)
parser.add_argument('--psd_cross_validate', action='store_true',
                    help='before fitting, cross-validate --psd_lr_nodes/--psd_lr_param/--psd_eta '
                         'on a small, fast validation subsample and use the best combination '
                         '(overrides those three flags)')
parser.add_argument('--psd_cv_n', type=int, default=200,
                    help='validation subsample size for --psd_cross_validate')
parser.add_argument('--psd_cv_m', type=int, default=10,
                    help='number of anchor nodes for --psd_cross_validate')
parser.add_argument('--psd_cv_bounce', type=int, default=20,
                    help='number of alternating minimization bounces for --psd_cross_validate')
parser.add_argument('--psd_cv_lr_nodes_grid', type=str, default='1e-4,1e-3,1e-2',
                    help='comma-separated candidate values for --psd_lr_nodes')
parser.add_argument('--psd_cv_lr_param_grid', type=str, default='1e-5,1e-4,1e-3')
parser.add_argument('--psd_cv_eta_grid', type=str, default='0.0,0.693147,1.609438',
                    help='comma-separated candidate LOG-precision starting points for --psd_eta')

# fast cross-validation of the anchor nodes themselves, run after the hyperparameter CV above
# (see psd_imputer.cross_validate_anchor_nodes)
parser.add_argument('--psd_anchor_cross_validate', action='store_true',
                    help='after psd_lr_nodes/psd_lr_param/psd_eta are chosen (by CV or by hand), '
                         'try --psd_cv_anchor_trials random m-row anchor-node subsamples (m = '
                         '--psd_m), score each with a Newton-only fit (no anchor/precision '
                         'movement) on a small validation subsample, and use the best one for '
                         'the real fit -- overrides --psd_anchor_init')
parser.add_argument('--psd_cv_anchor_trials', type=int, default=5,
                    help='number of candidate anchor-node subsamples to try for '
                         '--psd_anchor_cross_validate')
parser.add_argument('--psd_cv_anchor_newton_steps', type=int, default=10,
                    help='max Newton iterations per candidate for --psd_anchor_cross_validate')
parser.add_argument('--psd_cv_anchor_metric', type=str, default='rmse',
                    choices=['rmse', 'ed', 'ot'],
                    help='validation metric for --psd_anchor_cross_validate. ed/ot compare whole '
                         'rows, so the entries already missing before extra-hiding are imputed as '
                         'zero in both the imputed and true arrays first, so the score reflects '
                         'only recovery of the extra-hidden entries (see psd_imputer._cv_score)')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


def peak_rss_mb():
    """Process peak resident set size so far, in MB (ru_maxrss is KB on Linux, bytes on macOS)."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss / (1024 ** 2) if sys.platform == 'darwin' else maxrss / 1024



if __name__ == "__main__":

    OTLIM = 5000

    dataset = args.dataset

    dataset_loaded = scale(dataset_loader(dataset))

    if dataset_loaded.shape[0] > args.large_dataset_threshold:
        logging.info(f"dataset {dataset} has more than {args.large_dataset_threshold} observations, "
                     f"using {args.max_train_obs} for training and the rest as test set")

    METHODS = ["psd", "OT", "ice", "mean", "softimpute", "lin_rr", "mlp_rr"]

    psd_scores = {}
    ot_scores = {}
    ice_scores = {}
    mean_scores = {}
    softimpute_scores = {}
    lin_rr_scores = {}
    mlp_rr_scores = {}

    score_dicts = [psd_scores, ot_scores, ice_scores, mean_scores, softimpute_scores,
                   lin_rr_scores, mlp_rr_scores]

    for dic in score_dicts:
        for metric in ['MAE', 'RMSE', 'OT', 'OT_test', 'ED', 'runtime', 'memory']:
            dic[metric] = []
    psd_scores['avg_bounce_time'] = []
    psd_scores['MI_OT'] = []
    psd_scores['MI_ED'] = []

    p = args.p

    data = {"p": p, "M": [], "epsilon": [], "imp": {}, "params": vars(args)}

    for meth in METHODS:
        data["imp"][meth] = []

    batchsize = args.batchsize

    for n in range(args.nexp):

        if dataset_loaded.shape[0] > args.large_dataset_threshold:
            idx = np.random.default_rng(args.seed + n).permutation(dataset_loaded.shape[0])
            ground_truth = dataset_loaded[idx[:args.max_train_obs]]
            test_set = dataset_loaded[idx[args.max_train_obs:]]
        elif args.perc_test_set > 0:
            ground_truth, test_set = train_test_split(dataset_loaded, test_size=args.perc_test_set)
        else:
            ground_truth = dataset_loaded
            test_set = np.array([])
        X_true = torch.tensor(ground_truth)
        X_true_test = torch.tensor(test_set)
        n_train = ground_truth.shape[0]
        n_test = test_set.shape[0]
        n_total = n_train + n_test
        n_features = ground_truth.shape[1]
        test_split = n_test / n_total if n_total > 0 else 0.0

        logging.info(f"dataset: {dataset}\t"
                     f"features: {n_features}\t"
                     f"total obs: {n_total}\t"
                     f"train obs: {n_train}\t"
                     f"test obs: {n_test}\t"
                     f"test split: {test_split:.2%}\t"
                     f"train missingness (p): {p:.2%}")

        if args.psd_cross_validate:
            lr_nodes_grid = [float(x) for x in args.psd_cv_lr_nodes_grid.split(',')]
            lr_param_grid = [float(x) for x in args.psd_cv_lr_param_grid.split(',')]
            eta_grid = [float(x) for x in args.psd_cv_eta_grid.split(',')]
            logging.info(f"cross-validating psd_lr_nodes/psd_lr_param/psd_eta on a "
                         f"{min(args.psd_cv_n, ground_truth.shape[0])}-observation, "
                         f"{args.psd_cv_m}-anchor-node validation subsample "
                         f"({len(lr_nodes_grid) * len(lr_param_grid) * len(eta_grid)} combinations)...")
            best_lr_nodes, best_lr_param, best_eta, best_rmse, _ = cross_validate_hyperparams(
                ground_truth, args.p, lr_nodes_grid, lr_param_grid, eta_grid,
                m_cv=args.psd_cv_m, n_cv=args.psd_cv_n, nbr_bounce_cv=args.psd_cv_bounce,
                nbr_gradient_steps=args.psd_gradient_steps, nbr_newton_step_Q=args.psd_newton_step_Q,
                alpha=args.psd_alpha, lbd=args.psd_lbd, mu=args.psd_mu, seed=args.seed + n,
                verbose=True,
            )
            logging.info(f"cross-validation winner: psd_lr_nodes={best_lr_nodes:.1e}  "
                         f"psd_lr_param={best_lr_param:.1e}  psd_eta={best_eta:.4f}  "
                         f"(validation RMSE={best_rmse:.4f})")
            args.psd_lr_nodes = best_lr_nodes
            args.psd_lr_param = best_lr_param
            args.psd_eta = best_eta
            args.psd_eta_init_mode = 'fixed'

        cv_anchor_nodes = None
        if args.psd_anchor_cross_validate:
            logging.info(f"cross-validating psd anchor nodes: {args.psd_cv_anchor_trials} trials, "
                         f"m={args.psd_m}, up to {args.psd_cv_anchor_newton_steps} Newton "
                         f"iterations each, metric={args.psd_cv_anchor_metric}...")
            cv_anchor_nodes, best_anchor_score, _ = cross_validate_anchor_nodes(
                ground_truth, args.p, args.psd_m, args.psd_lr_nodes, args.psd_lr_param,
                args.psd_eta, n_trials=args.psd_cv_anchor_trials, n_cv=args.psd_cv_n,
                newton_steps=args.psd_cv_anchor_newton_steps,
                alpha=args.psd_alpha, lbd=args.psd_lbd, mu=args.psd_mu, seed=args.seed + n,
                verbose=True, cv_metric=args.psd_cv_anchor_metric,
            )
            logging.info(f"anchor-node cross-validation winner: "
                         f"{args.psd_cv_anchor_metric.upper()}={best_anchor_score:.4f}")

        ### Each entry from the second axis has a probability p of being NA

        if args.MAR:
            logging.info("MAR")
            mask = MAR_mask(X_true, p, args.p_obs).double()
        elif args.MNAR_log:
            logging.info("Logistic MNAR")
            mask = MNAR_mask_logistic(X_true, p, args.p_obs).double()
        elif args.MNAR_quant:
            logging.info("Quantile MNAR")
            mask = MNAR_mask_quantiles(X_true, p, args.q_mnar, 1-args.p_obs,
                                       cut='both', MCAR=False).double()
        else:  # MCAR
            mask = (torch.rand(ground_truth.shape) < p).double() 

        X_nas = X_true.clone()
        X_nas[mask.bool()] = np.nan

        M = mask.sum(1) > 0
        nimp = M.sum().item()

        data["M"].append(M.detach().cpu().numpy())

        data_nas = X_nas.cpu().numpy()

        logging.info("PSD Imputation")

        N = data_nas.shape[0]
        psd_lbd = args.psd_lbd / N
        psd_mu = args.psd_mu / N

        t_start = time.perf_counter()
        mem_start = peak_rss_mb()

        psd_imp_np, psd_history, psd_multiple_imp = psd_impute(
            data_nas, mask.cpu().numpy(),
            m=args.psd_m, eta_init=args.psd_eta, alpha=args.psd_alpha, lbd=psd_lbd,
            mu=psd_mu, l_rate_nodes=args.psd_lr_nodes, l_rate_param=args.psd_lr_param,
            nbr_bounce=args.psd_bounce, nbr_gradient_steps=args.psd_gradient_steps,
            nbr_newton_step_Q=args.psd_newton_step_Q, seed=args.seed + n, verbose=args.verbose,
            verbose_newton=args.verbose_newton, anchor_init=args.psd_anchor_init,
            eta_init_mode=args.psd_eta_init_mode, fixed_anchor_nodes=cv_anchor_nodes,
            dataset_true=ground_truth if args.psd_verbose_metrics else None,
            n_imputations=args.psd_n_imputations if n_test > 0 else 0,
        )

        psd_runtime = time.perf_counter() - t_start
        psd_scores['runtime'].append(psd_runtime)
        psd_scores['memory'].append(peak_rss_mb() - mem_start)
        psd_scores['avg_bounce_time'].append(psd_runtime / args.psd_bounce)

        psd_imp = torch.tensor(psd_imp_np)

        data["imp"]["psd"].append(psd_imp[mask.bool()].detach().cpu().numpy())

        psd_scores['MAE'].append(MAE(psd_imp, X_true, mask).item())
        psd_scores['RMSE'].append(RMSE(psd_imp, X_true, mask).item())
        psd_scores['ED'].append(energy_distance(psd_imp.detach().cpu().numpy(), X_true.detach().cpu().numpy()))
        if nimp < OTLIM:
            dists = ((psd_imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            psd_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                            np.ones(nimp) / nimp,
                                            dists.cpu().numpy()))
            if n_test > 0:
                dists_test = ((psd_imp[:, None, :] - X_true_test) ** 2).sum(2) / 2.
                psd_scores['OT_test'].append(ot.emd2(np.ones(n_train) / n_train,
                                                     np.ones(n_test) / n_test,
                                                     dists_test.cpu().numpy()))
            logging.info(f'psd imputation:\t '
                         f'MAE: {psd_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {psd_scores["RMSE"][-1]:.4f}\t'
                         f'OT: {psd_scores["OT"][-1]:.4f}\t'
                         f'ED: {psd_scores["ED"][-1]:.4f}\t'
                         f'Time: {psd_runtime:.4f}s\t'
                         f'Time/bounce: {psd_scores["avg_bounce_time"][-1]:.4f}s\t'
                         f'Mem: {psd_scores["memory"][-1]:.2f}MB')
        else:
            logging.info(f'psd imputation:\t '
                         f'MAE: {psd_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {psd_scores["RMSE"][-1]:.4f}\t'
                         f'ED: {psd_scores["ED"][-1]:.4f}\t'
                         f'Time: {psd_runtime:.4f}s\t'
                         f'Time/bounce: {psd_scores["avg_bounce_time"][-1]:.4f}s\t'
                         f'Mem: {psd_scores["memory"][-1]:.2f}MB')

        if psd_multiple_imp is not None:
            # stack the n_imputations independent completions into one (n_imputations * n_train, d)
            # point cloud and compare it against the held-out test set
            stacked_imp = psd_multiple_imp.reshape(-1, ground_truth.shape[1])
            n_stacked = stacked_imp.shape[0]

            psd_scores['MI_ED'].append(energy_distance(stacked_imp, test_set))
            if n_stacked < OTLIM:
                stacked_imp_t = torch.tensor(stacked_imp)
                dists_mi = ((stacked_imp_t[:, None, :] - X_true_test) ** 2).sum(2) / 2.
                psd_scores['MI_OT'].append(ot.emd2(np.ones(n_stacked) / n_stacked,
                                                    np.ones(n_test) / n_test,
                                                    dists_mi.cpu().numpy()))
                logging.info(f'psd multiple imputation:\t '
                             f'MI_OT: {psd_scores["MI_OT"][-1]:.4f}\t'
                             f'MI_ED: {psd_scores["MI_ED"][-1]:.4f}')
            else:
                logging.info(f'psd multiple imputation:\t '
                             f'MI_ED: {psd_scores["MI_ED"][-1]:.4f}')

        t_start = time.perf_counter()
        mem_start = peak_rss_mb()

        mean_imp = (1 - mask) * X_true + mask * nanmean(X_nas)

        mean_scores['runtime'].append(time.perf_counter() - t_start)
        mean_scores['memory'].append(peak_rss_mb() - mem_start)

        data["imp"]["mean"].append(mean_imp.cpu().numpy())

        mean_scores['MAE'].append(MAE(mean_imp, X_true, mask).cpu().numpy())
        mean_scores['RMSE'].append(RMSE(mean_imp, X_true, mask).cpu().numpy())
        mean_scores['ED'].append(energy_distance(mean_imp.detach().cpu().numpy(), X_true.detach().cpu().numpy()))
        if nimp < OTLIM:
            dists = ((mean_imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            mean_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                             np.ones(nimp) / nimp,
                                             dists.cpu().numpy()))
            if n_test > 0:
                dists_test = ((mean_imp[:, None, :] - X_true_test) ** 2).sum(2) / 2.
                mean_scores['OT_test'].append(ot.emd2(np.ones(n_train) / n_train,
                                                      np.ones(n_test) / n_test,
                                                      dists_test.cpu().numpy()))

            logging.info(f'mean imputation:\t '
                         f'MAE: {mean_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {mean_scores["RMSE"][-1]:.4f}\t'
                         f'OT: {mean_scores["OT"][-1]:.4f}\t'
                         f'ED: {mean_scores["ED"][-1]:.4f}\t'
                         f'Time: {mean_scores["runtime"][-1]:.4f}s\t'
                         f'Mem: {mean_scores["memory"][-1]:.2f}MB')
        else:
            logging.info(f'mean imputation:\t '
                         f'MAE: {mean_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {mean_scores["RMSE"][-1]:.4f}\t'
                         f'ED: {mean_scores["ED"][-1]:.4f}\t'
                         f'Time: {mean_scores["runtime"][-1]:.4f}s\t'
                         f'Mem: {mean_scores["memory"][-1]:.2f}MB')

        t_start = time.perf_counter()
        mem_start = peak_rss_mb()

        ice_mean = IterativeImputer(random_state=0, max_iter=50)
        ice_mean.fit(X_nas.cpu().numpy())
        ice_imp = torch.tensor(ice_mean.transform(data_nas))

        ice_scores['runtime'].append(time.perf_counter() - t_start)
        ice_scores['memory'].append(peak_rss_mb() - mem_start)

        data["imp"]["ice"].append(ice_imp.cpu().numpy())

        ice_scores['MAE'].append(MAE(ice_imp, X_true, mask).cpu().numpy())
        ice_scores['RMSE'].append(RMSE(ice_imp, X_true, mask).cpu().numpy())
        ice_scores['ED'].append(energy_distance(ice_imp.detach().cpu().numpy(), X_true.detach().cpu().numpy()))
        if nimp < OTLIM:
            dists = ((ice_imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            ice_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                            np.ones(nimp) / nimp,
                                            dists.cpu().numpy()))
            if n_test > 0:
                dists_test = ((ice_imp[:, None, :] - X_true_test) ** 2).sum(2) / 2.
                ice_scores['OT_test'].append(ot.emd2(np.ones(n_train) / n_train,
                                                     np.ones(n_test) / n_test,
                                                     dists_test.cpu().numpy()))
            logging.info(f'ice imputation:\t'
                         f'MAE: {ice_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {ice_scores["RMSE"][-1]:.4f}\t'
                         f'OT: {ice_scores["OT"][-1]:.4f}\t'
                         f'ED: {ice_scores["ED"][-1]:.4f}\t'
                         f'Time: {ice_scores["runtime"][-1]:.4f}s\t'
                         f'Mem: {ice_scores["memory"][-1]:.2f}MB')
        else:
            logging.info(f'ice imputation:\t'
                         f'MAE: {ice_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {ice_scores["RMSE"][-1]:.4f}\t'
                         f'ED: {ice_scores["ED"][-1]:.4f}\t'
                         f'Time: {ice_scores["runtime"][-1]:.4f}s\t'
                         f'Mem: {ice_scores["memory"][-1]:.2f}MB')

        t_start = time.perf_counter()
        mem_start = peak_rss_mb()

        cv_error, grid_lambda = cv_softimpute(data_nas, grid_len=15)
        lbda = grid_lambda[np.argmin(cv_error)]

        softimp = softimpute((data_nas), lbda)[1]

        softimpute_scores['runtime'].append(time.perf_counter() - t_start)
        softimpute_scores['memory'].append(peak_rss_mb() - mem_start)

        data["imp"]["softimpute"].append(softimp)
        softimp = torch.tensor(softimp)
        softimpute_scores['MAE'].append(
            MAE(softimp, X_true, mask).cpu().numpy())
        softimpute_scores['RMSE'].append(
            RMSE(softimp, X_true, mask).cpu().numpy())
        softimpute_scores['ED'].append(
            energy_distance(softimp.detach().cpu().numpy(), X_true.detach().cpu().numpy()))
        if nimp < OTLIM:
            dists = ((softimp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            softimpute_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                                   np.ones(nimp) / nimp,
                                                   dists.cpu().numpy()))
            if n_test > 0:
                dists_test = ((softimp[:, None, :] - X_true_test) ** 2).sum(2) / 2.
                softimpute_scores['OT_test'].append(ot.emd2(np.ones(n_train) / n_train,
                                                             np.ones(n_test) / n_test,
                                                             dists_test.cpu().numpy()))
            logging.info(f'softimpute:\t'
                         f'MAE: {softimpute_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {softimpute_scores["RMSE"][-1]:.4f}\t'
                         f'OT: {softimpute_scores["OT"][-1]:.4f}\t'
                         f'ED: {softimpute_scores["ED"][-1]:.4f}\t'
                         f'Time: {softimpute_scores["runtime"][-1]:.4f}s\t'
                         f'Mem: {softimpute_scores["memory"][-1]:.2f}MB')
        else:
            logging.info(f'softimpute:\t'
                         f'MAE: {softimpute_scores["MAE"][-1]:.4f}\t '
                         f'RMSE: {softimpute_scores["RMSE"][-1]:.4f}\t'
                         f'ED: {softimpute_scores["ED"][-1]:.4f}\t'
                         f'Time: {softimpute_scores["runtime"][-1]:.4f}s\t'
                         f'Mem: {softimpute_scores["memory"][-1]:.2f}MB')

        ### Automatic epsilon

        if args.quantile is not None:
            epsilon = pick_epsilon(X_nas, args.quantile, args.quantile_multiplier)
            logging.info(f"epsilon: {epsilon:.4f} "
                         f"({100 * args.quantile}th percentile times "
                         f"{args.quantile_multiplier})")

        else:
            epsilon = args.epsilon
            logging.info(f"epsilon: {epsilon:.4f} (fixed)")

        data["epsilon"].append(epsilon)

        logging.info("Sinkhorn Imputation")

        t_start = time.perf_counter()
        mem_start = peak_rss_mb()

        sk_imputer = OTimputer(eps=epsilon, niter=args.sinkhorn_niter, batchsize=batchsize, lr=args.lr)

        sk_imp, _, _ = sk_imputer.fit_transform(X_nas.clone(), report_interval=args.report_interval,
                                     verbose=True, X_true=X_true)
        sk_imp = sk_imp.detach()

        ot_scores['runtime'].append(time.perf_counter() - t_start)
        ot_scores['memory'].append(peak_rss_mb() - mem_start)

        ot_scores['MAE'].append(MAE(sk_imp, X_true, mask).item())
        ot_scores['RMSE'].append(RMSE(sk_imp, X_true, mask).item())
        ot_scores['ED'].append(energy_distance(sk_imp.detach().cpu().numpy(), X_true.detach().cpu().numpy()))
        if nimp < OTLIM:
            dists = ((sk_imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            ot_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                           np.ones(nimp) / nimp, \
                                           dists.cpu().numpy()))

            logging.info(f"Sinkhorn imputation:\t "
                         f"MAE: {ot_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {ot_scores['RMSE'][-1]:.4f}\t"
                         f"OT: {ot_scores['OT'][-1]:.4f}\t"
                         f"ED: {ot_scores['ED'][-1]:.4f}\t"
                         f"Time: {ot_scores['runtime'][-1]:.4f}s\t"
                         f"Mem: {ot_scores['memory'][-1]:.2f}MB")
        else:
            logging.info(f"Sinkhorn imputation:\t "
                         f"MAE: {ot_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {ot_scores['RMSE'][-1]:.4f}\t"
                         f"ED: {ot_scores['ED'][-1]:.4f}\t"
                         f"Time: {ot_scores['runtime'][-1]:.4f}s\t"
                         f"Mem: {ot_scores['memory'][-1]:.2f}MB")

        data["imp"]["OT"].append(sk_imp[mask.bool()].detach().cpu().numpy())

        logging.info("Linear Round Robin Imputation")

        t_start = time.perf_counter()
        mem_start = peak_rss_mb()

        n_, d = X_true.shape

        models = {}

        for i in range(d):
            ## predict the ith variable using d-1 others
            models[i] = torch.nn.Linear(d - 1, 1).to(device)

        linear_rr_imputer = RRimputer(models, max_iter=args.max_iter,
                                      niter=args.rr_niter,
                                      n_pairs=args.n_pairs,
                                      batchsize=batchsize,
                                      lr=args.lr,
                                      weight_decay=args.decay,
                                      order="random",
                                      eps=epsilon,
                                      opt=torch.optim.Adam,
                                      scaling=args.scaling)

        lin_imp, _, _ = linear_rr_imputer.fit_transform(X_nas.clone(), report_interval=1, verbose=True, X_true=X_true)
        lin_imp = lin_imp.detach()

        lin_rr_scores['runtime'].append(time.perf_counter() - t_start)
        lin_rr_scores['memory'].append(peak_rss_mb() - mem_start)

        lin_rr_scores['MAE'].append(MAE(lin_imp, X_true, mask).item())
        lin_rr_scores['RMSE'].append(RMSE(lin_imp, X_true, mask).item())
        lin_rr_scores['ED'].append(energy_distance(lin_imp.detach().cpu().numpy(), X_true.detach().cpu().numpy()))
        if nimp < OTLIM:
            dists = ((lin_imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            lin_rr_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                               np.ones(nimp) / nimp,
                                               dists.cpu().numpy()))
            logging.info(f"Linear RR imputation:\t"
                         f"MAE: {lin_rr_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {lin_rr_scores['RMSE'][-1]:.4f}\t"
                         f"OT: {lin_rr_scores['OT'][-1]:.4f}\t"
                         f"ED: {lin_rr_scores['ED'][-1]:.4f}\t"
                         f"Time: {lin_rr_scores['runtime'][-1]:.4f}s\t"
                         f"Mem: {lin_rr_scores['memory'][-1]:.2f}MB")
        else:
            logging.info(f"Linear RR imputation:\t"
                         f"MAE: {lin_rr_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {lin_rr_scores['RMSE'][-1]:.4f}\t"
                         f"ED: {lin_rr_scores['ED'][-1]:.4f}\t"
                         f"Time: {lin_rr_scores['runtime'][-1]:.4f}s\t"
                         f"Mem: {lin_rr_scores['memory'][-1]:.2f}MB")

        data["imp"]["lin_rr"].append(lin_imp[mask.bool()].detach().cpu().numpy())

        logging.info("MLP Round Robin Imputation")

        t_start = time.perf_counter()
        mem_start = peak_rss_mb()

        n_, d = X_true.shape
        d_ = d - 1

        models = {}

        for i in range(d):
            ## predict the ith variable using d-1 others
            models[i] = nn.Sequential(nn.Linear(d_, 2 * d_),
                                      nn.ReLU(),
                                      nn.Linear(2 * d_, d_),
                                      nn.ReLU(),
                                      nn.Linear(d_, 1)
                                      ).to(device)

        mlp_rr_imputer = RRimputer(models,
                                   max_iter=args.max_iter,
                                   niter=args.rr_niter,
                                   n_pairs=args.n_pairs,
                                   batchsize=batchsize,
                                   lr=args.lr,
                                   weight_decay=args.decay,
                                   order="random",
                                   eps=epsilon,
                                   opt=torch.optim.Adam,
                                   scaling=args.scaling)

        mlp_imp, _, _ = mlp_rr_imputer.fit_transform(X_nas.clone(), report_interval=1, verbose=True, X_true=X_true)
        mlp_imp = mlp_imp.detach()

        mlp_rr_scores['runtime'].append(time.perf_counter() - t_start)
        mlp_rr_scores['memory'].append(peak_rss_mb() - mem_start)

        mlp_rr_scores['MAE'].append(MAE(mlp_imp, X_true, mask).item())
        mlp_rr_scores['RMSE'].append(RMSE(mlp_imp, X_true, mask).item())
        mlp_rr_scores['ED'].append(energy_distance(mlp_imp.detach().cpu().numpy(), X_true.detach().cpu().numpy()))
        if nimp < OTLIM:
            dists = ((mlp_imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            mlp_rr_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                               np.ones(nimp) / nimp,
                                               dists.cpu().numpy()))
            logging.info(f"MLP RR imputation:\t"
                         f"MAE: {mlp_rr_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {mlp_rr_scores['RMSE'][-1]:.4f}\t"
                         f"OT: {mlp_rr_scores['OT'][-1]:.4f}\t"
                         f"ED: {mlp_rr_scores['ED'][-1]:.4f}\t"
                         f"Time: {mlp_rr_scores['runtime'][-1]:.4f}s\t"
                         f"Mem: {mlp_rr_scores['memory'][-1]:.2f}MB")
        else:
            logging.info(f"MLP RR imputation:\t"
                         f"MAE: {mlp_rr_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {mlp_rr_scores['RMSE'][-1]:.4f}\t"
                         f"ED: {mlp_rr_scores['ED'][-1]:.4f}\t"
                         f"Time: {mlp_rr_scores['runtime'][-1]:.4f}s\t"
                         f"Mem: {mlp_rr_scores['memory'][-1]:.2f}MB")

        data["imp"]["mlp_rr"].append(mlp_imp[mask.bool()].detach().cpu().numpy())

    scores = {}
    scores['psd'] = psd_scores
    scores['OT'] = ot_scores
    scores['ice'] = ice_scores
    scores['mean'] = mean_scores
    scores['softimpute'] = softimpute_scores
    scores['lin_rr'] = lin_rr_scores
    scores['mlp_rr'] = mlp_rr_scores

    mean_sd = {}
    for method, method_scores in scores.items():
        method_mean_sd = {}
        for metric, values in method_scores.items():
            if len(values) > 0:
                method_mean_sd[metric] = {'mean': np.mean(values), 'std': np.std(values)}
        mean_sd[method] = method_mean_sd
        logging.info(f'{method} averaged over {args.nexp} run(s):')
        for metric, stats in method_mean_sd.items():
            logging.info(f'  {metric}: mean {stats["mean"]:.4f}  std {stats["std"]:.4f}')

    if args.out_path is None:
        score_file = "_".join([dataset, "scores.pkl"])
    else:
        score_file = args.out_path

    pkl.dump(scores, open(os.path.join(args.out_dir, score_file), 'wb'))

    mean_std_file = "_".join([dataset, "mean_std.pkl"])
    pkl.dump(mean_sd, open(os.path.join(args.out_dir, mean_std_file), 'wb'))

    if args.out_data is None:
        data_file = "_".join([dataset, "data.pkl"])
    else:
        data_file = args.out_data

    pkl.dump(data, open(os.path.join(args.out_dir, data_file), 'wb'))
