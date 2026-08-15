#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn

from geomloss import SamplesLoss

import ot

import os
import pickle as pkl
import copy
import time

from sklearn.preprocessing import scale, quantile_transform
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


from MissingDataOT_master.utils import *
from MissingDataOT_master.softimpute import softimpute, cv_softimpute
from MissingDataOT_master.data_loaders import dataset_loader
from MissingDataOT_master.imputers import OTimputer, RRimputer
from MissingDataOT_master.psd_cv import cv_for_psd_model

import jax
import argparse
import logging
import os
import sys


# Adapted from original code by Boris Muzellec
# Source: https://github.com/BorisMuzellec/MissingDataOT


sys.path.append(os.path.abspath("/"))

from backbone.psd import energy_distance, multiple_imputation_psd, find_sub_matrix_minimal
from backbone.bouncing_adam import bouncing_function_real_data
from backbone.imputation_psd import imputation_by_psd_model_batch

logging.getLogger('jax').setLevel(logging.WARNING)
# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Print the path
print(f"Current file path: {current_file_path}")

# If you want just the directory of the file
current_directory = os.path.dirname(current_file_path)
print(f"Current directory: {current_directory}")


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,
                    help='seed for reproducibility')
parser.add_argument('--out_path', type=str, default=None,
                    help='filename for the results')
parser.add_argument('--out_data', type=str, default=None,
                    help='filename for the data')
parser.add_argument('--out_dir', type=str, default='exps',
                    help='directory name for results')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate (sinkhorn)')
parser.add_argument('--decay', type=float, default=1e-5,
                    help='weight decay (round robin)')
parser.add_argument('--scaling', type=float, default=.9,
                    help='sinkhorn scaling parameter (speed/precision tradeoff)')
parser.add_argument('-b', '--batchsize', type=int, default=128,
                    help='batchsize(s) for the experiments')
parser.add_argument('--niter', type=int, default=3,
                    help='number of GD iterations')
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

parser.add_argument('--nexp', type=int, default=2,
                    help='number of experiences per parameter setting')

parser.add_argument('-mi_nbr', '--mult_imp_nbr', type=int, default=5,
                    help='number of multiple imputation')

parser.add_argument('--dataset', type=str, default="iris",
                    help='dataset on which to run the experiments')

parser.add_argument('--constants_log_det', default="0.5", help='constant log det cross val psd model')
parser.add_argument('--inv_var', default="1.5, 2", help='inverse variance, parameter Gaussian cross val psd model')
parser.add_argument('--lim_nodes', type=int, default=50, help='set a max nbr of nodes that our model can bear')
parser.add_argument('--nbr_bounce', type=int, default=10, help='how many times bounce between optimizing the matrix and the nodes')
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

parser.add_argument('--perc_test_set', type=float, default=0, help='percentage of dataset to be considered test set')

parser.add_argument('--prior', default=None, help='define the prior')
parser.add_argument('--p', type=float, default=0.3, help='Proportion of imps')
parser.add_argument('--MAR', action='store_true')
parser.add_argument('--p_obs', type=float, default=0.3,
                    help='Proportion of variables that are fully observed (MAR & MNAR model)')
parser.add_argument('--MNAR_log', action='store_true')
parser.add_argument('--MNAR_quant', action='store_true')
parser.add_argument('--q_mnar', type=float, default=0.75,
                    help='quantile that will have imps (MNAR quantiles model)')
parser.add_argument('--max_iter_nwt_cv', type=int, default=20, help='max iter newton method for cv psd model')

parser.add_argument('--percentage_out', type=float, default=0.1,
                    help='percentage of ground_truth that we see for sure. This will be the nodes of the psd model')

parser.add_argument('--verbose', action='store_true')
parser.add_argument('--report_interval', type=int, default=500)
parser.add_argument('--verbose_psd', default=False, type=bool)


args = parser.parse_args()
grid_log_det = [float(x) for x in args.constants_log_det.split(",")]#ast.literal_eval(args.constants_log_det)
grid_inv_var = [float(x) for x in args.inv_var.split(",")]
print("const log_det ", grid_log_det, grid_log_det[0])
print("inv var ", grid_inv_var)
np.random.seed(args.seed)  # For CPU
torch.manual_seed(args.seed)  # For CPU
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

if __name__ == "__main__":
    # Now 'data' contains the deserialized object
    OTLIM = 5000

    dataset = args.dataset
    print("dataset:\n", dataset)
    dataset_loaded = scale(dataset_loader(dataset))
    print("size gt1 ", dataset_loaded.size)

    METHODS = ["OT", "ice", "mean", "softimpute", "psd", "psd_mi", "lin_rr", "mlp_rr"]

    ot_scores = {}
    ice_scores = {}
    mean_scores = {}
    softimpute_scores = {}
    psd_scores = {}
    results_psd_dict = {}
    results_psd1_dict = {}
    rmse_psd_bouncing = []


    score_dicts = [ot_scores, ice_scores, mean_scores, softimpute_scores, psd_scores]

    for dic in score_dicts:
        for metric in ['MAE', 'RMSE', 'OT', 'OT_test', 'ED', 'time', 'entropy', 'MI_OT_min', 'MI_OT_con', 'MI_OT_ideal', 'MI_RMSE_ideal', 'MI_ED_ideal', 'MI_ED_con']:
            dic[metric] = []

    p = args.p  # probability to hide a component
    perc_test_set = args.perc_test_set
    print("prob to hide a component: ", p)
    print("percentage test set: ", perc_test_set)
    data = {"p": p, "ground_truth": dataset_loaded, "mask": [], "M": [],
            "epsilon": [], "imp": {}, "train_set": [], "test_set": [], "params": vars(args)}
    for meth in METHODS:
        data["imp"][meth] = []
    batchsize = args.batchsize
    print("args.nexp: ", args.nexp)

    print("dataset considered: ", args.dataset, "shape: ", dataset_loaded.shape)
    for n in range(args.nexp):
        ### Each entry from the second axis has a probability p of being NA
        print("--------------- nbr iteration: ", n)
        print("dataset:\n", dataset)
        print("nbr bounce ", args.nbr_bounce)
        print("percentage missing ", args.p)
        print("l rate nodes ", args.l_rate_nodes)
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
        else:
            ## here we add the splitting
            percentage_out = args.percentage_out
            if perc_test_set > 0:
                ground_truth1, dataset_test_set = train_test_split(dataset_loaded, test_size=perc_test_set)
            else:
                ground_truth1 = dataset_loaded
                dataset_test_set = np.array([])
            n_train = ground_truth1.shape[0]
            n_test = dataset_test_set.shape[0]
            print("n_train ", n_train)
            print("n_test ", n_test)
            data["train_set"].append(ground_truth1)
            data["test_set"].append(dataset_test_set)
            print("percentage_out: ", percentage_out)
            nbr_obs = ground_truth1.shape[0]
            print("number of data: ", nbr_obs)
            lim_nodes = args.lim_nodes
            if nbr_obs * percentage_out > lim_nodes:  # lim is the max number of nodes that we can bear
                percentage_out = lim_nodes / nbr_obs
                print("new percentage_out: ", percentage_out)
            X_in, X_out = train_test_split(ground_truth1, test_size=percentage_out)
            out_size = X_out.shape[0]
            ground_truth = np.vstack((X_out, X_in))  # ground_truth = permutation(ground_truth1)
            X_true = torch.tensor(ground_truth)
            X_true_test = torch.tensor(dataset_test_set)
            mask_numpy = np.vstack((np.ones(X_out.shape) * p, np.random.rand(*X_in.shape))) < p
            print("X_in ", type(X_in), X_in.shape, "X_out ", X_out.shape, type(X_out), "mask ", type(mask_numpy))
            print("check mask of X_out: ", np.sum(mask_numpy[0:X_out.shape[0], 0:X_out.shape[1]]))
            np.testing.assert_allclose(np.sum(mask_numpy[0:X_out.shape[0], 0:X_out.shape[1]]), 0)
            print("check mask of X_out+1: ", np.sum(mask_numpy[0:X_out.shape[0] + 1, 0:X_out.shape[1]+1]))
            mask = torch.from_numpy(mask_numpy).double()  # m_ij = 1 iff component is missing

        X_nas = X_true.clone()
        X_nas[mask.bool()] = np.nan  # True=component is missing, False=component is not missing
        M = mask.sum(1) > 0  # M[i] == True iff component i has at least one missing component
        nimp = M.sum().item()

        data["mask"].append(mask.detach().cpu().numpy())
        data["M"].append(M.detach().cpu().numpy())
        #print("M : ", M)
        print("sum mask along 1st axis ", np.sum(mask_numpy, axis=1))

        # ice imputation
        start_time_ice = time.time()
        ice_mean = IterativeImputer(random_state=0, max_iter=50)
        data_nas = X_nas.cpu().numpy()
        ice_mean.fit(X_nas.cpu().numpy())
        ice_imp = torch.tensor(ice_mean.transform(data_nas))
        end_time_ice = time.time()
        elapsed_time_ice = end_time_ice - start_time_ice

        data["imp"]["ice"].append(ice_imp.cpu().numpy())
        ice_scores['MAE'].append(MAE(ice_imp, X_true, mask).cpu().numpy())
        ice_scores['RMSE'].append(RMSE(ice_imp, X_true, mask).cpu().numpy())
        ice_scores['ED'].append(energy_distance(ice_imp.numpy(), X_true.numpy()))
        ice_scores['time'].append(elapsed_time_ice)
        # ice
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
                         f'ED: {ice_scores["ED"][-1]:.4f}')
        else:
            logging.info(f'ice imputation:\t'
                         f'MAE: {ice_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {ice_scores["RMSE"][-1]:.4f}')

        # mean imputation
        start_time_mean = time.time()
        mean_imp = (1 - mask) * X_true + mask * nanmean(X_nas)
        end_time_mean = time.time()
        elapsed_time_mean = end_time_mean - start_time_mean

        data["imp"]["mean"].append(mean_imp.cpu().numpy())
        mean_scores['MAE'].append(MAE(mean_imp, X_true, mask).cpu().numpy())
        mean_scores['RMSE'].append(RMSE(mean_imp, X_true, mask).cpu().numpy())
        mean_scores['ED'].append(energy_distance(mean_imp.numpy(), X_true.numpy()))
        mean_scores['time'].append(elapsed_time_mean)

        # mean
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
                         f'ED: {mean_scores["ED"][-1]:.4f}')
        else:
            logging.info(f'mean imputation:\t '
                         f'MAE: {mean_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {mean_scores["RMSE"][-1]:.4f}')

        start_time_softimpute = time.time()
        cv_error, grid_lambda = cv_softimpute(data_nas, grid_len=15)
        lbda = grid_lambda[np.argmin(cv_error)]

        softimp = softimpute((data_nas), lbda)[1]
        end_time_softimpute = time.time()
        elapsed_time_softimpute = end_time_softimpute - start_time_softimpute

        data["imp"]["softimpute"].append(softimp)
        softimp = torch.tensor(softimp)
        softimpute_scores['MAE'].append(MAE(softimp, X_true, mask).cpu().numpy())
        softimpute_scores['RMSE'].append(RMSE(softimp, X_true, mask).cpu().numpy())
        softimpute_scores['ED'].append(energy_distance(softimp.numpy(), X_true.numpy()))
        softimpute_scores['time'].append(elapsed_time_softimpute)
        # softimpute
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
                         f'ED: {softimpute_scores["ED"][-1]:.4f}')
        else:
            logging.info(f'softimpute:\t'
                         f'MAE: {softimpute_scores["MAE"][-1]:.4f}\t '
                         f'RMSE: {softimpute_scores["RMSE"][-1]:.4f}')

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

        start_times_ot = time.time()
        sk_imputer = OTimputer(eps=epsilon, niter=args.niter, batchsize=batchsize, lr=args.lr)

        sk_imp, _, _ = sk_imputer.fit_transform(X_nas.clone(), report_interval=args.report_interval,
                                     verbose=True, X_true=X_true)
        sk_imp = sk_imp.detach()
        end_times_ot = time.time()
        elapsed_time_ot = end_times_ot - start_times_ot
        ot_scores['MAE'].append(MAE(sk_imp, X_true, mask).item())
        ot_scores['RMSE'].append(RMSE(sk_imp, X_true, mask).item())
        ot_scores['ED'].append(energy_distance(sk_imp.numpy(), X_true.numpy()))
        ot_scores['time'].append(elapsed_time_ot)
        # ot
        if nimp < OTLIM:
            dists = ((sk_imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            ot_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                           np.ones(nimp) / nimp,
                                           dists.cpu().numpy()))
            if n_test > 0:
                dists_test = ((sk_imp[:, None, :] - X_true_test) ** 2).sum(2) / 2.
                ot_scores['OT_test'].append(ot.emd2(np.ones(n_train) / n_train,
                                                     np.ones(n_test) / n_test,
                                                     dists_test.cpu().numpy()))
            logging.info(f"Sinkhorn imputation:\t "
                         f"MAE: {ot_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {ot_scores['RMSE'][-1]:.4f}\t"
                         f"OT: {ot_scores['OT'][-1]:.4f}\t"
                         f'ED: {ot_scores["ED"][-1]:.4f}')
        else:
            logging.info(f"Sinkhorn imputation:\t "
                         f"MAE: {ot_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {ot_scores['RMSE'][-1]:.4f}")

        data["imp"]["OT"].append(sk_imp[mask.bool()].detach().cpu().numpy())

        #psd
        print("psd method")
        start_times_psd = time.time()
        nb_sub = out_size
        print("subsample nodes", nb_sub)
        max_iter_newt = args.max_iter_nwt_cv
        mask_psd = 1 - mask_numpy  # our algo work with the convention that masks_ij = 1 iff component is seen
        nb_nodes = nb_sub
        linear_term_inv = np.eye(nb_nodes) / nb_nodes if (args.prior is None) else args.prior
        actual_nodes = X_out
        nbr_bounce = args.nbr_bounce
        iter_newt = np.array([args.intermediate_nwt] * (args.nbr_bounce + 1))  # we must run the algo at least once, even without bouncing
        iter_newt[0], iter_newt[-1] = args.first_nwt, args.last_nwt
        lbd = {'constraint': args.lbd_constraint, 'kernel': args.lbd_kernel, 'kernel_after': args.lbd_kernel_after}
        nwt = {'tolerance': args.tolerance, 'alpha': args.alpha, 'beta': args.beta, 'iter_nwt': iter_newt, 'iter_nwt_cv': args.max_iter_nwt_cv}
        c_mu_optimal, inverse_variance_optimal, results_psd, cv_error_psd = cv_for_psd_model(
            ground_truth=ground_truth, nodes=actual_nodes, mask_psd=mask_psd, x_out=actual_nodes,
            linear_term_inverse=linear_term_inv, lbd=lbd, nwt=nwt,
            grid_log_det=grid_log_det, grid_inv_var=grid_inv_var
        )
        results_psd_dict[n] = cv_error_psd
        print("cv_error ", cv_error_psd)
        print("best result at round ", n)
        print("best c_mu ", c_mu_optimal)
        print("best variance ", inverse_variance_optimal)
        bounce_properties = {'nbr_bounce': nbr_bounce, 'l_rate_nodes': args.l_rate_nodes, 'l_rate_param': args.l_rate_param, 'hit_optimiz_algo': args.hit_opt_algo}
        vector_inv_var = np.array([inverse_variance_optimal] * ground_truth.shape[1])
        q_sol, nodes_final, eta_final = bouncing_function_real_data(
            gt=ground_truth, initial_nodes=actual_nodes, masks=mask_psd, linear_term_inverse=linear_term_inv,
            c_mu=c_mu_optimal, eta=vector_inv_var, lbd=lbd, nwt=nwt,
            bounce_properties=bounce_properties, name_optimizer='Adam')
        best_imputation_psd = imputation_by_psd_model_batch(q_sol, nodes_final, ground_truth, mask_psd, eta_final)
        print("my best rmse psd outside tensor ", RMSE(torch.tensor(best_imputation_psd), X_true, mask))
        print("my best rmse psd outside ", RMSE(best_imputation_psd, ground_truth, mask_numpy))

        end_times_psd = time.time()
        elapsed_time_psd = end_times_psd - start_times_psd
        data["imp"]["psd"].append(best_imputation_psd)
        psd_scores['MAE'].append(MAE(best_imputation_psd, ground_truth, mask_numpy))  # da modificare
        psd_scores['RMSE'].append(RMSE(best_imputation_psd, ground_truth, mask_numpy))
        psd_scores['ED'].append(energy_distance(best_imputation_psd, X_true.numpy()))
        psd_scores['time'].append(elapsed_time_psd)
        mult_imp_score = []
        nimp = np.sum(M.numpy())
        mult_imp_nbr = args.mult_imp_nbr
        print("inv var optimal ", inverse_variance_optimal)
        mult_imp, entropy_psd = multiple_imputation_psd(mult_imp_nbr, q_sol, nodes_final, ground_truth, mask_psd, eta_final)
        psd_scores['entropy'].append(entropy_psd)
        best_dataset_mi_fct = find_sub_matrix_minimal(gt=ground_truth, mi=mult_imp)

        rmse_ideal = RMSE(best_dataset_mi_fct, ground_truth, mask_numpy)
        dists_best_dataset_of_mi = ((best_dataset_mi_fct[M][:, None] - ground_truth[M]) ** 2).sum(2) / 2.
        ot_ideal = ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, dists_best_dataset_of_mi)
        ed_ideal = energy_distance(best_dataset_mi_fct, ground_truth)

        psd_scores['MI_OT_ideal'].append(ot_ideal)
        psd_scores['MI_RMSE_ideal'].append(rmse_ideal)
        psd_scores['MI_ED_ideal'].append(ed_ideal)

        data["imp"]["psd_mi"].append(mult_imp)
        for i in range(mult_imp_nbr):
            dists = ((mult_imp[i][M][:, None] - ground_truth[M]) ** 2).sum(2) / 2.
            mult_imp_score.append(ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, dists))  # .cpu().numpy()))
        dists_psd_imp_true = ((best_imputation_psd[M][:, None] - ground_truth[M]) ** 2).sum(2) / 2.
        mult_imp_conc = np.concatenate(mult_imp[:, M, :], axis=0)  # concatenate
        mult_imp_conc_full = np.concatenate(mult_imp, axis=0)  # concatenate all the datasets
        dists_psd_imp_conc = ((mult_imp_conc[:, None] - ground_truth[M]) ** 2).sum(2) / 2.
        ed_psd_concatenate_score = energy_distance(mult_imp_conc_full, X_true.numpy())
        ot_psd_concatenate_score = ot.emd2(np.ones(nimp * mult_imp_nbr) / (nimp * mult_imp_nbr),
                                           np.ones(nimp) / nimp, dists_psd_imp_conc)
        ot_psd_imp = ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, dists_psd_imp_true)
        psd_scores['MI_OT_min'].append(np.min(mult_imp_score))
        psd_scores['MI_OT_con'].append(ot_psd_concatenate_score)
        psd_scores['MI_ED_con'].append(ed_psd_concatenate_score)

        print("mult imp min score ", mult_imp_score)
        print("ot psd imp ", ot_psd_imp)
        print("ot psd imp_concatenate ", ot_psd_concatenate_score)
        if nimp < OTLIM:
            dists = ((torch.tensor(best_imputation_psd)[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            psd_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                            np.ones(nimp) / nimp,
                                            dists.cpu().numpy()))
            if n_test > 0:
                mi_conc = torch.tensor(np.concatenate(mult_imp, axis=0))
                dists_test_conc = ((mi_conc[:, None, :] - X_true_test) ** 2).sum(2) / 2.
                n_psd, n_test_psd = dists_test_conc.shape
                psd_scores['OT_test'].append(ot.emd2(np.ones(n_psd) / n_psd,
                                                     np.ones(n_test_psd) / n_test_psd,
                                                     dists_test_conc.cpu().numpy()))

            logging.info(f'psd imputation:\t'
                         f'MAE: {psd_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {psd_scores["RMSE"][-1]:.4f}\t'
                         f'OT: {psd_scores["OT"][-1]:.4f}')
        else:
            logging.info(f'ice imputation:\t'
                         f'MAE: {psd_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {psd_scores["RMSE"][-1]:.4f}')
        print("psd_scores ", psd_scores)
        del X_true  # clean memory for next assignment
        del ground_truth
    
    scores = {}
    scores['OT'] = ot_scores
    scores['ice'] = ice_scores
    scores['mean'] = mean_scores
    scores['softimpute'] = softimpute_scores
    scores['psd'] = psd_scores
    mean_sd = {}

    print("dataset: ", args.dataset)
    np.set_printoptions(precision=2, suppress=True)
    print("cv_error_psd ", cv_error_psd)
    print("results psd after bouncing ", rmse_psd_bouncing)
    print("results_psd from cv:")
    for nbr, values in results_psd_dict.items():
        print("ite: ", nbr, ": ")
        for iperp, cv_err in values.items():
            print("iperp ", iperp, ": ", cv_err)

    for nbr, values in results_psd1_dict.items():
        print("ite: ", nbr, ": ")
        for iperp, res in values.items():
            print("iperp ", iperp, ": ", res)

    for method, dictio_of_metrics in scores.items():
        print("considered key: -------------------", method)
        dictio_metric = {}
        for metric, values in dictio_of_metrics.items():
            dictio_metric[metric] = {'mean': np.mean(values), 'std': np.std(values)}
            print(metric, f": mean {dictio_metric[metric]['mean']:.4f}  std {dictio_metric[metric]['std']:.4f}")
        mean_sd[method] = dictio_metric

    os.makedirs(args.out_dir, exist_ok=True)
    print("args.out_dir ", args.out_dir)
    print("dimension dataset ", ground_truth1.shape)
    print("dimension nodes ", X_out.shape)
    print("dimension nodes to mask ", X_in.shape)
    for key in data['params'].keys():
        print(" ", key, ": ", data['params'][key])

    perc_missing = str(p).replace('.', '')
    percentage_out = str(args.percentage_out).replace('.', '')
    nbr_bounce_str = str(args.nbr_bounce)
    lim_nodes = str(args.lim_nodes)
    nbr_exps = str(args.nexp)
    if args.out_path is None:
        score_file = "_".join([dataset, "p_missing"+perc_missing, "perc_out"+percentage_out, "nbr_bounce"+nbr_bounce_str, "lim_nodes"+lim_nodes, "nexp"+nbr_exps, "scores.pkl"])
    else:
        score_file = args.out_path

    pkl.dump(scores, open(os.path.join(args.out_dir, score_file), 'wb'))
    
    if args.out_data is None:
        data_file = "_".join([dataset, "p_missing"+perc_missing, "perc_out"+percentage_out, "nbr_bounce"+nbr_bounce_str, "lim_nodes"+lim_nodes, "nexp"+nbr_exps, "data.pkl"])
        mean_std_file = "_".join([dataset, "p_missing"+perc_missing, "perc_out"+percentage_out, "nbr_bounce"+nbr_bounce_str, "lim_nodes"+lim_nodes, "nexp"+nbr_exps, "mean_std.pkl"])
        results_psd_file = "_".join([dataset, "p_missing"+perc_missing, "perc_out"+percentage_out, "nbr_bounce"+nbr_bounce_str, "lim_nodes"+lim_nodes, "nexp"+nbr_exps, "result_psd.pkl"])
    else:
        data_file = args.out_data
        mean_std_file = args.out_data
        results_psd_file = args.out_data

    pkl.dump(data, open(os.path.join(args.out_dir, data_file), 'wb'))
    pkl.dump(mean_sd, open(os.path.join(args.out_dir, mean_std_file), 'wb'))
    pkl.dump(results_psd, open(os.path.join(args.out_dir, results_psd_file), 'wb'))



