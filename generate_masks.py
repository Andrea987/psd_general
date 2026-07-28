import numpy as np
from backbone.psd import (kernel_vector_matrices, c_masked, Matrices_Gaussian_Kernel_Preconditioned,
                          change_of_variable_of_the_parameters)
from backbone.useful_functions_for_the_tests import test_pd


def generate_masks_2d(nbr_of_sample, p_missing):
    # nbr_of_sample is the number of masks
    # p_missing=[p11, p01, p10], where p11 is the probability of seeing both components,
    # p01 is the probability of seeing the right component, p10 is the probability of seeing the left component
    masks = np.zeros((nbr_of_sample, 2))
    # p_missing = [0.3, 0.3, 0.4]
    v = np.random.choice(a=3, size=nbr_of_sample, p=p_missing)
    masks[v == 0, :] = np.array([1, 1])
    masks[v == 1, :] = np.array([0, 1])
    masks[v == 2, :] = np.array([1, 0])
    return masks


def generate_observation_and_masks_graphical_model(nbr_of_sample, p_missing):
    # nbr_of_sample is total nbr observation
    # p_missing is probability of missing of one component
    # observe, res[:, 1] and res[:, 0] depends on res[_, 2]
    dim = 3
    res = np.zeros((nbr_of_sample, dim))
    #print("res ", res)
    res[:, 2] = np.random.uniform(low=-1, high=1, size=nbr_of_sample)
    res[:, 1] = res[:, 2] + np.random.normal(loc=0, scale=0.1, size=nbr_of_sample)
    res[:, 0] = res[:, 2] + np.random.normal(loc=0, scale=0.1, size=nbr_of_sample)

    masks = np.zeros((nbr_of_sample, 3))
    v = np.random.choice(a=3, size=nbr_of_sample, p=p_missing)
    masks[v == 0, :] = np.array([1, 0, 1])
    masks[v == 1, :] = np.array([0, 1, 1])
    masks[v == 2, :] = np.array([0, 0, 1])
    min_seen_components = np.min(res * masks, axis=0)
    max_seen_components = np.max(res * masks, axis=0)
    return res, masks, min_seen_components, max_seen_components


def generate_matrices_to_plug_into_loss(h, v, linear_term, change_of_variable, lbd_kernel_after):
    # h: constraint matrix Tr(QH) = 1
    # v: kernel matrices
    # linear_term:
    nb_nodes = h.shape[0]
    ll = np.linalg.cholesky(h)  # Lower Triangular matrix of the cholesky decomposition H = L@(L.T)
    if change_of_variable:
        l_inv = np.linalg.inv(ll)
        constraint_matrix = np.eye(nb_nodes)
    else:
        l_inv = np.eye(nb_nodes)
        constraint_matrix = h
    v_new, linear_term_after_change_of_variable = change_of_variable_of_the_parameters(v, linear_term, l_inv)
    w = np.sum(v_new * v_new, axis=(-1, -2), keepdims=False)
    w[w <= 0] = np.sum(w, axis=0) / w.shape[0]  # if for some reason some matrices are singular
    v_new = v_new + np.eye(nb_nodes) * np.sqrt(w[:, None, None]) * lbd_kernel_after
    v_new = v_new / np.sum(v_new * np.eye(nb_nodes), axis=(-1, -2), keepdims=True)
    return v_new, linear_term_after_change_of_variable, l_inv, constraint_matrix
