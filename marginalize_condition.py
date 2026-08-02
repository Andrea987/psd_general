"""
Marginalize and condition a Gaussian PSD model on a subset of its variables, reusing the
K_S / H_NS / C_2eta_NS building blocks from psd.py (same ones the loss function is built on).

Convention (shared with psd.py): `mask` is a (d,) boolean array, True on the dimensions being
marginalized out / conditioned away (missing, NS), False on the dimensions kept/observed (S).
"""

import numpy as np
from psd import K_S, H_NS, C_2eta_NS


def partial_evaluation(anchor_nodes, precision, Q, mask, x):
    """
    Partially evaluate the model at x_S, without integrating the NS dimensions out yet.
    This is the "base matrix" that both marginalize and condition build on: Q o K_S(x),
    with o the Hadamard product.

    :param anchor_nodes: (m, d) anchor/inducing points
    :param precision: (d,) per-dimension precision of the Gaussian kernel
    :param Q: (m, m) PSD weight matrix
    :param mask: (d,) boolean, True on the dimensions not being evaluated (NS)
    :param x: (n, d) points; only the entries on the observed (S) dimensions matter, the
        entries on the NS dimensions are ignored
    :return: (n, m, m) stack of Q o K_S(x)_i, one per point
    """
    x = np.atleast_2d(x)
    masks = np.broadcast_to(mask, x.shape)
    info = {'dataset': x, 'anchor_nodes': anchor_nodes, 'precision': precision, 'masks': masks}
    return Q * K_S(info)  # (n, m, m), Hadamard product broadcasting Q over the n points


def marginalize(anchor_nodes, precision, Q, mask, x):
    """
    Evaluate the marginal of the model at x_S, having integrated the NS (mask == True)
    dimensions out over their whole domain.

    :param anchor_nodes: (m, d) anchor/inducing points
    :param precision: (d,) per-dimension precision of the Gaussian kernel
    :param Q: (m, m) PSD weight matrix
    :param mask: (d,) boolean, True on the dimensions to marginalize out (NS)
    :param x: (n, d) points; only the entries on the observed (S) dimensions matter
    :return: (n,) marginal value of the model at each point, i.e.
        C_2eta_NS * Tr((Q o K_S(x)) o H_NS), with o the Hadamard product
    """
    x = np.atleast_2d(x)
    masks = np.broadcast_to(mask, x.shape)
    info = {'dataset': x, 'anchor_nodes': anchor_nodes, 'precision': precision, 'masks': masks}

    base = partial_evaluation(anchor_nodes, precision, Q, mask, x)  # (n, m, m)
    A = base * H_NS(info)  # (n, m, m), Hadamard product
    trace_A = np.einsum('ikj->i', A)  # (n,): Tr(A_i)
    return C_2eta_NS(info) * trace_A


def condition(anchor_nodes, precision, Q, mask, x):
    """
    Condition the model on the observed (S, mask == False) dimensions taking value x there.

    :param anchor_nodes: (m, d) anchor/inducing points
    :param precision: (d,) per-dimension precision of the Gaussian kernel
    :param Q: (m, m) PSD weight matrix
    :param mask: (d,) boolean, True on the dimensions being conditioned on/removed (NS)
    :param x: (d,) or (1, d) point to condition on; only the entries on the observed (S)
        dimensions matter
    :return: (anchor_nodes_NS, precision_NS, Q_cond), the PSD model over the remaining (NS)
        dimensions such that phi_NS(x_NS)^T Q_cond phi_NS(x_NS) integrates to 1 over x_NS:
        Q_cond = (Q o K_S(x)) / eval, eval = marginalize(...)(x). Ready to use directly with
        sampling.sample_bisection(anchor_nodes_NS, precision_NS, Q_cond, N).
    """
    mask = np.asarray(mask, dtype=bool)
    x = np.atleast_2d(x)
    assert x.shape[0] == 1, "condition on a single point at a time"

    base = partial_evaluation(anchor_nodes, precision, Q, mask, x)[0]  # (m, m)
    eval_value = marginalize(anchor_nodes, precision, Q, mask, x)[0]
    Q_cond = base / eval_value

    anchor_nodes_NS = anchor_nodes[:, mask]
    precision_NS = precision[mask]
    return anchor_nodes_NS, precision_NS, Q_cond
