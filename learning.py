"""
Examples of np.einsum, building up to the exact patterns used in psd.py.

Rule of thumb: einsum(subscripts, *arrays) names every axis of every input
with a letter. Repeated letters across inputs mean "multiply these axes
together elementwise". Letters that appear on the input side but NOT in the
output are summed over. Letters that appear in the output are kept, in
whatever order you write them.

Run this file directly: python learning.py
"""
import numpy as np

np.random.seed(0)


def show(title, einsum_result, explicit_result):
    diff = np.max(np.abs(einsum_result - explicit_result))
    print(f"--- {title} ---")
    print("max abs diff vs explicit numpy:", diff)
    assert diff < 1e-10


# ---------------------------------------------------------------------------
# 1. Transpose: just reorder the output letters, nothing is summed.
# ---------------------------------------------------------------------------
A = np.random.randn(3, 4)
show("transpose: 'ij->ji'", np.einsum('ij->ji', A), A.T)


# ---------------------------------------------------------------------------
# 2. Sum over an axis: drop a letter from the output.
# ---------------------------------------------------------------------------
show("row sums: 'ij->i'", np.einsum('ij->i', A), A.sum(axis=1))
show("column sums: 'ij->j'", np.einsum('ij->j', A), A.sum(axis=0))
show("sum everything: 'ij->'", np.einsum('ij->', A), A.sum())


# ---------------------------------------------------------------------------
# 3. Elementwise product: same letters on both inputs, kept in the output.
# ---------------------------------------------------------------------------
B = np.random.randn(3, 4)
show("elementwise product: 'ij,ij->ij'", np.einsum('ij,ij->ij', A, B), A * B)


# ---------------------------------------------------------------------------
# 4. Dot product / inner product: shared letter, summed away (dropped from output).
# ---------------------------------------------------------------------------
x = np.random.randn(5)
y = np.random.randn(5)
show("dot product: 'i,i->'", np.einsum('i,i->', x, y), np.dot(x, y))


# ---------------------------------------------------------------------------
# 5. Outer product: different letters, both kept -> every combination.
# ---------------------------------------------------------------------------
show("outer product: 'i,j->ij'", np.einsum('i,j->ij', x, y[:3]), np.outer(x, y[:3]))


# ---------------------------------------------------------------------------
# 6. Matrix multiplication: the shared letter (k) is the contracted dimension.
# ---------------------------------------------------------------------------
M1 = np.random.randn(3, 4)
M2 = np.random.randn(4, 5)
show("matmul: 'ik,kj->ij'", np.einsum('ik,kj->ij', M1, M2), M1 @ M2)


# ---------------------------------------------------------------------------
# 7. Trace: Tr(M) = sum_i M[i, i]. Reusing the same letter twice on ONE input
# picks out the diagonal, and dropping it from the output sums that diagonal.
# ---------------------------------------------------------------------------
Q = np.random.randn(4, 4)
show("trace: 'ii->'", np.einsum('ii->', Q), np.trace(Q))

# Tr(Q @ M2') for square M2' -- combine "matmul" and "trace" in one call,
# without ever forming the product Q @ M2' explicitly:
M2sq = np.random.randn(4, 4)
show(
    "Tr(Q @ M2sq): 'ij,ji->'",
    np.einsum('ij,ji->', Q, M2sq),
    np.trace(Q @ M2sq),
)


# ---------------------------------------------------------------------------
# 8. Batched trace -- this is exactly the pattern used in psd.loss/gradient/
# hessian: trace_QA[i] = Tr(Q @ A_i) for a whole stack A of shape (N, m, m),
# with the SAME Q reused for every i in the stack.
# ---------------------------------------------------------------------------
N, m = 6, 4
Qb = np.random.randn(m, m)
A = np.random.randn(N, m, m)

trace_einsum = np.einsum('jk,ikj->i', Qb, A)
trace_explicit = np.array([np.trace(Qb @ A[i]) for i in range(N)])
show("batched Tr(Q @ A_i): 'jk,ikj->i'", trace_einsum, trace_explicit)


# ---------------------------------------------------------------------------
# 9. Batched outer product -- this is the K_S pattern: for each row i of a
# (n, m) matrix phi, form the m x m outer product phi[i] @ phi[i].T, stacked
# into an (n, m, m) array.
# ---------------------------------------------------------------------------
n = 5
phi = np.random.randn(n, m)

outer_einsum = np.einsum('ki,kj->kij', phi, phi)
outer_explicit = np.stack([np.outer(phi[k], phi[k]) for k in range(n)])
show("batched outer product: 'ki,kj->kij'", outer_einsum, outer_explicit)


# ---------------------------------------------------------------------------
# 10. Batched Tr(A_i @ dQ) -- this is the hessian_vector_product pattern:
# for each matrix A[k] in a stack, contract it against a single dQ.
# ---------------------------------------------------------------------------
dQ = np.random.randn(m, m)

trace_AdQ_einsum = np.einsum('kjl,lj->k', A, dQ)
trace_AdQ_explicit = np.array([np.trace(A[k] @ dQ) for k in range(N)])
show("batched Tr(A_i @ dQ): 'kjl,lj->k'", trace_AdQ_einsum, trace_AdQ_explicit)


# ---------------------------------------------------------------------------
# 11. Four-index outer product -- this is the hessian pattern: build the
# (m, m, m, m) tensor T[i, j, t, s] = Qinv[j, t] * Qinv[s, i] out of two
# independent matrices, with full control over the output axis order.
# ---------------------------------------------------------------------------
Qinv = np.linalg.inv(Qb + np.eye(m) * 5)  # make it invertible

T_einsum = np.einsum('jt,si->ijts', Qinv, Qinv)
T_explicit = np.zeros((m, m, m, m))
for i in range(m):
    for j in range(m):
        for t in range(m):
            for s in range(m):
                T_explicit[i, j, t, s] = Qinv[j, t] * Qinv[s, i]
show("four-index outer product: 'jt,si->ijts'", T_einsum, T_explicit)


print("\nall einsum examples match their explicit numpy equivalents")
