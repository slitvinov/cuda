"""
Cholesky solve for one CGP individual:
   1. Build  H = JTJ + lambdaI
   2. Build  g = JT * 1     (synthetic RHS for the demo)
   3. Cholesky factor       H = L LT
   4. Forward solve         L y = g
   5. Back solve            LT delta = y
   6. Verify by recomputing H * delta and comparing against g.

H/g build use numpy matrix ops (the "parallel" part of the CUDA
kernel).  Cholesky and triangular solves are written with explicit
loops to mirror the CUDA tid==0 serial section.

All linear algebra in fp64.
"""

import numpy as np
from jgp import (
    forward_jacobian, set_node,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SIN, OP_COS, OP_PAR_MUL, OP_PAR,
)


def cholesky_solve(J, lam):
    """
    J: float32 [m, n_p]
    lam: scalar
    Synthetic RHS g = J^T * 1.
    Returns (delta [n_p], max_err = max|H*delta - g|).
    """
    m, n = J.shape
    Jd = J.astype(np.float64)

    # Parallel: H build, g build.
    H = Jd.T @ Jd + lam * np.eye(n)
    g = Jd.T @ np.ones(m, dtype=np.float64)

    # Serial: Cholesky factorization.
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            v = H[i, j]
            for k in range(j):
                v -= L[i, k] * L[j, k]
            L[i, j] = np.sqrt(v) if i == j else v / L[j, j]

    # Serial: forward solve L y = g.
    y = np.zeros(n)
    for i in range(n):
        v = g[i]
        for k in range(i):
            v -= L[i, k] * y[k]
        y[i] = v / L[i, i]

    # Serial: back solve LT s = y.
    s = np.zeros(n)
    for i in range(n - 1, -1, -1):
        v = y[i]
        for k in range(i + 1, n):
            v -= L[k, i] * s[k]
        s[i] = v / L[i, i]

    # Verify.
    max_err = float(np.max(np.abs(H @ s - g)))
    return s, max_err


def main():
    G, gi, gn, go = 4, 1, 6, 1
    N = 16
    n_p = gn
    m = go * N
    ng_total = gi + gn + go
    lam = 1e-3

    x = np.linspace(-1, 1, N).astype(np.float32)
    inputs = x.reshape(gi, N)

    genomes = [np.zeros((ng_total, 3), dtype=np.uint8) for _ in range(G)]
    set_node(genomes[0], 1, OP_PAR_MUL, 0, 0)
    set_node(genomes[0], 2, OP_SIN,     1, 0)
    set_node(genomes[0], 3, OP_PAR_MUL, 0, 0)
    set_node(genomes[0], 4, OP_MUL,     3, 3)
    set_node(genomes[0], 5, OP_ADD,     2, 4)
    set_node(genomes[0], 7, 0, 5)
    set_node(genomes[1], 1, OP_PAR_MUL, 0, 0)
    set_node(genomes[1], 2, OP_MUL,     1, 1)
    set_node(genomes[1], 7, 0, 2)
    set_node(genomes[2], 1, OP_PAR_MUL, 0, 0)
    set_node(genomes[2], 2, OP_SIN,     1, 0)
    set_node(genomes[2], 7, 0, 2)
    set_node(genomes[3], 1, OP_PAR_MUL, 0, 0)
    set_node(genomes[3], 2, OP_SIN,     1, 0)
    set_node(genomes[3], 3, OP_PAR_MUL, 2, 0)
    set_node(genomes[3], 7, 0, 3)

    params = np.zeros((G, n_p), dtype=np.float32)
    params[0, 0] = 2.0; params[0, 2] = 1.5
    params[1, 0] = 1.5
    params[2, 0] = 0.7
    params[3, 0] = 1.5; params[3, 2] = 2.0

    deltas = np.zeros((G, n_p))
    errs   = np.zeros(G)
    for g in range(G):
        _, J = forward_jacobian(params[g], inputs, genomes[g], gi, gn, go)
        J_flat = J.reshape(m, n_p)
        deltas[g], errs[g] = cholesky_solve(J_flat, lam)

    print(f"\nCholesky solve verification.  lambda = {lam:.1e},  m = {m},  n_p = {n_p}.")

    print("\ndelta vector per individual (one row per individual, n_p=6 columns):")
    print("           q=0          q=1          q=2          q=3          q=4          q=5")
    for g in range(G):
        line = f"  i{g}  "
        for q in range(n_p):
            line += f"  {deltas[g, q]:+10.4e}"
        print(line)

    labels = [
        "i0  sin(a*x) + (b*x)^2   active q = {0, 2}",
        "i1  (a*x)^2              active q = {0}",
        "i2  sin(a*x)            active q = {0}",
        "i3  a*sin(b*x)          active q = {0, 2}",
    ]
    print("\nVerification: max |H*delta - g| per individual")
    for g in range(G):
        print(f"  {labels[g]:<40}   {errs[g]:.3e}")


if __name__ == "__main__":
    main()
