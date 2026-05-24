"""
Gauss-Newton CGP fitting for one individual.

  for it = 0 .. max_iter - 1:
      forward + Jacobian   ->  out, J
      r = out - target
      solve (JTJ + lambdaI) delta = JTr
      params -= delta

No trust region; every step is taken blindly with a fixed lambda.
"""

import numpy as np
from jgp import (
    forward_jacobian, set_node,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SIN, OP_COS, OP_PAR_MUL, OP_PAR,
)


def cholesky_solve(J, r, lam):
    """
    J: float32 [m, n_p]
    r: float32 [m]
    Solves (JTJ + lambdaI) delta = JTr using Cholesky + triangular solves.
    Returns delta as float64 [n_p].
    """
    m, n = J.shape
    Jd = J.astype(np.float64)
    rd = r.astype(np.float64)
    H = Jd.T @ Jd + lam * np.eye(n)
    g = Jd.T @ rd

    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            v = H[i, j]
            for k in range(j):
                v -= L[i, k] * L[j, k]
            L[i, j] = np.sqrt(v) if i == j else v / L[j, j]

    y = np.zeros(n)
    for i in range(n):
        v = g[i]
        for k in range(i):
            v -= L[i, k] * y[k]
        y[i] = v / L[i, i]

    delta = np.zeros(n)
    for i in range(n - 1, -1, -1):
        v = y[i]
        for k in range(i + 1, n):
            v -= L[k, i] * delta[k]
        delta[i] = v / L[i, i]

    return delta


def target_fn(ind, x):
    if ind == 0: return np.sin(x) + x * x      # a=b=1
    if ind == 1: return x * x                  # a=1
    if ind == 2: return np.sin(x)              # a=1
    if ind == 3: return np.sin(x)              # a=b=1


def main():
    G, gi, gn, go = 4, 1, 6, 1
    N = 16
    n_p = gn
    m = go * N
    ng_total = gi + gn + go
    max_iter = 8
    lam = 1e-3

    x = np.linspace(-1, 1, N).astype(np.float32)
    inputs = x.reshape(gi, N)

    targets = np.stack([target_fn(g, x) for g in range(G)]).astype(np.float32)
    targets = targets.reshape(G, go, N)

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

    print(f"\nGauss-Newton iteration (fixed lambda = {lam:.0e}):\n")
    print("iter   i0 loss        i1 loss        i2 loss        i3 loss")

    for it in range(max_iter + 1):
        losses = np.zeros(G)
        Js   = [None] * G
        rs   = [None] * G
        for g in range(G):
            out, J = forward_jacobian(params[g], inputs, genomes[g], gi, gn, go)
            r = (out - targets[g]).reshape(m).astype(np.float32)
            losses[g] = float(np.mean(r * r))
            Js[g] = J.reshape(m, n_p)
            rs[g] = r

        line = f"{it:4d} "
        for g in range(G):
            line += f"  {losses[g]:.6e}"
        print(line)

        if it == max_iter: break

        for g in range(G):
            delta = cholesky_solve(Js[g], rs[g], lam)
            params[g] -= delta.astype(np.float32)

    print("\nFinal parameters vs optimal (all targets are at a = b = 1):")
    print(f"  i0:  a = {params[0,0]:+9.6f}  (target 1.0)    b = {params[0,2]:+9.6f}  (target 1.0)")
    print(f"  i1:  a = {params[1,0]:+9.6f}  (target 1.0)")
    print(f"  i2:  a = {params[2,0]:+9.6f}  (target 1.0)")
    print(f"  i3:  b = {params[3,0]:+9.6f}  (target 1.0)    a = {params[3,2]:+9.6f}  (target 1.0)")


if __name__ == "__main__":
    main()
