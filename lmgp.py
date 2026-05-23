"""
Levenberg-Marquardt CGP fitting for one individual.

Per step:
  1. forward + J at current params              (out, J)
  2. r = out - target,   fnorm = ‖r‖
  3. solve (JᵀJ + λI) δ = Jᵀr;  also Js = J·δ, fnorm_lin = ‖r - Js‖
  4. params_trial = params - δ
  5. forward at trial → r_trial, fnorm_trial
  6. trust region:
       ratio = (‖r‖² - ‖r_trial‖²) / (‖r‖² - ‖r_lin‖²)
       accept iff ratio > 1e-4
       ratio > 0.75 → λ *= 0.5
       ratio < 0.25 → λ *= 2.0
  7. if accept: commit trial → params, r, fnorm

Each individual carries its own λ that adapts independently.
"""

import numpy as np
from jgp import (
    forward_jacobian, set_node,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SIN, OP_COS, OP_PAR_MUL, OP_PAR,
)


def cholesky_solve_with_js(J, r, lam):
    """
    Solves (JᵀJ + λI) δ = Jᵀr, and also returns
      Js        = J · δ                   [m]
      fnorm_lin = ‖r - Js‖
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

    Js = Jd @ delta
    fnorm_lin = float(np.sqrt(np.sum((rd - Js) ** 2)))
    return delta, Js.astype(np.float32), fnorm_lin


def trust_region_update(fnorm, fnorm_trial, fnorm_lin, lam):
    """Returns (new_lam, accept_bool)."""
    ssq   = fnorm       * fnorm
    ssq_t = fnorm_trial * fnorm_trial
    ssq_l = fnorm_lin   * fnorm_lin
    actred = ssq - ssq_t
    prered = ssq - ssq_l
    safe   = prered if prered > 0.0 else 1.0
    ratio  = actred / safe
    acc    = ratio > 1.0e-4
    if   ratio > 0.75 and acc: lam *= 0.5
    elif ratio < 0.25:         lam *= 2.0
    if lam < 1.0e-12: lam = 1.0e-12
    return lam, acc


def target_fn(ind, x):
    if ind == 0: return np.sin(x) + x * x
    if ind == 1: return x * x
    if ind == 2: return np.sin(x)
    if ind == 3: return np.sin(x)


def forward_value(params, inputs, genome, gi, gn, go):
    """Value-only forward (trial-point evaluation)."""
    out, _ = forward_jacobian(params, inputs, genome, gi, gn, go)
    return out


def main():
    G, gi, gn, go = 4, 1, 6, 1
    N = 16
    n_p = gn
    m = go * N
    ng_total = gi + gn + go
    max_iter = 8
    lam0 = 1e-3

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

    lam   = np.full(G, lam0, dtype=np.float32)
    rs    = [None] * G
    fnorm = np.zeros(G, dtype=np.float32)
    for g in range(G):
        out = forward_value(params[g], inputs, genomes[g], gi, gn, go)
        rs[g] = (out - targets[g]).reshape(m).astype(np.float32)
        fnorm[g] = float(np.sqrt(np.sum(rs[g] ** 2)))

    print(f"\nLevenberg-Marquardt iteration  (initial λ = {lam0:.0e}):\n")
    print("iter   i0 loss        i1 loss        i2 loss        i3 loss      acc")

    def print_losses(it, accepted):
        line = f"{it:4d} "
        for g in range(G):
            loss = float(fnorm[g] ** 2 / m)
            line += f"  {loss:.6e}"
        if accepted is None:
            line += "    -"
        else:
            line += f"    {sum(accepted)}/{G}"
        print(line)

    print_losses(0, None)

    for it in range(1, max_iter + 1):
        accepted = []
        for g in range(G):
            # Forward + J at current params.
            out, J = forward_jacobian(params[g], inputs, genomes[g], gi, gn, go)
            J_flat = J.reshape(m, n_p)
            # Step + linearized prediction.
            delta, _, fnorm_lin = cholesky_solve_with_js(J_flat, rs[g], lam[g])
            # Trial point.
            params_trial = params[g] - delta.astype(np.float32)
            out_trial = forward_value(params_trial, inputs, genomes[g], gi, gn, go)
            r_trial = (out_trial - targets[g]).reshape(m).astype(np.float32)
            fnorm_trial = float(np.sqrt(np.sum(r_trial ** 2)))
            # Trust region decides λ and accept.
            new_lam, acc = trust_region_update(fnorm[g], fnorm_trial,
                                                fnorm_lin, float(lam[g]))
            lam[g] = new_lam
            accepted.append(acc)
            if acc:
                params[g]   = params_trial
                rs[g]       = r_trial
                fnorm[g]    = fnorm_trial
        print_losses(it, accepted)

    print("\nFinal parameters vs optimal (all targets are at a = b = 1):")
    print(f"  i0:  a = {params[0,0]:+9.6f}  (target 1.0)    b = {params[0,2]:+9.6f}  (target 1.0)    λ = {lam[0]:.2e}")
    print(f"  i1:  a = {params[1,0]:+9.6f}  (target 1.0)                                  λ = {lam[1]:.2e}")
    print(f"  i2:  a = {params[2,0]:+9.6f}  (target 1.0)                                  λ = {lam[2]:.2e}")
    print(f"  i3:  b = {params[3,0]:+9.6f}  (target 1.0)    a = {params[3,2]:+9.6f}  (target 1.0)    λ = {lam[3]:.2e}")


if __name__ == "__main__":
    main()
