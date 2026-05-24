"""
Parameterized CGP: full Jacobian wrt per-node parameters, for one
individual.  Two-pass structure: forward values, then n_p tangent
sweeps, one per parameter.
"""

import numpy as np

OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SIN, OP_COS, OP_PAR_MUL, OP_PAR = range(8)


def apply_op_jvp(op, v0, v1, t0, t1, par, par_t):
    if op == OP_ADD: return v0 + v1, t0 + t1
    if op == OP_SUB: return v0 - v1, t0 - t1
    if op == OP_MUL: return v0 * v1, t0 * v1 + v0 * t1
    if op == OP_DIV:
        inv = 1.0 / v1
        v = v0 * inv
        return v, (t0 - v * t1) * inv
    if op == OP_SIN: return np.sin(v0),  np.cos(v0) * t0
    if op == OP_COS: return np.cos(v0), -np.sin(v0) * t0
    if op == OP_PAR_MUL: return par * v0, par_t * v0 + par * t0
    if op == OP_PAR:
        return (np.full_like(v0, par, dtype=np.float32),
                np.full_like(v0, par_t, dtype=np.float32))
    return np.zeros_like(v0), np.zeros_like(v0)


def forward_jacobian(params, inputs, genome, gi, gn, go):
    """
    params: float32 [n_p]      n_p = gn (p = 1 here)
    inputs: float32 [gi, N]
    genome: uint8   [gi + gn + go, 3]
    returns (out, J): float32 [go, N], float32 [go, N, n_p]
    """
    N = inputs.shape[1]
    n_p = gn
    state_v = np.zeros((gi + gn, N), dtype=np.float32)
    state_t = np.zeros((gi + gn, N), dtype=np.float32)

    for i in range(gi):
        state_v[i] = inputs[i]

    # Pass 1: forward values only.
    for j in range(gn):
        op, ptr0, ptr1 = genome[gi + j]
        v, _ = apply_op_jvp(op, state_v[ptr0], state_v[ptr1],
                            0.0, 0.0, float(params[j]), 0.0)
        state_v[gi + j] = v

    out = np.zeros((go, N), dtype=np.float32)
    for o in range(go):
        out[o] = state_v[genome[gi + gn + o, 1]]

    # Pass 2: tangent sweep per parameter q.
    J = np.zeros((go, N, n_p), dtype=np.float32)
    for q in range(n_p):
        for i in range(gi):
            state_t[i] = 0.0
        for j in range(gn):
            op, ptr0, ptr1 = genome[gi + j]
            par_t = 1.0 if j == q else 0.0
            _, t = apply_op_jvp(op, state_v[ptr0], state_v[ptr1],
                                state_t[ptr0], state_t[ptr1],
                                float(params[j]), par_t)
            state_t[gi + j] = t
        for o in range(go):
            J[o, :, q] = state_t[genome[gi + gn + o, 1]]

    return out, J


def set_node(g, row, op, p0=0, p1=0):
    g[row] = [op, p0, p1]


def main():
    G, gi, gn, go = 4, 1, 6, 1
    N = 16
    n_p = gn
    ng_total = gi + gn + go

    x = np.linspace(-1, 1, N).astype(np.float32)
    inputs = x.reshape(gi, N)

    # Same parameterized, non-linear-in-param individuals as jgp.cu:
    #   i0:  y = sin(a*x) + (b*x)^2    params a (q=0), b (q=2)
    #   i1:  y = (a*x)^2                a at q=0
    #   i2:  y = sin(a*x)              a at q=0
    #   i3:  y = a * sin(b*x)          b at q=0, a at q=2
    genomes = [np.zeros((ng_total, 3), dtype=np.uint8) for _ in range(G)]

    # i0
    set_node(genomes[0], 1, OP_PAR_MUL, 0, 0)
    set_node(genomes[0], 2, OP_SIN,     1, 0)
    set_node(genomes[0], 3, OP_PAR_MUL, 0, 0)
    set_node(genomes[0], 4, OP_MUL,     3, 3)
    set_node(genomes[0], 5, OP_ADD,     2, 4)
    set_node(genomes[0], 7, 0, 5)

    # i1
    set_node(genomes[1], 1, OP_PAR_MUL, 0, 0)
    set_node(genomes[1], 2, OP_MUL,     1, 1)
    set_node(genomes[1], 7, 0, 2)

    # i2
    set_node(genomes[2], 1, OP_PAR_MUL, 0, 0)
    set_node(genomes[2], 2, OP_SIN,     1, 0)
    set_node(genomes[2], 7, 0, 2)

    # i3
    set_node(genomes[3], 1, OP_PAR_MUL, 0, 0)
    set_node(genomes[3], 2, OP_SIN,     1, 0)
    set_node(genomes[3], 3, OP_PAR_MUL, 2, 0)
    set_node(genomes[3], 7, 0, 3)

    params = np.zeros((G, n_p), dtype=np.float32)
    params[0, 0] = 2.0   # i0: a
    params[0, 2] = 1.5   # i0: b
    params[1, 0] = 1.5   # i1: a
    params[2, 0] = 0.7   # i2: a
    params[3, 0] = 1.5   # i3: b
    params[3, 2] = 2.0   # i3: a

    Js = [forward_jacobian(params[g], inputs, genomes[g], gi, gn, go)[1]
          for g in range(G)]

    # Analytic gradients (depend on parameter values -- non-linear).
    def grad_i0_q0(x, p): return np.cos(p[0] * x) * x
    def grad_i0_q2(x, p): return 2.0 * p[2] * x * x
    def grad_i1_q0(x, p): return 2.0 * p[0] * x * x
    def grad_i2_q0(x, p): return np.cos(p[0] * x) * x
    def grad_i3_q0(x, p): return p[2] * np.cos(p[0] * x) * x
    def grad_i3_q2(x, p): return np.sin(p[0] * x)

    active = [
        [(0, "cos(a*x)*x",   grad_i0_q0), (2, "2*b*x^2",      grad_i0_q2)],
        [(0, "2*a*x^2",       grad_i1_q0)],
        [(0, "cos(a*x)*x",   grad_i2_q0)],
        [(0, "a*cos(b*x)*x", grad_i3_q0), (2, "sin(b*x)",    grad_i3_q2)],
    ]

    primary_labels = [
        "i0 dy/da = cos(a*x)*x",
        "i1 dy/da = 2*a*x^2",
        "i2 dy/da = cos(a*x)*x",
        "i3 dy/db = a*cos(b*x)*x",
    ]
    print(f"\n   x      {primary_labels[0]:<22}   {primary_labels[1]:<22}"
          f"   {primary_labels[2]:<22}   {primary_labels[3]:<22}")
    print("           GPU       ref          GPU       ref          "
          "GPU       ref          GPU       ref")
    for k in range(N):
        line = f"{x[k]:+6.3f} "
        for g in range(G):
            q, _, grad_fn = active[g][0]
            ref = grad_fn(x[k], params[g])
            line += f"  {Js[g][0, k, q]:+8.4f} {ref:+8.4f}"
        print(line)

    print("\nActive-column verification (max |GPU - analytic|):")
    for g in range(G):
        for q, label, grad_fn in active[g]:
            ref = grad_fn(x, params[g])
            err = float(np.max(np.abs(Js[g][0, :, q] - ref)))
            print(f"  i{g}  q={q}  dy/d{label:<18}  {err:.3e}")

    # Inactive-column sanity.
    inactive_max = 0.0
    for g in range(G):
        active_qs = {q for q, _, _ in active[g]}
        for q in range(n_p):
            if q in active_qs: continue
            inactive_max = max(inactive_max, float(np.max(np.abs(Js[g][0, :, q]))))
    print("\nInactive-column sanity (should all be 0):")
    print(f"  max |J[g, 0, k, q_inactive]| across all individuals  =  {inactive_max:.3e}")


if __name__ == "__main__":
    main()
