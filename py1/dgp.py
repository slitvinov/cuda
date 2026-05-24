"""
Differentiable CGP forward pass for one individual via forward-mode AD.

Mirrors dgp.cu: alongside each node's value we propagate a tangent
d(node)/dx, seeded with tangent = 1 on the input row.
"""

import numpy as np

OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SIN, OP_COS = 0, 1, 2, 3, 4, 5


def apply_op_ad(op, v0, v1, d0, d1):
    if op == OP_ADD: return v0 + v1, d0 + d1
    if op == OP_SUB: return v0 - v1, d0 - d1
    if op == OP_MUL: return v0 * v1, d0 * v1 + v0 * d1
    if op == OP_DIV:
        inv = 1.0 / v1
        v = v0 * inv
        return v, (d0 - v * d1) * inv
    if op == OP_SIN: return np.sin(v0),  np.cos(v0) * d0
    if op == OP_COS: return np.cos(v0), -np.sin(v0) * d0
    return np.zeros_like(v0), np.zeros_like(v0)


def forward_ad(inputs, genome, gi, gn, go):
    """
    inputs: float32 [gi, N]
    genome: uint8   [gi + gn + go, 3]
    returns (out_v, out_d): float32 [go, N], float32 [go, N]
    """
    N = inputs.shape[1]
    state_v = np.zeros((gi + gn, N), dtype=np.float32)
    state_d = np.zeros((gi + gn, N), dtype=np.float32)

    # Stage inputs.  Tangent seed: dx/dx = 1.
    for i in range(gi):
        state_v[i] = inputs[i]
        state_d[i] = 1.0

    # Sweep nodes, propagating (value, tangent) together.
    for j in range(gn):
        op, ptr0, ptr1 = genome[gi + j]
        v, d = apply_op_ad(op,
                           state_v[ptr0], state_v[ptr1],
                           state_d[ptr0], state_d[ptr1])
        state_v[gi + j] = v
        state_d[gi + j] = d

    # Materialize outputs (value AND tangent).
    out_v = np.zeros((go, N), dtype=np.float32)
    out_d = np.zeros((go, N), dtype=np.float32)
    for o in range(go):
        src = genome[gi + gn + o, 1]
        out_v[o] = state_v[src]
        out_d[o] = state_d[src]
    return out_v, out_d


def set_node(g, row, op, p0=0, p1=0):
    g[row] = [op, p0, p1]


def analytic_grad(ind, x):
    if ind == 0: return np.cos(x) + 2.0 * x   # sin(x) + x^2
    if ind == 1: return 2.0 * x               # x^2
    if ind == 2: return np.cos(x)             # sin(x)
    if ind == 3: return np.ones_like(x)       # x


def main():
    G, gi, gn, go = 4, 1, 6, 1
    N = 16
    ng_total = gi + gn + go

    x = np.linspace(-1, 1, N).astype(np.float32)
    inputs = x.reshape(gi, N)

    genomes = [np.zeros((ng_total, 3), dtype=np.uint8) for _ in range(G)]
    set_node(genomes[0], 1, OP_MUL, 0, 0)
    set_node(genomes[0], 2, OP_SIN, 0)
    set_node(genomes[0], 3, OP_ADD, 1, 2)
    set_node(genomes[0], 7, 0, 3)
    set_node(genomes[1], 1, OP_MUL, 0, 0)
    set_node(genomes[1], 7, 0, 1)
    set_node(genomes[2], 1, OP_SIN, 0)
    set_node(genomes[2], 7, 0, 1)
    set_node(genomes[3], 7, 0, 0)

    grads = [forward_ad(inputs, genomes[g], gi, gn, go)[1][0] for g in range(G)]

    print("\n   x          i0 (cos x + 2x)       i1 (2x)             i2 (cos x)          i3 (1)")
    print("          GPU        ref      GPU        ref      GPU        ref      GPU      ref")
    for k in range(N):
        line = f"{x[k]:+6.3f} "
        for g in range(G):
            ref = analytic_grad(g, x[k])
            line += f"  {grads[g][k]:+8.4f}  {ref:+8.4f}"
        print(line)

    print("\nMax |GPU - analytic| per individual:")
    names = ["i0 (cos x + 2x)", "i1 (2x)", "i2 (cos x)", "i3 (1)"]
    for g in range(G):
        ref = analytic_grad(g, x)
        err = float(np.max(np.abs(grads[g] - ref)))
        print(f"  {names[g]:<20}  {err:.3e}")


if __name__ == "__main__":
    main()
