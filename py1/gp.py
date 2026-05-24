"""
Cartesian Genetic Programming, forward pass for one individual.

Mirrors the kernel structure of gp.cu without the G / batching axis.
The outer loop over nodes stays as a Python loop; the per-sample work
(CUDA's per-thread stride over N) becomes a single numpy array op.

The demo runs the same four hand-crafted individuals as gp.cu so the
printed table and MSE match.
"""

import numpy as np

OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SIN, OP_COS = 0, 1, 2, 3, 4, 5


def apply_op(op, v0, v1):
    if op == OP_ADD: return v0 + v1
    if op == OP_SUB: return v0 - v1
    if op == OP_MUL: return v0 * v1
    if op == OP_DIV: return v0 / v1
    if op == OP_SIN: return np.sin(v0)
    if op == OP_COS: return np.cos(v0)
    return np.zeros_like(v0)


def forward(inputs, genome, gi, gn, go):
    """
    inputs: float32 [gi, N]
    genome: uint8   [gi + gn + go, 3]   per row = (op, ptr0, ptr1)
    returns out: float32 [go, N]
    """
    N = inputs.shape[1]
    state = np.zeros((gi + gn, N), dtype=np.float32)

    # 1. Stage inputs.
    for i in range(gi):
        state[i] = inputs[i]

    # 2. Sweep nodes.
    for j in range(gn):
        op, ptr0, ptr1 = genome[gi + j]
        state[gi + j] = apply_op(op, state[ptr0], state[ptr1])

    # 3. Materialize outputs.
    out = np.zeros((go, N), dtype=np.float32)
    for o in range(go):
        out[o] = state[genome[gi + gn + o, 1]]
    return out


def set_node(g, row, op, p0=0, p1=0):
    g[row] = [op, p0, p1]


def main():
    G, gi, gn, go = 4, 1, 6, 1
    N = 16
    ng_total = gi + gn + go

    x = np.linspace(-1, 1, N).astype(np.float32)
    inputs = x.reshape(gi, N)
    target = np.sin(x) + x * x

    genomes = [np.zeros((ng_total, 3), dtype=np.uint8) for _ in range(G)]

    # i0: y = sin(x) + x^2
    set_node(genomes[0], 1, OP_MUL, 0, 0)
    set_node(genomes[0], 2, OP_SIN, 0)
    set_node(genomes[0], 3, OP_ADD, 1, 2)
    set_node(genomes[0], 7, 0, 3)

    # i1: y = x^2
    set_node(genomes[1], 1, OP_MUL, 0, 0)
    set_node(genomes[1], 7, 0, 1)

    # i2: y = sin(x)
    set_node(genomes[2], 1, OP_SIN, 0)
    set_node(genomes[2], 7, 0, 1)

    # i3: y = x
    set_node(genomes[3], 7, 0, 0)

    outs = [forward(inputs, genomes[g], gi, gn, go)[0] for g in range(G)]

    names = [
        "i0 = sin(x)+x*x  (true)",
        "i1 = x*x",
        "i2 = sin(x)",
        "i3 = x",
    ]
    print("\n   x      target       i0          i1          i2          i3")
    for k in range(N):
        line = f"{x[k]:+6.3f}   {target[k]:+9.5f}"
        for g in range(G):
            line += f"  {outs[g][k]:+9.5f}"
        print(line)
    print("\nMSE per individual:")
    for g in range(G):
        mse = float(np.mean((outs[g] - target) ** 2))
        print(f"  {names[g]:<25}  {mse:.6e}")
    breakpoint()


if __name__ == "__main__":
    main()
