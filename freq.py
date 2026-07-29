#!/usr/bin/env python3
import sys
import numpy as np
import rec

def gaps(base, zmin=3.0):
    a = rec.load(base)
    line = np.asarray(a["id"], np.int64)
    lat = np.asarray(a["lat"], np.float64)
    it = np.asarray(a["iter"], np.int64)
    clk = np.asarray(a["clock64"], np.int64)
    n = int(line.max()) + 1
    order = np.argsort(line, kind="stable")
    S = lat[order].reshape(n, -1)
    med = np.median(S, 1)
    mad = np.median(np.abs(S - med[:, None]), 1)
    mad[mad == 0] = 1
    z = (lat - med[line]) / (1.4826 * mad[line])
    m = z > zmin
    clk, it = clk[m], it[m]
    o = np.argsort(clk)
    clk, it = clk[o], it[o]
    d = np.diff(clk)
    return d[(d > 0) & (np.diff(it) == 0)]


def main():
    base = sys.argv[1]
    zmin = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else 80000
    g = gaps(base, zmin)
    h, e = np.histogram(g, bins=4000, range=(0, hi))
    c = (e[:-1] + e[1:]) / 2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.semilogy(c, h)
    plt.xlabel("gap (cycles)")
    plt.ylabel("count")
    plt.tight_layout()
    out = base + ".freq.png"
    plt.savefig(out, dpi=130)
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
