#!/usr/bin/env python3
import sys

import numpy as np

import rec


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "mem"
    cutoff = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    a = rec.load(base)
    size = a["size"].astype(np.int64)
    lat = a["lat"].astype(np.float64)

    sizes = np.unique(size)
    med = np.empty(sizes.shape)
    mn = np.empty(sizes.shape)
    print("# size count min median")
    for i, s in enumerate(sizes):
        v = lat[size == s]
        mn[i] = v.min()
        med[i] = np.median(v)
        print(s, v.size, int(mn[i]), int(med[i]))

    m = sizes >= cutoff
    if m.sum() < 2:
        sys.exit("mem: need >=2 sizes >= %d to fit" % cutoff)
    b, aa = np.polyfit(sizes[m], med[m], 1)  # lat = aa + b*size, ns
    print("# fit over %d sizes >= %d" % (m.sum(), cutoff))
    print("# intercept %.1f ns  slope %.6f ns/byte  bandwidth %.1f GB/s"
          % (aa, b, 1.0 / b))


if __name__ == "__main__":
    main()
