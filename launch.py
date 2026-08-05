#!/usr/bin/env python3
import sys

import numpy as np

import rec


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "launch"
    a = rec.load(base)
    names = a.dtype.names
    cols = [np.asarray(a[n], np.uint64).tolist() for n in names]
    out = "\n".join(" ".join(map(str, row)) for row in zip(*cols))
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
