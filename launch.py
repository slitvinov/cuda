#!/usr/bin/env python3
import sys

import numpy as np

import rec


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "launch"
    a = rec.load(base)
    lat = np.asarray(a["lat"], np.int64)
    sys.stdout.write("\n".join(map(str, lat.tolist())) + "\n")

if __name__ == "__main__":
    main()
