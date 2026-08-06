#!/usr/bin/env python3
import sys
import numpy as np
import rec


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "launch"
    a = rec.load(base)
    lt = a["lat"].astype(np.int64)
    v, c = np.unique(lt, return_counts=True)
    cdf = np.cumsum(c)
    for x, p in zip(v, cdf):
        print(x, p, p / cdf[-1])


if __name__ == "__main__":
    main()
