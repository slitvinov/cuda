#!/usr/bin/env python3
import sys
import numpy as np
import rec


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "launch"
    a = rec.load(base)
    lt = a["lat"]
    print(lt.dtype)
    np.bincount(lt.astype(np.int64))


if __name__ == "__main__":
    main()
