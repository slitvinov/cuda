import os

import numpy as np


def meta(base):
    path = base + ".meta"
    rows, endian, cols = None, "little", []
    for ln in open(path):
        p = ln.split()
        if not p:
            continue
        if p[0] == "rows":
            try:
                rows = int(p[1])
            except (IndexError, ValueError):
                raise ValueError("%s: bad rows: %r" % (path, ln.strip()))
        elif p[0] == "endian":
            endian = p[1] if len(p) > 1 else ""
        elif len(p) == 2:
            cols.append((p[0], p[1]))
        else:
            raise ValueError("%s: bad line: %r" % (path, ln.strip()))
    if not cols:
        raise ValueError("%s: no columns" % path)
    if endian not in ("little", "big"):
        raise ValueError("%s: bad endian: %r" % (path, endian))
    e = "<" if endian == "little" else ">"
    try:
        dt = np.dtype([(n, e + t) for n, t in cols])
    except TypeError as ex:
        raise ValueError("%s: bad dtype (%s)" % (path, ex))
    return dt, rows


def load(base):
    dt, rows = meta(base)
    path = base + ".raw"
    sz = os.path.getsize(path)
    if sz % dt.itemsize:
        raise ValueError("%s: %d bytes not a multiple of record size %d" %
                         (path, sz, dt.itemsize))
    n = sz // dt.itemsize
    if rows is not None and n != rows:
        raise ValueError("%s: %d records but %s.meta says rows=%d" %
                         (path, n, base, rows))
    return np.memmap(path, dtype=dt, mode="r", shape=(n,))
