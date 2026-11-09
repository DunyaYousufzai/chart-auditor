"""Combine N readings of the same chart into one table plus an uncertainty per cell.

Each reading is a tables.Table (or None if the model wrote nothing usable).
Readings may disagree on labels ("ResNet" vs "Resnet") or on the number of rows,
so we first align every reading to a reference reading, then take per cell:
  value  = median of the N values
  spread = standard deviation of the N values (absolute units; checker makes it relative)
"""
import statistics
from collections import Counter

from checker import match_names
from tables import Table


def _reference(readings):
    """The reading whose shape (rows x series) is the most common; ties -> first."""
    shapes = Counter((len(r.rows), len(r.series)) for r in readings)
    shape = shapes.most_common(1)[0][0]
    return next(r for r in readings if (len(r.rows), len(r.series)) == shape)


def combine(readings):
    """Return (median_table, spread, info). median_table is None if no reading is usable."""
    ok = [r for r in readings if r is not None and r.rows and r.series]
    info = {"n": len(readings), "n_ok": len(ok)}
    if not ok:
        return None, {}, info
    ref = _reference(ok)
    collected = {(r, s): [] for r in ref.rows for s in ref.series}

    for rd in ok:
        rows = match_names(rd.rows, ref.rows)
        if not rows and len(rd.rows) == len(ref.rows):
            rows = dict(zip(rd.rows, ref.rows))
        cols = match_names(rd.series, ref.series)
        if len(cols) < len(ref.series) and len(rd.series) == len(ref.series):
            cols = dict(zip(rd.series, ref.series))
        cells = rd.cells()
        for (r, s), v in cells.items():
            key = (rows.get(r), cols.get(s))
            if key in collected:
                collected[key].append(v)

    values, spread = [], {}
    for r in ref.rows:
        row = []
        for s in ref.series:
            vs = collected[(r, s)]
            if vs:
                row.append(statistics.median(vs))
                spread[(r, s)] = statistics.pstdev(vs) if len(vs) > 1 else 0.0
            else:
                row.append(None)
        values.append(row)
    return Table(ref.x_name, list(ref.series), list(ref.rows), values), spread, info


def is_stable(prev, cur, eps=0.01):
    """True when the median table barely moved after adding readings (for early stopping)."""
    if prev is None or cur is None:
        return False
    a, b = prev.cells(), cur.cells()
    if a.keys() != b.keys():
        return False
    return all(abs(a[k] - b[k]) <= eps * max(abs(a[k]), 1e-9) for k in a)


def combine_until_stable(readings, n_min=3, eps=0.01):
    """Use readings one by one and stop once the median is stable.

    Returns (table, spread, info) with info["n_used"] = readings actually needed.
    Lets us measure how much early stopping saves without re-running the model.
    """
    prev = None
    for n in range(1, len(readings) + 1):
        cur, spread, info = combine(readings[:n])
        if n >= n_min and is_stable(prev, cur, eps):
            info["n_used"] = n
            return cur, spread, info
        prev = cur
    info["n_used"] = len(readings)
    return cur, spread, info
