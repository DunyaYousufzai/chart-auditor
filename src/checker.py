"""Compare the table read from a chart with the paper's table. Plain Python, no model.

Three steps:
  1. align   : link chart rows/series to table rows/series (exact, then fuzzy name match)
  2. scale   : find the unit factor (1, 1000, 1e6, ...) that makes them agree best
  3. compare : flag a cell when the gap is larger than the allowed tolerance

Tolerance:
  fixed     tol = tau                      (e.g. 5%)
  adaptive  tol = max(tau, k * s)          s = spread of the N chart readings for that cell

All gaps are relative: gap = |c - t| / ref, with ref = max(|t|, floor * largest |t|).
The floor stops tiny values (an error rate of 0.2 on an axis that goes to 30) from
giving huge relative gaps for a reading error that is small on the picture.
"""
import difflib
import re
import statistics

SCALES = [1, 1000, 1e-3, 1e6, 1e-6, 100, 0.01]


def norm(label):
    """'ResNet-50 ' -> 'resnet50', '2019.0' -> '2019'."""
    s = str(label).lower().strip()
    s = re.sub(r"\.0+$", "", s)
    s = re.sub(r"\(.*?\)", "", s)           # drop units in brackets
    return re.sub(r"[^a-z0-9%]", "", s)


def match_names(chart_names, table_names, cutoff=0.6):
    """Return {chart_name: table_name}. Exact (normalised) first, then fuzzy, one-to-one."""
    out, used = {}, set()
    tnorm = {norm(t): t for t in table_names}
    for c in chart_names:
        t = tnorm.get(norm(c))
        if t is not None and t not in used:
            out[c] = t
            used.add(t)
    for c in chart_names:
        if c in out:
            continue
        left = [t for t in table_names if t not in used]
        best = difflib.get_close_matches(norm(c), [norm(t) for t in left], n=1, cutoff=cutoff)
        if best:
            t = next(t for t in left if norm(t) == best[0])
            out[c] = t
            used.add(t)
    return out


def orient(chart, ref):
    """Return chart as is or transposed, whichever lines up better with ref.

    Qwen3-VL often writes categories as columns ("City | Zurich | Geneva" then
    "Patients | 350 | 315"). We count how many row and series names match ref in
    both orientations (plus a bonus when the shape matches) and keep the better one.
    """
    def score(c):
        s = len(match_names(c.rows, ref.rows, cutoff=0.8)) + len(match_names(c.series, ref.series, cutoff=0.8))
        if (len(c.rows), len(c.series)) == (len(ref.rows), len(ref.series)):
            s += 0.5
        return s

    t = chart.transpose()
    return t if score(t) > score(chart) else chart


def align(chart, table):
    """Pairs of (chart cell key, table cell key). chart and table are tables.Table.

    Series: by name; if both have the same number of series and names do not match
    (readers often write "series_1" or "Value"), we pair them by order.
    Rows: by name; if no row name matches and both have the same number of rows,
    by order (a reader may misread every label but still get the numbers).
    """
    col_map = match_names(chart.series, table.series)
    if len(col_map) < min(len(chart.series), len(table.series)) and len(chart.series) == len(table.series):
        col_map = dict(zip(chart.series, table.series))
    row_map = match_names(chart.rows, table.rows)
    if not row_map and len(chart.rows) == len(table.rows):
        row_map = dict(zip(chart.rows, table.rows))

    pairs = []
    for cr, tr in row_map.items():
        for cs, ts in col_map.items():
            pairs.append(((cr, cs), (tr, ts)))
    return pairs, row_map, col_map


def find_scale(c_vals, t_vals):
    """Unit factor f so that chart value c ~ f * table value t. Picks the f with the smallest median gap."""
    best, best_err = 1, float("inf")
    for f in SCALES:
        errs = [abs(c - f * t) / max(abs(f * t), 1e-9) for c, t in zip(c_vals, t_vals)]
        err = statistics.median(errs) if errs else float("inf")
        if err < best_err * 0.5 or (f == 1 and err <= best_err):  # prefer 1 unless clearly better
            best, best_err = f, err
    return best


def theil_sen(x, y):
    """Robust line y ~ a*x + b: median of pairwise slopes, so one or two odd cells do not move it."""
    slopes = sorted((y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x))
                    if x[j] != x[i])
    if not slopes:
        return 1.0, 0.0
    a = slopes[len(slopes) // 2]
    b = sorted(yi - a * xi for xi, yi in zip(x, y))[len(x) // 2]
    return a, b


def compare(chart, table, spread=None, tau=0.05, k=0.0, s_max=None, floor=0.05,
            use_scale=True, min_coverage=0.5, rel=None, calibrate=False, min_cells_rel=4):
    """Check a chart table against a paper table.

    chart, table : tables.Table (chart = median of the readings)
    spread       : {(row, series): absolute spread of that cell over the N readings}, or None
    tau          : base tolerance, relative (0.05 = 5%)
    k            : weight of the spread; k = 0 gives the fixed tolerance
    s_max        : if a cell's relative spread is above this, we cannot read it -> "not sure"
    rel          : relative rule. A cell is flagged only if its gap is also above rel x the median gap
                   of this chart (its reading noise): an altered cell stands out from the rest.
                   Used when the chart has at least min_cells_rel cells.
    calibrate    : first fit chart ~ a*table + b (Theil-Sen) and compare with the fitted values, so a
                   misread axis (all values off the same way) is not taken for a mismatch
    Returns a dict the web page can show: verdict, note, differences, plus details.
    """
    spread = spread or {}
    oriented = orient(chart, table)
    if oriented is not chart:  # transposed: the spread keys must follow
        spread = {(s, r): v for (r, s), v in spread.items()}
        chart = oriented
    pairs, row_map, col_map = align(chart, table)
    cc, tc = chart.cells(), table.cells()
    pairs = [(a, b) for a, b in pairs if a in cc and b in tc]

    n_table = len(tc)
    if not pairs:
        return {"verdict": "not_sure", "note": "No label of the chart matches a label of the table.",
                "differences": [], "checked": 0, "coverage": 0.0, "scale": 1, "cells": []}

    scale = find_scale([cc[a] for a, _ in pairs], [tc[b] for _, b in pairs]) if use_scale else 1
    biggest = max(abs(scale * tc[b]) for _, b in pairs) or 1.0

    xs, ys = [scale * tc[b] for _, b in pairs], [cc[a] for a, _ in pairs]
    fa, fb = theil_sen(xs, ys) if calibrate and len(pairs) >= min_cells_rel else (1.0, 0.0)
    gaps = [abs(c - (fa * t + fb)) / max(abs(t), floor * biggest, 1e-12) for t, c in zip(xs, ys)]
    noise = sorted(gaps)[len(gaps) // 2] if rel and len(gaps) >= min_cells_rel else 0.0

    cells, differences, unreadable = [], [], 0
    for (a, b), gap in zip(pairs, gaps):
        c, t = cc[a], scale * tc[b]
        ref = max(abs(t), floor * biggest, 1e-12)
        s = spread.get(a, 0.0) / ref
        tol = max(tau, k * s, (rel or 0) * noise)
        flagged = gap > tol
        readable = s_max is None or s <= s_max
        cell = {"row": b[0], "series": b[1], "chart": round(c, 6), "table": tc[b],
                "error": round(gap, 4), "spread": round(s, 4), "tol": round(tol, 4),
                "flag": bool(flagged and readable), "readable": readable}
        cells.append(cell)
        if not readable:
            unreadable += 1
        elif flagged:
            label = b[0] if len(table.series) == 1 else f"{b[0]} / {b[1]}"
            differences.append({"label": label, "chart": round(c, 4), "table": tc[b],
                                "error": round(gap, 4)})

    checked = len(cells)
    coverage = checked / max(1, n_table)
    unit = "" if scale == 1 else f" The table seems to use another unit (chart = {scale:g} x table)."

    if differences:
        verdict = "mismatch"
        note = (f"{len(differences)} of {checked} values differ by more than the allowed gap."
                f" Please check by hand.{unit}")
    elif unreadable or coverage < min_coverage:
        verdict = "not_sure"
        why = []
        if unreadable:
            why.append(f"{unreadable} values are hard to read on the chart")
        if coverage < min_coverage:
            why.append(f"only {checked} of {n_table} table values were found on the chart")
        note = "Not sure: " + " and ".join(why) + "." + unit
    else:
        verdict = "match"
        note = f"All {checked} values agree within the allowed gap ({tau:.0%} or more).{unit}"

    return {"verdict": verdict, "note": note, "differences": differences, "checked": checked,
            "coverage": round(coverage, 3), "scale": scale, "cells": cells}
