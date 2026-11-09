"""Make "paper tables" that agree or disagree with a chart, with a known answer.

We always change the TABLE, never the picture. So the same code works for our
synthetic charts and for real charts (ChartQA, arXiv) whose true table we know.

Kinds of example:
  match      the table is the chart's true table
  rescaled   still a match: the table uses another unit (e.g. thousands) and says so
             in its header. Tests that the checker finds the unit scale.
  value      one cell changed by a given size (2%, 5%, 10%, 25%, 50%)
  swap       the values of two rows are swapped (a label mix-up)
  decimal    one cell multiplied or divided by 10 (a decimal slip)

We do NOT make "drop a row" examples: in real papers a chart often shows only part
of a table, so a missing row is not clearly an error.
"""
import random

SIZES = [0.02, 0.05, 0.10, 0.25, 0.50]
RESCALE = [(1000, "thousands"), (1_000_000, "millions")]


def _round(v, dec):
    return round(v, dec) if dec is not None else v


def change_value(table, rng, size, dec=None):
    t = table.copy()
    cells = [(i, j) for i in range(len(t.rows)) for j in range(len(t.series))
             if t.values[i][j] not in (None, 0)]
    i, j = rng.choice(cells)
    old = t.values[i][j]
    sign = rng.choice([-1, 1])
    new = _round(old * (1 + sign * size), dec)
    if new == old or (old > 0 and new <= 0):  # rounding ate the change, or flipped sign
        new = _round(old * (1 + size), dec)
    if dec is not None and new == old:        # tiny numbers: move by one last digit
        new = round(old + 10 ** (-dec), dec)
    t.values[i][j] = new
    return t, [{"row": t.rows[i], "series": t.series[j], "true": old, "table": new}]


def swap_rows(table, rng, min_gap=0.05):
    """Swap two values of the same series that differ enough to matter."""
    t = table.copy()
    options = []
    for j in range(len(t.series)):
        for a in range(len(t.rows)):
            for b in range(a + 1, len(t.rows)):
                va, vb = t.values[a][j], t.values[b][j]
                if va is None or vb is None:
                    continue
                if abs(va - vb) / max(abs(va), abs(vb), 1e-9) >= min_gap:
                    options.append((a, b, j))
    if not options:
        return None, []
    a, b, j = rng.choice(options)
    va, vb = t.values[a][j], t.values[b][j]
    t.values[a][j], t.values[b][j] = vb, va
    return t, [{"row": t.rows[a], "series": t.series[j], "true": va, "table": vb},
               {"row": t.rows[b], "series": t.series[j], "true": vb, "table": va}]


def decimal_slip(table, rng, dec=None):
    t = table.copy()
    cells = [(i, j) for i in range(len(t.rows)) for j in range(len(t.series))
             if t.values[i][j] not in (None, 0)]
    i, j = rng.choice(cells)
    old = t.values[i][j]
    factor = rng.choice([10, 0.1])
    d = None if dec is None else (dec + 1 if factor == 0.1 else dec)
    new = _round(old * factor, d)
    t.values[i][j] = new
    return t, [{"row": t.rows[i], "series": t.series[j], "true": old, "table": new}]


def rescale(table, rng):
    """Same numbers in another unit, with the unit written in the header.

    Only for big numbers (largest value >= 1000), as in real tables.
    Returns (None, 1) when the table has no big numbers.
    """
    options = [(f, w) for f, w in RESCALE if table.max_abs() >= f]
    if not options:
        return None, 1
    t = table.copy()
    factor, word = rng.choice(options)
    t.values = [[None if v is None else v / factor for v in r] for r in t.values]
    t.series = [f"{s} ({word})" for s in t.series]
    return t, factor


def make_paper_table(table, rng, p_match=0.5, p_rescaled=0.2, dec=None):
    """Pick a kind at random and return (paper_table, info).

    info = {"label": "match" | "mismatch", "kind": ..., "size": ..., "changed": [...]}
    Half matches, half mismatches by default. Mismatches: 70% value, 15% swap, 15% decimal.
    """
    if rng.random() < p_match:
        if rng.random() < p_rescaled:
            t, factor = rescale(table, rng)
            if t is not None:
                return t, {"label": "match", "kind": "rescaled", "size": None,
                           "scale": factor, "changed": []}
        return table.copy(), {"label": "match", "kind": "match", "size": None,
                              "scale": 1, "changed": []}

    r = rng.random()
    if r < 0.15:
        t, changed = swap_rows(table, rng)
        if t is not None:
            return t, {"label": "mismatch", "kind": "swap", "size": None,
                       "scale": 1, "changed": changed}
    elif r < 0.30:
        t, changed = decimal_slip(table, rng, dec)
        return t, {"label": "mismatch", "kind": "decimal", "size": None,
                   "scale": 1, "changed": changed}

    size = rng.choice(SIZES)
    t, changed = change_value(table, rng, size, dec)
    return t, {"label": "mismatch", "kind": "value", "size": size,
               "scale": 1, "changed": changed}


if __name__ == "__main__":
    from tables import Table
    demo = Table("Model", ["Acc"], ["A", "B", "C"], [[80.0], [72.0], [65.0]])
    rng = random.Random(0)
    for _ in range(6):
        t, info = make_paper_table(demo, rng, dec=1)
        print(info["kind"], info["size"], t.to_csv().replace("\n", "  "))
