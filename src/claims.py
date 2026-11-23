"""Claims a paper's text makes about a chart, and how to check them against a reading.

PRISMM-Bench shows that most real chart errors flagged by reviewers are not a wrong table cell but a
sentence that the chart contradicts ("the text says POGM uses less GPU memory, the figure shows the
opposite"). We make such sentences from the chart's true data, true and false ones, and check them
against what the reader got from the picture.

Kinds
  value     "At x = 10, Accuracy is 82.4."             false: the number is off by 5-25 %
  max       "Accuracy is highest at x = 10."           false: a point clearly below the maximum
  trend     "Accuracy increases as x grows."           false: the opposite direction
  above     "A is above B for every x."                 false: B is clearly above A somewhere
  above_at  "At x = 10, A is higher than B."           false: the order at that point is reversed

check(claim, reading, true_table, tau) -> "match" (supported), "mismatch" (contradicted) or
"not_sure" (the reading is missing the cells, or the evidence is inside the tolerance).
"""
from checker import align, num_label, orient
from tables import fmt_number

MIN_GAP = 0.10   # a false claim must be wrong by at least this much (relative), so it is a clear error


def _ref(v, biggest, floor=0.05):
    return max(abs(v), floor * biggest, 1e-12)


def _cols(table):
    return {s: {r: table.values[i][j] for i, r in enumerate(table.rows) if table.values[i][j] is not None}
            for j, s in enumerate(table.series)}


def _name(table, s):
    return s if len(table.series) > 1 else (s or "the value")


def _x(table, r):
    return f"{table.x_name} = {r}" if table.x_name else str(r)


def make_claims(table, rng):
    """True and false claims for one chart: [(claim, label)], label "match" (true) or "mismatch" (false)."""
    out = []
    cols = _cols(table)
    biggest = table.max_abs() or 1.0
    rows = list(table.rows)

    for s, col in cols.items():
        if len(col) < 3:
            continue
        # value
        r = rng.choice(list(col))
        v = col[r]
        out.append(({"type": "value", "series": s, "x": r, "v": v}, "match"))
        size = rng.choice([0.05, 0.10, 0.25]) * rng.choice([-1, 1])
        out.append(({"type": "value", "series": s, "x": r, "v": v + size * _ref(v, biggest), "size": abs(size)},
                     "mismatch"))
        # max: only when the maximum is clear (second best at least MIN_GAP below)
        ordered = sorted(col.items(), key=lambda kv: kv[1], reverse=True)
        (r_max, v_max), (_, v_2) = ordered[0], ordered[1]
        if (v_max - v_2) / _ref(v_max, biggest) >= MIN_GAP / 2:
            out.append(({"type": "max", "series": s, "x": r_max}, "match"))
            low = [r for r, v in col.items() if (v_max - v) / _ref(v_max, biggest) >= MIN_GAP]
            if low:
                out.append(({"type": "max", "series": s, "x": rng.choice(low)}, "mismatch"))
        # trend: numeric x in order and a clear overall change
        xs = [num_label(r) for r in rows if r in col]
        if all(x is not None for x in xs) and xs == sorted(xs):
            first, last = col[rows[0]] if rows[0] in col else None, col[rows[-1]] if rows[-1] in col else None
            if first is not None and last is not None and abs(last - first) / _ref(first, biggest) >= 0.2:
                d = "up" if last > first else "down"
                out.append(({"type": "trend", "series": s, "dir": d}, "match"))
                out.append(({"type": "trend", "series": s, "dir": "down" if d == "up" else "up"}, "mismatch"))

    names = list(cols)
    if len(names) >= 2:
        pairs = [(a, b) for a in names for b in names if a != b]
        rng.shuffle(pairs)
        for a, b in pairs[:4]:
            common = [r for r in rows if r in cols[a] and r in cols[b]]
            if len(common) < 2:
                continue
            gaps = [(cols[a][r] - cols[b][r]) / _ref(cols[b][r], biggest) for r in common]
            if min(gaps) >= MIN_GAP:
                out.append(({"type": "above", "a": a, "b": b}, "match"))
                out.append(({"type": "above", "a": b, "b": a}, "mismatch"))
            clear = [r for r, g in zip(common, gaps) if abs(g) >= MIN_GAP]
            if clear:
                r = rng.choice(clear)
                hi, lo = (a, b) if cols[a][r] > cols[b][r] else (b, a)
                out.append(({"type": "above_at", "a": hi, "b": lo, "x": r}, "match"))
                out.append(({"type": "above_at", "a": lo, "b": hi, "x": r}, "mismatch"))
    seen, unique = set(), []
    for c, label in out:
        c["text"] = text(c, table)
        if c["text"] not in seen:
            seen.add(c["text"])
            unique.append((c, label))
    return unique


def text(c, table):
    t = c["type"]
    if t == "value":
        return f"At {_x(table, c['x'])}, {_name(table, c['series'])} is {fmt_number(round(c['v'], 6))}."
    if t == "max":
        return f"{_name(table, c['series'])} is highest at {_x(table, c['x'])}."
    if t == "trend":
        word = "increases" if c["dir"] == "up" else "decreases"
        return f"{_name(table, c['series'])} {word} as {table.x_name or 'x'} grows."
    if t == "above":
        return f"{c['a']} is above {c['b']} for every {table.x_name or 'x'}."
    if t == "above_at":
        return f"At {_x(table, c['x'])}, {c['a']} is higher than {c['b']}."
    raise ValueError(t)


def map_reading(reading, true_table):
    """{(true row, true series): value read} using the checker's own alignment."""
    if reading is None:
        return {}
    reading = orient(reading, true_table)
    pairs, _, _ = align(reading, true_table)
    cells = reading.cells()
    return {b: cells[a] for a, b in pairs if a in cells}


def check(c, read, true_table, tau=0.05):
    """read = map_reading(...). Returns match / mismatch / not_sure."""
    vals = list(read.values())
    if not vals:
        return "not_sure"
    biggest = max(abs(v) for v in vals) or 1.0

    def get(r, s):
        return read.get((r, s))

    t = c["type"]
    if t == "value":
        v = get(c["x"], c["series"])
        if v is None:
            return "not_sure"
        return "mismatch" if abs(v - c["v"]) / _ref(c["v"], biggest) > tau else "match"

    if t == "max":
        col = {r: v for (r, s), v in read.items() if s == c["series"]}
        if c["x"] not in col or len(col) < 2:
            return "not_sure"
        top = max(col.values())
        return "mismatch" if (top - col[c["x"]]) / _ref(top, biggest) > tau else "match"

    if t == "trend":
        rows = [r for r in true_table.rows if get(r, c["series"]) is not None]
        if len(rows) < 2:
            return "not_sure"
        first, last = get(rows[0], c["series"]), get(rows[-1], c["series"])
        d = (last - first) / _ref(first, biggest)
        if abs(d) <= tau:
            return "not_sure"
        return "match" if (d > 0) == (c["dir"] == "up") else "mismatch"

    if t in ("above", "above_at"):
        rows = [c["x"]] if t == "above_at" else true_table.rows
        seen, worst = 0, 0.0
        for r in rows:
            va, vb = get(r, c["a"]), get(r, c["b"])
            if va is None or vb is None:
                continue
            seen += 1
            worst = max(worst, (vb - va) / _ref(vb, biggest))
        if not seen:
            return "not_sure"
        return "mismatch" if worst > tau else "match"
    raise ValueError(t)
