"""One small table format shared by the generator, the readers and the checker.

A table looks like this (CSV or " | " separated):

    Model,CIFAR-10,ImageNet        <- header: x label name, then one column per series
    ResNet,93.1,76.2
    ViT,98.1,81.0

A chart with one series is the same with a single value column.
In Python we keep it as a Table: row labels, series names and a grid of numbers.
"""
import csv
import io
import re
from dataclasses import dataclass, field


@dataclass
class Table:
    x_name: str                 # name of the first column, e.g. "Model"
    series: list                # names of the value columns, e.g. ["CIFAR-10", "ImageNet"]
    rows: list                  # row labels, e.g. ["ResNet", "ViT"]
    values: list = field(default_factory=list)  # values[i][j] = row i, series j (float or None)

    def cells(self):
        """Dictionary {(row, series): value} without empty cells."""
        out = {}
        for i, r in enumerate(self.rows):
            for j, s in enumerate(self.series):
                v = self.values[i][j]
                if v is not None:
                    out[(r, s)] = v
        return out

    def max_abs(self):
        vals = [abs(v) for v in self.cells().values()]
        return max(vals) if vals else 0.0

    def copy(self):
        return Table(self.x_name, list(self.series), list(self.rows),
                     [list(r) for r in self.values])

    def to_csv(self, dec=None):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow([self.x_name] + self.series)
        for r, vals in zip(self.rows, self.values):
            w.writerow([r] + [fmt_number(v, dec) for v in vals])
        return buf.getvalue()


def fmt_number(v, dec=None):
    if v is None:
        return ""
    if dec is None:
        return f"{v:g}" if abs(v) < 1e15 else str(v)
    if dec == 0:
        return str(int(round(v)))
    return f"{v:.{dec}f}"


# ---------- Reading numbers written by people or models ----------
_NUM = re.compile(r"[-+−]?\d[\d,\s]*\.?\d*(?:[eE][-+]?\d+)?|[-+−]?\.\d+")


def parse_number(text):
    """'82.4%' -> 82.4, '1,234' -> 1234, '$5.2' -> 5.2, '—' -> None."""
    if text is None:
        return None
    t = str(text).strip().replace("−", "-")
    if not t:
        return None
    m = _NUM.search(t)
    if not m:
        return None
    s = m.group(0).replace(" ", "")
    # "1,234" or "1,234,567" = thousands separator; "3,5" = decimal comma
    if "," in s:
        if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", s):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def from_csv(text):
    """Read a table we wrote ourselves (clean CSV, header first)."""
    rows = list(csv.reader(io.StringIO(text.strip())))
    header, body = rows[0], rows[1:]
    return Table(
        x_name=header[0],
        series=header[1:],
        rows=[r[0] for r in body],
        values=[[parse_number(c) for c in r[1:len(header)]] +
                [None] * (len(header) - len(r)) for r in body],
    )


def parse_model_output(text):
    """Turn what a chart reader wrote into a Table, or None if it is unusable.

    Handles DePlot style ("TITLE | x <0x0A> Model | Acc <0x0A> A | 80"),
    pipe tables, markdown tables, and plain CSV, with or without ``` fences.
    """
    if not text or not text.strip():
        return None
    t = text.replace("<0x0A>", "\n").replace("\\n", "\n")
    t = re.sub(r"```[a-zA-Z]*", "", t)
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    # drop markdown separator lines like |---|---|
    lines = [l for l in lines if not re.fullmatch(r"[|\s:\-]+", l)]
    if not lines:
        return None

    sep = "|" if sum("|" in l for l in lines) >= len(lines) / 2 else ","
    if sep == "|":
        grid = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
    else:
        grid = [[c.strip() for c in r] for r in csv.reader(io.StringIO("\n".join(lines)))]

    # DePlot starts with "TITLE | ..."; drop that line
    if grid and grid[0] and grid[0][0].upper().startswith("TITLE"):
        grid = grid[1:]
    grid = [g for g in grid if len(g) >= 2]
    if not grid:
        return None

    def numeric_share(row):
        cells = row[1:]
        return sum(parse_number(c) is not None for c in cells) / max(1, len(cells))

    # header = first row whose value cells are mostly not numbers
    if numeric_share(grid[0]) < 0.5:
        header, body = grid[0], grid[1:]
    else:
        header, body = ["x"] + [f"series_{j}" for j in range(1, len(grid[0]))], grid
    width = len(header)
    body = [r for r in body if numeric_share(r) > 0]
    if not body:
        return None

    values = []
    for r in body:
        r = (r + [""] * width)[:width]
        values.append([parse_number(c) for c in r[1:]])
    return Table(x_name=header[0], series=header[1:],
                 rows=[r[0] for r in body], values=values)
