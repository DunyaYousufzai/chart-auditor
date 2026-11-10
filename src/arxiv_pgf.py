"""Real paper charts with exact data: pgfplots figures from arXiv LaTeX sources.

Many papers draw their bar/line charts with pgfplots and write the numbers in the
source (\\addplot coordinates {...} or \\addplot table {...}). We compile each such figure
on its own, so we get the chart exactly as it looks in the paper, plus its true table.
Then perturb.py makes match / mismatch paper tables, like for ChartQA.

Steps (run on UBELIX; "compile" needs `module load texlive`):
    python src/arxiv_pgf.py fetch   --out data/arxiv_raw --from 2026-01-01 --max 1500
    python src/arxiv_pgf.py extract --raw data/arxiv_raw --out data/arxiv_pgf/figs
    python src/arxiv_pgf.py compile --figs data/arxiv_pgf/figs
    python src/arxiv_pgf.py pairs   --figs data/arxiv_pgf/figs --out data/arxiv_pgf

Safety: sources are untrusted. We only unpack .tex/.dat/.csv/.txt/.sty files with safe
paths, and compile with -no-shell-escape in a scratch folder.
"""
import argparse
import gzip
import io
import json
import random
import re
import shutil
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from perturb import make_paper_table
from tables import Table

UA = "chart-auditor-course-project/0.1 (University of Neuchatel)"
CATS = ["cs.LG", "cs.CL", "cs.CV", "cs.AI", "stat.ML", "cs.NE", "cs.RO", "eess.SP"]
KEEP_EXT = {".tex", ".dat", ".csv", ".txt", ".sty", ".tsv"}


# ---------------------------------------------------------------- fetch
def arxiv_ids(cat, start, max_results, date_from):
    q = urllib.parse.urlencode({"search_query": f"cat:{cat}", "sortBy": "submittedDate",
                                "sortOrder": "descending", "start": start, "max_results": max_results})
    req = urllib.request.Request(f"http://export.arxiv.org/api/query?{q}", headers={"User-Agent": UA})
    xml = urllib.request.urlopen(req, timeout=60).read()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(xml).findall("a:entry", ns):
        pid = e.find("a:id", ns).text.rsplit("/abs/", 1)[-1]
        published = e.find("a:published", ns).text[:10]
        if published >= date_from:
            out.append((re.sub(r"v\d+$", "", pid), published))
    return out


def safe_unpack(data, dest):
    """Unpack an arXiv e-print (tar.gz, gz or plain tex) keeping only text files with safe names."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            for m in tar.getmembers():
                p = Path(m.name)
                if not m.isfile() or p.is_absolute() or ".." in p.parts or p.suffix.lower() not in KEEP_EXT:
                    continue
                if m.size > 5_000_000:
                    continue
                target = dest / p
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(m).read())
                n += 1
        return n
    except tarfile.TarError:
        pass
    try:  # a single gzipped .tex file
        text = gzip.decompress(data)
    except OSError:
        text = data
    if b"\\documentclass" in text[:20000] or b"\\begin{document}" in text:
        (dest / "main.tex").write_bytes(text)
        return 1
    return 0


def cmd_fetch(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.jsonl"
    done = {json.loads(l)["id"] for l in index.open()} if index.exists() else set()
    ids = []
    for cat in CATS:
        for start in range(0, args.per_cat, 200):
            ids += arxiv_ids(cat, start, 200, args.date_from)
            time.sleep(3)
    seen, todo = set(), []
    for pid, pub in ids:
        if pid not in seen and pid not in done:
            seen.add(pid)
            todo.append((pid, pub))
    todo = todo[:args.max]
    print(f"{len(todo)} papers to fetch ({len(done)} already)", flush=True)
    with index.open("a") as f:
        for k, (pid, pub) in enumerate(todo):
            try:
                req = urllib.request.Request(f"https://arxiv.org/e-print/{pid}", headers={"User-Agent": UA})
                data = urllib.request.urlopen(req, timeout=120).read()
                n = safe_unpack(data, out / pid.replace("/", "_"))
                status = "ok" if n else "no_source"
            except Exception as e:  # network errors, 403, pdf-only papers
                n, status = 0, f"error:{type(e).__name__}"
            f.write(json.dumps({"id": pid, "published": pub, "files": n, "status": status}) + "\n")
            f.flush()
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(todo)}", flush=True)
            time.sleep(3)  # arXiv asks for one request every 3 seconds


# ---------------------------------------------------------------- extract
def balanced(text, i, open_ch, close_ch):
    """text[i] == open_ch; return index just after the matching close_ch (or -1)."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == open_ch:
            depth += 1
        elif text[j] == close_ch:
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def strip_comments(tex):
    return re.sub(r"(?<!\\)%.*", "", tex)


def option_list(opts, key):
    """Value of key={a,b,c} inside an options string, as a list, or None."""
    m = re.search(re.escape(key) + r"\s*=\s*\{", opts)
    if not m:
        return None
    end = balanced(opts, m.end() - 1, "{", "}")
    body = opts[m.end():end - 1]
    items, depth, cur = [], 0, ""
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        items.append(cur.strip())
    return [clean_label(x) for x in items]


def clean_label(s):
    s = s.strip().strip("{}").strip()
    s = re.sub(r"\\(textbf|textit|emph|text|mathrm|textsc|texttt)\{([^{}]*)\}", r"\2", s)
    s = s.replace("\\_", "_").replace("\\%", "%").replace("\\&", "&").replace("$", "").replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


def to_float(s):
    try:
        return float(s.strip().replace("{", "").replace("}", ""))
    except ValueError:
        return None


def parse_coordinates(body, horizontal):
    pts = []
    for a, b in re.findall(r"\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)", body):
        if horizontal:
            a, b = b, a
        y = to_float(b)
        if y is None:
            return None
        pts.append((clean_label(a), y))
    return pts


def parse_inline_table(body, opts, src_dir):
    body = body.strip()
    if "\n" not in body and not re.search(r"\s\d", body):  # {file.dat}
        f = (src_dir / body).resolve()
        if not f.is_file() or src_dir.resolve() not in f.parents:
            return None
        body = f.read_text(errors="ignore")
    rows = [r.split() if "," not in r else [c.strip() for c in r.split(",")]
            for r in body.replace("\\\\", "\n").splitlines() if r.strip() and not r.strip().startswith("#")]
    if len(rows) < 2:
        return None
    header = rows[0] if to_float(rows[0][-1]) is None else None
    data = rows[1:] if header else rows
    xi, yi = 0, 1
    if header:
        mx = re.search(r"\bx\s*=\s*\{?([^,\]}]+)", opts)
        my = re.search(r"\by\s*=\s*\{?([^,\]}]+)", opts)
        if mx and mx.group(1).strip() in header:
            xi = header.index(mx.group(1).strip())
        if my and my.group(1).strip() in header:
            yi = header.index(my.group(1).strip())
    pts = []
    for r in data:
        if max(xi, yi) >= len(r):
            return None
        y = to_float(r[yi])
        if y is None:
            return None
        pts.append((clean_label(r[xi]), y))
    return pts


PLOT_RE = re.compile(r"\\addplot\+?\s*(\[[^\]]*\])?\s*(coordinates|table)\s*(\[[^\]]*\])?\s*\{", re.S)


def extract_figures(tex, src_dir):
    """Yield dicts {tikz, table, chart_type} for pgfplots bar/line charts with inline data."""
    tex = strip_comments(tex)
    for m in re.finditer(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", tex, re.S):
        block = m.group(0)
        if block.count("\\begin{axis}") != 1:  # one plain linear axis (no log axes, no groupplots)
            continue
        a = block.index("\\begin{axis}") + len("\\begin{axis}")
        opts = ""
        if block[a:a + 1] == "[":
            e = balanced(block, a, "[", "]")
            opts = block[a + 1:e - 1]
        if any(k in opts for k in ("polar", "3d", "view=", "colorbar")):
            continue
        horizontal = "xbar" in opts
        bar = horizontal or "ybar" in opts
        series = []
        legend = option_list(opts, "legend entries")
        for pm in PLOT_RE.finditer(block):
            popts = (pm.group(1) or "") + (pm.group(3) or "")
            if any(k in popts for k in ("only marks", "fill between", "forget plot", "error bars")):
                continue
            end = balanced(block, pm.end() - 1, "{", "}")
            body = block[pm.end():end - 1]
            if pm.group(2) == "coordinates":
                pts = parse_coordinates(body, horizontal or "xbar" in popts)
            else:
                pts = parse_inline_table(body, popts, src_dir)
            if not pts:
                continue
            after = block[end:end + 300]
            lm = re.match(r"\s*;?\s*\\addlegendentry\s*\{", after)
            name = None
            if lm:
                le = balanced(after, lm.end() - 1, "{", "}")
                name = clean_label(after[lm.end():le - 1])
            series.append((name, pts))
            bar = bar or "ybar" in popts or "xbar" in popts
        if not 1 <= len(series) <= 4:
            continue
        xs = [x for x, _ in series[0][1]]
        if not 2 <= len(xs) <= 15 or len(set(xs)) != len(xs):
            continue
        if any([x for x, _ in pts] != xs for _, pts in series):
            continue
        ticks = option_list(opts, "xticklabels") if not horizontal else option_list(opts, "yticklabels")
        rows = ticks if ticks and len(ticks) == len(xs) else xs
        names = [n or (legend[k] if legend and k < len(legend) else f"series_{k + 1}")
                 for k, (n, _) in enumerate(series)]
        if len(set(names)) != len(names):
            continue
        values = [[pts[i][1] for _, pts in series] for i in range(len(xs))]
        mx = re.search(r"\b(y|x)label\s*=\s*\{([^{}]*)\}", opts)
        x_name = clean_label(mx.group(2)) if mx and mx.group(1) == ("y" if horizontal else "x") else "x"
        yield {"tikz": block, "chart_type": ("h_bar" if horizontal else "v_bar") if bar else "line",
               "table": Table(x_name or "x", names, rows, values).to_csv(),
               "value_labels": "nodes near coords" in block}


def preamble_lines(tex):
    pre = tex.split("\\begin{document}")[0]
    keep = []
    for line in strip_comments(pre).splitlines():
        s = line.strip()
        if s.startswith(("\\usetikzlibrary", "\\usepgfplotslibrary", "\\definecolor", "\\colorlet")):
            keep.append(s)
        elif s.startswith("\\pgfplotsset") and s.count("{") == s.count("}"):
            keep.append(s)
    return keep


def cmd_extract(args):
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_papers = n_figs = 0
    for paper in sorted(p for p in raw.iterdir() if p.is_dir()):
        texs = list(paper.rglob("*.tex"))
        if not texs:
            continue
        pre = []
        for t in texs:
            txt = t.read_text(errors="ignore")
            if "\\begin{document}" in txt:
                pre += preamble_lines(txt)
        found = 0
        for t in texs:
            txt = t.read_text(errors="ignore")
            if "\\addplot" not in txt:
                continue
            for fig in extract_figures(txt, t.parent):
                fid = f"{paper.name}_{found:02d}"
                d = out / fid
                d.mkdir(exist_ok=True)
                (d / "table.csv").write_text(fig["table"])
                (d / "fig.tex").write_text(
                    "\\documentclass[border=4pt]{standalone}\n\\usepackage[T1]{fontenc}\n"
                    "\\usepackage{amsmath,amssymb,xcolor}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n"
                    "\\usetikzlibrary{patterns,arrows.meta,calc,positioning}\n"
                    + "\n".join(dict.fromkeys(pre)) + "\n\\begin{document}\n" + fig["tikz"] + "\n\\end{document}\n")
                (d / "meta.json").write_text(json.dumps({"paper": paper.name, "source": str(t.relative_to(raw)),
                                                         "chart_type": fig["chart_type"],
                                                         "value_labels": fig["value_labels"]}))
                for f in paper.rglob("*"):  # data files the figure may read
                    if f.suffix.lower() in {".dat", ".csv", ".tsv", ".txt"} and f.is_file():
                        shutil.copy(f, d / f.name)
                found += 1
        n_papers += bool(found)
        n_figs += found
    print(f"{n_figs} candidate figures from {n_papers} papers")


# ---------------------------------------------------------------- compile
def cmd_compile(args):
    figs = sorted(p for p in Path(args.figs).iterdir() if (p / "fig.tex").exists())
    ok = 0
    for k, d in enumerate(figs):
        if (d / "chart.png").exists():
            ok += 1
            continue
        try:
            r = subprocess.run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"],
                               cwd=d, capture_output=True, timeout=90)
            if r.returncode == 0 and (d / "fig.pdf").exists():
                subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", "fig.pdf", "chart"],
                               cwd=d, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            pass
        ok += (d / "chart.png").exists()
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(figs)} compiled, {ok} ok", flush=True)
    print(f"{ok}/{len(figs)} figures compiled to PNG")


# ---------------------------------------------------------------- pairs
def cmd_pairs(args):
    from tables import from_csv
    figs = sorted(p for p in Path(args.figs).iterdir() if (p / "chart.png").exists())
    out = Path(args.out) / args.split
    for sub in ("images", "tables", "paper_tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(out / "manifest.jsonl", "w") as man:
        for d in figs:
            table = from_csv((d / "table.csv").read_text())
            if not table.cells() or table.max_abs() == 0:
                continue
            meta = json.loads((d / "meta.json").read_text())
            sid = f"arxiv_{d.name}"
            rng = random.Random(f"{args.seed}-{sid}")
            dec = max((len(f"{v:g}".split(".")[1]) if "." in f"{v:g}" else 0) for v in table.cells().values())
            paper, info = make_paper_table(table, rng, p_match=0.5, dec=min(dec, 3))
            shutil.copy(d / "chart.png", out / "images" / f"{sid}.png")
            (out / "tables" / f"{sid}.csv").write_text(table.to_csv())
            (out / "paper_tables" / f"{sid}.csv").write_text(paper.to_csv())
            man.write(json.dumps({
                "id": sid, "source": "arxiv_pgf", "paper": meta["paper"], "split": args.split,
                "chart_type": meta["chart_type"], "n_points": len(table.rows), "n_series": len(table.series),
                "value_labels": meta["value_labels"],
                "image": f"{args.split}/images/{sid}.png", "table": f"{args.split}/tables/{sid}.csv",
                "paper_table": f"{args.split}/paper_tables/{sid}.csv",
                "label": info["label"], "kind": info["kind"], "size": info["size"],
                "scale": info["scale"], "changed": info["changed"], "target": table.to_csv()}) + "\n")
            kept += 1
    print(f"{kept} pairs written to {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", default="data/arxiv_raw")
    f.add_argument("--from", dest="date_from", default="2026-01-01")
    f.add_argument("--per-cat", type=int, default=400, help="listing size per category")
    f.add_argument("--max", type=int, default=1500)
    e = sub.add_parser("extract")
    e.add_argument("--raw", default="data/arxiv_raw")
    e.add_argument("--out", default="data/arxiv_pgf/figs")
    c = sub.add_parser("compile")
    c.add_argument("--figs", default="data/arxiv_pgf/figs")
    p = sub.add_parser("pairs")
    p.add_argument("--figs", default="data/arxiv_pgf/figs")
    p.add_argument("--out", default="data/arxiv_pgf")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"fetch": cmd_fetch, "extract": cmd_extract, "compile": cmd_compile, "pairs": cmd_pairs}[args.cmd](args)


if __name__ == "__main__":
    main()
