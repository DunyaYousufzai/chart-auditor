"""Re-render DaTikZ charts from their TikZ code at a higher resolution.

DaTikZ ships every figure as a 420x420 px image. We have the LaTeX code of each one, so we
compile it again (pdflatex, no shell escape), crop the page to the figure (pdfcrop) and
rasterise it with the longest side = --size px. Tables are not touched: the new manifest
points to the same tables, so old and new readings are compared on exactly the same charts.

    python src/rerender.py --manifest data/datikz_pgf/test/manifest.jsonl --figs data/datikz_pgf/figs \
        --split-out test_hr --size 1260 --jobs 16

Writes data/datikz_pgf/<split-out>/images/*.png and data/datikz_pgf/<split-out>/manifest.jsonl
(only the charts that compiled). Needs pdflatex, pdfcrop and pdftoppm (texlive + poppler).
"""
import argparse
import json
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def variants(code):
    """LaTeX sources to try, in order: standalone class (tight box around the figure, like the DaTikZ
    images), then the original class with page numbers and headers turned off (some preambles
    need their own class)."""
    m = re.search(r"\\documentclass(\[[^\]]*\])?\{[^}]*\}", code)
    out = []
    if m:
        out.append(code[:m.start()] + r"\documentclass[border=4pt]{standalone}" + code[m.end():])
    out.append(code.replace(r"\begin{document}",
                            r"\AtBeginDocument{\pagestyle{empty}\thispagestyle{empty}}" "\n" r"\begin{document}", 1))
    return out


def render(job):
    tex, out_png, size = job
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            for src in variants(Path(tex).read_text(errors="ignore")):
                (tmp / "fig.tex").write_text(src)
                (tmp / "fig.pdf").unlink(missing_ok=True)
                r = subprocess.run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
                                    "fig.tex"], cwd=tmp, capture_output=True, timeout=120)
                if r.returncode == 0 and (tmp / "fig.pdf").exists():
                    break
            else:
                return False
            subprocess.run(["pdfcrop", "--margins", "4", "fig.pdf", "crop.pdf"], cwd=tmp, capture_output=True,
                           timeout=60)
            pdf = "crop.pdf" if (tmp / "crop.pdf").exists() else "fig.pdf"
            subprocess.run(["pdftoppm", "-png", "-f", "1", "-l", "1", "-scale-to", str(size), "-singlefile",
                            pdf, "chart"], cwd=tmp, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False
        if not (tmp / "chart.png").exists():
            return False
        shutil.copy(tmp / "chart.png", out_png)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--figs", default="data/datikz_pgf/figs")
    ap.add_argument("--split-out", required=True)
    ap.add_argument("--size", type=int, default=1260, help="longest side in px (DaTikZ images are 420)")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    root = manifest.parent.parent
    out = root / args.split_out
    (out / "images").mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in manifest.open()]
    jobs = []
    for r in recs:
        fig = Path(args.figs) / r["id"].removeprefix("arxiv_") / "fig.tex"
        jobs.append((str(fig), str(out / "images" / f"{r['id']}.png"), args.size))
    ok = 0
    with ProcessPoolExecutor(args.jobs) as ex, open(out / "manifest.jsonl", "w") as man:
        for k, (r, done) in enumerate(zip(recs, ex.map(render, jobs))):
            if done:
                ok += 1
                man.write(json.dumps({**r, "image": f"{args.split_out}/images/{r['id']}.png"}) + "\n")
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(recs)} done, {ok} rendered", flush=True)
    print(f"{ok}/{len(recs)} charts re-rendered to {out}")


if __name__ == "__main__":
    main()
