"""Real paper charts with exact data from DaTikZ-v3 (nllg/datikz-v3 on Hugging Face).

DaTikZ holds ~145k TikZ figures scraped from arXiv and other sources, each with its
LaTeX code and the rendered image. We keep pgfplots bar/line charts whose numbers are
written in the code (same parser as arxiv_pgf.py), so we get a real paper chart and its
exact table without compiling anything.

    python src/datikz_pgf.py --parquet-dir data/datikz_raw --out data/datikz_pgf/figs [--origin arxiv]
    python src/arxiv_pgf.py pairs --figs data/datikz_pgf/figs --out data/datikz_pgf

Only figures whose code holds exactly one tikzpicture are used, so the image is that chart.
"""
import argparse
import collections
import json
from pathlib import Path

from arxiv_pgf import extract_figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="data/datikz_raw")
    ap.add_argument("--out", default="data/datikz_pgf/figs")
    ap.add_argument("--origin", default=None, help="keep only this origin (e.g. arxiv); default: all")
    args = ap.parse_args()
    import pyarrow.parquet as pq

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen_origin, kept, scanned = collections.Counter(), collections.Counter(), 0
    for f in sorted(Path(args.parquet_dir).glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=500, columns=["code", "origin", "date", "uri", "image"]):
            for row in batch.to_pylist():
                scanned += 1
                origin = row["origin"] or "unknown"
                seen_origin[origin] += 1
                code = row["code"] or ""
                if args.origin and origin != args.origin:
                    continue
                if "\\addplot" not in code or code.count("\\begin{tikzpicture}") != 1:
                    continue
                figs = list(extract_figures(code, Path("/nonexistent")))
                if len(figs) != 1 or not row["image"]:
                    continue
                fig = figs[0]
                fid = f"datikz_{f.stem.split('-')[1]}_{scanned:06d}"
                d = out / fid
                d.mkdir(exist_ok=True)
                (d / "chart.png").write_bytes(row["image"]["bytes"])
                (d / "table.csv").write_text(fig["table"])
                (d / "fig.tex").write_text(code)
                (d / "meta.json").write_text(json.dumps({
                    "paper": row["uri"], "origin": origin, "date": str(row["date"])[:10],
                    "chart_type": fig["chart_type"], "value_labels": fig["value_labels"]}))
                kept[origin] += 1
        print(f"{f.name}: scanned {scanned}, kept {sum(kept.values())}", flush=True)
    print("origins seen:", dict(seen_origin))
    print("kept by origin:", dict(kept))


if __name__ == "__main__":
    main()
