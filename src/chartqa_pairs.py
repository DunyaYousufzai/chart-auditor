"""Build chart/table pairs from ChartQA with a known answer ("semi-synthetic real" set).

ChartQA gives real charts (Statista, Pew, OECD, OWID) together with the table they were
drawn from. We keep that table as the chart's truth and make a "paper table" from it
with perturb.py: unchanged (match) or changed on purpose (mismatch). The picture is
never touched, so the reader faces real chart styles.

Input : the unzipped "ChartQA Dataset" folder (train/val/test with png/, tables/, annotations/)
Output: same layout as the synthetic set, so run_reader.py and evaluate.py work unchanged:
    out/<split>/images/*.png, tables/*.csv, paper_tables/*.csv, manifest.jsonl

    python src/chartqa_pairs.py --src "data/chartqa_raw/ChartQA Dataset" --out data/chartqa_pairs \
        --split test --limit 500
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from perturb import make_paper_table
from tables import from_csv

KEEP_TYPES = {"v_bar", "h_bar", "line", "bar", "vbar", "hbar"}


def chart_type(ann_path):
    """ChartQA annotations have a "type" field (v_bar, h_bar, line, pie). None if unknown."""
    if not ann_path.exists():
        return None
    try:
        return str(json.loads(ann_path.read_text()).get("type", "")).lower() or None
    except (json.JSONDecodeError, AttributeError):
        return None


def decimals(table):
    d = 0
    for v in table.cells().values():
        s = f"{v:g}"
        if "." in s and "e" not in s:
            d = max(d, len(s.split(".")[1]))
    return min(d, 3)


def usable(table, max_rows=15, max_series=4):
    if table is None or not table.series or len(table.series) > max_series:
        return False
    if not 2 <= len(table.rows) <= max_rows:
        return False
    n_cells = len(table.rows) * len(table.series)
    return len(table.cells()) == n_cells and table.max_abs() > 0  # no empty or text cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/chartqa_raw/ChartQA Dataset")
    ap.add_argument("--out", default="data/chartqa_pairs")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="max pairs to keep")
    ap.add_argument("--p-match", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src) / args.split
    out = Path(args.out) / args.split
    for d in ("images", "tables", "paper_tables"):
        (out / d).mkdir(parents=True, exist_ok=True)

    pngs = sorted((src / "png").glob("*.png"))
    random.Random(args.seed).shuffle(pngs)
    kept, skipped = 0, {"no_table": 0, "type": 0, "shape": 0}
    with open(out / "manifest.jsonl", "w") as manifest:
        for png in pngs:
            if args.limit and kept >= args.limit:
                break
            tab = src / "tables" / (png.stem + ".csv")
            if not tab.exists():
                skipped["no_table"] += 1
                continue
            ctype = chart_type(src / "annotations" / (png.stem + ".json"))
            if ctype is not None and ctype not in KEEP_TYPES:
                skipped["type"] += 1
                continue
            try:
                table = from_csv(tab.read_text(encoding="utf-8-sig"))
            except Exception:
                table = None
            if not usable(table):
                skipped["shape"] += 1
                continue

            sid = f"chartqa_{args.split}_{png.stem}"
            rng = random.Random(f"{args.seed}-{sid}")
            dec = decimals(table)
            paper, info = make_paper_table(table, rng, p_match=args.p_match, dec=dec)

            shutil.copy(png, out / "images" / f"{sid}.png")
            (out / "tables" / f"{sid}.csv").write_text(table.to_csv())
            (out / "paper_tables" / f"{sid}.csv").write_text(paper.to_csv())
            source = "pew" if png.stem.isdigit() else "statista"  # ChartQA names Pew charts by number
            manifest.write(json.dumps({
                "id": sid, "source": f"chartqa_{source}", "split": args.split,
                "chart_type": ctype or "unknown", "n_points": len(table.rows),
                "n_series": len(table.series), "value_labels": None,
                "image": f"{args.split}/images/{sid}.png",
                "table": f"{args.split}/tables/{sid}.csv",
                "paper_table": f"{args.split}/paper_tables/{sid}.csv",
                "label": info["label"], "kind": info["kind"], "size": info["size"],
                "scale": info["scale"], "changed": info["changed"],
                "target": table.to_csv(),
            }) + "\n")
            kept += 1
            if kept % 100 == 0:
                print(f"  {args.split}: {kept} pairs", flush=True)
    print(f"{args.split}: kept {kept} pairs, skipped {skipped}")


if __name__ == "__main__":
    main()
