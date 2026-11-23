"""Copy a subset of the DaTikZ benchmark (v1 and v2) into a small folder that can live in git.

Used for frontier_eval/: 200 test charts and 100 val charts, so API models can be compared with ours
on exactly the same charts and checks without the full data folder.

    python src/pack_subset.py --src data/datikz_pgf --out frontier_eval/data \
        --test-ids frontier_eval/test_ids.txt --val-ids frontier_eval/val_ids.txt

The folder keeps the same layout as data/datikz_pgf, so every script works on it unchanged:
  test_hr/manifest.jsonl  test_hr/images/  test/tables/  test/paper_tables/  test_v2/manifest.jsonl ...
"""
import argparse
import json
import shutil
from pathlib import Path


def pack(src, out, split, ids):
    ids = set(ids)
    files = set()
    for man, key in ((f"{split}_hr/manifest.jsonl", "id"), (f"{split}_v2/manifest.jsonl", "chart_id")):
        recs = [json.loads(l) for l in open(src / man)]
        keep = [r for r in recs if r[key] in ids]
        (out / man).parent.mkdir(parents=True, exist_ok=True)
        with open(out / man, "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
                files.update(r[k] for k in ("image", "table", "paper_table") if r.get(k))
        print(f"{man}: {len(keep)} records")
    for rel in sorted(files):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, out / rel)
    print(f"{split}: {len(files)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/datikz_pgf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-ids", required=True)
    ap.add_argument("--val-ids", required=True)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    pack(src, out, "test", Path(args.test_ids).read_text().split())
    pack(src, out, "val", Path(args.val_ids).read_text().split())


if __name__ == "__main__":
    main()
