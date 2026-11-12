"""Export the PRISMM-Bench figure-vs-table cases with a numeric chart/table conflict.

Of 384 reviewer-flagged inconsistencies, 10 are "figure-table" and only these 3 are a
number in a chart that disagrees with a table (the others are qualitative: images,
environments, iteration counts). We use them as a case study, not as a benchmark.

    python src/prismm_cases.py --parquet data/prismm_raw/test.parquet --out data/prismm_cases

Writes, per case: <uid>/part0.png, part1.png, meta.json (description, which part is which),
plus a manifest.jsonl that run_reader.py can read (one line per image, task "read").
"""
import argparse
import json
from pathlib import Path

NUMERIC_CASES = ["8TyGCAuCGd_0", "yEnJvc7ogD_1", "CfXRcN4iUw_0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/prismm_raw/test.parquet")
    ap.add_argument("--out", default="data/prismm_cases")
    args = ap.parse_args()
    import pyarrow.parquet as pq

    rows = pq.read_table(args.parquet, columns=["uid", "category", "visual_elements", "parts", "description",
                                                "review_text"]).to_pylist()
    out = Path(args.out)
    (out / "cases" / "images").mkdir(parents=True, exist_ok=True)
    with open(out / "cases" / "manifest.jsonl", "w") as man:
        for r in rows:
            if r["uid"] not in NUMERIC_CASES:
                continue
            parts = r["parts"]
            for k, img in enumerate(parts["content_image"]):
                if not img:
                    continue
                name = f"{r['uid']}_part{k}"
                (out / "cases" / "images" / f"{name}.png").write_bytes(img["bytes"])
                man.write(json.dumps({"id": name, "case": r["uid"], "image": f"cases/images/{name}.png",
                                      "element": r["visual_elements"][k] if k < len(r["visual_elements"]) else None,
                                      "page": parts["page"][k]}) + "\n")
            (out / "cases" / f"{r['uid']}.json").write_text(json.dumps(
                {"uid": r["uid"], "elements": r["visual_elements"], "description": r["description"],
                 "review": r["review_text"]}, indent=2))
            print(r["uid"], r["visual_elements"], "-", r["description"][:120])


if __name__ == "__main__":
    main()
