"""Collect outputs/results/*.json into one markdown table (for the report).

    python src/summarize.py outputs/results > outputs/results/summary.md
"""
import json
import sys
from pathlib import Path


def f(x):
    return "" if x is None else f"{x:.3f}"


def main(folder):
    rows = []
    for p in sorted(Path(folder).glob("*.json")):
        r = json.loads(p.read_text())
        base = {"file": p.stem, "model": r["model"].split("/")[-1], "N": r.get("n_use"),
                "s/chart": r.get("seconds_per_chart")}
        read = r.get("reading", {}).get("median_of_n", {})
        for key in [k for k in r if k.startswith("detection")]:
            d = r[key]
            rows.append({**base, "setting": key.replace("detection_", ""),
                         "cell_acc": read.get("cell_acc"), "rnss": read.get("rnss"),
                         "P": d["precision"], "R": d["recall"], "F1": d["f1"],
                         "not_sure": d["not_sure_rate"]})
    cols = ["file", "model", "N", "setting", "cell_acc", "rnss", "P", "R", "F1", "not_sure", "s/chart"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/results")
