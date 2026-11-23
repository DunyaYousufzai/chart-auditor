"""Benchmark v2: more realistic errors on the same DaTikZ charts.

v1 gives each chart one paper table (match or one changed cell). v2 makes many checks per chart from
the chart's true data, with error kinds taken from real errata and reviewer-flagged errors
(docs/results/PROBLEM_VALIDATION.md). The pictures do not change, so a reader reads each chart once
and every check reuses that reading.

Table checks (the paper's table against the chart)
  match / rescaled               no error
  value 2-50 %, swap, decimal    as in v1
  rank_flip   two methods swapped in one row: the table ranks them the other way
  stale       every cell off by a few percent, as when the table is left from an older run
  dup_series  one series copied over another (copy-paste of a curve)
Text checks (a sentence of the paper against the chart): see claims.py

    python src/bench_v2.py --manifest data/datikz_pgf/test_hr/manifest.jsonl --out data/datikz_pgf/test_v2
    python src/bench_v2.py --manifest data/datikz_pgf/val_hr/manifest.jsonl --out data/datikz_pgf/val_v2
    python src/bench_v2.py --manifest data/datikz_pgf/test_hr/manifest.jsonl --subset 200 \
        --subset-out data/datikz_pgf/subset200_ids.txt

Writes <out>/manifest.jsonl (one line per check, with "chart_id" = the chart to read) and
<out>/paper_tables/*.csv. Paths in the manifest are relative to <out>/.. like the v1 manifests.
"""
import argparse
import collections
import json
import random
from pathlib import Path

from claims import make_claims
from perturb import SIZES, change_value, decimal_slip, rescale, swap_rows
from tables import from_csv


def rank_flip(table, rng, min_gap=0.05):
    """Swap the values of two series in one row (the table ranks two methods the other way)."""
    t = table.copy()
    opts = []
    for i in range(len(t.rows)):
        for a in range(len(t.series)):
            for b in range(a + 1, len(t.series)):
                va, vb = t.values[i][a], t.values[i][b]
                if va is not None and vb is not None and abs(va - vb) / max(abs(va), abs(vb), 1e-9) >= min_gap:
                    opts.append((i, a, b))
    if not opts:
        return None, []
    i, a, b = rng.choice(opts)
    va, vb = t.values[i][a], t.values[i][b]
    t.values[i][a], t.values[i][b] = vb, va
    return t, [{"row": t.rows[i], "series": t.series[a], "true": va, "table": vb},
               {"row": t.rows[i], "series": t.series[b], "true": vb, "table": va}]


def stale(table, rng, sigma):
    """Every cell moved by a relative Gaussian error of sigma (an older run of the same experiment)."""
    t = table.copy()
    big = table.max_abs() or 1.0
    changed = []
    for i, row in enumerate(t.values):
        for j, v in enumerate(row):
            if v is not None:
                new = v + rng.gauss(0, sigma) * max(abs(v), 0.05 * big)
                changed.append({"row": t.rows[i], "series": t.series[j], "true": v, "table": new})
                row[j] = new
    return t, changed


def dup_series(table, rng, min_gap=0.10):
    """Series b replaced by series a (a curve pasted twice)."""
    t = table.copy()
    big = table.max_abs() or 1.0
    opts = []
    for a in range(len(t.series)):
        for b in range(len(t.series)):
            if a == b:
                continue
            gaps = [abs(r[a] - r[b]) / max(abs(r[b]), 0.05 * big) for r in t.values
                    if r[a] is not None and r[b] is not None]
            if gaps and sum(gaps) / len(gaps) >= min_gap:
                opts.append((a, b))
    if not opts:
        return None, []
    a, b = rng.choice(opts)
    changed = []
    for i, r in enumerate(t.values):
        if r[a] is not None and r[b] is not None:
            changed.append({"row": t.rows[i], "series": t.series[b], "true": r[b], "table": r[a]})
            r[b] = r[a]
    return t, changed


def table_variants(table, rng):
    """[(kind, size, label, paper_table, changed)] for one chart."""
    out = [("match", None, "match", table.copy(), [])]
    t, factor = rescale(table, rng)
    if t is not None:
        out.append(("rescaled", None, "match", t, []))
    for size in SIZES:
        t, ch = change_value(table, rng, size)
        out.append(("value", size, "mismatch", t, ch))
    for kind, fn in (("swap", swap_rows), ("rank_flip", rank_flip), ("dup_series", dup_series)):
        t, ch = fn(table, rng)
        if t is not None:
            out.append((kind, None, "mismatch", t, ch))
    t, ch = decimal_slip(table, rng)
    out.append(("decimal", None, "mismatch", t, ch))
    for sigma in (0.02, 0.05):
        t, ch = stale(table, rng, sigma)
        out.append(("stale", sigma, "mismatch", t, ch))
    return out


def build(manifest, out, seed=0):
    manifest, out = Path(manifest), Path(out)
    root = manifest.parent.parent
    (out / "paper_tables").mkdir(parents=True, exist_ok=True)
    rel = out.name
    counts = collections.Counter()
    with open(out / "manifest.jsonl", "w") as f:
        for line in manifest.open():
            rec = json.loads(line)
            table = from_csv((root / rec["table"]).read_text())
            rng = random.Random(f"{seed}-{rec['id']}")
            base = {k: rec.get(k) for k in ("source", "paper", "chart_type", "n_points", "n_series",
                                            "value_labels", "image", "table")}
            base["chart_id"] = rec["id"]
            for k, (kind, size, label, t, ch) in enumerate(table_variants(table, rng)):
                vid = f"{rec['id']}__t{k}_{kind}"
                (out / "paper_tables" / f"{vid}.csv").write_text(t.to_csv())
                f.write(json.dumps({**base, "id": vid, "check": "table", "label": label, "kind": kind,
                                    "size": size, "paper_table": f"{rel}/paper_tables/{vid}.csv",
                                    "changed": ch}) + "\n")
                counts[("table", kind, label)] += 1
            for k, (claim, label) in enumerate(make_claims(table, rng)):
                vid = f"{rec['id']}__c{k}_{claim['type']}"
                f.write(json.dumps({**base, "id": vid, "check": "claim", "label": label, "kind": claim["type"],
                                    "size": claim.get("size"), "claim": claim}) + "\n")
                counts[("claim", claim["type"], label)] += 1
    for k, v in sorted(counts.items()):
        print(f"  {k[0]:5s} {k[1]:11s} {k[2]:8s} {v:6d}")


def subset(manifest, n, seed=0, per_paper=2):
    """n charts, stratified by chart type x written values x single/multi series, at most per_paper per paper."""
    recs = [json.loads(l) for l in open(manifest)]
    rng = random.Random(seed)
    rng.shuffle(recs)
    strata = collections.defaultdict(list)
    for r in recs:
        strata[(r.get("chart_type"), bool(r.get("value_labels")), (r.get("n_series") or 1) > 1)].append(r)
    total = len(recs)
    chosen, papers = [], collections.Counter()
    for key, rs in sorted(strata.items(), key=str):
        want = max(1, round(n * len(rs) / total))
        for r in rs:
            if want == 0:
                break
            if papers[r.get("paper")] < per_paper:
                chosen.append(r["id"])
                papers[r.get("paper")] += 1
                want -= 1
    return chosen[:n] if len(chosen) >= n else chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--subset-out")
    args = ap.parse_args()
    if args.subset:
        ids = subset(args.manifest, args.subset, args.seed)
        Path(args.subset_out).write_text("\n".join(ids) + "\n")
        print(f"{len(ids)} charts -> {args.subset_out}")
    if args.out:
        build(args.manifest, args.out, args.seed)


if __name__ == "__main__":
    main()
