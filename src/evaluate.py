"""Turn saved readings into results.

    python src/evaluate.py \
        --manifest data/synthetic_v2/test_seen/manifest.jsonl \
        --readings outputs/readings/qwen4b_test_seen_n5.jsonl \
        --val-manifest data/synthetic_v2/val/manifest.jsonl \
        --val-readings outputs/readings/qwen4b_val_n5.jsonl \
        --out outputs/results/qwen4b_test_seen_n5.json

What it reports
  reading   : how well the chart was read (first reading alone vs median of N)
  detection : mismatch precision / recall / F1 for
                fixed     tol = tau                 (tau tuned on val, k = 0)
                adaptive  tol = max(tau, k*s)       (tau, k, s_max tuned on val)
              and recall split by size and kind of change
  speed     : seconds per chart, readings used with early stopping
For --task judge readings (baseline B0) only detection is reported.
"""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

from checker import compare
from ensemble import combine, combine_until_stable
from metrics import detection_scores, mean_dict, reading_scores, recall_by
from readers.prompts import parse_judge
from tables import from_csv, parse_model_output

TAUS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
KS = [0.5, 1, 1.5, 2, 3, 4]
SMAX = [None, 0.15, 0.3, 0.5]
RELS = [2, 3, 4, 6, 8]


def load(manifest, readings, root=None, n_use=None):
    manifest = Path(manifest)
    root = Path(root) if root else manifest.parent.parent
    recs = {json.loads(l)["id"]: json.loads(l) for l in manifest.open()}
    rows = []
    for line in Path(readings).open():
        rd = json.loads(line)
        rec = recs.get(rd["id"])
        if rec is None:
            continue
        outs = rd["outputs"][:n_use] if n_use else rd["outputs"]
        rows.append({"rec": rec, "rd": rd, "outs": outs,
                     "true": from_csv((root / rec["table"]).read_text()),
                     "paper": from_csv((root / rec["paper_table"]).read_text())})
    return rows


def prepare(rows, early_stop=False):
    """Parse and combine the readings of every chart once."""
    for r in rows:
        parsed = [parse_model_output(t) for t in r["outs"]]
        r["parsed"] = parsed
        if early_stop:
            r["median"], r["spread"], r["info"] = combine_until_stable(parsed)
        else:
            r["median"], r["spread"], r["info"] = combine(parsed)
    return rows


def verdicts(rows, **params):
    out = []
    for r in rows:
        if r["median"] is None:
            out.append("not_sure")
        else:
            out.append(compare(r["median"], r["paper"], r["spread"], **params)["verdict"])
    return out


def objective(scores):
    # F1 over decided cases, discounted by how often we say "not sure"
    return scores["f1"] * (1 - scores["not_sure_rate"])


def tune(rows, mode):
    """Best parameters on rows. mode: fixed (tau), adaptive (tau, k, s_max), relative (tau, rel, calibrate)."""
    labels = [r["rec"]["label"] for r in rows]
    if mode == "adaptive":
        grid = [{"tau": t, "k": k, "s_max": s} for t, k, s in itertools.product(TAUS, KS, SMAX)]
    elif mode == "relative":
        grid = [{"tau": t, "k": 0.0, "s_max": None, "rel": r, "calibrate": c}
                for t, r, c in itertools.product(TAUS, RELS, [False, True])]
    else:
        grid = [{"tau": t, "k": 0.0, "s_max": None} for t in TAUS]
    best, best_obj = None, -1
    for params in grid:
        sc = detection_scores(labels, verdicts(rows, **params))
        if objective(sc) > best_obj:
            best, best_obj = params, objective(sc)
    return best


def detection_block(rows, params):
    labels = [r["rec"]["label"] for r in rows]
    v = verdicts(rows, **params)
    sc = detection_scores(labels, v)
    sc["params"] = params
    sc["recall_by_size"] = recall_by([r["rec"].get("size") for r in rows], labels, v)
    sc["recall_by_kind"] = recall_by([r["rec"].get("kind") for r in rows], labels, v)
    match_kinds = [r["rec"]["kind"] for r in rows if r["rec"]["label"] == "match"]
    match_v = [vv for r, vv in zip(rows, v) if r["rec"]["label"] == "match"]
    fa = defaultdict(lambda: [0, 0])
    for kd, vv in zip(match_kinds, match_v):
        fa[kd][0] += vv == "mismatch"
        fa[kd][1] += 1
    sc["false_alarm_by_kind"] = {kd: round(a / b, 3) for kd, (a, b) in fa.items()}
    return sc


def reading_block(rows):
    first = [reading_scores(r["parsed"][0] if r["parsed"] else None, r["true"]) for r in rows]
    med = [reading_scores(r["median"], r["true"]) for r in rows]
    by = defaultdict(list)
    for r, m in zip(rows, med):
        by[("n_series", r["rec"].get("n_series", 1))].append(m)
        by[("value_labels", r["rec"].get("value_labels"))].append(m)
        by[("chart_type", r["rec"].get("chart_type"))].append(m)
    return {"first_reading": mean_dict(first), "median_of_n": mean_dict(med),
            "median_by": {f"{a}={b}": mean_dict(v) for (a, b), v in sorted(by.items(), key=str)}}


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--readings", required=True)
    ap.add_argument("--val-manifest")
    ap.add_argument("--val-readings")
    ap.add_argument("--root")
    ap.add_argument("--n-use", type=int, default=None, help="use only the first n readings (ablation 1/3/5/9)")
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = load(args.manifest, args.readings, args.root, args.n_use)
    task = rows[0]["rd"].get("task", "read")
    res = {"manifest": args.manifest, "readings": args.readings, "n_charts": len(rows),
           "model": rows[0]["rd"]["model"], "task": task, "n_use": args.n_use or rows[0]["rd"]["n"],
           "seconds_per_chart": sum(r["rd"]["seconds_per_chart"] for r in rows) / len(rows)}

    if task == "judge":
        labels = [r["rec"]["label"] for r in rows]
        v = [parse_judge(r["outs"][0]) for r in rows]
        res["detection_B0"] = detection_scores(labels, v)
        res["detection_B0"]["recall_by_size"] = recall_by([r["rec"].get("size") for r in rows], labels, v)
    else:
        prepare(rows, args.early_stop)
        res["reading"] = reading_block(rows)
        res["readings_used"] = sum(r["info"].get("n_used", r["info"]["n"]) for r in rows) / len(rows)
        tune_rows = rows
        if args.val_manifest and args.val_readings:
            tune_rows = prepare(load(args.val_manifest, args.val_readings, args.root, args.n_use), args.early_stop)
        else:
            print("WARNING: no validation readings; tuning on the test set itself (optimistic).")
        res["detection_fixed_5pct"] = detection_block(rows, {"tau": 0.05, "k": 0.0, "s_max": None})
        res["detection_fixed"] = detection_block(rows, tune(tune_rows, "fixed"))
        res["detection_adaptive"] = detection_block(rows, tune(tune_rows, "adaptive"))
        res["detection_relative"] = detection_block(rows, tune(tune_rows, "relative"))

    # ---- print a short summary ----
    print(f"\n{res['model']}  task={task}  charts={len(rows)}  N={res['n_use']}  "
          f"{res['seconds_per_chart']:.2f}s/chart")
    if task == "read":
        for name in ("first_reading", "median_of_n"):
            m = res["reading"][name]
            print(f"  reading {name:14s} cell_acc@5%={fmt(m['cell_acc'])}  rel_err={fmt(m['mean_rel_err'])}  "
                  f"rnss={fmt(m['rnss'])}  parsed={fmt(m['parsed'])}")
    for key in [k for k in res if k.startswith("detection")]:
        d = res[key]
        print(f"  {key:22s} P={fmt(d['precision'])} R={fmt(d['recall'])} F1={fmt(d['f1'])} "
              f"not_sure={fmt(d['not_sure_rate'])}  params={d.get('params', '')}")
        print(f"  {'':22s} recall by size: {d['recall_by_size']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
