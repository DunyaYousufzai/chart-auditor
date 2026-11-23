"""Oracle reading: how well does the checker do if the chart were read perfectly?

Replaces the model's reading with the chart's true data (from the LaTeX source), optionally with
noise, and runs the same checker and the same threshold tuning on val as evaluate.py.

    python src/oracle.py \
        --manifest data/datikz_pgf/test_hr/manifest.jsonl \
        --val-manifest data/datikz_pgf/val_hr/manifest.jsonl \
        --readings outputs/readings/<model>_datikz_hr_test_n9.jsonl \
        --val-readings outputs/readings/<model>_datikz_hr_val_n9.jsonl \
        --out outputs/results_v3/oracle_datikz_hr_test.json

Reports
  noise sweep : F1 when each cell of the true table is off by a random relative error of sigma
                (0 = perfect reading). Tells how good the reading must be for a given F1.
  breakdown   : for every chart the model got wrong, whether the oracle gets it right too
                (then the error is the reading) or also fails (then it is the checker or the benchmark).
"""
import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

from evaluate import detection_block, load, prepare, tune, verdicts
from metrics import detection_scores

SIGMAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]


def noisy(table, sigma, rng):
    """Copy of table with every value moved by a relative Gaussian error of sigma.
    Near-zero cells use 5 % of the largest value as scale, like the checker's floor."""
    t = copy.deepcopy(table)
    vals = [abs(v) for row in t.values for v in row if v is not None]
    big = max(vals) if vals else 1.0
    for row in t.values:
        for j, v in enumerate(row):
            if v is not None and sigma > 0:
                row[j] = v + rng.gauss(0, sigma) * max(abs(v), 0.05 * big)
    return t


def as_oracle(rows, sigma, seed):
    rng = random.Random(seed)
    out = []
    for r in rows:
        o = dict(r)
        o["median"] = noisy(r["true"], sigma, rng)
        o["spread"] = {}
        out.append(o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--readings", required=True, help="model readings, only to know which charts and for the breakdown")
    ap.add_argument("--val-readings", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out")
    args = ap.parse_args()

    test = prepare(load(args.manifest, args.readings))
    val = prepare(load(args.val_manifest, args.val_readings))
    labels = [r["rec"]["label"] for r in test]
    res = {"manifest": args.manifest, "n_charts": len(test), "sweep": []}

    print(f"charts={len(test)}  (label counts {dict(Counter(labels))})")
    print(f"{'sigma':>6} {'F1 val-tau':>10} {'F1 5%':>7} {'false alarm':>11} {'recall':>7}  tau")
    for sigma in SIGMAS:
        f1s, f5s, fas, recs, taus = [], [], [], [], []
        for seed in range(1 if sigma == 0 else args.seeds):
            o_test, o_val = as_oracle(test, sigma, seed), as_oracle(val, sigma, 1000 + seed)
            params = tune(o_val, "fixed")
            d = detection_block(o_test, params)
            d5 = detection_block(o_test, {"tau": 0.05, "k": 0.0, "s_max": None})
            v = verdicts(o_test, **params)
            match = [vv for vv, l in zip(v, labels) if l == "match"]
            f1s.append(d["f1"]); f5s.append(d5["f1"]); recs.append(d["recall"]); taus.append(params["tau"])
            fas.append(sum(x == "mismatch" for x in match) / max(1, len(match)))
        row = {"sigma": sigma, "f1": sum(f1s) / len(f1s), "f1_fixed_5pct": sum(f5s) / len(f5s),
               "false_alarm": sum(fas) / len(fas), "recall": sum(recs) / len(recs), "taus": taus}
        if sigma == 0:
            row["recall_by_size"] = d["recall_by_size"]
            row["recall_by_kind"] = d["recall_by_kind"]
            row["false_alarm_by_kind"] = d["false_alarm_by_kind"]
        res["sweep"].append(row)
        print(f"{sigma:6.2f} {row['f1']:10.3f} {row['f1_fixed_5pct']:7.3f} {row['false_alarm']:11.3f} "
              f"{row['recall']:7.3f}  {taus}")

    # ---- breakdown of the model's errors, both at the model's own val-tuned tau ----
    p_model = tune(val, "fixed")
    v_model = verdicts(test, **p_model)
    v_oracle = verdicts(as_oracle(test, 0.0, 0), **p_model)
    model_scores = detection_scores(labels, v_model)
    want = {"match": "match", "mismatch": "mismatch"}
    cases = Counter()
    examples = {"reading": [], "checker_or_benchmark": []}
    for r, l, vm, vo in zip(test, labels, v_model, v_oracle):
        if vm == want[l]:
            continue
        kind = "false_alarm" if l == "match" else ("missed" if vm == "match" else "not_sure")
        cause = "reading" if vo == want[l] else "checker_or_benchmark"
        cases[(kind, cause)] += 1
        if len(examples[cause]) < 15:
            examples[cause].append({"id": r["rec"]["id"], "label": l, "model": vm, "oracle": vo,
                                    "kind": r["rec"].get("kind"), "size": r["rec"].get("size")})
    res["model"] = {"params": p_model, "f1": model_scores["f1"], "readings": args.readings}
    res["breakdown"] = {f"{k}/{c}": n for (k, c), n in sorted(cases.items())}
    res["examples"] = examples
    print(f"\nmodel F1={model_scores['f1']:.3f} at {p_model}")
    total = sum(cases.values())
    for (k, c), n in sorted(cases.items()):
        print(f"  {k:12s} caused by {c:22s} {n:4d}  ({100 * n / total:.0f} % of the model's errors)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
