"""Score a reader on benchmark v2 (bench_v2.py): every table check and text claim of every chart.

    python src/evaluate_v2.py \
        --manifest data/datikz_pgf/test_v2/manifest.jsonl --readings outputs/readings/X_datikz_hr_test_n9.jsonl \
        --val-manifest data/datikz_pgf/val_v2/manifest.jsonl --val-readings outputs/readings/X_datikz_hr_val_n9.jsonl \
        [--subset data/datikz_pgf/subset200_ids.txt] [--oracle] --out outputs/results_v4/X_v2.json

Each chart is read once (the median of its readings); all its checks reuse that reading.
The tolerance tau is tuned on val, separately for table checks and claims, on balanced accuracy:
  balanced = (share of no-error checks passed + mean over error kinds of the share caught) / 2
"not_sure" counts as not caught for errors and as not passed for no-error checks.
--oracle replaces the reading with the chart's true data (upper bound of the checks).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from checker import compare
from claims import check, map_reading
from ensemble import combine
from tables import from_csv, parse_model_output

TAUS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]


def load(manifest, readings, subset=None, oracle=False):
    manifest = Path(manifest)
    root = manifest.parent.parent
    recs = [json.loads(l) for l in manifest.open()]
    if subset:
        keep = set(Path(subset).read_text().split())
        recs = [r for r in recs if r["chart_id"] in keep]
    reads = {}
    for line in Path(readings).open():
        rd = json.loads(line)
        reads[rd["id"]] = rd
    charts, tables = {}, {}
    for r in recs:
        cid = r["chart_id"]
        if cid not in tables:
            tables[cid] = from_csv((root / r["table"]).read_text())
            if oracle:
                charts[cid] = (tables[cid], {})
            elif cid in reads:
                med, spread, _ = combine([parse_model_output(t) for t in reads[cid]["outputs"]])
                charts[cid] = (med, spread)
        if r["check"] == "table":
            r["_paper"] = from_csv((root / r["paper_table"]).read_text())
    recs = [r for r in recs if r["chart_id"] in charts]
    secs = [reads[c]["seconds_per_chart"] for c in charts if c in reads]
    info = {"model": next(iter(reads.values()))["model"] if reads and not oracle else "oracle",
            "seconds_per_chart": sum(secs) / len(secs) if secs else None}
    return recs, charts, tables, info


def verdicts(recs, charts, tables, tau_table, tau_claim):
    out, mapped = [], {}
    for r in recs:
        med, spread = charts[r["chart_id"]]
        if med is None:
            out.append("not_sure")
        elif r["check"] == "table":
            out.append(compare(med, r["_paper"], spread, tau=tau_table)["verdict"])
        else:
            if r["chart_id"] not in mapped:
                mapped[r["chart_id"]] = map_reading(med, tables[r["chart_id"]])
            out.append(check(r["claim"], mapped[r["chart_id"]], tables[r["chart_id"]], tau=tau_claim))
    return out


def scores(recs, v):
    """Per kind: share passed (no-error) or caught (error), and not_sure; plus balanced accuracy per check type."""
    by = defaultdict(lambda: {"n": 0, "ok": 0, "not_sure": 0})
    for r, vv in zip(recs, v):
        key = f"{r['check']}/{r['kind']}" + (f"/{r['size']}" if r.get("size") is not None else "") + f"/{r['label']}"
        want = r["label"]
        b = by[key]
        b["n"] += 1
        b["ok"] += vv == want
        b["not_sure"] += vv == "not_sure"
    kinds = {k: {"n": b["n"], "ok": round(b["ok"] / b["n"], 3), "not_sure": round(b["not_sure"] / b["n"], 3)}
             for k, b in sorted(by.items())}
    summary = {}
    for check_type in ("table", "claim"):
        ok_neg = [r for r, vv in zip(recs, v) if r["check"] == check_type and r["label"] == "match"]
        passed = sum(vv == "match" for r, vv in zip(recs, v) if r["check"] == check_type and r["label"] == "match")
        caught = defaultdict(lambda: [0, 0])
        for r, vv in zip(recs, v):
            if r["check"] == check_type and r["label"] == "mismatch":
                caught[r["kind"]][0] += vv == "mismatch"
                caught[r["kind"]][1] += 1
        if not ok_neg or not caught:
            continue
        pass_rate = passed / len(ok_neg)
        catch = {k: a / b for k, (a, b) in caught.items()}
        macro = sum(catch.values()) / len(catch)
        summary[check_type] = {"no_error_passed": round(pass_rate, 3), "false_alarm": round(1 - pass_rate, 3),
                               "caught_by_kind": {k: round(x, 3) for k, x in sorted(catch.items())},
                               "caught_macro": round(macro, 3), "balanced": round((pass_rate + macro) / 2, 3),
                               "n_no_error": len(ok_neg), "n_error": sum(b for _, b in caught.values())}
    return kinds, summary


def tune(recs, charts, tables, check_type):
    sub = [r for r in recs if r["check"] == check_type]
    best, best_b = None, -1
    for tau in TAUS:
        v = verdicts(sub, charts, tables, tau, tau)
        b = scores(sub, v)[1].get(check_type, {}).get("balanced", -1)
        if b > best_b:
            best, best_b = tau, b
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--readings", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--val-readings", required=True)
    ap.add_argument("--subset")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    recs, charts, tables, info = load(args.manifest, args.readings, args.subset, args.oracle)
    vrecs, vcharts, vtables, _ = load(args.val_manifest, args.val_readings, None, args.oracle)
    tau_t, tau_c = tune(vrecs, vcharts, vtables, "table"), tune(vrecs, vcharts, vtables, "claim")
    v = verdicts(recs, charts, tables, tau_t, tau_c)
    kinds, summary = scores(recs, v)
    res = {"manifest": args.manifest, "readings": args.readings, "subset": args.subset, "oracle": args.oracle,
           **info, "n_charts": len(charts), "n_checks": len(recs), "tau_table": tau_t, "tau_claim": tau_c,
           "summary": summary, "by_kind": kinds}

    print(f"{info['model']}  charts={len(charts)} checks={len(recs)}  tau table={tau_t} claim={tau_c}")
    for ct, s in summary.items():
        print(f"  {ct:5s} balanced={s['balanced']:.3f}  false_alarm={s['false_alarm']:.3f}  "
              f"caught_macro={s['caught_macro']:.3f}  {s['caught_by_kind']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
