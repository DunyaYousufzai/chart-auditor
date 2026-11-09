"""Scores for the two questions of the project.

1. How well does the reader read a chart?   (reading_scores)
   - cell_acc@5%   share of true cells read within 5% (after aligning labels)
   - mean_rel_err  mean relative error of the aligned cells
   - coverage      share of true cells the reader found at all
   - rnss          Relative Number Set Similarity (ChartQA / Pix2Struct metric):
                   ignores labels, matches the numbers one-to-one, 1 = perfect
2. How well are mismatches found?           (detection_scores)
   - precision, recall, F1 of "mismatch", over the examples where the tool decided
   - not_sure_rate share of "not sure" answers
   - strict_*      the same, counting "not sure" as a flag (it is sent to a person)
"""
from collections import defaultdict

from checker import align


def rnss(pred_vals, true_vals):
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # small fallback: greedy matching
        linear_sum_assignment = None
    if not pred_vals or not true_vals:
        return 0.0

    def cost(p, t):
        return min(1.0, abs(p - t) / max(abs(t), 1e-9))

    if linear_sum_assignment is None:
        left, total = list(pred_vals), 0.0
        for t in true_vals:
            if not left:
                total += 1
                continue
            i = min(range(len(left)), key=lambda i: cost(left[i], t))
            total += cost(left.pop(i), t)
        return 1 - (total + len(left)) / max(len(pred_vals), len(true_vals))
    import numpy as np
    m = np.array([[cost(p, t) for t in true_vals] for p in pred_vals])
    r, c = linear_sum_assignment(m)
    return float(1 - (m[r, c].sum() + abs(len(pred_vals) - len(true_vals))) /
                 max(len(pred_vals), len(true_vals)))


def reading_scores(pred, true, tol=0.05):
    """pred, true: tables.Table (pred may be None)."""
    tc = true.cells()
    if pred is None:
        return {"cell_acc": 0.0, "mean_rel_err": 1.0, "coverage": 0.0, "rnss": 0.0, "parsed": 0}
    pc = pred.cells()
    pairs, _, _ = align(pred, true)
    pairs = [(a, b) for a, b in pairs if a in pc and b in tc]
    errs = [min(1.0, abs(pc[a] - tc[b]) / max(abs(tc[b]), 1e-9)) for a, b in pairs]
    n = max(1, len(tc))
    return {
        "cell_acc": sum(e <= tol for e in errs) / n,
        "mean_rel_err": (sum(errs) + (len(tc) - len(errs))) / n,  # missing cells count as 100% error
        "coverage": len(errs) / n,
        "rnss": rnss(list(pc.values()), list(tc.values())),
        "parsed": 1,
    }


def mean_dict(rows):
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def detection_scores(labels, verdicts):
    """labels: 'match'/'mismatch'; verdicts: 'match'/'mismatch'/'not_sure'."""
    tp = fp = fn = tn = ns = 0
    stp = sfp = sfn = 0
    for y, v in zip(labels, verdicts):
        pos = y == "mismatch"
        if v == "not_sure":
            ns += 1
        elif v == "mismatch":
            tp += pos
            fp += not pos
        else:
            fn += pos
            tn += not pos
        flag = v in ("mismatch", "not_sure")
        stp += flag and pos
        sfp += flag and not pos
        sfn += (not flag) and pos

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    p, r, f = prf(tp, fp, fn)
    sp, sr, sf = prf(stp, sfp, sfn)
    n = len(labels)
    return {"n": n, "precision": p, "recall": r, "f1": f, "not_sure_rate": ns / max(1, n),
            "strict_precision": sp, "strict_recall": sr, "strict_f1": sf,
            "accuracy": (tp + tn) / max(1, n)}


def recall_by(groups, labels, verdicts):
    """Recall of mismatches split by a group (e.g. size of change or kind)."""
    hit, tot = defaultdict(int), defaultdict(int)
    for g, y, v in zip(groups, labels, verdicts):
        if y != "mismatch":
            continue
        tot[g] += 1
        hit[g] += v == "mismatch"
    return {g: round(hit[g] / tot[g], 3) for g in sorted(tot, key=str)}
