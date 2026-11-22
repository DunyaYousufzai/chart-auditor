"""Generate per-dataset diagnostic plots from evaluate.py output JSONs.

    python src/plot_dataset_behavior.py outputs/results --out outputs/plots
    python src/plot_dataset_behavior.py outputs/results/qwen4b_test_seen_n5.json
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(text)).strip("_") or "dataset"


def _dataset_name(data: dict, fallback: str) -> str:
    manifest = data.get("manifest", "")
    if manifest:
        parts = Path(manifest).parts
        # e.g. data/synthetic_v2/test_seen/manifest.jsonl -> synthetic_v2_test_seen
        if len(parts) >= 3 and parts[-1].endswith(".jsonl"):
            return _slug("_".join(parts[-3:-1]))
        return _slug(Path(manifest).stem)
    return _slug(fallback)


def plot_detection_modes(data: dict, out_dir: Path, title_prefix: str) -> None:
    det_keys = [k for k in data if k.startswith("detection")]
    if not det_keys:
        return

    modes = [k.replace("detection_", "") for k in det_keys]
    metrics = ["precision", "recall", "f1", "not_sure_rate"]
    labels = ["Precision", "Recall", "F1", "Not sure rate"]

    x = list(range(len(modes)))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(max(7, len(modes) * 1.8), 4.5))
    for idx, (m_key, m_label) in enumerate(zip(metrics, labels)):
        vals = [data[k].get(m_key) or 0.0 for k in det_keys]
        offsets = [xi + (idx - (len(metrics) - 1) / 2) * width for xi in x]
        bars = ax.bar(offsets, vals, width=width, label=m_label)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score / Rate")
    ax.set_title(f"{title_prefix} — Detection by mode")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "detection_modes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_recall_by_size(data: dict, out_dir: Path, title_prefix: str) -> None:
    det_keys = [k for k in data if k.startswith("detection") and data[k].get("recall_by_size")]
    if not det_keys:
        return

    sizes = []
    for k in det_keys:
        for sz in data[k]["recall_by_size"].keys():
            if sz not in sizes:
                sizes.append(sz)
    if not sizes:
        return

    x = list(range(len(sizes)))
    width = 0.8 / len(det_keys)

    fig, ax = plt.subplots(figsize=(max(7, len(sizes) * 1.4), 4.5))
    for idx, k in enumerate(det_keys):
        mode = k.replace("detection_", "")
        rbs = data[k]["recall_by_size"]
        vals = [rbs.get(sz) if rbs.get(sz) is not None else 0.0 for sz in sizes]
        offsets = [xi + (idx - (len(det_keys) - 1) / 2) * width for xi in x]
        bars = ax.bar(offsets, vals, width=width, label=mode)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall")
    ax.set_xlabel("Change size")
    ax.set_title(f"{title_prefix} — Recall by mismatch size")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "recall_by_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_by_kind(data: dict, out_dir: Path, title_prefix: str) -> None:
    # Choose preferred detection block for recall_by_kind
    preferred = ["detection_adaptive", "detection_fixed", "detection_fixed_5pct", "detection_relative"]
    det_key = next((k for k in preferred if k in data and data[k].get("recall_by_kind")), None)
    if det_key is None:
        det_key = next((k for k in data if k.startswith("detection") and data[k].get("recall_by_kind")), None)
    if det_key is None:
        return

    rbk = data[det_key].get("recall_by_kind") or {}
    fak = data[det_key].get("false_alarm_by_kind") or {}
    kinds = sorted(set(rbk.keys()) | set(fak.keys()))
    if not kinds:
        return

    x = list(range(len(kinds)))
    width = 0.36
    rec_vals = [rbk.get(kd) if rbk.get(kd) is not None else 0.0 for kd in kinds]
    fa_vals = [fak.get(kd) if fak.get(kd) is not None else 0.0 for kd in kinds]

    fig, ax = plt.subplots(figsize=(max(7, len(kinds) * 1.4), 4.5))
    b1 = ax.bar([xi - width / 2 for xi in x], rec_vals, width=width, label="Recall (mismatch)")
    b2 = ax.bar([xi + width / 2 for xi in x], fa_vals, width=width, label="False alarm rate (match)")
    ax.bar_label(b1, labels=[f"{v:.2f}" for v in rec_vals], padding=2, fontsize=7)
    ax.bar_label(b2, labels=[f"{v:.2f}" for v in fa_vals], padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(kinds, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.set_title(f"{title_prefix} — Recall & false alarms by kind ({det_key.replace('detection_', '')})")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "recall_and_false_alarms_by_kind.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reading_breakdown(data: dict, out_dir: Path, title_prefix: str) -> None:
    reading = data.get("reading")
    if not reading:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: first_reading vs median_of_n
    r_modes = ["first_reading", "median_of_n"]
    r_metrics = ["cell_acc", "rnss", "parsed"]
    x0 = list(range(len(r_metrics)))
    w0 = 0.35
    for idx, rm in enumerate(r_modes):
        block = reading.get(rm) or {}
        vals = [block.get(m) or 0.0 for m in r_metrics]
        offsets = [xi + (idx - 0.5) * w0 for xi in x0]
        bars = axes[0].bar(offsets, vals, width=w0, label=rm)
        axes[0].bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=2, fontsize=7)

    axes[0].set_xticks(x0)
    axes[0].set_xticklabels(["cell_acc@5%", "RNSS", "parsed"])
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title("First reading vs. Median of N")
    axes[0].grid(True, axis="y", alpha=0.35)
    axes[0].legend(fontsize=8)

    # Right: median_by breakdown (cell_acc)
    med_by = reading.get("median_by") or {}
    if med_by:
        keys = list(med_by.keys())
        vals = [(med_by[k] or {}).get("cell_acc") or 0.0 for k in keys]
        bars = axes[1].barh(keys, vals)
        axes[1].bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=2, fontsize=7)
        axes[1].set_xlim(0, 1.15)
        axes[1].set_xlabel("cell_acc@5%")
        axes[1].set_title("Median cell_acc by slice")
        axes[1].grid(True, axis="x", alpha=0.35)
    else:
        axes[1].axis("off")

    fig.suptitle(f"{title_prefix} — Chart reading behavior")
    fig.tight_layout()
    fig.savefig(out_dir / "reading_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_file(json_path: Path, out_root: Path) -> Path:
    data = json.loads(json_path.read_text())
    ds_name = _dataset_name(data, json_path.stem)
    run_folder = out_root / ds_name / _slug(json_path.stem)
    run_folder.mkdir(parents=True, exist_ok=True)

    model_short = str(data.get("model", "model")).split("/")[-1]
    prefix = f"{model_short} | {ds_name} ({json_path.stem})"

    plot_detection_modes(data, run_folder, prefix)
    plot_recall_by_size(data, run_folder, prefix)
    plot_by_kind(data, run_folder, prefix)
    plot_reading_breakdown(data, run_folder, prefix)
    return run_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Result JSON files or directories containing *.json")
    parser.add_argument("--out", default="outputs/plots", help="Output root directory for plots")
    args = parser.parse_args()

    out_root = Path(args.out)
    files: list[Path] = []
    for item in args.inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif p.is_file() and p.suffix == ".json":
            files.append(p)

    for f in files:
        dest = process_file(f, out_root)
        print(f"Saved plots for {f.name} -> {dest}")


if __name__ == "__main__":
    main()