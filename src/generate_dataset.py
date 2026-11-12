"""Make a synthetic dataset of bar and line charts with known true tables.

For every chart we save:
  - the picture
  - the chart's true table (what the reader must learn to write)
  - a "paper table": either the same numbers (match) or changed on purpose (mismatch),
    made by perturb.py. This is the pair the auditor has to check.
"""
import argparse
import json
import random
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt

from perturb import make_paper_table
from tables import Table

# ---------- 1. What the charts are about ----------
MODELS = ["ResNet", "ViT", "BERT", "T5", "GPT-2", "LSTM", "CNN", "MLP", "SVM", "XGBoost"]
CITIES = ["Zurich", "Geneva", "Bern", "Basel", "Lausanne", "Lugano", "Fribourg", "Lucerne"]
COUNTRIES = ["Chile", "Peru", "Kenya", "Nepal", "Spain", "Italy", "Japan", "Ghana"]
METHODS = ["Baseline", "Method A", "Method B", "Ours", "Ablation 1", "Ablation 2"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Names for the series when a chart has more than one (the legend)
DATASETS = ["CIFAR-10", "ImageNet", "SVHN", "MNIST", "COCO"]
GROUPS = ["Control", "Treatment", "Placebo"]
SEXES = ["Female", "Male"]
LINE_METHODS = ["Baseline", "Ours", "Method A", "Method B"]

# "category" = a bar graph
# Minimum bound (low=40): The very bottom of the chart represents 40% accuracy.
# Any model with an accuracy score lower than 40% will not show up on the graph.
# Maximum bound (high=99): The very top of the chart represents 99% accuracy.
# The vertical scale will stop exactly at 99%.
# dec=1: Rounds and displays the numeric accuracy values on the chart to 1 decimal place (e.g., 85.4%)
#  year - line chart
# series = possible legend names. With series=None the chart has one series, named after the y axis.

TOPICS = [
    dict(title="Model accuracy", x="Model", y="Accuracy (%)", kind="category", labels=MODELS, low=40, high=99, dec=1, series=DATASETS),
    dict(title="Training time", x="Model", y="Time (hours)", kind="category", labels=MODELS, low=1, high=120, dec=1, series=None),
    dict(title="Patients enrolled", x="City", y="Patients", kind="category", labels=CITIES, low=20, high=400, dec=0, series=GROUPS),
    dict(title="Population", x="Country", y="Population (thousands)", kind="category", labels=COUNTRIES, low=500, high=90000, dec=0, series=SEXES),
    dict(title="Error rate", x="Method", y="Error rate (%)", kind="category", labels=METHODS, low=1, high=30, dec=1, series=DATASETS),
    dict(title="Concentration", x="City", y="Concentration (mg/L)", kind="category", labels=CITIES, low=0.1, high=5.0, dec=2, series=None),
    dict(title="Annual revenue", x="Year", y="Revenue (million USD)", kind="years", labels=None, low=50, high=900, dec=0, series=None),
    dict(title="Active users", x="Year", y="Users (thousands)", kind="years", labels=None, low=10, high=5000, dec=0, series=LINE_METHODS),
    dict(title="Rainfall", x="Month", y="Rainfall (mm)", kind="months", labels=None, low=10, high=200, dec=0, series=CITIES),
]

# ---------- 2. How the charts look (12 styles) ----------
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
          "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#222222", "#E15759"]
FONTS = ["sans-serif", "serif", "monospace"]
# (width, height): (5, 4): A compact, slightly landscape chart. It will be 5 inches wide by 4 inches high (500 × 400 pixels)
SIZES = [(5, 4), (6, 4), (7, 4.5), (4.5, 4.5)]

# Unseen styles use a whole matplotlib style sheet (other background, grid, fonts,
# colour cycle, tick style). Before, styles 8-11 only changed the colour, because
# every other property came from i % 2, i % 3, i % 4 and repeated the seen styles.
UNSEEN_SHEETS = {8: "ggplot", 9: "bmh", 10: "fivethirtyeight", 11: "Solarize_Light2"}

SEEN_STYLES = list(range(0, 8))     # the model can learn these
UNSEEN_STYLES = list(range(8, 12))  # only for the hardest test

SPLITS = {
    "train": SEEN_STYLES,
    "val": SEEN_STYLES,
    "test_seen": SEEN_STYLES,
    "test_unseen": UNSEEN_STYLES,
}


def style_for(i):
    return {
        "id": i,
        "color": COLORS[i],
        # font : is a clever programming trick used to automatically loop through a list of 3 different fonts,
        # assigning one to each chart so that neighboring charts don't look exactly the same.
        # i % 3 forces the index to constantly cycle through 0, 1, 2 and reset, preventing an "index out of bounds" crash
        "font": FONTS[i % 3],
        # "figsize": SIZES[i % 4]: Sets the chart size.
        # It cycles through your 4 size options, ensuring chart 0 gets the 1st size, chart 1 gets the 2nd, and so on, resetting every 4 charts.
        "figsize": SIZES[i % 4],
        # "grid": i % 2 == 0: Turns the background grid lines ON for even-numbered charts (0, 2, 4...) and OFF for odd-numbered charts (1, 3, 5...).
        "grid": i % 2 == 0,
        # "value_labels": i % 3 == 1: Shows the exact data numbers directly on top of the bars/lines,
        # but only on every 3rd chart (charts 1, 4, 7...). The rest will hide them.
        "value_labels": i % 3 == 1,   # numbers written on the chart
        # "marker": "o" if i % 2 == 1 else None: Adds dot markers (o) to the data points on odd-numbered charts (1, 3, 5...).
        #   Even-numbered charts get no dots (None)
        "marker": "o" if i % 2 == 1 else None,
        "bg": "#F5F5F5" if i % 5 == 0 else "white",
        # This line sets the image sharpness and quality (DPI) dots per inch
        # Chart 0: 100 + 50 * 0 = 100 DPI (Standard resolution; smaller file size)
        # Chart 1: 100 + 50 * 1 = 150 DPI (Medium resolution; sharper)
        # Chart 2: 100 + 50 * 2 = 200 DPI (High resolution; very crisp and detailed)
        # Chart 3: Resets back to 100 DPI, and the loop repeats.
        "dpi": 100 + 50 * (i % 3),
        "sheet": UNSEEN_SHEETS.get(i),  # None = plain matplotlib look
    }


# ---------- 3. Make a random table ----------
def make_table(rng, topic, chart_type, max_series=3):
    kind = topic["kind"]
    if kind == "category":
        n = rng.randint(3, 8)
        labels = rng.sample(topic["labels"], min(n, len(topic["labels"])))
    elif kind == "years":
        n = rng.randint(4, 10)
        start = rng.randint(2005, 2018)
        labels = [str(start + k) for k in range(n)]
    else:  # months
        n = rng.randint(4, 12)
        start = rng.randint(0, 12 - n)
        labels = MONTHS[start:start + n]

    # How many series (legend entries). Grouped bars get crowded, so fewer bars per group.
    if topic["series"] and max_series > 1 and rng.random() < 0.5:
        k = rng.randint(2, min(max_series, len(topic["series"])))
        series = rng.sample(topic["series"], k)
        if chart_type == "bar":
            labels = labels[:max(3, 8 // k)]
    else:
        series = [topic["y"]]

    low, high, dec = topic["low"], topic["high"], topic["dec"]
    span = high - low

    columns = []
    for _ in series:
        if chart_type == "line":
            # a wandering line, like a real time series
            v = rng.uniform(low + 0.1 * span, low + 0.5 * span)
            values = []
            for _ in labels:
                values.append(v)
                v = min(high, max(low, v + rng.gauss(0, 0.12 * span)))
        else:
            values = [rng.uniform(low, high) for _ in labels]
        values = [round(v, dec) for v in values]
        if dec == 0:
            values = [int(v) for v in values]
        columns.append(values)

    rows = [list(r) for r in zip(*columns)]
    return Table(x_name=topic["x"], series=series, rows=labels, values=rows)


# ---------- 4. Draw the chart ----------
def draw(table, chart_type, style, topic, path):
    dec = topic["dec"]
    labels = table.rows
    k = len(table.series)
    multi = k > 1
    sheet = [style["sheet"]] if style["sheet"] else []

    with plt.style.context(sheet), plt.rc_context({"font.family": style["font"]}):
        fig, ax = plt.subplots(figsize=style["figsize"])
        if not style["sheet"]:
            fig.patch.set_facecolor(style["bg"])
            ax.set_facecolor(style["bg"])

        # one colour for a single series; the style sheet's colour cycle for several
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", COLORS)
        colors = [style["color"]] if not multi else [cycle[(style["id"] + j) % len(cycle)] for j in range(k)]
        all_values = [v for row in table.values for v in row]

        if chart_type == "bar":
            width = 0.8 / k
            xs = range(len(labels))
            for j, name in enumerate(table.series):
                vals = [row[j] for row in table.values]
                offset = (j - (k - 1) / 2) * width
                bars = ax.bar([x + offset for x in xs], vals, width=width,
                              color=colors[j], label=name)
                if style["value_labels"]:
                    ax.bar_label(bars, labels=[f"{v:.{dec}f}" for v in vals],
                                 padding=2, fontsize=8 if k == 1 else 6)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(labels)
            ax.set_ylim(0, max(all_values) * 1.15)  # bars always start at zero
        else:
            for j, name in enumerate(table.series):
                vals = [row[j] for row in table.values]
                ax.plot(labels, vals, color=colors[j], marker=style["marker"],
                        linewidth=2, label=name)
                if style["value_labels"]:
                    for x, v in zip(labels, vals):
                        ax.annotate(f"{v:.{dec}f}", (x, v), textcoords="offset points",
                                    xytext=(0, 6), ha="center", fontsize=8 if k == 1 else 6)
            ax.margins(y=0.2)

        ax.set_title(topic["title"])
        ax.set_xlabel(topic["x"])
        ax.set_ylabel(topic["y"])
        if multi:
            ax.legend(fontsize=8)
        if not style["sheet"]:
            # Bug fix: ax.grid(False, alpha=0.4) still turns the grid ON, because
            # matplotlib enables it whenever line properties are passed.
            if style["grid"]:
                ax.grid(True, axis="y", alpha=0.4)
            else:
                ax.grid(False)
        ax.set_axisbelow(True)
        if len(labels) > 6 or max(len(l) for l in labels) > 6:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        fig.savefig(path, dpi=style["dpi"], bbox_inches="tight")
        plt.close(fig)


# ---------- 5. Build one group ----------
def build_split(out_dir, split, count, max_series=3, p_match=0.5):
    img_dir = out_dir / split / "images"
    tab_dir = out_dir / split / "tables"
    pap_dir = out_dir / split / "paper_tables"
    for d in (img_dir, tab_dir, pap_dir):
        d.mkdir(parents=True, exist_ok=True)

    with open(out_dir / split / "manifest.jsonl", "w") as manifest:
        for i in range(count):
            rng = random.Random(f"{split}-{i}")  # same seed = same chart
            style = style_for(rng.choice(SPLITS[split]))
            topic = rng.choice(TOPICS)
            chart_type = "bar" if topic["kind"] == "category" else rng.choice(["bar", "line"])
            table = make_table(rng, topic, chart_type, max_series)

            sample_id = f"{split}_{i:05d}"
            img_path = img_dir / f"{sample_id}.png"
            tab_path = tab_dir / f"{sample_id}.csv"
            pap_path = pap_dir / f"{sample_id}.csv"

            csv_text = table.to_csv(topic["dec"])
            tab_path.write_text(csv_text)
            draw(table, chart_type, style, topic, img_path)

            paper, info = make_paper_table(table, rng, p_match=p_match, dec=topic["dec"])
            pap_path.write_text(paper.to_csv())

            record = {
                "id": sample_id,
                "source": "synthetic",
                "split": split,
                "chart_type": chart_type,
                "style_id": style["id"],
                "seen_style": style["id"] in SEEN_STYLES,
                "topic": topic["title"],
                "n_points": len(table.rows),
                "n_series": len(table.series),
                "value_labels": style["value_labels"],
                "image": str(img_path.relative_to(out_dir)),
                "table": str(tab_path.relative_to(out_dir)),
                "paper_table": str(pap_path.relative_to(out_dir)),
                "label": info["label"],      # match / mismatch: the answer for the auditor
                "kind": info["kind"],        # match, rescaled, value, swap, decimal
                "size": info["size"],        # size of the change for kind=value
                "scale": info["scale"],      # unit factor for kind=rescaled
                "changed": info["changed"],  # which cells were changed
                "target": csv_text,  # the answer the model must learn to write
            }
            manifest.write(json.dumps(record) + "\n")
            if (i + 1) % 100 == 0 or i + 1 == count:
                print(f"  {split}: {i + 1}/{count}", flush=True)

    print(f"{split}: {count} charts done")


def main():
    # 2600 charts
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_v2")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test-seen", type=int, default=200)
    parser.add_argument("--n-test-unseen", type=int, default=200)
    parser.add_argument("--max-series", type=int, default=3, help="1 = only single-series charts, as in v1")
    parser.add_argument("--p-match", type=float, default=0.5, help="share of matching pairs")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    out_dir = Path(args.out)
    for split, n in [("train", args.n_train), ("val", args.n_val),
                     ("test_seen", args.n_test_seen), ("test_unseen", args.n_test_unseen)]:
        build_split(out_dir, split, n, args.max_series, args.p_match)


if __name__ == "__main__":
    main()


# How the code works
# The file has five small parts:
# TOPICS says what charts are about. Each topic has names for the axes,
# the range of the numbers (low to high), and how many decimals (dec). We have decimals like 82.4 and big numbers like 45210. The model must handle both.
# Some topics also have legend names (series), so half of their charts show 2-3 series:
# grouped bars or several lines, like most charts in real papers.

# style_for(i) builds the look of style number i. The look comes from i by simple rules.
# For example, i % 2 == 0 means “even styles have grid lines”. So there is no long list to keep.
# Styles 8-11 (only in test_unseen) use a full matplotlib style sheet, so they really look different.

# make_table makes random labels and numbers.
# A line chart uses a wandering line.
# A wandering line (also known as a random walk) is a mathematical trick used to generate mock data that looks like a realistic financial or scientific trend over time.
# A bar chart uses free random numbers. We round the numbers before drawing.
# So the picture and the true table always have the same numbers.

# draw draws the picture with matplotlib.
# build_split makes one group. For each chart it saves four things:
# the picture (images/train_00000.png),
# the true table (tables/train_00000.csv),
# the paper table (paper_tables/train_00000.csv): same as the true table, or changed on purpose,
# one line in manifest.jsonl. This is a list file. Each line is one example, with its facts
# (chart type, style, topic, match or mismatch and what was changed) and the target text.


# The target text is what the model learns to write. It looks like this:
# Model,Accuracy (%)
# ResNet,82.4
# ViT,91.0
# BERT,77.3
# or, with two series:
# Model,CIFAR-10,ImageNet
# ResNet,93.1,76.2
# ViT,98.1,81.0
