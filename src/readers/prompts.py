"""Prompts shared by the instruction-following readers (Qwen3-VL and later the LoRA model).

We ask for " | " between cells (like DePlot) instead of commas, so numbers such as
"1,234" cannot break the table.
"""

READ_PROMPT = """Extract the data table behind this chart.
Rules:
- Output only the table, nothing else.
- Separate cells with " | ". One line per x-axis category, in the order shown.
- First line is the header: the x-axis title, then one column per series. Use the legend names as series names; if there is no legend, use the y-axis title.
- Value cells contain only a number: no units, no % sign, no thousands separators.
- Read each value as precisely as the axis allows.

Example for a bar chart of accuracy of two models on two datasets:
Model | CIFAR-10 | ImageNet
ResNet | 93.1 | 76.2
ViT | 98.1 | 81.0"""

# Baseline B0: one model sees the chart and the paper's table and decides directly.
JUDGE_PROMPT = """Here is a chart from a scientific paper and a table from the same paper.
Table:
{table}

Do the numbers in the chart agree with the table? Small reading differences (about 5%) are fine; a unit change written in the header (e.g. "thousands") is fine.
Answer with one word on the first line: MATCH or MISMATCH. Then, if MISMATCH, name the cells that differ."""


def parse_judge(text):
    t = (text or "").strip().upper()
    first = t.splitlines()[0] if t else ""
    if "MISMATCH" in first:
        return "mismatch"
    if "MATCH" in first:
        return "match"
    return "not_sure"
