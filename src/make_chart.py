import matplotlib
matplotlib.use("Agg")   # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

# 1. A small table (this is the "truth")
table = pd.DataFrame({
    "Model": ["A", "B", "C", "D"],
    "Accuracy": [80, 72, 65, 90],
})
table.to_csv("outputs/table.csv", index=False)


# 2. A function that draws a bar chart from a table
def draw_bar(df, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(df["Model"], df["Accuracy"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model accuracy")
    ax.set_ylim(0, 100)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# 3. MATCH: chart drawn from the same numbers
draw_bar(table, "outputs/chart_match.png")

# 4. MISMATCH: we change model B from 72 to 60, then draw
changed = table.copy()
changed.loc[changed["Model"] == "B", "Accuracy"] = 60
draw_bar(changed, "outputs/chart_mismatch.png")

print("Done. Files are in outputs/")