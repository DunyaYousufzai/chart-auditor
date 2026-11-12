"""A fake reader that knows the true table and adds noise. No model, no GPU.

Used only to test the whole pipeline (reading -> ensemble -> checker -> metrics) on a
laptop, and as a sanity check: with noise 0 the checker must be perfect.
Some charts are "hard": their readings are noisier, which is exactly the case the
uncertainty-aware threshold is meant for.
"""
import random
from pathlib import Path

from tables import from_csv


class SimulatedReader:
    def __init__(self, noise=0.02, hard_share=0.3, hard_noise=0.10, **_):
        self.noise, self.hard_share, self.hard_noise = noise, hard_share, hard_noise
        self.name = f"simulated(noise={noise},hard={hard_share}x{hard_noise})"

    def read(self, image_paths, n=1, temperature=0.0, seed=0, tables=None, **_):
        out = []
        for p in image_paths:
            p = Path(p)
            true = from_csv((p.parent.parent / "tables" / (p.stem + ".csv")).read_text())
            chart_rng = random.Random(f"chart-{p.stem}")
            sigma = self.hard_noise if chart_rng.random() < self.hard_share else self.noise
            # the chart's own bias: the same misreading in every sample
            bias = {k: chart_rng.gauss(0, sigma / 2) for k in true.cells()}
            texts = []
            for s in range(n):
                rng = random.Random(f"{seed}-{p.stem}-{s}")
                lines = [" | ".join([true.x_name] + true.series)]
                for r, vals in zip(true.rows, true.values):
                    cells = [f"{v * (1 + bias[(r, sname)] + rng.gauss(0, sigma)):.3f}"
                             for v, sname in zip(vals, true.series)]
                    lines.append(" | ".join([r] + cells))
                texts.append("\n".join(lines))
            out.append(texts)
        return out
