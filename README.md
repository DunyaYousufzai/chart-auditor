# Cross-Modal Scientific Data Auditor

A small open-model tool that reads numbers from bar and line charts
and checks them against the table in the same paper.

Course project: Generative AI (University of Neuchatel, Autumn 2026).
## Team members
Dunya | Adriana | Bill

## Quick start (CPU, any computer)

    docker build -t chart-auditor:dev .
    docker run --rm -v "$(pwd)/outputs:/app/outputs" chart-auditor:dev python src/make_chart.py

## Pilot pipeline (branch `bill-base`)

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python src/generate_dataset.py --out data/synthetic_v2          # charts + true tables + paper tables
    python src/chartqa_pairs.py --split test --limit 500             # real charts, perturbed tables
    python src/run_reader.py --reader simulated --manifest data/synthetic_v2/val/manifest.jsonl \
        --n 5 --out outputs/readings/sim_val.jsonl                    # no GPU needed
    python src/evaluate.py --manifest data/synthetic_v2/val/manifest.jsonl \
        --readings outputs/readings/sim_val.jsonl
    python -m pytest tests

Real readers (DePlot, Qwen3-VL) run on a GPU: see `slurm/README.md` (UBELIX).
Plan and design notes: `docs/PLAN_PILOTO.md`.

## Project status

Work in progress.