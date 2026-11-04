# Cross-Modal Scientific Data Auditor

A small open-model tool that reads numbers from bar and line charts
and checks them against the table in the same paper.

Course project: Generative AI (University of Neuchatel, Autumn 2026).
## Team members
Dunya | Adriana | Bill

## Quick start (CPU, any computer)

    docker build -t chart-auditor:dev .
    docker run --rm -v "$(pwd)/outputs:/app/outputs" chart-auditor:dev python src/make_chart.py

## Project status

Work in progress.