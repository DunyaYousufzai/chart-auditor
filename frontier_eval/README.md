# Frontier model comparison (GPT, Gemini, Claude)

Goal: compare API models with our Qwen models on **exactly the same charts, prompt and checker**. The
only thing that changes is who reads the chart. Our reference results on the same charts are in
`results/`. Everything runs on a laptop: only the API calls leave your machine.

## What is in this folder

| Path | What |
|---|---|
| `data/test_hr/` | **200 test charts** (images + `manifest.jsonl`), real arXiv charts from DaTikZ, 1260 px |
| `data/val_hr/` | **100 val charts**, only used to pick the tolerance (never report val numbers) |
| `data/test/tables/`, `data/val/tables/` | the true data of each chart (from the LaTeX), used for scoring. **Never send these to a model** |
| `data/test/paper_tables/` | benchmark v1: one "paper table" per chart (match or one error) |
| `data/test_v2/` | benchmark v2: 4,503 checks on the 200 charts (table errors and text claims) |
| `test_ids.txt`, `val_ids.txt` | the chart ids (200 + 100), stratified by chart type, written values and number of series |
| `readings/` | our Qwen readings of the same charts (9 readings per chart) |
| `results/` | our scores on the same charts (`v1_*`, `v2_*`, N = 1 and N = 9) |

Chart mix in the 200: bar and line charts; with and without the numbers written on the chart; one and
several series; at most 2 charts per paper.

## Setup

```bash
pip install anthropic openai google-genai
```

Keys go in environment variables only (never in a file you commit):

```bash
export ANTHROPIC_API_KEY=...
```

```bash
export OPENAI_API_KEY=...
```

```bash
export GEMINI_API_KEY=...
```

Run every command from the repository root.

## The specification (keep it identical for every model)

- **Prompt:** `READ_PROMPT` in `src/readers/prompts.py`, unchanged. No system prompt, no examples.
- **Input:** the PNG as it is (no resizing), the image first or alongside the prompt, one chart per request.
- **Output:** whatever text the model writes; `src/tables.py` parses it (pipe, markdown or CSV tables).
- **Readings:** **1 per chart** (`--n 1`). Our models also have N = 1 results for the fair comparison.
  If budget allows, a second run with `--n 3` shows how much repeated reading helps.
- **Tolerance:** picked on the 100 val charts by the evaluation scripts, never on test.
- **Settings per provider** (record them in the results table):
  - Claude: `--model claude-opus-5-5 --effort high`. Current Claude models take no temperature. Server-side
    fallbacks are on: if a request is declined it is retried on another model and the reading records
    `usage.served_by`; report how many charts that happened to (normally zero).
  - GPT: `--model <current GPT vision model id> --detail high` (check the model list in the OpenAI docs;
    `--detail original` is worth one extra run if the model supports it).
  - Gemini: `--model <current Gemini model id>` (check the Gemini docs).
- Each reading saves the input and output tokens (`usage`), so the cost per chart can be computed.

## 1. Smoke test (5 charts, a few cents)

```bash
python src/run_reader.py --reader claude --model claude-opus-5-5 --effort high --manifest frontier_eval/data/test_hr/manifest.jsonl --out frontier_eval/readings/claude-opus-5-5_test.jsonl --limit 5 --batch 5
```

Open the output file and check that `outputs` holds a table. The file can be resumed: ids already in it
are skipped, failed requests are left out and retried on the next run. So after the smoke test, run the
full command below on the same `--out` file.

## 2. Read the 200 test + 100 val charts

One command per model; replace `<reader>` / `<model>` (`claude`, `openai`, `gemini`):

```bash
python src/run_reader.py --reader <reader> --model <model> --manifest frontier_eval/data/test_hr/manifest.jsonl frontier_eval/data/val_hr/manifest.jsonl --out frontier_eval/readings/<model>_test.jsonl frontier_eval/readings/<model>_val.jsonl --n 1 --batch 20 --workers 8
```

300 requests per model. Lower `--workers` if you hit rate limits.

## 3. Score

Benchmark v1 (chart vs. one paper table):

```bash
python src/evaluate.py --manifest frontier_eval/data/test_hr/manifest.jsonl --readings frontier_eval/readings/<model>_test.jsonl --val-manifest frontier_eval/data/val_hr/manifest.jsonl --val-readings frontier_eval/readings/<model>_val.jsonl --out frontier_eval/results/v1_<model>_n1.json
```

Benchmark v2 (all table errors and text claims):

```bash
python src/evaluate_v2.py --manifest frontier_eval/data/test_v2/manifest.jsonl --readings frontier_eval/readings/<model>_test.jsonl --val-manifest frontier_eval/data/val_v2/manifest.jsonl --val-readings frontier_eval/readings/<model>_val.jsonl --out frontier_eval/results/v2_<model>_n1.json
```

What to report per model (all printed by the two scripts):

- **Reading:** `cell_acc@5%` (share of cells read within 5 %) — the main number, since the oracle showed
  that reading is what limits the system (`docs/results/ORACLE.md`).
- **v2 table:** `balanced`, `false_alarm`, `caught_by_kind`.
- **v2 claims:** `balanced`, `false_alarm`, `caught_by_kind`.
- **v1:** `F1` (careful: F1 rewards flagging everything; read it next to the false alarms).
- **Cost:** mean input/output tokens per chart × the provider's price, and seconds per chart.

## 4. Optional: the direct judge (no pipeline)

The model sees the chart **and** the paper table and answers MATCH / MISMATCH (`JUDGE_PROMPT`). This
tests whether a big model needs our pipeline at all.

```bash
python src/run_reader.py --reader <reader> --model <model> --task judge --manifest frontier_eval/data/test_hr/manifest.jsonl --out frontier_eval/readings/<model>_judge_test.jsonl --n 1 --batch 20
```

```bash
python src/evaluate.py --manifest frontier_eval/data/test_hr/manifest.jsonl --readings frontier_eval/readings/<model>_judge_test.jsonl --out frontier_eval/results/v1_<model>_judge.json
```

## Our reference results (same 200 test charts, tolerance picked on the same 100 val charts)

| Model | Size | Readings | Cells read ±5 % | v2 table balanced | v2 table false alarms | v2 claims balanced | v1 F1 |
|---|---|---|---|---|---|---|---|
| Oracle (true data) | — | — | 100 % | 0.981 | 0 % | 1.000 | — |
| **Qwen3-VL-4B + our LoRA** | 4B | 1 | 67.9 % | 0.677 | 54 % | 0.782 | 0.766 |
| **Qwen3-VL-4B + our LoRA** | 4B | 9 | 71.3 % | 0.697 | 50 % | 0.785 | 0.752 |
| Qwen3-VL-8B base (FP8) | 8B | 1 | 57.8 % | 0.628 | 56 % | 0.726 | 0.718 |
| Qwen3-VL-4B base | 4B | 1 | 47.3 % | 0.563 | 64 % | 0.670 | 0.726 |
| Qwen3-VL-4B base | 4B | 9 | 48.7 % | 0.571 | 64 % | 0.685 | 0.725 |

Our models run locally on one RTX 4090: ~0.6 s per chart for 9 readings, no per-chart cost.

## When you are done

Commit `frontier_eval/readings/<model>_*.jsonl` and `frontier_eval/results/*_<model>_*.json` to this
branch, plus the exact model ids and settings you used. Do not commit keys.
