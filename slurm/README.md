# Correr en UBELIX

## Configuración (`slurm/site.env`)

| | Valor | Nota |
|---|---|---|
| Cuenta / partición / QoS | `gratis` / `gpu` / `job_gratis` | La misma que funcionó en spec-compression. Alternativa: `teaching` / `teaching` / `job_teaching`. |
| GPU | `gpu:rtx4090:1` (24 GB) | Qwen3-VL-4B en bf16 cabe con margen. Para LoRA/GRPO: `gpu:h100:1` o `gpu:a100:1`. |
| Python con GPU | contenedor `vllm.sif` (vLLM 0.32, transformers 5.19, torch 2.13 cu130) | Reutilizado de spec-compression (enlace duro, no ocupa espacio extra). |
| Python CPU | `~/venvs/chart-auditor-cpu` (módulo `Python/3.11.3`) | matplotlib, pandas, scipy: generar datos y evaluar. |

## Carpetas

| Qué | Dónde | Por qué |
|---|---|---|
| Código | `~/chart-auditor` (rama `bill-base`) | Pequeño. Se actualiza con rsync desde el portátil (o `git pull` cuando haya push). |
| Entorno CPU | `~/venvs/chart-auditor-cpu` | Tiene que durar. |
| Datos, lecturas, pesos, contenedor | `/storage/scratch/users/$USER/chart-auditor` (`PROJ_STORE`) | Grande y regenerable. Cuota de scratch: 30 TB. |
| Resultados finales | `outputs/results/*.json`, `summary.md` | Pequeños; se copian al portátil para el informe. |

`setup_ubelix.sh` crea enlaces simbólicos dentro del clon (`data/synthetic_v2`, `data/chartqa_*`,
`outputs/readings`, `outputs/results`) que apuntan a `PROJ_STORE`, así los comandos son iguales en
local y en el cluster, y git no los ve (están en `.gitignore`).

```
/storage/scratch/users/$USER/chart-auditor/
├── images/vllm.sif
├── data/
│   ├── synthetic_v2/        train / val / test_seen / test_unseen
│   ├── chartqa_raw/         ChartQA original
│   └── chartqa_pairs/       val / test con tabla alterada
├── outputs/
│   ├── readings/            textos crudos de los modelos (.jsonl)
│   └── results/             métricas (.json) + summary.md
├── hf_cache/                pesos de Qwen3-VL y DePlot
└── vllm_cache/
```

## Pasos

```bash
cd ~/chart-auditor
bash slurm/setup_ubelix.sh                 # una vez
bash slurm/sbatch.sh slurm/smoke.sbatch    # prueba con 20 gráficos
cat slurm/logs/ca-smoke-*.out
bash slurm/pilot.sh                        # todos los jobs del piloto
squeue -u $USER
bash slurm/pilot_eval.sh                   # cuando terminen: métricas + summary.md
```

`slurm/sbatch.sh` añade cuenta, partición, QoS y GPU desde `site.env`; cualquier otro argumento
pasa a `sbatch` (por ejemplo `--gres=gpu:h100:1` para cambiar de GPU).
