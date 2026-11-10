# LoRA de Qwen3-VL-4B (9 nov 2026)

**Entrenamiento:** LoRA r=16 / alpha=32 sobre las capas lineales del modelo de lenguaje (33 M parámetros,
0.74 %), visión congelada. 2.000 gráficos sintéticos (estilos vistos) + 8.000 de ChartQA train, 2 épocas,
lote efectivo 16, lr 1e-4 coseno. RTX 4090, 2 h 17 min, 13.9 GB. Pérdida val 0.57 → 0.10.
Config y curva: `lora_json/train_*`.

## Antes y después (test, mismos sets que el piloto)

| Lector | Set | Celdas ±5 % | F1 τ ajustado (N=1) | F1 adaptativo (N=9) | s/gráfico |
|---|---|---|---|---|---|
| Qwen3-VL-4B base | Sint. vistos | 94.1 % | 0.77 | 0.80 | 0.25 |
| **Qwen + LoRA** | Sint. vistos | **99.1 %** | 0.89 | **0.93** | 0.24 |
| DePlot | Sint. vistos | 98.8 % | — | 0.86 | 0.83 |
| Qwen3-VL-4B base | Sint. no vistos | 92.3 % | 0.80 | 0.84 | 0.26 |
| **Qwen + LoRA** | Sint. no vistos | **98.2 %** | 0.86 | **0.92** | 0.26 |
| DePlot | Sint. no vistos | 97.6 % | — | 0.85 | 0.78 |
| Qwen3-VL-4B base | ChartQA | 87.7 % | 0.89 | 0.90 | 0.28 |
| **Qwen + LoRA** | ChartQA | **97.7 %** | 0.94 | **0.95** | 0.28 |
| DePlot | ChartQA | 96.8 % | — | 0.93 | 1.47 |

DePlot: N=5, mejor F1 adaptativo del piloto.

## Observaciones

1. **El LoRA cierra y supera la brecha con DePlot**: en ChartQA, 87.7 % → 97.7 % de celdas bien leídas
   (DePlot 96.8 %), y F1 de detección 0.95 frente a 0.93, siendo ~5x más rápido.
2. **Generaliza a estilos no vistos** (98.2 %), aunque solo entrenó con los estilos 0–7.
3. **Después del LoRA, la repetición sí ayuda.** Con el modelo base las 9 lecturas eran idénticas; con el
   LoRA hay dispersión útil, y el umbral adaptativo sube el F1 en los tres sets (+0.02 a +0.05 frente al
   umbral fijo ajustado). Es la primera evidencia a favor de la hipótesis principal del proposal.
4. Ojo: ChartQA test es de la misma distribución que ChartQA train. Hace falta un set real fuera de
   distribución (CharXiv / arXiv / PRISMM-Bench) para afirmar que generaliza a papers.
