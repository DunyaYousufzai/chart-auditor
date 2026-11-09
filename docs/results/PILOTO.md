# Resultados del piloto (9 nov 2026)

Modelos sin ajustar. Lecturas en UBELIX (RTX 4090, vLLM 0.32). Umbral τ y k ajustados en val de cada fuente.
Tabla completa: `pilot_summary_raw.md`; métricas detalladas por ejecución: `pilot_json/`.

## Sets

| Set | Gráficos | Qué es |
|---|---|---|
| Sintético, estilos vistos | 200 | matplotlib, estilos 0–7, 1–3 series |
| Sintético, estilos no vistos | 200 | hojas de estilo ggplot / bmh / fivethirtyeight / Solarize |
| ChartQA test | 1000 | gráficos reales (Statista, Pew, OWID) con la tabla alterada (sin tortas) |

Mitad match, mitad mismatch (valor cambiado 2–50 %, swap de etiquetas, coma decimal).

## Tabla principal (test)

| Lector | Set | Celdas ±5 % | F1 τ=5 % | F1 τ ajustado | F1 adaptativo | s/gráfico |
|---|---|---|---|---|---|---|
| DePlot (N=5) | Sint. vistos | 98.8 % | 0.87 | 0.86 | 0.86 | 0.83 |
| | Sint. no vistos | 97.6 % | 0.83 | 0.86 | 0.85 | 0.78 |
| | ChartQA | 96.8 % | 0.85 | 0.93 | 0.93 | 1.47 |
| Qwen3-VL-4B (N=9) | Sint. vistos | 94.3 % | 0.81 | 0.81 | 0.80 | 0.25 |
| | Sint. no vistos | 92.1 % | 0.80 | 0.80 | 0.84 | 0.26 |
| | ChartQA | 87.4 % | 0.83 | 0.90 | 0.90 | 0.28 |
| B0: Qwen juez directo | Sint. vistos | — | 0.28 | — | — | 0.09 |
| | Sint. no vistos | — | 0.19 | — | — | 0.10 |
| | ChartQA | — | 0.28 | — | — | 0.11 |

s/gráfico de Qwen incluye las 9 lecturas (vLLM las genera en paralelo).

## Observaciones

1. **Pipeline >> juez directo.** B0 casi siempre responde MATCH (recall 10–16 %). Coincide con Ho et al. y PRISMM-Bench.
2. **DePlot es el mejor lector, pero juega en casa en ChartQA**: el checkpoint público está ajustado sobre ChartQA. Qwen es ~5x más rápido por lectura. El LoRA busca cerrar la brecha 87 % → 97 %.
3. **La repetición no da señal de incertidumbre con Qwen**: las 9 muestras (temperatura 0.7) son casi idénticas (dispersión relativa mediana 0). N=1, 3, 5 y 9 dan lo mismo y el umbral adaptativo apenas cambia el F1 (salvo +0.04 en estilos no vistos). La hipótesis del proposal no se confirma con muestreo; alternativas: probabilidades de los tokens numéricos, o lecturas con variaciones de la imagen (recorte, escala).
4. **τ = 5 % es demasiado laxo para ChartQA**: el valor ajustado en val es 2 % y sube el F1 de ~0.84 a ~0.91.
5. **Errores de formato de Qwen** (tabla transpuesta, filas sin etiqueta) explicaban la mayor parte de su error inicial (cell acc 60 % → 94 % tras arreglar el parser). El LoRA también enseña el formato.
