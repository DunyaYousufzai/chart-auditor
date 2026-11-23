# Comparación con otros modelos abiertos

Mismos 987 gráficos de papers (DaTikZ test, 1260 px), mismo checker, tolerancia elegida en val. Solo
cambia quién lee el gráfico. Benchmark v2 (`bench_v2.py`): "balanceado" = promedio entre aceptar lo
que está bien y detectar cada tipo de error; las falsas alarmas son la parte de lo que está bien que se
marca como error.

| Lector | Tamaño | Lecturas | Celdas ±5 % | v2 tabla: balanceado | v2 tabla: falsas alarmas | v2 afirmaciones: balanceado | v2 afirmaciones: falsas alarmas |
|---|---|---|---|---|---|---|---|
| Oráculo (datos reales) | — | — | 100 % | 0.984 | 0 % | 1.000 | 0 % |
| **Qwen3-VL-4B + nuestro LoRA** | 4B | 9 | **69.0 %** | **0.682** | **49 %** | **0.788** | **25 %** |
| Qwen3-VL-8B base (FP8) | 8B | 9 | 59.8 % | 0.620 | 58 % | 0.727 | 31 % |
| Qwen3-VL-4B base | 4B | 9 | 51.9 % | 0.571 | 64 % | 0.675 | 37 % |
| Qwen3-VL-2B base | 2B | 9 | 40.7 % | 0.498 | 76 % | 0.597 | 45 % |
| DePlot (gráfico → tabla) | 0.3B | 1 | 44.3 % | 0.522 | 72 % | 0.502 | 53 % |
| ChartGemma | 3B | 1 | 31.9 % | 0.400 | 78 % | 0.388 | 64 % |
| UniChart (gráfico → tabla) | 0.3B | 1 | 24.5 % | 0.338 | 88 % | 0.298 | 72 % |

- **Las herramientas que ya existen para pasar gráficos a tablas (DePlot, UniChart, ChartGemma) leen mal
  los gráficos de papers**: fueron entrenadas con gráficos web. Es el mismo problema de dominio del
  Qwen base.
- **La adaptación pesa más que el tamaño**: nuestro 4B adaptado lee 9 puntos mejor que el 8B sin
  adaptar (69.0 % frente a 59.8 %) y gana en las dos partes de v2.
- **El LoRA mejora también las afirmaciones del texto** (0.675 → 0.788) sin haber sido entrenado para eso:
  mejora la lectura y la lectura sirve para cualquier verificación.
- TinyChart no está: necesita el código de su propio repositorio. ChartGemma no documenta un prompt de
  tabla; usamos el único de cuatro probados que devolvía tablas.
- En v1, DePlot tiene un F1 (0.765) casi igual al nuestro leyendo mucho peor, porque el F1 premia marcar
  todo como error. Por eso v2 usa la métrica balanceada.

Resultados: `v2_json/v2_<modelo>_full.json` (987 gráficos) y `_sub.json` (los 200 de la comparación con
modelos frontera), lectura y v1 en `oracle_json/`.
