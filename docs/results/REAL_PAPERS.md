# Gráficos de papers reales (11 nov 2026)

Hasta aquí todo era sintético o ChartQA (Statista/Pew), que es justo donde entrenamos el LoRA y donde
DePlot fue ajustado. Aquí probamos fuera de distribución: gráficos pgfplots de papers de arXiv.

## Set: DaTikZ-v3 (arXiv)

[DaTikZ-v3](https://huggingface.co/datasets/nllg/datikz-v3) trae el código TikZ de figuras de arXiv junto
con la imagen renderizada. De ahí sacamos la **tabla exacta** desde el código LaTeX
(`src/datikz_pgf.py` + `src/arxiv_pgf.py`), sin leer nada a ojo:

- solo pgfplots de barras o líneas, 1–4 series con la misma x, 2–15 puntos;
- fuera: ejes log, polares/3D, colorbar, puntos fuera de los límites del eje, etiquetas de eje ocultas,
  leyendas dibujadas fuera de la imagen (`legend to name`);
- 2.252 gráficos válidos; **995 test** (620 líneas, 330 barras verticales, 45 horizontales) y **399 val**,
  sin solapamiento; mitad match / mitad mismatch con las mismas perturbaciones que antes.

Subset **"con nombres"** (836 test / 333 val): gráficos de una serie, o con leyenda que nombra las series.
Los otros 159 tienen varias series sin leyenda: ni una persona sabe qué línea es cuál, así que el
emparejamiento de series es ambiguo.

## Resultados (test)

| Lector | Set | Celdas ±5 % | F1 τ ajustado | F1 adaptativo | s/gráfico |
|---|---|---|---|---|---|
| Qwen3-VL-4B base (N=9) | arXiv, todo | 44.3 % | 0.745 | 0.740 | 0.36 |
| **Qwen + LoRA** (N=9) | arXiv, todo | **52.0 %** | 0.742 | 0.723 | 0.30 |
| DePlot (N=5) | arXiv, todo | 40.9 % | 0.740 | 0.742 | 2.81 |
| Qwen3-VL-4B base | arXiv, con nombres | 46.0 % | 0.736 | 0.745 | |
| **Qwen + LoRA** | arXiv, con nombres | **58.8 %** | 0.737 | 0.723 | |
| DePlot | arXiv, con nombres | 45.4 % | 0.744 | 0.742 | |
| *Marcar todo como mismatch* | | — | 0.667 | | |
| *Referencia: Qwen + LoRA en ChartQA* | | *97.7 %* | *0.94* | *0.95* | |

τ y k ajustados en val (τ = 2–5 %). JSONs: `real_json/`.

Celdas ±5 % por tipo (todo el test, mediana de N):

| | Líneas (620) | Barras V (330) | Barras H (45) | Con números escritos (241) | Sin números (754) |
|---|---|---|---|---|---|
| Qwen base | 39.2 % | 52.4 % | 55.3 % | 58.5 % | 39.7 % |
| Qwen + LoRA | 42.5 % | **69.2 %** | 57.6 % | **69.8 %** | 46.3 % |
| DePlot | 34.3 % | 53.4 % | 38.6 % | 51.9 % | 37.3 % |

## Observaciones

1. **El salto de dominio es enorme.** El mismo LoRA que lee 97.7 % de las celdas en ChartQA lee 52 % en
   arXiv. Los gráficos de papers son líneas densas sin números escritos, con ejes en notación científica,
   multiplicadores (`·10⁻²`), muchas series y leyendas pequeñas.
2. **El LoRA sigue siendo el mejor lector** (+8 puntos sobre la base, +13 en el subset con nombres, 69 % en
   barras), aunque nunca vio un gráfico LaTeX. DePlot, que en ChartQA iba casi a la par, queda último y es
   ~9x más lento.
3. **La detección apenas supera a no hacer nada.** F1 ≈ 0.74 con los tres lectores, frente a 0.667 de marcar
   todo como mismatch. El recall es alto (0.90–0.96; decimal y swap > 0.85), el problema es la precisión
   (~0.61): la mitad de los gráficos que sí coinciden se marcan como mismatch, porque la lectura de líneas sin
   números (interpolar píxeles) no llega a la precisión de 2–5 % que pide el comparador. Cambios del 2 % se
   detectan solo ~55 % de las veces.
4. **Más lectura no se traduce en mejor detección.** El LoRA lee mejor que DePlot pero tiene el mismo F1:
   el umbral ajustado se acomoda al error de cada lector. Hace falta un salto grande de precisión, no uno
   pequeño.
5. **El umbral adaptativo no ayuda aquí** (con el LoRA incluso baja de 0.742 a 0.723): la dispersión entre
   las 9 lecturas no anticipa cuándo la lectura está mal.
6. **Construir el benchmark también fue parte del trabajo.** La primera corrida (`*_v1_*`) dio 41–49 % de
   celdas; revisando falsas alarmas vimos que muchas eran culpa de nuestras tablas de referencia, no del
   lector: etiquetas como `($3\%, 20\%$)` partidas en la coma, etiquetas asignadas sin mirar `xtick`, ejes con
   etiquetas ocultas y leyendas fuera de la imagen. Con el extractor corregido (commit c1b13a3) el LoRA pasa
   de 48.9 % a 52.0 %.

## Qué sigue (ideas)

- **LoRA con gráficos de papers reales**: quedan ~850 gráficos DaTikZ sin usar (más los de arXiv 2026), con
  tabla exacta. Es un set de entrenamiento con el estilo LaTeX, que no tiene ningún trabajo previo que
  conozcamos.
- **Lectura agéntica**: leer primero los ticks y el multiplicador del eje, recortar/ampliar cada serie y
  convertir píxeles a valores; el modelo decide qué zona mirar.
- **Tolerancia según lo que el gráfico permite**: un valor escrito sobre la barra se puede verificar al
  0.1 %; un punto de una línea sin número, como mucho al ~5 % del rango del eje.

## PRISMM-Bench: errores reales señalados por revisores

[PRISMM-Bench](https://huggingface.co/datasets/daluggas/PRISMM-Bench) recoge 384 inconsistencias reales
que revisores encontraron en papers de OpenReview. 10 son figura-tabla y **solo 3 son numéricas**
(`src/prismm_cases.py`: `8TyGCAuCGd_0`, `yEnJvc7ogD_1`, `CfXRcN4iUw_0`). Ninguna es "la celda X vale
distinto en el gráfico y en la tabla": son afirmaciones derivadas (el máximo de una curva frente a una
celda, un orden entre métodos, un gráfico de varios paneles). Quedan como caso de estudio: muestran que los
errores reales piden razonar sobre lo leído, que es el argumento para un enfoque con agente.

## CharXiv

CharXiv tiene gráficos reales de arXiv, pero solo preguntas y respuestas, sin la tabla de datos, así que no
sirve para medir lectura celda a celda ni para verificar contra una tabla.

## Reproducir

```bash
python src/datikz_pgf.py --origin arxiv
python src/arxiv_pgf.py pairs --figs data/datikz_pgf/figs --out data/datikz_pgf --origin arxiv --limit 1000
python src/arxiv_pgf.py pairs --figs data/datikz_pgf/figs --out data/datikz_pgf --origin arxiv --split val --skip 1000 --limit 400
# lecturas: slurm/read.sbatch (ver slurm/README.md); evaluación: src/evaluate.py con --val-manifest
```

Subset "con nombres": líneas del manifest cuya tabla tiene una sola serie o ninguna llamada `series_*`.
