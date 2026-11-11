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

## LoRA etapa 2: entrenado con gráficos LaTeX reales

Partimos del LoRA anterior (ya fusionado) y entrenamos un segundo LoRA r=16 con **1.047 gráficos pgfplots**
de DaTikZ (arXiv + tex.stackexchange) ×3, más 1.100 sintéticos y 1.100 de ChartQA para no olvidar.
**Ningún paper de test o val aparece en el entrenamiento** (se excluye el paper entero, no solo la figura:
`arxiv_pgf.py pairs --exclude`). 668 pasos, lote efectivo 16, lr 1e-4, RTX 4090, 1 h 05 min, 13.9 GB.
Pérdida val en papers 0.79 → 0.57; en ChartQA no sube (0.02–0.06). Log y config: `real_json/train_*`.

| Lector | Set | Celdas ±5 % | F1 τ ajustado | F1 adaptativo |
|---|---|---|---|---|
| Qwen + LoRA | arXiv, todo | 52.0 % | 0.742 | 0.723 |
| **Qwen + LoRA + papers** | arXiv, todo | **60.6 %** | **0.747** | 0.738 |
| Qwen + LoRA | arXiv, con nombres | 58.8 % | 0.737 | 0.723 |
| **Qwen + LoRA + papers** | arXiv, con nombres | **62.9 %** | 0.745 | 0.737 |
| Qwen + LoRA | ChartQA | 97.7 % | 0.94 | 0.95 |
| **Qwen + LoRA + papers** | ChartQA | 97.6 % | 0.941 | 0.951 |

Por tipo: barras verticales 69 % → **74 %**, líneas 43 % → **53 %**, con números escritos 70 % → **80 %**.

7. **Entrenar con el estilo de los papers sí mejora la lectura** (+8.6 puntos, +10 en líneas) **sin olvidar
   ChartQA** (97.6 %, F1 0.95). La lectura en papers pasa de 44 % (base) a 61 % en dos etapas de LoRA.
8. **Pero la detección no se mueve** (F1 0.742 → 0.747). La mitad de los gráficos correctos se siguen
   marcando como mismatch (56 %). El motivo es estructural: el veredicto del gráfico dice mismatch si
   *cualquier* celda se pasa del umbral, y un gráfico de papers tiene 10–40 celdas. Con 60 % de celdas
   dentro del ±5 %, casi todos los gráficos tienen alguna celda fuera aunque coincidan con la tabla. Para
   bajar las falsas alarmas haría falta ~99 % por celda, o una regla de decisión distinta.

## Regla relativa (resultado negativo) y de dónde vienen las falsas alarmas

**Idea probada:** marcar una celda solo si su error sobresale frente al error típico de ese gráfico
(`rel` × mediana de los errores), opcionalmente tras una calibración robusta chart ≈ a·tabla + b
(Theil-Sen) para absorber un eje mal leído (`checker.compare(rel=..., calibrate=...)`, ajustado en val).

**No funciona.** En papers baja el F1 en los cuatro lectores (LoRA + papers: 0.753 → 0.719); en ChartQA y
sintéticos queda igual o algo peor (salvo DePlot en sintéticos vistos, +0.02). Queda como opción apagada.

**Por qué:** la suposición era que el error de lectura es ruido parejo. Midiendo en los gráficos que sí
coinciden (LoRA + papers, test):

| | Gráficos correctos | Gráficos alterados |
|---|---|---|
| Con alguna celda > 5 % | 59 % | 91 % |
| Celdas fuera, cuando hay | 5 de 14 | 3 de 11 |
| Error típico (mediana) | 1.0 % | 1.7 % |
| Error máximo (mediana) | 9.9 % | 59.7 % |

La lectura es muy precisa en la mayoría de las celdas, pero cuando falla en un gráfico correcto falla en
**varias celdas a la vez y por mucho** (58 % de las falsas alarmas tienen alguna celda > 30 %). Ese patrón
no se distingue de una tabla alterada mirando solo errores. Variante "si hay muchas celdas fuera, decir *no
sé*": baja las falsas alarmas de 51 % a 20 % en val, pero a costa de responder *no sé* en 39 % de los
gráficos; tampoco sube el F1.

Desglose de las 267 falsas alarmas: 79 % en gráficos de varias series, 71 % en líneas; cambiar el orden de
las series solo arregla 1, así que el lector no confunde colores de la leyenda.

**Bug encontrado en el camino (commit 9a8ebe6):** en el 20 % de las falsas alarmas el comparador
emparejaba etiquetas numéricas distintas: `norm()` borra el punto, así que "1.5" era igual a "15", y la
coincidencia aproximada juntaba "0.5" con "5". En ChartQA casi no pasa (las etiquetas son categorías y
años); en papers, con ejes 0.1, 0.5, 1e-3, mucho. Ahora las etiquetas numéricas solo se emparejan si son el
mismo número, y nunca por orden. Todo se re-evaluó (`checker_v3_json/`): sintético y ChartQA no cambian
(±0.01), papers sube un poco (LoRA + papers 0.747 → **0.753**) y más gráficos pasan a *no sé* (5 % → 9 %)
porque ya no se comparan celdas mal emparejadas.

## Revisión a ojo de las falsas alarmas

12 falsas alarmas al azar del LoRA + papers (imagen, tabla real y lectura, lado a lado):

| Tipo | Casos | Ejemplo |
|---|---|---|
| Tolerancia demasiado estricta: el lector acierta a 1–2 px, pero el valor es chico frente al eje | 5 | 1.7 vs 1.29 en un eje 0–60 (0.7 % del eje, 32 % del valor); lecturas redondeadas a enteros en un eje 20–60 |
| Error real del lector | 6 | números escritos sobre las barras, chicos o rotados, mal leídos o cambiados de serie; una barra chica leída como 0; dos curvas inventadas planas; puntos donde las líneas se tocan |
| Imagen rota | 1 | etiquetas del eje encimadas ("FReent Own") |

Medido sobre las 246 falsas alarmas, con el peor error expresado como fracción del rango de valores del
gráfico (≈ el eje):

| Peor error / rango | Falsas alarmas | Detecciones correctas |
|---|---|---|
| < 4 % (problema de tolerancia) | 37 (15 %) | 32 |
| 4–10 % | 31 | 43 |
| ≥ 10 % (lectura mal) | **178 (72 %)** | 363 |

La muestra exageraba el problema de tolerancia: **la mayoría de las falsas alarmas son lecturas muy
erróneas** (≥ 10 % del eje), no umbrales mal puestos. Medir el error contra el rango del eje quitaría
~37 falsas alarmas pero también ~32 detecciones, así que no compensa. El cuello de botella es el lector.
Dos pistas concretas: las imágenes son de 420 px (texto de 5–6 px de alto) y los errores grandes se
concentran en números escritos chicos/rotados y en líneas que se tocan.

## Alta resolución: los mismos gráficos recompilados a 1260 px

Recompilamos el código TikZ de cada gráfico (`src/rerender.py`, `slurm/rerender.sbatch`: pdflatex sin
shell-escape, clase `standalone`, recorte y rasterizado con el lado mayor = 1260 px). Compilan 987 de 995
gráficos de test y 389 de 399 de val. De paso: DaTikZ **estira todo a un cuadrado de 420×420**; la versión
nueva conserva la proporción real. Comparamos sobre exactamente los 987 gráficos y las mismas tablas
(umbral elegido en val, como siempre); lo único que cambia es la imagen.

| Lector | Imagen | Celdas ±5 % | Error rel. | P | R | F1 | Falsa alarma en match | Not sure |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B base | 420 px | 43.7 % | 0.457 | 0.625 | 0.923 | 0.745 | 49.4 % | 22.8 % |
| Qwen3-VL-4B base | 1260 px | 51.9 % | 0.388 | 0.641 | 0.947 | 0.764 | 49.2 % | 18.3 % |
| LoRA + papers | 420 px | 59.4 % | 0.304 | 0.640 | 0.914 | 0.753 | 52.1 % | 9.1 % |
| LoRA + papers | 1260 px | **68.6 %** | **0.238** | **0.689** | 0.892 | **0.777** | **41.7 %** | 7.8 % |

- **La resolución explica una parte grande del error**: +8.2 puntos de lectura para Qwen base y +9.2 para
  el LoRA, sin cambiar el modelo. El LoRA se entrenó con imágenes de 420 px y aun así mejora con 1260 px.
- **Resolución y LoRA se suman**: de 43.7 % (base, 420 px) a 68.6 % (LoRA, 1260 px).
- **Es la primera mejora clara de detección en papers**: el F1 del LoRA sube de 0.753 a 0.777 y las falsas
  alarmas sobre gráficos correctos bajan de 52 % a 42 %. Para Qwen base sube el F1 (0.745 → 0.764) por
  recall y menos "not sure", pero las falsas alarmas siguen en ~49 %: leer mejor no alcanza si la lectura
  todavía falla en algunas celdas de casi todos los gráficos.
- Costo: ~0.6 s por gráfico con N = 9 (frente a ~0.4 s a 420 px).
- Siguiente paso obvio: re-entrenar el LoRA etapa 2 con imágenes en alta resolución.

## Qué sigue (ideas)

- **LoRA con imágenes en alta resolución**: el set de entrenamiento también tiene código TikZ, así que se
  puede recompilar a 1260 px y re-entrenar.
- **Más datos de papers**: con 1.047 gráficos el LoRA ya sube 8.6 puntos; DaTikZ tiene más figuras TikZ
  que se podrían recompilar con variaciones de estilo.
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
# alta resolución: sbatch --nodes=1 --partition=epyc2 --account=gratis --qos=job_gratis slurm/rerender.sbatch
# lecturas: slurm/read.sbatch (ver slurm/README.md); evaluación: src/evaluate.py con --val-manifest
```

Subset "con nombres": líneas del manifest cuya tabla tiene una sola serie o ninguna llamada `series_*`.
