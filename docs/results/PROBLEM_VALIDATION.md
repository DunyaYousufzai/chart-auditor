# ¿El problema existe en papers reales?

Nuestro benchmark crea los errores a propósito (valor cambiado 2–50 %, filas intercambiadas, decimal
movido). Antes de seguir hay que saber si ese tipo de error aparece de verdad en papers. No encontramos
ningún dataset de inconsistencias reales gráfico–tabla ni un estudio que mida su frecuencia, así que
revisamos los dos datasets que sí tienen **errores reales**:

- **PRISMM-Bench** ([arXiv 2510.16505](https://arxiv.org/abs/2510.16505), ICLR 2026): 384 inconsistencias
  señaladas por revisores en OpenReview (ICLR 2024–2025), 353 papers.
- **SPOT** ([arXiv 2505.11855](https://arxiv.org/abs/2505.11855),
  [datos](https://huggingface.co/datasets/amphora/SPOT-MetaData), CC-BY 4.0): 91 errores que llevaron a
  una errata (59) o a una retractación (32), en 83 papers publicados de todas las áreas.

## Método

Cada caso se etiquetó a mano, a partir de su descripción, según si para detectarlo **hay que leer los
datos de un gráfico** y con qué se compara:

| Etiqueta | Qué es | Ejemplo real |
|---|---|---|
| `numeric_chart_vs_table_or_text` | un valor del gráfico no coincide con la tabla o el texto (nuestro caso actual) | "la mejor accuracy de la Tabla 1 no coincide con la Figura 2(c)" |
| `ranking_chart_vs_table` | el orden entre métodos cambia entre gráfico y tabla | "la tabla dice que A supera a B; en el gráfico B supera a A" |
| `text_claim_vs_chart` | una afirmación del texto (tendencia, comparación) contradice el gráfico | "el texto dice que usa menos memoria GPU; la figura muestra lo contrario" |
| `chart_vs_chart` | dos gráficos del paper se contradicen o uno es copia del otro | "la curva de [H+]=3.5 es una copia de la de 3.0" |
| `chart_internal_sanity` | el gráfico es imposible o incoherente en sí mismo | "barras de error por debajo de 0 en una tasa de éxito" |
| `no_chart_data` | diagramas, imágenes, nombres, ecuaciones, referencias | — |

Etiquetas por caso: `validation/prismm_chart_tags.csv` y `validation/spot_chart_tags.csv`.

## Resultados

| Etiqueta | PRISMM-Bench (n = 384) | SPOT (n = 91) |
|---|---|---|
| `numeric_chart_vs_table_or_text` | 3 (0.8 %) | 1 (1.1 %) |
| `ranking_chart_vs_table` | 3 (0.8 %) | 0 |
| `text_claim_vs_chart` | 20 (5.2 %) | 0 |
| `chart_vs_chart` | 5 (1.3 %) | 1 (1.1 %) |
| `chart_internal_sanity` | 20 (5.2 %) | 1 (1.1 %) |
| **Total que necesita leer un gráfico** | **51 (13.3 %)** | **3 (3.3 %)** |
| `no_chart_data` | 333 (86.7 %) | 88 (96.7 %) |

En SPOT dominan las demostraciones matemáticas (37) y las imágenes duplicadas (27); no hay casi gráficos
de datos porque muchos papers son de matemática y biología.

## Qué significa

1. **Nuestro caso exacto es real pero raro.** "Una celda del gráfico distinta de la tabla" aparece en
   ~1 % de los errores reales. Si el proyecto solo afirma resolver eso, resuelve un problema marginal.
2. **El problema más amplio sí es frecuente.** En PRISMM, **1 de cada 7–8 inconsistencias** que
   encontraron los revisores necesita leer los datos de un gráfico; la mayoría compara el gráfico con una
   **afirmación del texto** o revisa que el gráfico tenga sentido. Y los modelos actuales resuelven
   PRISMM mal (27.8–53.9 % según sus autores).
3. **La pieza común es la lectura.** Todas esas categorías necesitan lo que ya construimos: leer valores
   exactos del gráfico. Cambia con qué se comparan (tabla, texto, otro gráfico, sentido común).

**Conclusión:** el proyecto se sostiene si el objetivo pasa de "detectar celdas distintas entre gráfico y
tabla" a **"leer los datos de gráficos científicos y verificarlos contra el resto del paper"**. El
benchmark actual queda como la parte gráfico–tabla de esa tarea.

## Cómo hacer el benchmark más realista

Tipos de error a agregar, sacados de los casos reales y generables con los datos exactos que ya tenemos
de DaTikZ:

| Error nuevo | Basado en | Cómo generarlo |
|---|---|---|
| Afirmación del texto falsa | `text_claim_vs_chart` (20 casos) | frases desde los datos ("A supera a B en todo x", "la curva se estanca en ~75 %") y su versión falsa |
| Orden cambiado | `ranking_chart_vs_table` | invertir el orden de dos métodos en la tabla sin cambiar mucho los valores |
| Versión vieja | errata típicas de "se volvió a correr" | mover todas las celdas un poco a la vez (ruido de 1–5 %) |
| Curva duplicada | SPOT (XRD copiado), PRISMM 297 | copiar una serie sobre otra en la tabla |
| Valor imposible / ejes | `chart_internal_sanity` | sin tabla: tasas > 100 % o < 0, ticks no uniformes |

## Límites de esta revisión

- Etiquetado por una sola persona desde descripciones cortas; falta un segundo anotador y medir acuerdo.
- PRISMM solo tiene lo que los revisores **notaron**: errores numéricos finos (un 5 % de diferencia en una
  curva) son justo los que un humano no ve, así que la frecuencia real puede ser mayor.
- SPOT es chico y mezcla todas las áreas.
- Falta la medición directa: comparar, en el código LaTeX de papers de arXiv, los números de los gráficos
  pgfplots con las tablas y el texto del mismo paper.
