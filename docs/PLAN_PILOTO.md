# Plan piloto: Cross-Modal Scientific Data Auditor

Borrador para el equipo (Dunya, Adriana, Bill). 9 nov 2026.
Se basa en el proposal del 20 oct y en el estado del repo en `cd2fa33`.

---

## 1. Dónde estamos

| Pieza | Estado | Comentario |
|---|---|---|
| Generador sintético (`src/generate_dataset.py`) | Funciona | Solo pares limpios ("match"). No genera mismatches todavía. |
| Backend FastAPI (`backend/app`) | Esqueleto | `reader.py` es falso (no mira la imagen). `checker.py` usa tolerancia fija 5 %. |
| Frontend React | Esqueleto | `handleCheck` devuelve un resultado falso; no llama al backend. |
| Modelos (lectores, mapper) | Nada | Ni DePlot, ni Qwen3-VL, ni scripts de Slurm. |
| Extracción desde PDF | Nada | Es el riesgo más grande según el propio proposal. |
| Evaluación / métricas | Nada | |

### Problemas encontrados al correr el generador

1. **Bug de la grilla**: `ax.grid(style["grid"], axis="y", alpha=0.4)` activa la grilla aunque
   `style["grid"]` sea `False`, porque matplotlib la enciende cuando recibe kwargs
   (sale un `UserWarning` en cada gráfico). Hoy todos los gráficos tienen grilla.
   Arreglo: `if style["grid"]: ax.grid(True, axis="y", alpha=0.4)` y si no, `ax.grid(False)`.
2. **Los estilos "unseen" no son realmente nuevos**: todas las propiedades salen de `i % 2`,
   `i % 3`, `i % 4`, `i % 5`, así que los estilos 8–11 reutilizan las mismas fuentes, tamaños,
   fondos y dpi que 0–7; solo cambia el color y la combinación. Para medir generalización de
   verdad hace falta otra *librería* de gráficos (Plotly o Vega-Altair) o, mejor, gráficos reales.
3. **Solo una serie por gráfico**. En papers casi todo son barras agrupadas o varias líneas
   (método × dataset). Sin multi-serie, el mapeo de etiquetas (paso 3) es trivial y no se puede evaluar.
4. **Tablas aleatorias**, no de papers reales (el proposal dice PubMed Central). Para el piloto
   está bien; se puede mejorar después.

---

## 2. Los dos enfoques

| | A. Pipeline modular (el del proposal) | B. Extremo a extremo |
|---|---|---|
| Idea | VLM lee el gráfico → tabla; LLM mapea etiquetas; Python compara | Un VLM ve gráfico + tabla y responde "match / mismatch" |
| Ventajas | Explicable (dice qué número difiere), aritmética exacta, permite el umbral con incertidumbre | Simple, un solo modelo |
| Desventajas | Más piezas, errores se acumulan | Los modelos pequeños son malos en esto (Ho et al., PRISMM-Bench); no explica |
| Uso en el proyecto | **Sistema principal** | **Baseline B0** |

Recomendación: A como sistema, B como baseline. Así ambas propuestas quedan en el informe y
la comparación A vs. B es en sí un resultado.

---

## 3. Las bases de datos

| Dataset | Para qué sirve de verdad | Veredicto |
|---|---|---|
| **ChartQA** | Gráficos reales (Statista, Pew, OWID) **con su tabla original**. Sirve para LoRA y para evaluar lectura gráfico→tabla. | **Imprescindible.** Es nuestro "real" principal en volumen. |
| **CharXiv** | Gráficos de arXiv (estilo paper real), pero solo pregunta-respuesta, no tablas completas. | **Útil para evaluar lectura de números sueltos** en estilo científico. No sirve para entrenar. |
| **PRISMM-Bench** | Inconsistencias reales marcadas por revisores (384 en 353 papers ICLR). Multiple choice. | **Solo evaluación final, y pequeña.** Hay que filtrar los casos gráfico-vs-tabla; probablemente queden decenas. Revisar pronto cuántos son. |

Propuesta extra que resuelve la falta de datos reales con etiqueta:
**"semi-sintético real"**: tomar gráficos *reales* de ChartQA (y unos 30–50 de papers arXiv
con gráfico + tabla de los mismos datos, pareados a mano) y **perturbar la tabla**, no el
gráfico (cambiar un valor 2/5/10/25/50 %, cambiar escala ×1000, intercambiar dos etiquetas,
quitar una fila). Así tenemos imágenes reales con verdad conocida y miles de ejemplos
de mismatch, sin dibujar nada.

---

## 3b. Qué tomamos de otros trabajos y qué hacemos distinto

| Tomamos | De | Lo que cambiamos / añadimos |
|---|---|---|
| Leer el gráfico N veces, mediana por celda, parar cuando es estable | Self-ensembling (Berkane et al. 2026) | Ellos solo extraen datos. Nosotros usamos la **dispersión como tolerancia** del chequeo (`max(τ, k·s)`) y como criterio de "no sé". |
| DePlot como lector gráfico → tabla, métrica RNSS | DePlot / Pix2Struct / ChartQA | Lo usamos como baseline, con el mismo chequeo encima. |
| Gráficos reales con su tabla | ChartQA | **Alteramos la tabla, no el gráfico**: imágenes reales con verdad conocida y mismatches de tamaño controlado (2–50 %, swap, coma decimal, cambio de unidad). No existe ese set. |
| Inconsistencias reales de revisores | PRISMM-Bench | Lo convertimos de multiple choice a "gráfico vs. tabla → veredicto". |
| Unidad (miles, millones) la resuelve un LLM en el proposal | — | Búsqueda determinista del factor de escala en Python; el LLM solo para lo que Python no resuelve. |

## 4. Alcance del piloto (1–2 días)

Objetivo: **un corte vertical que funcione de punta a punta** con números reales, antes de
entrenar nada.

- **Entrada**: imagen del gráfico + CSV de la tabla (sin PDF todavía).
- **Datos**: sintético v1 (con mismatches) + 300 ejemplos de ChartQA test con tabla perturbada.
- **Lectores**: DePlot y Qwen3-VL-4B-Instruct zero-shot, servido con vLLM en UBELIX.
- **Repetición**: N = 5 lecturas con temperatura > 0; mediana por celda, dispersión = incertidumbre.
- **Checker**: fijo (τ = 5 %) vs. con incertidumbre `|c − t| > max(τ, k·s)`; k y τ ajustados en val.
- **Mapper**: para series simples, matching de strings (exacto / fuzzy). LLM solo cuando haga falta (fase 2).
- **Salida**: una tabla de resultados con P/R/F1 de mismatch y error de lectura por lector.

**Criterio de éxito del piloto**: tener la fila "Qwen3-VL-4B zero-shot + N=5 + umbral adaptativo"
con números, comparada con DePlot y con B0, en sintético y en ChartQA-perturbado.

---

## 5. Fases después del piloto

| Fase | Contenido | Tema del curso | Cómputo |
|---|---|---|---|
| 1. Piloto | Lo de arriba | Prompting, self-consistency, salida estructurada | Inferencia, 1 GPU |
| 2. LoRA SFT | Qwen3-VL-4B en sintético multi-serie + ChartQA train. Evaluar en estilos no vistos y en CharXiv | **Fine-tuning (LoRA/QLoRA)** | 1 GPU A100/H100, pocas horas |
| 3. Mapper agente | LLM con herramientas: `read_chart`, `parse_table`, `compare` (Python). Decide unidades, pide relectura si la dispersión es alta, vota | **Agentes / tool calling** | Inferencia |
| 4. GRPO (opcional) | Recompensa = % de celdas dentro de tolerancia + formato válido. Sin etiquetas humanas | **RL con recompensa verificable** | 1–2 GPU grandes; cortar si es inestable |
| 5. Aceleración | FP8 / AWQ 4-bit / GGUF, imágenes más chicas, early stop del muestreo | **Inferencia eficiente** | Medición |
| 6. PDF + real | PDFFigures 2.0 / GROBID, pareo gráfico-tabla por caption; evaluación en PRISMM-Bench filtrado | Sistema completo | CPU + inferencia |

Fase 3 cubre el tema "agentes" del curso sin inventar nada: el mapeo y el "¿vuelvo a leer?" son
decisiones naturales para un agente, y la aritmética la hace siempre una herramienta Python.

---

## 6. UBELIX / Slurm

Uso previsto (Bill tiene acceso):

- **Inferencia por lotes** (N lecturas × miles de gráficos): vLLM con Qwen3-VL-4B en una GPU. Un job por lector/configuración.
- **LoRA**: Qwen3-VL-4B en bf16 + LoRA cabe en una GPU de 40–80 GB; con ~10k ejemplos y 2 épocas, del orden de horas.
- **GRPO**: el más caro (generación + entrenamiento); solo si sobra tiempo.
- Pasos prácticos: ver particiones y tipos de GPU con `sinfo -p gpu` / la doc de UBELIX; entorno con
  conda o Apptainer; modelos y datos en `$SCRATCH` (caché de Hugging Face fuera del home); un `slurm/`
  en el repo con un `.sbatch` por experimento y logs versionados.
- Todo lo de CPU (generador, checker, métricas, frontend) se corre en local.

---

## 7. Estructura del código (rama `bill-base`)

```
src/
  tables.py            formato de tabla común + parser de lo que escriben los modelos
  perturb.py           crea la "tabla del paper": match / rescaled / value / swap / decimal
  generate_dataset.py  sintético v2: mismatches, multi-serie, estilos no vistos reales
  chartqa_pairs.py     pares semi-sintéticos sobre gráficos reales de ChartQA
  readers/             deplot.py, qwen_vl.py (vLLM y transformers), simulated.py, prompts.py
  run_reader.py        corre un lector sobre un manifest y guarda los textos crudos
  ensemble.py          mediana + dispersión por celda, parada temprana
  checker.py           alineación de etiquetas, factor de escala, tolerancia fija/adaptativa
  metrics.py           lectura (cell acc, error relativo, RNSS) y detección (P/R/F1)
  evaluate.py          ajusta τ, k en val y evalúa en test
  summarize.py         junta los resultados en una tabla markdown
slurm/                 setup, prueba rápida, jobs de lectura, piloto completo (ver slurm/README.md)
tests/                 tests de las partes en Python puro
```

El backend importa `checker` y `tables` desde `src/`; ya no tiene copia propia.

---

## 8. Calendario comprimido (~2 semanas, depende de la cola de UBELIX)

| Cuándo | Entregable |
|---|---|
| Día 1–2 | Generador arreglado (grilla, mismatches, multi-serie). ChartQA con tablas alteradas. Comparador con umbral adaptativo y métricas. Lectores DePlot y Qwen3-VL. Primer job en UBELIX. **Primera tabla de resultados (fin del piloto).** |
| Resto de la semana 1 | LoRA SFT. Evaluación en estilos no vistos, ChartQA, CharXiv. Mapper agente. Cuantización. |
| Semana 2 | GRPO si hay margen; si no, ablaciones. Pareo PDF en 30–50 papers. PRISMM-Bench real, frontend conectado, informe. |

---

## 9. Decisiones pendientes

1. Fecha de entrega final y si hay presentación intermedia.
2. ¿Confirmamos A como sistema y B como baseline?
3. ¿Aceptamos el set semi-sintético (tabla perturbada sobre gráfico real) como evaluación "real" principal, con PRISMM-Bench como caso de estudio pequeño?
4. ¿GRPO entra o queda opcional desde ya?
