# Oráculo: ¿el límite es la lectura o el checker?

Pregunta: cuando el sistema se equivoca (sobre todo las falsas alarmas en gráficos que sí coinciden), ¿es
porque el modelo leyó mal o porque el checker compara mal?

Experimento (`src/oracle.py`): en lugar de la lectura del modelo, se le da al checker **los datos reales
del gráfico** (sacados del LaTeX), sin ruido o con un error relativo aleatorio de σ por celda. El resto
es igual que en `evaluate.py`: mismo checker, umbral elegido en val, mismos 987 gráficos de DaTikZ test
(463 coinciden, 524 tienen un error puesto a propósito).

## 1. Con lectura perfecta el checker casi no falla

| | F1 | Falsas alarmas (gráficos que coinciden) | Recall |
|---|---|---|---|
| **Lectura perfecta (oráculo)** | **0.958** | **0 %** | 0.920 |
| Nuestro mejor modelo (LoRA etapa 2, alta resolución) | 0.793 | 38 % (umbral 5 %) | 0.971 |
| Qwen3-VL-4B base, misma imagen | 0.763 | 44 % (umbral 5 %) | 0.946 |

Lo único que el oráculo no detecta son los cambios más chicos: recall 59 % para cambios de 2 % (umbral
elegido = 2 %), 93 % para 5 %, ≥ 98.5 % de 10 % para arriba. Es decir, falla solo donde el cambio es del
tamaño de la tolerancia.

## 2. Casi todos los errores del modelo son de lectura

Para cada gráfico en que el modelo se equivoca, ¿el oráculo acierta (→ culpa de la lectura) o también
falla (→ culpa del checker o del benchmark)?

| Error del modelo | Por la lectura | Por checker/benchmark |
|---|---|---|
| Falsa alarma | 276 | 0 |
| Error no detectado | 5 | 9 |
| "No estoy seguro" | 31 | 3 |
| **Total (324)** | **312 (96 %)** | **12 (4 %)** |

Los 12 casos del checker/benchmark son cambios de 2 %, justo en el límite de la tolerancia.

## 3. ¿Qué tan bien hay que leer?

F1 si cada celda se lee con un error relativo aleatorio de σ (promedio de 5 semillas, umbral elegido en
val con el mismo ruido):

| σ (error de lectura) | 0 % | 1 % | 2 % | 3 % | 5 % | 8 % | 10 % | 15 % |
|---|---|---|---|---|---|---|---|---|
| F1 | 0.958 | 0.919 | 0.826 | 0.799 | 0.727 | 0.700 | 0.692 | 0.692 |

- Para F1 > 0.9 hay que leer **cada celda con ~1 % de error**. Con 5 % ya cae a 0.73.
- Desde 8 % el F1 es el de marcar todo como error (~0.69): el checker no puede compensar una lectura así.
- Nuestro modelo lee el 69 % de las celdas dentro del 5 %, pero con una cola larga (error relativo medio
  24 %), por eso queda en 0.79.

## 4. Un bug que encontró el oráculo

Con lectura perfecta, el 4.8 % de los gráficos que coinciden salían como error. Causa: `norm()` borra los
símbolos, así que "1.3B" y "13B", "+" y "+++", "I^+" y "I^-" quedaban iguales y el checker emparejaba
columnas cambiadas. Arreglo en `match_names`: primero etiquetas idénticas, después las normalizadas solo
si no son ambiguas (test `test_labels_that_differ_only_in_symbols`). Con eso el oráculo pasa a 0 falsas
alarmas (F1 0.939 → 0.958). En los modelos casi no cambia nada (±0.003 de F1), porque sus errores son
de lectura.

## Qué significa

1. **El checker no es el problema; la lectura sí.** Ajustar tolerancias o el emparejamiento no va a
   mejorar mucho el resultado. El esfuerzo tiene que ir a leer mejor: más precisión por celda (zoom,
   calibrar los ejes con los ticks, lectura por partes) o modelos más fuertes.
2. **Hay un límite físico.** Un cambio de 2 % en una línea sin números escritos es, en una imagen de 1260
   px, del orden de unos pocos píxeles. El benchmark tiene que reportar los resultados por tamaño de
   error y decir hasta dónde es razonable detectar.
3. **Esto ordena la comparación con modelos frontera**: la pregunta es si leen con ~1–2 % de error. Si
   no, tampoco van a verificar bien, por grandes que sean.

Resultados: `oracle_json/` (oráculo y los tres modelos re-evaluados con el checker arreglado).
