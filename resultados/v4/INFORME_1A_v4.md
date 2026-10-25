# Línea 1A · versión 4: más muestra de test y objetivo normalizado

Pre-registro `preregistro/preregistro_1A_v4.yaml`, congelado el 2026-10-26 (SHA-256 `fa2a3067…3ba8`) **después**
de comprobar con datos sintéticos que el procedimiento alcanza el objetivo de potencia fijado antes de medirlo.
Tercera reapertura del pre-registro; los ensayos de v1-v3 siguen en `resultados/trials.jsonl`.

## Cambios frente a la v3

| | v3 | v4 |
| --- | --- | --- |
| Datos diarios | desde 2003 | desde 1988 (índice de Jansen desde 1980, VIX desde 1990) |
| Descubrimiento | 2007-04 a 2018-12 | 1993-01 a 2018-12 |
| Bloques de test | 2011-2018, ~83 semanas por fold | **1997-2018, ~230 semanas por fold** (incluye 2000-2002 y 2008-2009) |
| Candidatas | 14 | **10 con historia larga**: F01-F07, F11, F13, F14 (F08-F10 y F12 se quedan con el resultado de la v3) |
| Objetivo | y | **y / σ̂₅** (retorno normalizado por la volatilidad prevista en t) |
| Criterios | media de folds, U2 agregado, GBM con G1 + G3 | iguales |

Las celdas de estrés ya son evaluables: VIX alto en 4 de 5 folds y mercado bajista en los 5. Con la v1-v3 no lo eran en ninguno.

## Potencia (datos sintéticos, `resultados/potencia_v4` y `resultados/potencia_completa_v4`)

El ruido conserva la volatilidad de cada semana (crisis con ruido grande), el efecto se planta en retornos y el R²
se mide en los bloques de test. Es un modelo de ruido más realista que el de las pruebas de la v1-v3.

1. **Objetivo** (regla: y_norm solo si sube la potencia de la etapa 1 en ≥ 5 puntos): F01 lineal R² 2 %:
   88 % con y_norm frente a 68 % con y → **y_norm**.
2. **Procedimiento completo** (objetivo: ≥ 70 % a R² 2 % y falsos positivos ≤ 5 %), 30 réplicas por celda:

   | Efecto plantado | Etapa 1 | Completo |
   | --- | --- | --- |
   | Ninguno | 0/30 | **0/30** |
   | F01 lineal, R² 2 % | 97 % | **93 %** |
   | F01 lineal, R² 4 % | 100 % | 100 % |
   | F06 umbral, R² 4 % | 70 % | 30 % |
   | F06 lineal, R² 4 % | 43 % | 0 % |

   **Limitación**: con ruido alineado con las crisis, una variable como F06 (persistente, de colas gruesas, que
   concentra su varianza en las crisis) casi no se detecta ni con R² 4 %. La potencia de F06 en la v3 estaba
   sobreestimada porque su prueba usaba ruido sin alinear.

## Resultado con los datos reales

**Nulo**: catálogo vacío, validación con la media histórica (R² OOS = 0; Sharpe neto 0,19 con la media desde 1993),
holdout sin abrir.

### Etapa 1 (50 contrastes evaluables; q1 = 1,81; q2 = 0,143)

| Variable | Corte | U1 = t medio | U2 = IC de rangos | Folds con signo | p perm. individual | Mejor sombra la supera |
| --- | --- | --- | --- | --- | --- | --- |
| F14 cambio de mes | mercado alcista | **1,65** | — (dummy) | 4/5 | 0,009 | 12 % |
| F05 cambio del VIX | mercado alcista | 1,31 | 0,142 | 5/5 | 0,002 | 50 % |
| F06 VRP | mercado alcista | 1,27 | 0,048 | 5/5 | 0,011 | 56 % |
| F01 reversión 5d / σ | global | 1,22 | 0,093 | 4/5 | 0,002 | 63 % |
| F05 cambio del VIX | global | 1,22 | 0,112 | 4/5 | 0,002 | 63 % |
| F06 VRP | global | 1,12 | 0,044 | 4/5 | 0,020 | 78 % |

Contraste analítico agregado (U4, todo 1993-2018, NW): **F05 t = 3,06 (p = 0,002)** y **F01 t = −2,84
(p = 0,005)**, ambas con el signo pre-registrado; F06 t = 1,76 (p = 0,08). Con la muestra larga la reversión
semanal es significativa en el agregado, pero no supera el umbral familywise de estabilidad.

### Etapa 2 (excepciones F03, F06, F13)

| Variable | L1 ridge [umbral] | L1 enet [umbral] | G1 [umbral] | G3 [umbral] | Clase |
| --- | --- | --- | --- | --- | --- |
| F03 | 0,026 [0,040] | 0,013 [0,032] | 0,053 [0,084] | 0,098 [0,111] | descartada |
| F06 | 0,005 [0,040] | 0,001 [0,032] | 0,014 [0,084] | 0,061 [0,111] | descartada |
| **F13** | **0,051** [0,040] | **0,042** [0,032] | 0,052 [0,084] | 0,043 [0,111] | descartada |

F13 (semana FOMC) cumple en los dos lineales pero no en el GBM.

## Nula del procedimiento y criterios 7.6

- Nula del procedimiento (100 corridas con y_norm desplazado): 0 variables estables en todas; falsos positivos
  0/100 (IC 95 % 0-3,6 %), coherente con el 0/30 de la prueba de potencia.
- Criterios 7.6: fallan los cinco (`08_criterios_holdout.json`). **El holdout sigue sellado.**

## Qué significa este nulo (el más informativo de las cuatro versiones)

- Con potencia del 93 % para un efecto tipo F01 de R² 2 %, no encontrarlo es evidencia fuerte en contra: la razón
  de verosimilitudes es 0,07 (7 % de fallos si existiera frente a 100 % si no). Para efectos de R² ≥ 4 % es
  prácticamente concluyente.
- El tamaño compatible con lo observado es menor. En la potencia de la etapa 1, el t medio mediano de F01 plantado
  es 2,09 con R² 1 % y 2,96 con R² 2 %; el valor real (1,22) apunta, escalando con la raíz del R², a un R² semanal del orden del 0,3 %. Es
  coherente con el U4 significativo: hay algo de reversión semanal, pero es pequeño, por debajo de lo que este
  diseño puede certificar como estable.
- Para F06 y efectos de umbral persistentes el nulo sigue siendo poco informativo (potencia 0-30 % a R² 4 %).
