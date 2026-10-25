# Línea 1A · versión 2: mediana de los folds

Pre-registro `preregistro/preregistro_1A_v2_mediana.yaml` (2026-10-25). Es una **reapertura del pre-registro**
hecha después de ver los resultados de la v1, así que es un ensayo nuevo: sus entradas en `resultados/trials.jsonl`
llevan `version_prereg = v2_mediana` y se suman a las de la v1 para el DSR y el PBO.

**Único cambio**: los estadísticos de estabilidad se resumen con la **mediana de los 5 folds** en vez del peor fold,
en la etapa 1 (U1, U2) y en la etapa 2 (L1 de ridge y elastic net, G1, G2, I1). G3 ya era una mediana. Datos,
candidatas, signos, folds, sombras, semillas y modelos son los mismos de la v1.

## Resultado

**Sigue siendo nulo.** Ninguna candidata pasa la etapa 1; F06 (VRP) cumple ahora en los dos modelos lineales pero
no en el GBM, así que como excepción teórica tampoco entra. Catálogo vacío, validación igual a la v1 (media
histórica, R² OOS = 0) y holdout sin abrir.

## Etapa 1

La mediana sube las candidatas, pero también sube la nula: el ruido con mejor mediana entre 42 contrastes es más
alto que el ruido con mejor peor fold.

| | v1 (peor fold) | v2 (mediana) |
| --- | --- | --- |
| q1 (p95 del máximo de U1 en las sombras) | 0,88 | **1,84** |
| q2 (p95 del máximo de U2) | 0,50 | **0,80** |

![t por fold, v2](figuras/etapa1_t_por_fold.png)

| Variable | Corte | Signo | U1 = t mediano | U2 = monotonía mediana | p perm. individual | Réplicas en que la mejor sombra la supera |
| --- | --- | --- | --- | --- | --- | --- |
| **F06 VRP** | global | + | **1,59** | 0,5 | 0,009 | 16 % |
| F02 distancia a SMA20 | mercado alcista | − | 1,32 | 0,4 | 0,019 | |
| F09 amplitud sectorial | mercado alcista | − | 1,31 | 0,0 | 0,008 | |
| F02 distancia a SMA20 | global | − | 1,10 | 0,3 | 0,040 | |
| F09 amplitud sectorial | global | − | 1,09 | 0,3 | 0,010 | |
| F08 RSP − SPY | mercado alcista | + | 1,05 | 0,4 | 0,048 | |
| F01 reversión 5d / σ | mercado alcista | − | 1,01 | 0,1 | 0,061 | |
| F11 ΔT10Y2Y (sin signo) | global | 0 | 0,96 | 0,6 | 0,31 | |
| F01 reversión 5d / σ | global | − | 0,85 | 0,3 | 0,067 | |

Lectura:

- **Cambia el orden.** Con la mediana, el mejor candidato es F06 (VRP): t mediano 1,59, el más cercano al umbral.
  Pasaría en un umbral alrededor del percentil 84 de la nula; con el p95 pre-registrado no pasa.
  Su monotonía (0,5) tampoco llega al q2 de 0,80.
- **Aparece la amplitud sectorial (F09)** con p individual 0,008-0,010, pero sin forma (U2 0,0-0,3).
- **Reversión (F01, F02)** mejora algo, sobre todo en mercado alcista, sin acercarse a 1,84.
- Momentum, crédito y calendario siguen con t mediano negativo.

## Etapa 2 (solo las excepciones F03, F06, F13)

| Variable | L1 ridge [umbral] | L1 enet [umbral] | G1 [umbral] | G2 [umbral] | G3 [umbral] | Clase |
| --- | --- | --- | --- | --- | --- | --- |
| F03 | 0,00013 [0,00086] | −0,00019 [0,00214] | −0,0019 [0,0048] | −0,41 [0,85] | 0,0014 [0,0046] | descartada |
| **F06** | **0,0023** [0,00086] | **0,0055** [0,00214] | 0,0039 [0,0048] | −0,00 [0,85] | 0,0027 [0,0046] | descartada |
| F13 | 0,00029 [0,00086] | 0,00030 [0,00214] | 0 [0,0048] | 0,66 [0,85] | 0 [0,0046] | descartada |

F06 **cumple en lineal** (ridge y elastic net por encima de sus sombras con la mediana) pero **no en el GBM**: la
pendiente SHAP se queda a un 80 % del umbral y la forma no es monótona (el efecto de umbral de la v1, plano hasta
~+0,5 desviaciones y luego alto). Una excepción solo entra como "supresión sospechosa" si cumple en lineal y GBM.

![dependencia SHAP, v2](figuras/dependencia_shap.png)

## Validación, nula y criterios

- Validación 2019-2022: catálogo vacío → media histórica (Sharpe neto 0,16, R² OOS = 0). DSR y PBO no aplican.
- Nula del procedimiento con la mediana: __NULA_V2__
- Criterios 7.6: fallan todos. **El holdout sigue sellado.**

## Qué aprendemos de la v2

- El peor fold no era el único freno: con la mediana el umbral del ruido sube de 0,88 a 1,84, porque la mediana
  también favorece a las sombras afortunadas. Relajar la agregación no regala significación.
- F06 (VRP) es la candidata más sólida en las dos versiones: significativa en todo el periodo (t = 2,19), primera
  con la mediana y lineal por encima de sus sombras. Lo que falla es la forma: es un efecto de umbral, no lineal ni
  monótono. Si se quiere seguir, la siguiente reapertura natural es pre-registrar F06 en forma de umbral
  (p. ej. VRP por encima de su p70 de los 3 años previos). Sería la v3 y sumaría otro ensayo.
