# Línea 1B · Meta-label sobre la regla de reversión

Pre-registro `preregistro/preregistro_1B.yaml` (2026-10-26, SHA-256 `222d62e1…481f`), fechado antes de construir
eventos o meta-labels. Periodos de la 1A v5: descubrimiento 1993-2018, validación 2019-2022, holdout sellado.
Meta-features previstas: F05 (estable condicional en la 1A v5), R_vol y R_trend.

## Paso 11a · Capa de volatilidad (8.1)

HAR-RV en logaritmos con RV diaria, semanal y mensual y VIX², reestimado cada mes con ventana expansiva desde 1990
y solo con objetivos ya realizados. Comparación por QLIKE semanal (menor es mejor) frente a un GJR-GARCH(1,1,1)
t de Student con la misma reestimación (`01_volatilidad_qlike.csv`):

| Periodo | Semanas | QLIKE HAR-VIX | QLIKE GJR-GARCH | Correlación |
| --- | --- | --- | --- | --- |
| Descubrimiento 1993-2018 | 1.356 | **0,330** | 0,354 | 0,96 |
| Validación 2019-2022 | 207 | **0,402** | 0,494 | 0,94 |

El HAR con VIX es mejor en los dos periodos. σ̂₅ mediana en descubrimiento: 1,8 % (rango 0,8 %-11,9 %).
No cuenta como hallazgo de 1B.

## Paso 11b · Validez de la regla primaria (8.2): **no se cumple → 1B se detiene**

Regla: comprar al cierre de la semana t si el retorno a 5 días es menor que −1,5 · σ̂₅; barreras ±1 σ̂₅ y vertical de
5 sesiones, comprobadas con máximos y mínimos diarios (stop primero si toca las dos el mismo día; hueco de apertura
ejecutado a la apertura); 4 pb de coste ida y vuelta (`02_eventos_descubrimiento.csv`, `02_regla_primaria.json`).

| | Resultado | Requisito |
| --- | --- | --- |
| Eventos en 1993-2018 | **54** (2,1 al año) | — (la especificación esperaba 150-300 en 2007-2022) |
| Retorno neto medio por evento | +0,08 % | > 0 ✓ |
| t Newey-West | **0,27** | > 1,5 ✗ |
| Meta-label = 1 (acierto) | 52 % | — |
| Salidas | 24 verticales, 16 tomas de beneficio, 14 stops | — |

**La regla sola no tiene una ventaja distinguible de cero**, así que, según 8.2, la línea 1B se detiene aquí: el
meta-label filtra un edge, no lo crea. No se han construido meta-features, sombras ni modelos de la etapa 1B.
Ensayo registrado en `trials.jsonl` (línea 1B).

## Observaciones

- **Muy pocos eventos.** Con σ̂₅ de un HAR que incluye el VIX, la volatilidad prevista sube justo cuando el mercado
  cae, y una caída de 1,5 σ̂₅ en una semana es rara: 54 eventos en 26 años, frente a los 150-300 en 16 años que
  preveía la especificación. Aun con 54 eventos, t = 0,27 está lejos de 1,5; no es solo un problema de muestra.
- **Coherencia con la 1A.** La reversión a 5 días (F01) no fue estable en ninguna versión de la 1A y su efecto
  estimado era de R² ~0,3 %. Una regla basada en ella sin ventaja propia es lo esperable.
- **Cómo seguir, si se quiere.** Cada regla primaria alternativa es un ensayo nuevo que se registra (8.2): otro
  umbral (p. ej. −1,0 σ̂₅, que daría más eventos), una regla basada en el hallazgo de la 1A (por ejemplo, comprar tras
  subidas fuertes del VIX en mercado alcista, coherente con F05) o la regla pre-FOMC que menciona la especificación.
  Cualquiera de ellas tendría que pasar la misma puerta antes de construir el meta-label.

## 1B-v2 · Regla primaria basada en la 1A (subida del VIX en mercado alcista)

Pre-registro `preregistro/preregistro_1B_v2.yaml` (SHA-256 `28903287…21bf`), fechado antes de construir sus eventos.
Regla: comprar al cierre de la semana t si F05 (cambio log del VIX a 5 días) supera su percentil 80 de las 156
semanas anteriores y el índice está sobre su SMA200; mismas barreras, costes y puerta que la v1.
Resultados en `resultados/1B/v2/`.

| | Resultado | Requisito |
| --- | --- | --- |
| Eventos 1993-2018 | **199** (7,7 al año) | — |
| Retorno neto medio por evento | **+0,44 %** | > 0 ✓ |
| t Newey-West | **3,88** | > 1,5 ✓ |
| Meta-label = 1 (acierto) | 64 % | — |
| Salidas | 83 tomas de beneficio, 79 verticales, 37 stops | — |

Retorno neto medio por subperiodo: 1993-97 +0,60 % (48), 1998-01 +0,02 % (26), 2002-05 +0,46 % (21),
2006-10 +0,49 % (35), 2011-14 +0,35 % (32), 2015-18 +0,54 % (37). Positivo en los seis.

**Pasa la puerta de 8.2**, pero con una advertencia fuerte: **es circular**. F05 se seleccionó en la 1A con los
mismos datos de 1993-2018, así que esta t no es evidencia independiente. La prueba real es la validación 2019-2022
(y, en su día, el holdout).

**Problema para el meta-label**: dentro de los eventos, R_trend vale siempre 1 (forma parte de la regla) y F05 está
truncada por encima de su p80. Las meta-features previstas apenas tienen variación: solo quedan R_vol (78 eventos
con VIX alto, 121 con VIX bajo) y la magnitud de F05 dentro de los eventos.

### Meta-label de la 1B-v2 (anexo del pre-registro fijado antes de calcularlo, SHA-256 `a6ecb170…076e`)

**Calibración del percentil** (100 conjuntos con los meta-labels desplazados sobre la secuencia de eventos,
`03_calibracion_*.csv`): fracción con alguna meta-feature en el meta-modelo = 1 % con p de 95 a 70, 3 % con p = 65,
6 % con p = 60 y 8 % con p = 50 → **p\* = 65**.

**Etapa 1** (AUC por fold frente al meta-label, media de folds; umbral 0,078 con p = 65; `04_etapa1.csv`):

| Meta-feature | U | Umbral | p perm. | Acierto tercil alto − bajo |
| --- | --- | --- | --- | --- |
| F05 (magnitud dentro de los eventos) | −0,009 | 0,078 | 1,00 | +3,0 pp |
| R_vol | 0,070 | 0,078 | 0,49 | +0,5 pp |

**Ninguna pasa**: no hay etapa 2 y el meta-modelo es la tasa base de acierto. El filtro con p > 0,55 opera todos
los eventos con tamaño constante, es decir, equivale a la regla primaria sola.

### Validación 2019-2022 (primera prueba no circular de la regla; `05_*`)

| | Descubrimiento 1993-2018 | **Validación 2019-2022** |
| --- | --- | --- |
| Eventos | 199 (7,7/año) | **25** (5 en 2019, 8 en 2020, 12 en 2021, 0 en 2022) |
| Retorno neto medio por evento | +0,44 % | **+0,39 %** |
| t Newey-West | 3,88 | **0,99** |
| Acierto (meta-label = 1) | 64 % | 56 % |
| Salidas | 83 TP · 79 vertical · 37 stop | 7 TP · 10 vertical · 8 stop |

| Estrategia (P&L semanal, 0 sin operación) | Operaciones | Sharpe neto | Rentab. anual | Vol. anual |
| --- | --- | --- | --- | --- |
| Primario solo | 25 | 0,11 | 0,42 % | 3,8 % |
| Primario filtrado (tasa base) | 25 | 0,10 | 0,11 % | 1,1 % |

- El retorno medio por evento fuera de muestra (+0,39 %) es casi igual al de descubrimiento (+0,44 %), lo que es
  **coherente con la regla**, pero con 25 eventos el t es 0,99: **no es significativo**. 2022 no tiene eventos
  porque el índice pasó casi todo el año bajo su SMA200.
- El filtro no aporta (no hay meta-features); el AUC de 0,42 que sale en la tabla de `05_validacion.csv` no es
  interpretable: la probabilidad es la tasa base, constante dentro de cada año.
- Nula de filtros aleatorios: el filtro no la supera (p = 0,99). DSR del líder (primario) = 0,66 con 35 ensayos
  acumulados de 1A y 1B; PBO = 0,38 con 2 estrategias distintas.

### Conclusión de la 1B

- La regla F05 en mercado alcista mantiene el signo y el tamaño fuera de muestra, pero la validación es
  demasiado corta para confirmarla (25 eventos, t ≈ 1), y su t de descubrimiento es circular.
- El meta-label no encuentra nada que filtre: ni la intensidad del pico de VIX ni el régimen de volatilidad predicen
  qué eventos aciertan.
- Criterios del holdout para la 1B: no hay filtro que validar (nula de filtros no superada, DSR 0,66 < 0,95).
  Sumado a que la 1A tampoco cumplió los suyos, **el holdout 2023-2025 no se abre**.

### Nula del procedimiento de la 1B (100 corridas, meta-labels desplazados, p = 65; `06_nula_procedimiento.csv`)

6/100 corridas dejan alguna meta-feature en el meta-modelo (R_vol: 5, F05: 1).
Es un 6 % (IC 95 % aprox. 2-13 %), algo por encima del 3 % de la calibración y del 5 % objetivo, dentro del margen de 100 corridas; el procedimiento es ligeramente menos conservador de lo calibrado. No afecta a la decisión: el meta-modelo real no tiene features.
