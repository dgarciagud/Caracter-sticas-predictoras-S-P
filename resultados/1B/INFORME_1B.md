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
