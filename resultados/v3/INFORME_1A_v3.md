# Línea 1A · versión 3: criterios con potencia comprobada

Pre-registro `preregistro/preregistro_1A_v3.yaml` (2026-10-25, SHA-256 `edd093f9…21dc`). Segunda reapertura del
pre-registro, después de ver la v1, la v2 y la prueba de potencia. Ensayos acumulados en `resultados/trials.jsonl`.

## Cómo se eligieron los criterios

1. **Regla fijada antes de ver la potencia de las variantes**: primera variante, en orden de preferencia por
   estabilidad, con potencia de la etapa 1 ≥ 50 % a R² 2 % (F01 lineal) y falsos positivos ≤ 5 %.

   | Variante (U1 / U2) | Potencia F01 lineal 2 % | Falsos positivos | |
   | --- | --- | --- | --- |
   | mediana / U2 agregado | 40 % | 0 % | no |
   | mediana / sin U2 | 47 % | 11 % | no |
   | **media / U2 agregado** | **53 %** | **0 %** | **elegida** |
   | media / sin U2 | 60 % | 9 % | (no se llega) |

   U2 agregado es la correlación de rangos x-y con todas las semanas de test juntas. La versión con medias por
   quintil agregadas se descartó antes de medir potencia porque satura: el p95 del máximo del ruido vale 1.
   Detalle en `resultados/potencia_v3/`.

2. **Diagnóstico de la etapa 2 con efectos plantados**: G2 (Spearman x-SHAP) rechazaba incluso efectos lineales
   en F06 (colas gruesas: el GBM solo corta en los extremos). Con G1 + G3 la potencia completa sube sin falsos
   positivos (`resultados/potencia_completa_v3*`):

   | Efecto plantado | Etapa 1 | Completo G1+G3 | Completo G1+G2+G3 |
   | --- | --- | --- | --- |
   | F01 lineal, R² 2 % | 50 % | 25 % | 25 % |
   | F01 lineal, R² 4 % | 100 % | 75 % | 75 % |
   | F06 lineal, R² 2 % | 25 % | 20 % | 10 % |
   | F06 umbral, R² 4 % | 55 % | 30 % | 5 % |

   Para comparar: con la v1, la potencia de la etapa 1 sola era del 0-7 % en esas mismas celdas.

**Criterios de la v3**: media de los folds en U1, L1, G1, G2 e I1; U2 = correlación de rangos agregada; el GBM
cumple con G1 y G3. Todo lo demás es igual que en la v1.

## Resultado con los datos reales

**Sigue siendo nulo**: catálogo vacío, validación igual (media histórica, R² OOS = 0), holdout sin abrir.

### Etapa 1 (q1 = 1,66; q2 = 0,143)

| Variable | Corte | U1 = t medio | U2 = IC de rangos | p perm. individual | Réplicas en que la mejor sombra la supera |
| --- | --- | --- | --- | --- | --- |
| **F06 VRP** | global | **1,40** | 0,069 | 0,001 | 18 % |
| F09 amplitud sectorial | mercado alcista | 1,17 | 0,071 | 0,008 | 51 % |
| F09 amplitud sectorial | global | 1,08 | 0,077 | 0,007 | 62 % |
| F02 distancia a SMA20 | global | 1,00 | 0,086 | 0,009 | 76 % |
| F01 reversión 5d / σ | global | 0,97 | 0,084 | 0,024 | 79 % |

Ninguna llega: F06 se queda en el percentil ~82 del máximo del ruido en U1, y ninguna candidata alcanza en U2 un
IC de rangos de 0,14 (los mejores rondan 0,07-0,09).

### Etapa 2 (excepciones F03, F06, F13)

| Variable | L1 ridge [umbral] | L1 enet [umbral] | G1 [umbral] | G3 [umbral] | Clase |
| --- | --- | --- | --- | --- | --- |
| F03 | 0,00010 [0,00078] | −0,00025 [0,00164] | −0,0020 [0,0041] | 0,0014 [0,0046] | descartada |
| **F06** | **0,0020** [0,00078] | **0,0036** [0,00164] | **0,0043** [0,0041] | 0,0027 [0,0046] | descartada |
| F13 | 0,00028 [0,00078] | 0,00033 [0,00164] | 0,00004 [0,0041] | 0 [0,0046] | descartada |

F06 cumple ahora en los dos lineales y en la dirección del GBM (G1), y solo falla en la magnitud SHAP media (G3,
60 % del umbral). Le falta eso para clasificarse como "supresión sospechosa", que además solo entraría al modelo
final si la validación la confirmase.

## Nula del procedimiento y criterios 7.6

__NULA_V3__

## Lectura conjunta de v1, v2 y v3

- Con criterios que sí detectan efectos realistas (25-75 % de potencia completa a R² 2-4 %), el S&P 500 a 5 días
  en 2011-2018 sigue sin dar ninguna característica estable. El nulo de la v3 es más informativo que el de la v1:
  descarta efectos lineales de R² ≥ 4 % en F01 con bastante confianza; no descarta efectos de R² ~2 %.
- F06 (VRP) es la candidata más cercana en las tres versiones y en todas las etapas, pero no llega a ningún umbral
  familywise.
- Una cuarta reapertura para salvar a F06 (por ejemplo, quitar G3) sería ya ajustar el procedimiento a una
  variable concreta. Cada reapertura suma ensayos y, sobre todo, erosiona el valor del pre-registro.
