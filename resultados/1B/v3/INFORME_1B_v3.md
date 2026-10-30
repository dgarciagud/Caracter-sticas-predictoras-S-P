# Línea 1B-v3 · Regla laxa de reversión + meta-label (F05, R_vol, R_trend)

Pre-registro: `preregistro/preregistro_1B_v3.yaml` (hash 59742465…; corregido solo un error de sintaxis YAML
antes de construir eventos). Holdout 2023-2025 sellado.

## 1. Regla primaria (descubrimiento 1993-2018; puerta informada, no obligatoria)

Comprar si log(P_t/P_{t-5}) < −0,5·σ̂₅ (σ̂₅ de t−5); barreras ±1·σ̂₅_t, vertical 5 sesiones, 4 pb.

| Eventos | Por año | % semanas | Por fold de test | Acierto | Media neta | t NW |
| --- | --- | --- | --- | --- | --- | --- |
| 357 | 13,7 | 26 % | 72 / 69 / 63 / 52 / 55 | 59,4 % | +23 pb | 1,99 |

Salidas: 133 toma de beneficio, 129 vertical, 95 stop. La regla, sola, ya cumpliría la puerta de 8.2.

## 2. Calibración del percentil (meta-labels reales desplazados, 100 corridas)

| Percentil | 95 | **90** | 85 | 80 | 75 | 70 | 65 | 60 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| % nulas con alguna meta-feature | 4 | **5** | 7 | 12 | 14 | 18 | 22 | 26 | 39 |

**p\* = 90.** Los falsos positivos son casi todos de R_vol (sin signo previo).

## 3. Potencia (meta-labels sintéticos sobre eventos, features y pesos reales; 40 réplicas; publicada antes de la selección real)

Detección = la feature plantada entra en el meta-modelo (etapas 1 y 2) con p\* = 90.

| Celda | Pasa etapa 1 | Entra al meta-modelo | Otras features |
| --- | --- | --- | --- |
| Sin efecto | 17,5 % alguna | 2,5 % alguna (R_vol) | — |
| F05, AUC 0,55 | 40 % | **12,5 %** | 0 |
| F05, AUC 0,60 | 87,5 % | **55 %** | 0 |
| F05, AUC 0,65 | 100 % | **82,5 %** | 0 |
| R_trend, +10 pp en alcista (AUC ≈ 0,55) | 25 % | **12,5 %** | 0,025 |

Lectura: con ~310 eventos de test el procedimiento solo detecta de forma fiable efectos de AUC ≥ 0,65; un
AUC 0,60 es una moneda al aire y 0,55 (o +10 pp por régimen) es prácticamente indetectable. La etapa 2
(logística + EBM contra sombras) es el cuello de botella: con AUC 0,60 la etapa 1 pasa en el 87,5 % de los casos
pero solo el 55 % termina en el meta-modelo. Conforme al pre-registro, la potencia es informativa y no cambia la
configuración.
