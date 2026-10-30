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

## 4. Selección real (descubrimiento, p\* = 90; 1.000 sombras en etapa 1, 100 réplicas en etapa 2)

| Feature | Signo previo | U (media de folds, signo·(AUC − 0,5)) | Umbral p90 | p perm. | Acierto tercil alto − bajo | Etapa 1 |
| --- | --- | --- | --- | --- | --- | --- |
| F05 | + | −0,023 | 0,069 | 1,00 | −1,7 pp | no (signo contrario) |
| R_vol | sin signo | 0,076 | 0,069 | 0,056 | −5,7 pp | **sí** |
| R_trend | + | 0,044 | 0,069 | 0,60 | +10,1 pp | no |

Etapa 2 de R_vol: L = 0,019 (umbral 0,040), G1 = 0,025 (0,124), G3 = 0,028 (0,125) → no entra.
**Meta-features: ninguna.** F05 va en sentido contrario al esperado; R_trend apunta en la dirección prevista
(+10 pp de acierto en mercado alcista, justo el efecto de la celda de potencia, que se detecta solo un 12,5 % de las
veces) pero no alcanza el umbral.

## 5. Validación 2019-2022 (53 eventos)

Sin meta-features, el "filtro" es la tasa base ponderada de cada año (opera con tamaño reducido cuando supera 0,55).

| Estrategia | Operaciones | Sharpe neto | Rent. anual | Vol. anual |
| --- | --- | --- | --- | --- |
| Primario solo | 53 | 0,23 | 1,6 % | 7,1 % |
| Filtrado (logística = EBM = tasa base) | 53 | 0,21 | 0,2 % | 1,0 % |

Primario en validación: acierto 51 %, media neta +43 pb, t NW 1,18. Nula de filtros aleatorios: p95 = 0,71,
p-valor 0,93. DSR (41 ensayos de 1A y 1B) = 0,85; PBO = 0,56.

| Criterio de éxito | Resultado |
| --- | --- |
| Sharpe filtrado > primario | no (0,21 < 0,23) |
| Sharpe filtrado > p95 de filtros aleatorios | no (0,21 < 0,71) |
| DSR > 0,95 | no (0,85) |

## 6. Conclusión

**1B-v3 nula**, como prevé el pre-registro cuando no hay meta-features. La regla laxa de reversión tiene una
ventaja propia modesta y estable en signo (t 1,99 en descubrimiento, +43 pb por operación en validación pero t 1,18),
y ni F05 ni los regímenes añaden un filtro detectable: F05 tiene el signo contrario y R_trend, en la dirección
esperada, queda en el rango de efectos (~+10 pp) que esta muestra no puede distinguir del ruido. No se ejecuta la nula
del procedimiento: sin meta-features seleccionadas no hay nada que contrastar más allá de la calibración (sección 2).
Holdout sin abrir.
