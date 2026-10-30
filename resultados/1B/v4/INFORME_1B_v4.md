# Línea 1B-v4 · Reversión laxa con barreras de ±0,5 σ + meta-label (F05, R_vol, R_trend)

Pre-registro: `preregistro/preregistro_1B_v4.yaml` (hash f0ff17a3…). Idéntica a la 1B-v3 salvo las barreras
(±0,5·σ̂₅_t en lugar de ±1). **Propuesta posterior a ver la validación de la 1B-v3** (motivo de diseño declarado:
simetría con la entrada y etiqueta más limpia); la validación 2019-2022 no es limpia para esta línea. Holdout sellado.

## 1. Regla primaria (descubrimiento 1993-2018; puerta informada)

| | v3 (±1σ) | **v4 (±0,5σ)** |
| --- | --- | --- |
| Eventos | 357 | 357 |
| Toma de beneficio / stop / vertical | 133 / 95 / 129 | 205 / 151 / 1 |
| Acierto | 59,4 % | 57,7 % |
| Media neta por operación | +23 pb | +10 pb |
| t NW | 1,99 | 1,25 (no cumpliría la puerta) |
| Stops en días que tocan ambas barreras | — | 4 |

## 2. Calibración (p\*)

| Percentil | 95 | **90** | 85 | 80 | 75 | 70 | 65 | 60 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| % nulas con alguna meta-feature | 2 | **3** | 7 | 9 | 11 | 13 | 18 | 22 | 29 |

## 3. Potencia con p\* = 90 (40 réplicas; calculada antes de la selección real)

| Celda | Entra al meta-modelo |
| --- | --- |
| Sin efecto | 0 % |
| F05 AUC 0,55 / 0,60 / 0,65 | 7,5 % / 55 % / 87,5 % |
| R_trend +10 pp | 12,5 % |

## 4. Selección real

| Feature | U | Umbral p90 | p perm. | Acierto tercil alto − bajo | Etapa 1 |
| --- | --- | --- | --- | --- | --- |
| F05 (+) | −0,017 | 0,072 | 1,00 | −5,0 pp | no (signo contrario) |
| R_vol (sin signo) | 0,090 | 0,072 | 0,008 | −9,0 pp | **sí** |
| R_trend (+) | 0,040 | 0,072 | 0,72 | +7,3 pp | no |

Etapa 2 de R_vol: L 0,046 (umbral 0,102), G1 0,058 (0,148), G3 0,060 (0,140) → no entra. **Meta-features: ninguna.**

## 5. Validación 2019-2022 (53 eventos)

| Estrategia | Operaciones | Sharpe neto | Rent. anual |
| --- | --- | --- | --- |
| Primario solo | 53 | **−0,25** | −1,1 % |
| Filtrado (tasa base < 0,55) | 0 | — | 0 % |

Primario: acierto 51 %, media +4 pb, t NW 0,17. Sin meta-features la tasa base anual no supera 0,55 y el filtro no
opera; con una sola serie con varianza no se calculan DSR ni PBO (con Sharpe negativo serían ≈ 0).

## 6. Exploratorio: meta-modelo con F05, R_vol y R_trend sin selección (no confirmatorio)

| Estrategia | Operaciones | Sharpe neto | AUC | Acierto si opera | p-valor nula filtros |
| --- | --- | --- | --- | --- | --- |
| Filtrado logística | 35 | −0,35 | 0,40 | 46 % | 0,81 |
| Filtrado EBM | 27 | −0,83 | 0,32 | 37 % | 0,97 |

DSR (50 ensayos) ≈ 0; PBO 0,08.

## 7. Conclusión

**1B-v4 nula y peor que la v3.** Las barreras de ±0,5 σ dan una etiqueta más limpia (la vertical desaparece), pero
cortan la reversión, que necesita varios días: la ventaja de la regla baja a la mitad en descubrimiento (+10 pb, t 1,25)
y desaparece en validación (+4 pb, t 0,17, Sharpe −0,25). El meta-label no encuentra features (mismo patrón que la v3:
F05 al revés, R_trend en la dirección esperada pero débil, R_vol pasa la etapa 1 y cae en la 2) y, forzado, filtra
contra el resultado (AUC 0,32-0,40).

Nota técnica: la primera ejecución de la validación falló al calcular el DSR con una serie de varianza cero (filtro
sin operaciones); se borraron de `trials.jsonl` las entradas de las dos ejecuciones fallidas y se repitió.
