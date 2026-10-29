# Prueba sintética de la metodología screening OR + SFI/MDA

`scripts/probar_screening.py` (2026-10-29). 40 réplicas por celda; fase 1 con 300 sombras y p_s = 80; fase 2 con 20
réplicas de sombras. Base: objetivo de la 1A v5 (y / σ̂₅, test 1997-2018) desplazado; efectos plantados en unidades
del objetivo normalizado. No usa el objetivo real ni suma en `trials.jsonl`.

## Calibración del error (celda sin efecto)

| p (percentil final) | 95 | 90 | **85** | 80 | 75 | 70 | 65 | 60 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Falsos positivos del procedimiento | 0 % | 0 % | **2,5 %** | 12,5 % | 12,5 % | 12,5 % | 17,5 % | 20 % | 30 % |

**p\* = 85.** Con el OR, el 40 % de los conjuntos sin señal llevan alguna candidata a la fase 2 (es lo buscado: la
fase 1 es laxa y la fase 2 decide).

## Potencia con p\* = 85

| Efecto plantado | Pasa C1 | Pasa C2 | Pasa C3 | Pasa OR | **Seleccionada (procedimiento completo)** | 1A v5 (mismo R², referencia) |
| --- | --- | --- | --- | --- | --- | --- |
| F01 lineal, R² 0,5 % | 38 % | 3 % | 28 % | 40 % | **25 %** | 70 % |
| F01 lineal, R² 1 % | 75 % | 3 % | 75 % | 80 % | **68 %** | 97 % |
| F01 lineal, R² 2 % | 100 % | 3 % | 100 % | 100 % | **100 %** | 100 % |
| F05 solo en mercado alcista, R² 1 % | 50 % | 10 % | 35 % | 55 % | **45 %** | — |
| F06 con coeficiente que deriva, R² 2 % | 13 % | 5 % | 23 % | 40 % | **8 %** | — |

(La v5 de la 1A usó un modelo de ruido algo distinto —efecto plantado en retornos— y 30 réplicas; la comparación es
orientativa.)

## Veredicto frente al criterio de adopción (fijado en la sección 6 de la metodología)

- Falsos positivos ≤ 5 %: **✓** (2,5 %).
- Potencia ≥ 70 % para F01 lineal con R² 1 %: **✗** (68 %; con 40 réplicas, IC 95 % ≈ 51-81 %).
- **No se adopta tal cual.** Es menos potente que el procedimiento de la 1A v5 para efectos incondicionales.

## Lo que enseña la prueba

1. **C2 (interacción con estados) casi no tiene potencia**: 3-10 % incluso con un efecto plantado solo en mercado
   alcista. Con ~230 semanas por fold y un estado que vale 1 el 73 % del tiempo, el t de la interacción es muy
   ruidoso. El efecto condicional lo recogen C1 y C3, no C2.
2. **C3 (IC latente) no rescata las señales que derivan**: con un coeficiente AR(1) de media 0 y R² 2 %, la fase 1
   la deja pasar un 23 % de las veces y el procedimiento completo la selecciona un 8 %. Para el resto de efectos se
   comporta como C1 (con q ≈ 0 el signo previsto es el de la media).
3. **El OR sí cumple su papel**: la fase 1 deja pasar más (40-80 % con efectos pequeños) y el error lo controla la
   fase 2. El límite de la potencia está en la fase 2 (el SFI y el MDA con 20 réplicas son ruidosos).
4. **Fallo de diseño detectado**: el umbral del SFI puede ser negativo (el máximo de las sombras empeora el MSE), así
   que una candidata que empeora la predicción fuera de muestra puede "pasar" el SFI. Debe exigirse además SFI > 0.
