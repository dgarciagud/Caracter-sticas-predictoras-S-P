# Evaluación de sensibilidad: screening OR + SFI/MDA sobre la 1A v5

`scripts/sensibilidad_screening.py` (2026-10-29). **Declarada como sensibilidad**: no modifica el catálogo de la 1A
(F05 estable condicional en mercado alcista) ni reabre su pre-registro. Registrada como ensayo en `trials.jsonl`.
Diseño de la 1A v5 (descubrimiento 1993-2018, test 1997-2018, objetivo y / σ̂₅, 10 candidatas); p_s = 80 y
p\* = 85 de la calibración sintética, fijados antes de mirar el objetivo real; 1.000 sombras en la fase 1 y 100 en la 2.

## Fase 1 · screening OR (`fase1.csv`)

| Variable | C1 [1,18] | C2 [1,46] | C3 [0,95] | Pasa OR |
| --- | --- | --- | --- | --- |
| F01 reversión 5d / σ | **1,22** | 0,32 | 0,37 | ✓ (C1) |
| F05 cambio del VIX | **1,22** | 0,55 | 0,66 | ✓ (C1) |
| F06 VRP | 1,12 | 0,56 | **1,03** | ✓ (C3) |
| F14 cambio de mes | 1,00 | 1,33 | 0,44 | ✗ |
| Resto (F02, F03, F04, F07, F11, F13) | ≤ 0,69 | ≤ 0,98 | ≤ 0,80 | ✗ |

## Fase 2 · SFI y MDA clusterizado (`fase2.csv`)

| Variable | Cluster | SFI [−0,006] | MDA del cluster [0,015] | Resultado |
| --- | --- | --- | --- | --- |
| **F01** | {F01, F02, F05} | **+0,0007** | **0,047** | **seleccionada** (representante del cluster) |
| F05 | {F01, F02, F05} | −0,0046 | 0,047 | candidata, representada por F01 |
| F06 | {F06} | −0,0135 | 0,006 | no |

## Lectura

- **Coincide con la 1A v5 en lo esencial**: el único bloque de señal es el cluster de reversión / miedo
  {F01, F02, F05}. La v5 lo recogió como F05 condicional a mercado alcista; esta metodología, como F01 incondicional
  (su representante por SFI). Son la misma información: F01 y F05 tienen |ρ| ≈ 0,8.
- **F06 (VRP)** entra al screening por C3 (su IC latente) pero no aporta fuera de muestra en el modelo completo.
- **F14** no pasa: con la interacción (C2 = 1,33 < 1,46) se queda cerca, igual que en la v4.
- La señal es débil: el SFI de F01 es apenas positivo (+0,0007 en MSE de y / σ̂₅) y el MDA del cluster es el que
  decide. Con el fallo de diseño señalado en la prueba sintética (umbral de SFI negativo), F05 "pasa" el SFI aunque
  empeora el MSE; no cambia el resultado porque el cluster entra por MDA.

**Conclusión**: la metodología alternativa no descubre nada nuevo y confirma que la única señal de la 1A es el cluster
de reversión semanal / pico de VIX. El catálogo de la 1A no cambia.
