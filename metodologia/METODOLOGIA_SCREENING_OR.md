# Metodología · Screening OR + selección por contribución fuera de muestra

Versión 0.1 (2026-10-29). Propuesta para **líneas o datos nuevos** (no reabre la 1A ni la 1B). Implementación de
prueba en `screening/` y `scripts/probar_screening.py`; antes de usarla con datos reales debe pasar la verificación
de la sección 6.

## 1. Principio

No existe una métrica única que decida si una característica sirve. El error típico es filtrar con un **AND** de
criterios exigentes: cada filtro al p95 de su nula, encadenados, deja el error real del procedimiento muy por debajo
del 5 % y la potencia en el suelo (en la 1A v1-v4: error 0/100 y potencia del 0-8 % para R² 0,5-2 %).

La alternativa es:

- **Fase 1 (screening laxo, OR)**: una característica pasa si cumple **alguno** de varios criterios. Su papel es no
  perder candidatas (pocos falsos negativos), no decidir.
- **Fase 2 (selección, decide)**: contribución fuera de muestra en el modelo completo, con las variables de estado
  como features, contra sombras entrenadas en el mismo modelo.
- **El error se controla una sola vez, al final**: un único percentil p de la nula fija los umbrales de la fase 2 y
  se calibra con la nula del procedimiento completo (objetivo desplazado) para que la tasa de falsos positivos del
  procedimiento sea ≈ 5 %.

El IC medio sirve para **priorizar**, no para **descartar**.

## 2. Elementos fijados a priori (en el pre-registro de cada línea)

- Lista de candidatas, con signo esperado o "sin signo".
- Lista **corta** de variables de estado S (p. ej. R_vol, R_trend; como máximo 3-4). No se amplía después de ver datos.
- Folds: walk-forward expansivo con purga y embargo (como en la 1A v4/v5).
- Clusters de correlación de las candidatas (Spearman, enlace medio, corte |ρ| = 0,6), calculados en descubrimiento.
- Percentil del screening (p_s) y rejilla para calibrar el percentil final (p*).

## 3. Fase 1 · Screening OR

Todas las estadísticas se calculan en los bloques de test y se resumen con la **media de folds**. Cada criterio tiene
su umbral = percentil p_s (laxo, p. ej. 80) del máximo de sus sombras sobre todas las candidatas (y estados, si
procede). Pasa la fase 1 quien supere **al menos uno** de C1, C2 o C3.

| Criterio | Estadístico | Qué rescata | Nula |
| --- | --- | --- | --- |
| **C1 · ICIR incondicional** | t de Newey-West de la pendiente de y sobre x, con el signo pre-registrado (|t| sin signo) | señal "clásica" robusta | máximo sobre candidatas de las sombras |
| **C2 · IC condicional** | t NW de la interacción x·s en y = a + b x + c s + d x·s, para cada estado s; se toma el máximo sobre estados | señales de régimen | máximo sobre candidatas × estados (Westfall-Young); sustituye a Bonferroni y al DSR, que no corresponde aquí |
| **C3 · IC latente predecible** | modelo de nivel local (Kalman) sobre el IC instantáneo z_t = x̃_t·ỹ_t; q = σ²_η/σ²_ε estimado por máxima verosimilitud en el entrenamiento; estadístico = t de la regresión de z_t sobre la predicción a un paso μ̂_{t|t-1} en el test | IC que deriva de forma **predecible** | máximo sobre candidatas de las sombras (con la misma persistencia) |

Matices de C3, que no se incorpora tal cual:

1. Una SNR alta (q grande) **no basta**: un IC que deriva con media ≈ 0 solo es útil si su estado estimado predice el
   IC siguiente. Por eso el estadístico es la capacidad predictiva a un paso, no q (q se informa como descriptivo).
2. Una candidata persistente frente a un objetivo autocorrelacionado genera deriva espuria del IC; las sombras
   conservan esa persistencia, así que el umbral la descuenta.

**C4 · IC marginal** no se usa como criterio de screening: depende del orden y de "las señales que ya tienes", que no
existen al empezar. Su función (redundancia y sustitución) la cumple el **MDA clusterizado** de la fase 2.

## 4. Fase 2 · Selección por contribución fuera de muestra

Modelo: LightGBM con los parámetros fijos de la 1A (profundidad 3, 7 hojas, 40 por hoja, lr 0,02, 300 árboles,
feature_fraction 0,8). Pérdida: MSE del objetivo (y / σ̂₅). En cada réplica de sombras (desplazamiento circular de
todas las candidatas, mismo k):

- **SFI (Single Feature Importance)**: para cada candidata j, modelo con {j} ∪ S frente al modelo con solo S;
  SFI_j = reducción media del MSE fuera de muestra en los folds. Igual para su sombra. Mide si la característica,
  sola y con las variables de estado, aporta: así captura la dependencia de régimen.
- **MDA clusterizado**: modelo con todas las candidatas que pasaron la fase 1, sus sombras y S; en el test de cada
  fold se permutan **juntas** las columnas de cada cluster y se mide el aumento del MSE. Los clusters de sombras
  replican los de las reales.

Valor real = mediana sobre réplicas; umbral = percentil p* del máximo sobre sombras. Una candidata se **selecciona**
si su cluster supera el umbral de MDA **o** ella supera el de SFI. Dentro de un cluster seleccionado se conserva la
candidata con mayor SFI (las demás quedan "representadas por").

## 5. Control del error y de los ensayos

- **Error del procedimiento**: p* = el percentil más bajo de la rejilla (95 … 50) con ≤ 5 % de conjuntos sin señal
  (objetivo desplazado circularmente) que acaben con alguna candidata seleccionada. La calibración no usa el objetivo
  real.
- **Ensayos**: cada aplicación con datos reales es un ensayo en `trials.jsonl`; el DSR y el PBO de la evaluación
  posterior los cuentan todos.
- **Holdout**: igual que en la 1A/1B, sellado hasta congelar la especificación.

## 6. Verificación antes de usarla con datos reales

Con datos sintéticos (objetivo desplazado con su volatilidad alineada a cada semana + efecto plantado sobre una
candidata real), `scripts/probar_screening.py` mide:

| Celda | Qué comprueba |
| --- | --- |
| Sin efecto | tasa de falsos positivos del procedimiento (objetivo ≤ 5 % con p*) |
| F01 lineal, R² 0,5 / 1 / 2 % | potencia para un efecto incondicional |
| F05 solo en mercado alcista, R² 1 % | potencia de C2 y del SFI con estados |
| F06 con coeficiente que deriva (paseo aleatorio, media 0) | potencia de C3 y si la fase 2 la usa |

Criterio de adopción (fijado aquí): falsos positivos ≤ 5 % y potencia ≥ 70 % para F01 lineal con R² 1 %. Se compara con
la v5 de la 1A, que en las mismas condiciones dio 97 % con R² 1 % y 0/30 falsos positivos.

## 7. Qué no cambia

Pre-registro antes de mirar datos, sombras circulares como nula, separación descubrimiento / validación / holdout,
evaluación fuera de muestra (R² OOS, Clark-West, económicas, DSR, PBO) y criterios de apertura del holdout.

## 8. Resultado de la verificación (2026-10-29)

Detalle en `resultados/screening_prueba/INFORME.md`. p\* = 85 (2,5 % de falsos positivos). Potencia del procedimiento
completo: F01 lineal 25 % (R² 0,5 %), **68 % (R² 1 %)**, 100 % (R² 2 %); F05 condicional 45 %; F06 con deriva 8 %.

**No cumple el criterio de adopción** (68 % < 70 %) y es menos potente que la 1A v5 (97 % con R² 1 %). Además:

- C2 (interacción) apenas tiene potencia (3-10 %); C3 no rescata las señales con deriva (8 %).
- El umbral del SFI puede quedar por debajo de 0: hay que exigir también SFI > 0.

Uso recomendado: el **principio** (filtro laxo, decisión fuera de muestra y error calibrado al final) se mantiene, pero
la implementación por defecto sigue siendo la de la 1A v5. Una versión 0.2 debería (a) exigir SFI > 0, (b) prescindir
de C2 o sustituirlo por el SFI con estados, (c) usar más réplicas en la fase 2, y volver a pasar esta verificación.

Evaluación de sensibilidad sobre la 1A v5 (`resultados/sensibilidad_screening/INFORME.md`): selecciona el cluster
{F01, F02, F05} (representante F01) y nada más; el catálogo de la 1A no cambia.
