# Línea 1A según la guía · versión g2: early stopping y mínimo por hoja en el GBM

**Reapertura declarada de g1** ([informe g1](../g1/INFORME_1A_guia.md)). Se escribió después de ver los resultados
de g1 y a petición del autor. El pre-registro es
[`preregistro/preregistro_1A_guia_g2.yaml`](../../../preregistro/preregistro_1A_guia_g2.yaml), con commit `daae635`
antes de ejecutar.

**Qué cambia: solo el GBM.**

- **Número de árboles:** se decide por early stopping (paciencia 50, máximo 2.000, learning rate 0,02). La validación
  interna es el último 20 % de cada entrenamiento, separado con una purga de 5 sesiones.
- **Mínimo de observaciones por hoja:** se elige entre 50, 100, 200, 400 y 800 en esa misma validación interna.
  Después se reentrena con todo el entrenamiento.
- **Variantes fijas:** cada valor de la rejilla, con early stopping, se evalúa también como variante fija y entra en
  el registro.

Datos, candidatas, clusters, umbrales, particiones, sombras, regímenes y estrategia son idénticos a g1. La fase 1 y
los clusters dan exactamente lo mismo.

## Resultado

**El catálogo sigue vacío, pero ahora el cluster de corto plazo falla en otro punto:**

| Paso | g1 (300 árboles fijos, mínimo por hoja 200) | g2 (early stopping + rejilla) |
| --- | --- | --- |
| Clustered MDA | sobrevive solo el corto plazo (ΔMSE 9,7×10⁻³) | sobrevive solo el corto plazo (ΔMSE 2,5×10⁻³; p95 sombras 0,85×10⁻³) |
| Puerta de skill (CPCV) | **pasa**: IC 0,047, t 2,24, R² +0,18 % | **falla**: IC 0,025, t 1,17, R² +0,09 % |
| SHAP: signo en los 6 bloques | ✔ | ✔ |
| SHAP: monotonía mínima | **0,64 ✘** | **0,76 ✔** (p95 sombras 0,52) |
| SHAP: forma | 0,92 ✔ | 0,92 ✔ |
| SHAP: relevancia | ✔ | ✔ |
| Walk-forward GBM: R² OOS | −0,98 % | −0,79 % |
| Walk-forward GBM: Sharpe neto (B&H 0,10) | 0,143 | 0,077 |
| Walk-forward GBM: rotación anual | 16,4 | 6,5 |
| Decisión | descartada por la monotonía SHAP | descartada por la puerta de skill |

Con early stopping, el GBM aprende curvas más suaves: la monotonía SHAP pasa de 0,64 a 0,76 y el cluster supera
todos los criterios de estabilidad. Pero el modelo predice peor: su IC fuera de muestra deja de ser significativo y no
supera la puerta de skill. La guía es explícita aquí: sin skill, el SHAP describe ruido ajustado, por estable que sea.

**Lectura.** En g1 la señal tenía skill pero una forma demasiado irregular. En g2 tiene una forma limpia, pero no
tiene skill. Que el veredicto dependa de un hiperparámetro es, en sí mismo, evidencia de fragilidad.

## Qué eligió la validación interna

Hay 15 reentrenos en walk-forward y 15 particiones CPCV ([`04_gbm_hiperparametros_elegidos.csv`](04_gbm_hiperparametros_elegidos.csv)):

| | Walk-forward | CPCV |
| --- | --- | --- |
| Mínimo por hoja elegido | 50 ×3, 100 ×4, 200 ×4, 400 ×1, 800 ×3 | 50 ×4, 100 ×4, 200 ×1, 400 ×2, 800 ×4 |
| Árboles, mediana (mín-máx) | 17 (1-444) | 62 (1-1.035) |

- **Ningún valor de la rejilla domina.** La elección salta de un extremo a otro según el año, porque la validación
  interna no distingue entre valores.
- **El early stopping corta casi enseguida.** En walk-forward la mediana es de 17 árboles, y a veces solo 1, que
  equivale a predecir la media. La validación interna está diciendo que apenas hay señal que aprender.

## Variantes fijas de la rejilla

Las cinco variantes llevan early stopping y se evalúan sobre el cluster de corto plazo
([`04_gbm_rejilla_min_child.csv`](04_gbm_rejilla_min_child.csv)):

| Mínimo por hoja | CPCV IC (t) | CPCV R² OOS | WF R² OOS | WF Clark-West t | WF Sharpe | Δ Sharpe vs B&H (t) |
| --: | --: | --: | --: | --: | --: | --: |
| 50 | 0,018 (0,87) | +0,03 % | −0,80 % | −0,66 | 0,114 | +0,015 (0,69) |
| 100 | 0,025 (1,17) | +0,21 % | −0,80 % | −0,75 | 0,105 | +0,006 (0,22) |
| 200 | 0,019 (0,87) | +0,25 % | −0,50 % | −0,35 | 0,095 | −0,004 (−0,15) |
| 400 | 0,004 (0,15) | +0,16 % | −0,59 % | −0,57 | 0,117 | +0,018 (0,83) |
| 800 | 0,016 (0,65) | +0,15 % | −0,33 % | −0,32 | 0,102 | +0,003 (0,07) |

Ningún valor da un IC significativo en CPCV ni un R² positivo en walk-forward. Ninguno mejora al buy & hold de forma
significativa.

## Regímenes, DSR y PBO

- **Regímenes:** igual que en g1, no se acepta ninguno. En el GBM, permitir las interacciones empeora el CPCV
  (t −2,24).
- **Intentos:** la cuenta conjunta g1 + g2 da 41 intentos únicos; las series idénticas entre versiones se cuentan
  una vez ([`07_dsr_pbo.json`](07_dsr_pbo.json)).

| Matriz 1998-2012 | Variantes | PBO | DSR del líder | Líder |
| --- | --: | --: | --: | --- |
| Modelos en walk-forward (g1 + g2) | 18 | **0,80** | 0,75 | GBM de g1 (Sharpe 0,14) |
| Modelos + reglas univariantes | 40 | 0,33 | 0,96 | regla FOMC (Sharpe 0,47) |

- **El PBO de los modelos empeora** respecto a g1 (0,72 → 0,80), y el DSR del mejor modelo baja (0,81 → 0,75).
- **El DSR de la regla FOMC sube** de 0,952 a 0,961 aunque se añadan intentos. El DSR penaliza según la dispersión de
  Sharpe entre intentos, y las variantes GBM añadidas tienen Sharpe parecidos. No es más evidencia a favor de FOMC,
  y valen las mismas reservas que en g1.

## Conclusión

Calibrar el GBM con early stopping y el mínimo por hoja no rescata ninguna característica. Sí aclara el diagnóstico:

1. **El cluster de corto plazo** puede tener un SHAP estable o un modelo con skill, pero no las dos cosas a la vez.
2. **La validación interna** apenas encuentra señal: muy pocos árboles y ningún mínimo por hoja preferido.
3. **En tiempo real** sigue sin batir a la media histórica.

El resultado nulo de g1 se mantiene, ahora más sólido. El holdout 2013-2026 sigue sellado.

## Reproducir

```bash
GUIA1A_PREREG=preregistro/preregistro_1A_guia_g2.yaml TQDM_DISABLE=1 python scripts/run_guia_1a.py todo   # unos 15 minutos con 4 núcleos
```
