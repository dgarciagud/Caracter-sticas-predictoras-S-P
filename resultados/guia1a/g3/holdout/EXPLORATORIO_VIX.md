# Holdout · análisis exploratorio a posteriori: cambio del VIX (VOL_dvix5)

**No es confirmatorio.** Se ejecutó el 2026-11-20, después de abrir y leer el holdout congelado. VOL_dvix5 no estaba
en el catálogo g3: se descartó por relevancia SHAP por debajo de las sombras cuando REV_r5 está en el mismo modelo.
Cuenta como intento y no cambia el veredicto del [holdout](INFORME_HOLDOUT.md). Usa el mismo código
(`scripts/holdout_guia_1a.py --exploratorio`) y el mismo esquema: walk-forward desde 1992, reentreno anual, 2 pb por
lado. Resultados en [`exploratorio_VOL_dvix5/`](exploratorio_VOL_dvix5) y
[`exploratorio_REV_r5_VOL_dvix5/`](exploratorio_REV_r5_VOL_dvix5).

## Pendiente de y sobre VOL_dvix5

Descubrimiento: b = +0,092 (t 4,15), con intervalo del 95 % [0,048, 0,135].

| Periodo | b | t NW | b holdout / b descubrimiento | ¿Mismo signo? | ¿Dentro del IC 95 %? |
| --- | --: | --: | --: | :-: | :-: |
| 2013-2026 | +0,022 | 0,89 | 0,24 (encoge 76 %) | ✔ | ✘ |
| 2013-2022 | +0,023 | 0,78 | 0,25 | ✔ | ✘ |
| 2023-2026 (virgen) | +0,019 | 0,42 | 0,21 | ✔ | ✘ |

## Predicción y estrategia (combinación = OLS)

| Variables | Periodo | R² OOS | Clark-West t | Sharpe neto | Δ Sharpe vs B&H (t) |
| --- | --- | --: | --: | --: | --: |
| VOL_dvix5 | 2013-2026 | −0,38 % | 0,87 | 0,49 | −0,19 (**−2,21**) |
| VOL_dvix5 | 2023-2026 | −0,16 % | 0,45 | 0,74 | −0,22 (−1,55) |
| REV_r5 + VOL_dvix5 | 2013-2026 | −0,16 % | 1,26 | 0,57 | −0,11 (−1,53) |
| REV_r5 + VOL_dvix5 | 2023-2026 | +0,16 % | 0,91 | 0,78 | −0,19 (−1,43) |
| *REV_r5 sola (holdout congelado)* | *2013-2026* | *−0,12 %* | *1,47* | *0,56* | *−0,12 (−1,66)* |
| *Buy & hold* | *2013-2026* | | | *0,69* | |

## Lectura

- **El cambio del VIX se debilita más que REV_r5.** Mantiene el signo positivo (tras subidas del VIX, algo más de
  retorno), pero conserva solo una cuarta parte de su efecto, con t 0,4-0,9. Queda fuera del intervalo del
  descubrimiento en todos los tramos, también en el virgen.
- **Como estrategia es significativamente peor que el buy & hold** en 2013-2026 (t −2,21).
- **Añadirlo a REV_r5 no aporta.** El Sharpe pasa de 0,56 a 0,57 y el R² empeora, lo que confirma la lectura del
  descubrimiento: es en buena parte un proxy de REV_r5.
