# Características predictoras del S&P 500 · Línea 1A

Implementación de la línea 1A de la especificación *Líneas 1A y 1B · Especificación S&P 500 a 5 días*
(22-10-2026): qué características tienen un efecto de signo y forma estables sobre el retorno a 5 sesiones
del S&P 500, juzgadas contra sombras circulares y con validación sellada.

Datos y librerías del repositorio de Stefan Jansen, *Machine Learning for Trading, 3.ª ed.*
([stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading)):

| Uso | De Jansen |
| --- | --- |
| Índice S&P 500 diario | `data/equities/market/sp500/sp500.csv` vía `data.load_sp500_index()` |
| ETFs (SPY, RSP, HYG, IEF, SPDR sectoriales) | `data/etfs/market/download.py` + `data.load_etfs()` |
| Posicionamiento CFTC (E-mini S&P 500) | `data/futures/positioning/cot_download.py` + `data.load_cot()` (2017-2025) |
| Amplitud real de componentes (7.8) | barras diarias AlgoSeek, `data.load_sp500_daily_bars()` |
| Volatilidades y medias | `ml4t-engineer`: `ewma_volatility`, `realized_volatility`, `sma` |
| DSR y PBO | `ml4t-diagnostic`: `deflated_sharpe_ratio`, `compute_pbo` |

**Informes:** [`resultados/INFORME_1A.md`](resultados/INFORME_1A.md) (v1, peor fold, pre-registro original) y
[`resultados/v2_mediana/INFORME_1A_v2.md`](resultados/v2_mediana/INFORME_1A_v2.md) (v2, mediana de los folds,
reapertura del pre-registro tras ver la v1) y
[`resultados/v3/INFORME_1A_v3.md`](resultados/v3/INFORME_1A_v3.md) (v3, criterios elegidos por potencia) y
[`resultados/v4/INFORME_1A_v4.md`](resultados/v4/INFORME_1A_v4.md) (v4, test 1997-2018 y objetivo normalizado; potencia 93 % a R² 2 %) y
[`resultados/v5/INFORME_1A_v5.md`](resultados/v5/INFORME_1A_v5.md) (v5, umbrales calibrados al 5 % de error total; **F05 estable condicional**, holdout sin abrir).

Línea 1B: [`resultados/1B/INFORME_1B.md`](resultados/1B/INFORME_1B.md) (la regla primaria no pasa la puerta de 8.2; 1B se detiene).

Réplica externa: [`replicaciones/sg_reversion_semanal/INFORME.md`](replicaciones/sg_reversion_semanal/INFORME.md) (SG, reversión semanal con ML, 13-05-2026; no se replica con precios del S&P 500 2017-2021; no escribe en `trials.jsonl`).

Prueba de potencia de la etapa 1 (datos sintéticos): [`resultados/potencia/INFORME_POTENCIA.md`](resultados/potencia/INFORME_POTENCIA.md).

Para ejecutar una versión distinta de la v1: `export LINEA1A_PREREG=preregistro/preregistro_1A_v2_mediana.yaml`.
Los resultados van a `resultados/<version>/` y `trials.jsonl` acumula los ensayos de todas las versiones.

## Estructura

```
preregistro/preregistro_1A.yaml   pre-registro fechado (su SHA-256 va en cada ensayo)
data_ref/fomc_decision_dates.csv  reuniones FOMC programadas (federalreserve.gov)
linea1a/                          paquete: datos, etapa1, etapa2, seleccion, evaluacion, figuras, registro
scripts/00_descargar_datos.sh     descarga con los scripts de Jansen
scripts/run_linea_1a.py           orquestador por pasos (sección 9.3)
resultados/                       tablas, figuras, trials.jsonl e informe
```

## Reproducir

```bash
git clone --depth 1 https://github.com/stefan-jansen/machine-learning-for-trading.git ../ml4t/machine-learning-for-trading
uv venv --python 3.13 && uv pip install -r requirements.txt     # o pip
bash scripts/00_descargar_datos.sh
python scripts/run_linea_1a.py datos
python scripts/run_linea_1a.py descubrimiento   # etapas 1 y 2, catálogo
python scripts/run_linea_1a.py validacion       # 2019-2022
python scripts/run_linea_1a.py nula             # 100 corridas con la sombra del objetivo (~2-3 h, 4 núcleos)
python scripts/run_linea_1a.py muestra_corta    # 7.8 con AlgoSeek
python scripts/run_linea_1a.py criterios        # 7.6
```

`ML4T_REPO` apunta al clon de Jansen si no está en `../ml4t/machine-learning-for-trading`.
Con `FRED_API_KEY` y `python data/macro/download_alfred.py` en el repo de Jansen, VIXCLS pasa a la primera publicación de ALFRED.

El holdout 2023-2025 está sellado en el código: `datos.cargar_diario()` se niega a cargar precios de 2023 en adelante.
