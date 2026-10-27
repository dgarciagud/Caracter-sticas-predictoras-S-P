"""Rutas y parámetros de la réplica (SG Global Style Counselling, 13-05-2026).

Nada de aquí escribe en resultados/trials.jsonl: la réplica no es un ensayo de la línea 1A y no
importa linea1a.registro. Todo lo que produce va a replicaciones/sg_reversion_semanal/resultados/.
"""

from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
CACHE = RAIZ / "cache"
# v2: sólo precio y volumen (resultados/). v3: + fundamentales XBRL, dividendos y sustitutos de las
# revisiones de BPA (resultados/v3/). Mismo modelo, cartera y análisis en las dos.
VERSION = os.environ.get("SG_VERSION", "v3")
RESULTADOS = RAIZ / "resultados" if VERSION == "v2" else RAIZ / "resultados" / VERSION
FIGURAS = RESULTADOS / "figuras"
SUFIJO = "" if VERSION == "v2" else f"_{VERSION}"

# Universo: componentes del S&P 500 en barras diarias AlgoSeek (Jansen), 2017-01-03 a 2021-12-31.
INICIO_ETF = "2016-01-01"
# Ventanas de las características (sesiones); min_periods reducidos porque el panel empieza en 2017.
V_BETA, MIN_BETA = 252, 63
V_VOL, MIN_VOL = 126, 63
V_SECTOR, MIN_SECTOR = 252, 63

# Modelo: GBM sobre el rango transversal del retorno a 5 sesiones, reentrenado cada trimestre con
# ventana expansiva (el paper usa ventana móvil de 10 años; aquí no hay tanta historia).
HORIZONTE = 5
REENTRENO_SESIONES = 63
MIN_ENTRENO_SESIONES = 189
EMBARGO_SESIONES = HORIZONTE + 1
# Árboles poco profundos y hojas grandes (como LGBM_PARAMS de linea1a): con 1-4 años de entrenamiento y un
# objetivo tan ruidoso, la versión con 31 hojas sobreajustaba (IC fuera de muestra 0,007 frente a 0,016 de la
# reversión simple). Ver INFORME.md, «Iteraciones».
LGBM_PARAMS = dict(
    objective="regression",
    num_leaves=7,
    max_depth=3,
    learning_rate=0.02,
    n_estimators=300,
    min_child_samples=2000,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    reg_lambda=1.0,
    verbose=-1,
    n_jobs=4,
    deterministic=True,
    force_row_wise=True,
    random_state=7,
)

# Cartera: larga el quintil superior y corta el inferior dentro de cada sector (neutral a sector),
# pesos iguales, 1 de nominal largo y 1 corto. La señal se calcula al cierre del día de rebalanceo.
CUANTIL = 0.2
DIA_REBALANCEO = 2  # miércoles, como en el análisis de lags del paper
LAG_INVERTIBLE = 1  # ejecución al cierre siguiente
COSTE_PB = 10.0  # por lado y por unidad negociada (grandes valores de EE. UU.)
COSTES_SENSIBILIDAD = [0.0, 5.0, 10.0, 20.0]
SEMANAS_ANIO = 52
MESES_ANIO = 12

for _d in (CACHE, RESULTADOS, FIGURAS):
    _d.mkdir(parents=True, exist_ok=True)
