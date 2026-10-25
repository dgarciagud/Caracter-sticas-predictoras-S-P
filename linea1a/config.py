"""Rutas, pre-registro y constantes de la línea 1A."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
# La versión activa del pre-registro se elige con LINEA1A_PREREG (por defecto, la v1 original).
PREREG_PATH = Path(os.environ.get("LINEA1A_PREREG", RAIZ / "preregistro" / "preregistro_1A.yaml")).resolve()
CACHE = RAIZ / "cache"
DATA_REF = RAIZ / "data_ref"
# Un único registro de ensayos para todas las versiones: el DSR y el PBO usan el recuento acumulado.
TRIALS_PATH = RAIZ / "resultados" / "trials.jsonl"

# Repositorio de Stefan Jansen (ML for Trading, 3.ª ed.): sus loaders `data.*` y sus datos.
ML4T_REPO = Path(os.environ.get("ML4T_REPO", RAIZ.parent / "ml4t" / "machine-learning-for-trading"))


def usar_repo_ml4t() -> None:
    """Pone el repo de Jansen en sys.path para importar `data` (sus loaders)."""
    if not ML4T_REPO.exists():
        raise FileNotFoundError(
            f"No encuentro el repositorio de ML4T en {ML4T_REPO}. "
            "Clónalo (scripts/00_descargar_datos.sh) o fija ML4T_REPO."
        )
    os.environ.setdefault("ML4T_DATA_PATH", str(ML4T_REPO / "data"))
    if str(ML4T_REPO) not in sys.path:
        sys.path.insert(0, str(ML4T_REPO))


def cargar_prereg() -> dict:
    with open(PREREG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def hash_prereg() -> str:
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


PREREG = cargar_prereg()
VERSION = str(PREREG.get("version", "v1"))
RESULTADOS = RAIZ / "resultados" if VERSION == "v1" else RAIZ / "resultados" / VERSION
FIGURAS = RESULTADOS / "figuras"

# Cómo se resume un estadístico sobre los folds: "min" (peor fold, v1), "mediana" (v2) o "media".
AGREGACION = PREREG.get("agregacion_folds", "min")
# U2 de la etapa 1: "por_fold" (v1, v2), "agregado" (quintiles con todas las semanas de test) o "ninguno".
MODO_U2 = PREREG.get("modo_U2", "por_fold")
# Estadísticos que debe superar el GBM en la etapa 2 (v1, v2: G1, G2 y G3).
CRITERIO_GBM: list[str] = list(PREREG.get("criterio_gbm", ["G1", "G2", "G3"]))


def agregar_folds(a, axis: int = -1):
    f = {"min": np.nanmin, "mediana": np.nanmedian, "media": np.nanmean}[AGREGACION]
    return f(a, axis)


CANDIDATAS: list[str] = list(PREREG["candidatas"])
SIGNOS: dict[str, int] = {k: int(v["signo"]) for k, v in PREREG["candidatas"].items()}
EXCEPCIONES: list[str] = list(PREREG["excepciones_teoricas"])
PRIORIDAD: list[str] = list(PREREG["clusters"]["prioridad_representante"])

_m = PREREG["muestra"]
DESC_INI, DESC_FIN = (pd.Timestamp(d) for d in _m["descubrimiento"])
VAL_INI, VAL_FIN = (pd.Timestamp(d) for d in _m["validacion"])
HOLDOUT_INI = pd.Timestamp(_m["holdout"][0])
H = int(_m["horizonte_sesiones"])

_tests = PREREG["folds"].get("ventana_test", ["2011-01-01", "2018-12-31"])
TEST_INI, TEST_FIN = (pd.Timestamp(d) for d in _tests)
# Fecha desde la que se cargan los datos diarios (lookbacks); el panel de cada inicio se cachea aparte.
INICIO_CARGA = str(_m.get("inicio_carga", "2003-01-01"))
PANEL_PATH = CACHE / ("panel.parquet" if INICIO_CARGA == "2003-01-01" else f"panel_{INICIO_CARGA[:4]}.parquet")
# Objetivo de la selección: "y" (retorno log en exceso) o "y_norm" (y / sigma_5 prevista en t).
OBJETIVO = str(_m.get("objetivo", "y"))
N_FOLDS = 5
EMBARGO_SEMANAS = int(PREREG["folds"]["embargo_semanas"])
MIN_CELDA = int(PREREG["regimenes"]["minimo_semanas_celda"])

_s = PREREG["sombras"]
K_MIN = int(_s["k_min_semanas"])
R1 = int(_s["replicas_etapa1"])
R2 = int(_s["replicas_etapa2"])
RY = int(_s["replicas_objetivo"])
SEMILLA = int(_s["semilla_base"])

GAMMA = 3.0
W_MAX = 1.5
COSTE_PB = float(PREREG["evaluacion"]["costes_pb_por_lado"])

CORTES = ["global", "R_vol=1", "R_vol=0", "R_trend=1", "R_trend=0"]

LGBM_PARAMS = dict(
    max_depth=3,
    num_leaves=7,
    min_child_samples=40,
    learning_rate=0.02,
    n_estimators=300,
    colsample_bytree=0.8,
    subsample=1.0,
    verbose=-1,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
)

for _d in (RESULTADOS, FIGURAS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)
