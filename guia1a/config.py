"""Constantes de la línea 1A según la guía del TFM (Guia_TFM_ML_gestion_Inversiones_L1_v26-27_v5).

Todo lo que fija el pre-registro se lee de preregistro/preregistro_1A_guia.yaml; su SHA-256 va
en cada línea del registro de intentos.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
PREREG_PATH = Path(os.environ.get("GUIA1A_PREREG", RAIZ / "preregistro" / "preregistro_1A_guia.yaml")).resolve()
CACHE = RAIZ / "cache"
DATA_REF = RAIZ / "data_ref"
ML4T_REPO = Path(os.environ.get("ML4T_REPO", RAIZ.parent / "ml4t" / "machine-learning-for-trading"))

with open(PREREG_PATH, encoding="utf-8") as _f:
    PREREG = yaml.safe_load(_f)

VERSION = str(PREREG["version"])
RESULTADOS = RAIZ / "resultados" / "guia1a" / VERSION
FIGURAS = RESULTADOS / "figuras"
REGISTRO = RESULTADOS / "registro_intentos.csv"
SERIES_OOS = RESULTADOS / "series_oos"

_m = PREREG["muestra"]
INICIO_DATOS = pd.Timestamp(_m["inicio_datos"])
DESC_INI, DESC_FIN = (pd.Timestamp(d) for d in _m["descubrimiento"])
HOLD_INI, HOLD_FIN = (pd.Timestamp(d) for d in _m["holdout"])
H = int(_m["horizonte_sesiones"])
SPAN_SIGMA = int(_m["span_sigma"])

_cv = PREREG["cpcv"]
N_BLOQUES = int(_cv["bloques"])
K_TEST = int(_cv["bloques_prueba"])
PURGA = int(_cv["purga_sesiones"])
EMBARGO = int(_cv["embargo_sesiones"])

_wf = PREREG["walk_forward"]
WF_TEST_INI = pd.Timestamp(_wf["primer_test"])
WF_MIN_TRAIN = int(_wf["minimo_sesiones_entrenamiento"])

NW_LAGS = int(PREREG["inferencia"]["nw_lags"])
SEMILLA = int(PREREG["semilla"])
SEMILLAS_GBM = [SEMILLA + i for i in range(int(PREREG["modelos"]["gbm"]["semillas"]))]

_s = PREREG["sombras"]
SOMBRA_MIN = int(_s["desplazamiento_minimo"])
SOMBRA_BORDE = float(_s["excluir_bordes"])
N_NULA_F1 = int(_s["replicas_fase1"])
N_SOMBRAS_SHAP = int(_s["copias_por_variable"])
N_NULA_SKILL = int(_s["replicas_puerta"])
N_NULA_REG = int(_s["replicas_regimen"])

COSTE_PB = float(PREREG["estrategia"]["coste_pb_por_lado"])

CANDIDATAS: dict[str, dict] = PREREG["candidatas"]
REGIMENES: dict[str, dict] = PREREG["regimenes"]
INTERACCIONES: dict[str, dict] = PREREG["interacciones"]
SIGNOS = {k: int(v["signo"]) for k, v in CANDIDATAS.items()}
FAMILIA = {k: v["familia"] for k, v in CANDIDATAS.items()}

_g = PREREG["modelos"]["gbm"]
LGBM = dict(
    n_estimators=int(_g["n_estimators"]),
    learning_rate=float(_g["learning_rate"]),
    max_depth=int(_g["max_depth"]),
    num_leaves=int(_g["num_leaves"]),
    min_child_samples=int(_g["min_child_samples"]),
    colsample_bytree=float(_g["colsample_bytree"]),
    subsample=1.0,
    verbose=-1,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
)
ALPHAS_RIDGE = [float(a) for a in PREREG["modelos"]["ridge"]["alphas"]]

# Llave del holdout: sin esta variable de entorno con este valor exacto el código no carga 2013+.
LLAVE_HOLDOUT = "ABRIR-HOLDOUT-UNA-SOLA-VEZ"


def hash_prereg() -> str:
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def holdout_abierto() -> bool:
    return os.environ.get("GUIA1A_HOLDOUT") == LLAVE_HOLDOUT


for _d in (RESULTADOS, FIGURAS, SERIES_OOS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)
