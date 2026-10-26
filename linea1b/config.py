"""Pre-registro, rutas y constantes de la línea 1B."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
PREREG_PATH = Path(os.environ.get("LINEA1B_PREREG", RAIZ / "preregistro" / "preregistro_1B.yaml")).resolve()
PREREG_1A = RAIZ / "preregistro" / "preregistro_1A_v5.yaml"  # catálogo y definiciones de las meta-features
RESULTADOS = RAIZ / "resultados" / "1B"
FIGURAS = RESULTADOS / "figuras"
CACHE = RAIZ / "cache"

with open(PREREG_PATH, encoding="utf-8") as f:
    PREREG = yaml.safe_load(f)


def hash_prereg() -> str:
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


_m = PREREG["muestra"]
INICIO_CARGA = pd.Timestamp(_m["inicio_carga"])
DESC_INI, DESC_FIN = (pd.Timestamp(d) for d in _m["descubrimiento"])
TEST_INI, TEST_FIN = (pd.Timestamp(d) for d in _m["ventana_test"])
VAL_INI, VAL_FIN = (pd.Timestamp(d) for d in _m["validacion"])
HOLDOUT_INI = pd.Timestamp(_m["holdout"][0])

H = 5
UMBRAL_REGLA = 1.5
BARRERA = 1.0
COSTE_IDA_VUELTA = 4e-4
T_MIN_REGLA = 1.5
INICIO_HAR = pd.Timestamp("1990-01-01")
META_FEATURES: list[str] = list(PREREG["meta_features"])

for _d in (RESULTADOS, FIGURAS):
    _d.mkdir(parents=True, exist_ok=True)
