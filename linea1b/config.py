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
CACHE = RAIZ / "cache"

with open(PREREG_PATH, encoding="utf-8") as f:
    PREREG = yaml.safe_load(f)

VERSION = str(PREREG["version"])
RESULTADOS = RAIZ / "resultados" / "1B" if VERSION == "1B-v1" else RAIZ / "resultados" / "1B" / VERSION.split("-")[1]
FIGURAS = RESULTADOS / "figuras"
# Regla primaria: "reversion" (1B-v1), "vix_alcista" (1B-v2) o "reversion_laxa" (1B-v3).
REGLA = {"1B-v1": "reversion", "1B-v2": "vix_alcista"}.get(VERSION, "reversion_laxa")


def hash_prereg() -> str:
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


_m = PREREG["muestra"]
INICIO_CARGA = pd.Timestamp(_m["inicio_carga"])
DESC_INI, DESC_FIN = (pd.Timestamp(d) for d in _m["descubrimiento"])
TEST_INI, TEST_FIN = (pd.Timestamp(d) for d in _m["ventana_test"])
VAL_INI, VAL_FIN = (pd.Timestamp(d) for d in _m["validacion"])
HOLDOUT_INI = pd.Timestamp(_m["holdout"][0])

H = 5
UMBRAL_REGLA = 0.5 if REGLA == "reversion_laxa" else 1.5
BARRERA = 1.0
COSTE_IDA_VUELTA = 4e-4
T_MIN_REGLA = 1.5
INICIO_HAR = pd.Timestamp("1990-01-01")
META_FEATURES: list[str] = list(PREREG.get("meta_features") or PREREG["anexo_meta_label"]["meta_features_efectivas"])
# La puerta de 8.2 detiene la línea en la 1B-v1/v2; en la 1B-v3 solo se informa.
PUERTA_OBLIGATORIA = REGLA != "reversion_laxa"

for _d in (RESULTADOS, FIGURAS):
    _d.mkdir(parents=True, exist_ok=True)
