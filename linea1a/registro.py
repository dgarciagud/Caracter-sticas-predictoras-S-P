"""Registro de ensayos (trials.jsonl, sección 9.1)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from . import config as C


def _limpio(v):
    if isinstance(v, dict):
        return {k: _limpio(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_limpio(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def registrar(
    *,
    periodo: str,
    variables: list[str],
    modelo: str,
    hiperparametros: dict,
    metricas: dict,
    semilla: int | None = None,
    variante: str = "semanal_sin_solape",
    nota: str = "",
) -> None:
    entrada = dict(
        fecha=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        linea="1A",
        periodo=periodo,
        hash_prereg=C.hash_prereg(),
        variables=variables,
        modelo=modelo,
        hiperparametros=hiperparametros,
        horizonte=C.H,
        variante_decision=variante,
        semilla=semilla,
        metricas=metricas,
        nota=nota,
    )
    with open(C.TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(_limpio(entrada), ensure_ascii=False) + "\n")


def leer(linea: str = "1A") -> list[dict]:
    if not C.TRIALS_PATH.exists():
        return []
    with open(C.TRIALS_PATH, encoding="utf-8") as f:
        return [e for e in map(json.loads, f) if e["linea"] == linea]
