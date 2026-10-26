"""Registro de ensayos de la línea 1B en el mismo trials.jsonl que la 1A (8.9: DSR y PBO con ambos)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from linea1a.registro import _limpio

from . import config as C

TRIALS_PATH = C.RAIZ / "resultados" / "trials.jsonl"


def registrar(*, periodo: str, modelo: str, variables: list[str], hiperparametros: dict, metricas: dict, nota: str = "") -> None:
    entrada = dict(
        fecha=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        linea="1B",
        periodo=periodo,
        version_prereg=C.PREREG["version"],
        hash_prereg=C.hash_prereg(),
        variables=variables,
        modelo=modelo,
        hiperparametros=hiperparametros,
        horizonte=C.H,
        variante_decision="semanal_eventos_regla",
        metricas=metricas,
        nota=nota,
    )
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(_limpio(entrada), ensure_ascii=False) + "\n")
