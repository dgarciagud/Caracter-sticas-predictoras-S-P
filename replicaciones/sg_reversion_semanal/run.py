"""Réplica completa: python -m replicaciones.sg_reversion_semanal.run (desde la raíz del repositorio).

SG_VERSION=v2 reproduce la versión sólo de precio y volumen (resultados/); por defecto, v3 (resultados/v3/).

Lee los datos con linea1a.datos y el repo de Jansen (scripts/00_descargar_datos.sh) y escribe sólo en
replicaciones/sg_reversion_semanal/{cache,resultados}; resultados/trials.jsonl no se toca.
"""

from __future__ import annotations

from . import caracteristicas, datos, modelo
from . import config as C
from .analisis import Replica


def main() -> None:
    print("datos")
    d = datos.cargar()
    panel, pxf = caracteristicas.construir(d)
    if C.VERSION != "v2":
        print("fundamentales SEC y dividendos")
        panel = caracteristicas.anadir_v3(panel, d)
    panel = caracteristicas.anadir_objetivo(panel, pxf)
    print("modelo walk-forward")
    pred, extra = modelo.walk_forward(panel)
    print("análisis")
    Replica(panel, pred, pxf, d["etf"], extra).todo()


if __name__ == "__main__":
    main()
