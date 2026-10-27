"""Réplica completa: python -m replicaciones.sg_reversion_semanal.run (desde la raíz del repositorio).

Lee los datos con linea1a.datos y el repo de Jansen (scripts/00_descargar_datos.sh) y escribe sólo en
replicaciones/sg_reversion_semanal/{cache,resultados}; resultados/trials.jsonl no se toca.
"""

from __future__ import annotations

from . import caracteristicas, datos, modelo
from .analisis import Replica


def main() -> None:
    print("datos")
    d = datos.cargar()
    panel, pxf = caracteristicas.construir(d)
    panel = caracteristicas.anadir_objetivo(panel, pxf)
    print("modelo walk-forward")
    pred, extra = modelo.walk_forward(panel)
    print("análisis")
    Replica(panel, pred, pxf, d["etf"], extra).todo()


if __name__ == "__main__":
    main()
