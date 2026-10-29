"""Evaluación de SENSIBILIDAD: metodología screening OR + SFI/MDA sobre los datos reales de la 1A (v5).

Declarada como sensibilidad: NO modifica el catálogo de la 1A (F05 condicional, v5) ni reabre su pre-registro.
Se registra como ensayo en trials.jsonl (cuenta para el DSR y el PBO). El percentil p* se toma de la calibración
sintética (resultados/screening_prueba/falsos_positivos_por_percentil.csv), fijada antes de mirar el objetivo real.
Mismo diseño que la 1A v5: descubrimiento 1993-2018, test 1997-2018, objetivo y / sigma5, 10 candidatas.

Uso: LINEA1A_PREREG=preregistro/preregistro_1A_v5.yaml python scripts/sensibilidad_screening.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linea1a import config as C  # noqa: E402
from linea1a import registro  # noqa: E402
from linea1a.etapa1 import Periodo  # noqa: E402
from screening.fase1 import fase1  # noqa: E402
from screening.fase2 import fase2, seleccion  # noqa: E402

warnings.filterwarnings("ignore")
P_SCREEN = 80.0


def p_estrella() -> float:
    fp = pd.read_csv(C.RAIZ / "resultados" / "screening_prueba" / "falsos_positivos_por_percentil.csv", index_col=0).iloc[:, 0]
    validos = fp[fp <= 0.05].index
    return float(min(validos)) if len(validos) else 95.0


def main():
    assert C.VERSION == "v5", "la sensibilidad se hace sobre el diseño de la 1A v5"
    p = p_estrella()
    per = Periodo.desde_panel(pd.read_parquet(C.PANEL_PATH))
    f1 = fase1(per, R=1000, semilla=C.SEMILLA + 11, p_s=P_SCREEN)
    t1 = f1["tabla"]
    pasan = t1[t1.pasa_OR].variable.tolist()
    f2 = fase2(per, pasan, R=100, semilla=C.SEMILLA + 12, n_jobs=4) if pasan else None
    sel, t2 = seleccion(f2, p)
    dest = C.RAIZ / "resultados" / "sensibilidad_screening"
    dest.mkdir(parents=True, exist_ok=True)
    t1.to_csv(dest / "fase1.csv", index=False)
    if len(t2):
        t2.to_csv(dest / "fase2.csv", index=False)
    res = dict(p_screen=P_SCREEN, p_estrella=p, pasan_fase1=pasan, seleccionadas=sel,
               nota="evaluación de sensibilidad; no modifica el catálogo de la 1A v5")
    json.dump(res, open(dest / "resumen.json", "w"), indent=2)
    registro.registrar(periodo="descubrimiento", variables=list(per.nombres), modelo="sensibilidad_screening_OR_SFI_MDA",
                       hiperparametros=dict(p_screen=P_SCREEN, p_estrella=p, R1=1000, R2=100),
                       metricas=dict(pasan_fase1=pasan, seleccionadas=sel),
                       nota="evaluación de sensibilidad declarada; no cambia el catálogo de la 1A")
    pd.set_option("display.width", 220)
    print(t1.round(3).to_string(index=False))
    print(t2.round(4).to_string(index=False) if len(t2) else "sin fase 2")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
