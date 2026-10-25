"""Potencia del procedimiento completo (etapas 1 y 2 + clasificación) con datos sintéticos.

Como scripts/potencia.py, pero ejecuta seleccionar_catalogo() y mide si la variable plantada acaba en el catálogo
como "estable_incondicional" (o "estable_condicional"). Usa la versión de pre-registro activa (LINEA1A_PREREG) o
fija la agregación y el modo de U2 con --agregacion / --modo-u2. No usa el retorno real ni suma en trials.jsonl.

Uso: python scripts/potencia_completa.py --runs 20 --r1 200 --r2 20 --agregacion media --modo-u2 agregado
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linea1a import config as C  # noqa: E402
from linea1a.etapa1 import Periodo, folds, indices_sombra, offsets  # noqa: E402
from linea1a.seleccion import seleccionar_catalogo  # noqa: E402
from potencia import senal  # noqa: E402

warnings.filterwarnings("ignore")
CELDAS = [("F01", "lineal", 0.02), ("F06", "lineal", 0.02), ("F01", "lineal", 0.04), ("F06", "umbral", 0.04)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--r1", type=int, default=200)
    ap.add_argument("--r2", type=int, default=20)
    ap.add_argument("--agregacion", default=None)
    ap.add_argument("--modo-u2", default=None)
    ap.add_argument("--criterio-gbm", default=None, help="p. ej. G1,G3")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--salida", default="potencia_completa")
    a = ap.parse_args()
    if a.agregacion:
        C.AGREGACION = a.agregacion
    if a.modo_u2:
        C.MODO_U2 = a.modo_u2
    if a.criterio_gbm:
        C.CRITERIO_GBM = a.criterio_gbm.split(",")

    per = Periodo.desde_panel(pd.read_parquet(C.PANEL_PATH))
    T = len(per.fechas)
    en_test = np.zeros(T, bool)
    for _, te in folds(per):
        en_test[te] = True
    ks = offsets(T, a.runs, C.SEMILLA + 7777)  # mismos desplazamientos de y que scripts/potencia.py
    dest = C.RAIZ / "resultados" / a.salida
    dest.mkdir(parents=True, exist_ok=True)
    filas = []
    for r, k in enumerate(ks):
        idx, ok = indices_sombra(T, int(k))
        y0, m0 = per.y[idx], per.ym[idx] & ok
        sd = np.nanstd(np.where(m0 & en_test, y0, np.nan))
        for var, forma, r2 in CELDAS:
            j = per.nombres.index(var)
            y = y0 + sd * np.sqrt(r2 / (1 - r2)) * C.SIGNOS[var] * senal(per.X[:, j], forma, en_test)
            sel = seleccionar_catalogo(per.con_y(y, m0), R1=a.r1, R2=a.r2, semilla=C.SEMILLA + 9000 + r, n_jobs=a.n_jobs)
            cat = sel["catalogo"].set_index("variable")
            estables = cat[cat.clase.str.startswith("estable")].index.tolist()
            fila = dict(
                replica=r,
                variable=var,
                forma=forma,
                r2=r2,
                clase_e1=cat.loc[var, "clase_e1"],
                clase=cat.loc[var, "clase"],
                motivo=cat.loc[var, "motivo"],
                detectada=var in estables,
                otras_estables=";".join(v for v in estables if v != var),
            )
            filas.append(fila)
            print(fila, flush=True)
            pd.DataFrame(filas).to_csv(dest / "potencia_completa_detalle.csv", index=False)
    d = pd.DataFrame(filas)
    res = d.groupby(["variable", "forma", "r2"]).agg(
        potencia_etapa1=("clase_e1", lambda s: s.isin(["pasa_global", "condicional"]).mean()),
        potencia_completa=("detectada", "mean"),
        n=("detectada", "size"),
    )
    res.to_csv(dest / "potencia_completa_resumen.csv")
    print(res)
    print(d.groupby(["variable", "forma", "r2"]).clase.value_counts())


if __name__ == "__main__":
    main()
