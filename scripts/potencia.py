"""Prueba de potencia de la etapa 1 (datos sintéticos; no usa el retorno real, no suma en trials.jsonl).

Cada réplica parte de y desplazado circularmente (sin relación con X, misma distribución y autocorrelación que el
retorno real) y le añade un efecto conocido sobre una candidata real:

    y* = y_sombra + beta * g(x),   g = x estandarizada (lineal) o 1[x > p70] estandarizada (umbral)

beta se calibra para que g explique un R² semanal dado en los bloques de test (2011-2018). Se mide con qué frecuencia la etapa 1 detecta la variable
plantada (pasa global: U1 > q1 y U2 > q2) con el peor fold (v1) y con la mediana (v2), y cuántas otras variables
pasan (falsos positivos). La etapa 1 es condición necesaria para "estable incondicional", así que esta potencia
es una cota superior de la del procedimiento completo.

Uso: python scripts/potencia.py [--runs 100] [--r1 200] [--n-jobs 4] [--variantes a,b] [--salida carpeta]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linea1a import config as C  # noqa: E402
from linea1a.etapa1 import Periodo, etapa1, folds, indices_sombra, offsets  # noqa: E402

warnings.filterwarnings("ignore")
VARIABLES = ["F06", "F01"]
FORMAS = ["lineal", "umbral"]
R2S = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
# (agregación de U1 sobre folds, modo de U2)
VARIANTES = {
    "v1_peor_fold": ("min", "por_fold"),
    "v2_mediana": ("mediana", "por_fold"),
    "mediana_U2agregado": ("mediana", "agregado"),
    "mediana_sinU2": ("mediana", "ninguno"),
    "media_U2agregado": ("media", "agregado"),
    "media_sinU2": ("media", "ninguno"),
}


def senal(x: np.ndarray, forma: str, en_test: np.ndarray) -> np.ndarray:
    """g(x) con media 0 y varianza 1 en los bloques de test, que es donde se mide la etapa 1."""
    ok = np.isfinite(x)
    g = np.where(ok, x, np.nan)
    if forma == "umbral":
        g = np.where(ok, (x > np.nanpercentile(x[en_test], 70)).astype(float), np.nan)
    g = (g - np.nanmean(g[en_test])) / np.nanstd(g[en_test])
    return np.nan_to_num(g)


def una_replica(r: int, k: int, r1: int, variantes: list[str]) -> list[dict]:
    per = Periodo.desde_panel(pd.read_parquet(C.CACHE / "panel.parquet"))
    T = len(per.fechas)
    idx, ok = indices_sombra(T, int(k))
    y0 = per.y[idx]
    m0 = per.ym[idx] & ok
    en_test = np.zeros(T, bool)
    for _, te in folds(per):
        en_test[te] = True
    # R² calibrado en los bloques de test 2011-2018 (F06 concentra su varianza en 2008-2009).
    sd = np.nanstd(np.where(m0 & en_test, y0, np.nan))
    filas = []
    celdas = [("ninguna", "ninguna", 0.0)] + [(v, f, r2) for v in VARIABLES for f in FORMAS for r2 in R2S if r2 > 0]
    for var, forma, r2 in celdas:
        y = y0.copy()
        if r2 > 0:
            j = per.nombres.index(var)
            beta = sd * np.sqrt(r2 / (1 - r2)) * C.SIGNOS[var]
            y = y0 + beta * senal(per.X[:, j], forma, en_test)
        p = per.con_y(y, m0)
        for version in variantes:
            C.AGREGACION, C.MODO_U2 = VARIANTES[version]
            e1 = etapa1(p, R=r1, semilla=C.SEMILLA + 5000 + r)
            t = e1["tabla"]
            g = t[t.corte == "global"].set_index("variable")
            pasan = set(t[t.pasa].variable)
            fila = dict(replica=r, k=int(k), variable=var, forma=forma, r2=r2, version=version, q1=e1["q1"], q2=e1["q2"])
            if var != "ninguna":
                fila.update(
                    U1=g.loc[var, "U1"],
                    U2=g.loc[var, "U2"],
                    pasa_global=bool(g.loc[var, "pasa"]),
                    pasa_algun_corte=var in pasan,
                    U1_supera_q1=bool(g.loc[var, "U1"] > e1["q1"]),
                )
            fila["n_otras_pasan"] = len(pasan - {var})
            filas.append(fila)
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--r1", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--variantes", default="v1_peor_fold,v2_mediana")
    ap.add_argument("--salida", default="potencia")
    a = ap.parse_args()
    variantes = a.variantes.split(",")
    per = Periodo.desde_panel(pd.read_parquet(C.CACHE / "panel.parquet"))
    ks = offsets(len(per.fechas), a.runs, C.SEMILLA + 7777)
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(una_replica)(r, k, a.r1, variantes) for r, k in enumerate(ks))
    d = pd.DataFrame([f for filas in res for f in filas])
    dest = C.RAIZ / "resultados" / a.salida
    dest.mkdir(parents=True, exist_ok=True)
    d.to_csv(dest / "potencia_detalle.csv", index=False)

    plantadas = d[d.variable != "ninguna"]
    pot = (
        plantadas.groupby(["version", "variable", "forma", "r2"])
        .agg(
            potencia=("pasa_global", "mean"),
            potencia_solo_U1=("U1_supera_q1", "mean"),
            pasa_algun_corte=("pasa_algun_corte", "mean"),
            U1_mediano=("U1", "median"),
            U2_mediano=("U2", "median"),
            q1_medio=("q1", "mean"),
            falsos_otras=("n_otras_pasan", "mean"),
        )
        .reset_index()
    )
    nula = d[d.variable == "ninguna"].groupby("version").agg(
        tasa_falsos_positivos=("n_otras_pasan", lambda s: (s > 0).mean()), q1_medio=("q1", "mean")
    )
    pot.to_csv(dest / "potencia_resumen.csv", index=False)
    nula.to_csv(dest / "potencia_nula.csv")
    pd.set_option("display.width", 220)
    print(nula.round(3))
    print(pot.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
