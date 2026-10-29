"""Prueba sintética de la metodología screening OR + SFI/MDA (metodologia/METODOLOGIA_SCREENING_OR.md, sección 6).

No usa el objetivo real ni suma en trials.jsonl. Base sin señal: objetivo de la 1A v5 (y / sigma5, test 1997-2018)
desplazado circularmente; sobre él se planta un efecto conocido en una candidata real (en unidades del objetivo
normalizado; R² medido en los bloques de test):

    ninguna            sin efecto (falsos positivos; calibra p*)
    F01 lineal         R² 0,5 / 1 / 2 %
    F05 condicional    efecto solo con R_trend = 1, R² 1 %
    F06 deriva         coeficiente AR(1) con phi = 0,99 y media 0, R² 2 %

Uso: LINEA1A_PREREG=preregistro/preregistro_1A_v5.yaml python scripts/probar_screening.py --runs 40
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
from linea1a.etapa1 import Periodo, folds, indices_sombra, offsets  # noqa: E402
from screening.fase1 import fase1  # noqa: E402
from screening.fase2 import fase2, seleccion  # noqa: E402

warnings.filterwarnings("ignore")
CELDAS = [
    ("ninguna", "ninguna", 0.0),
    ("F01", "lineal", 0.005),
    ("F01", "lineal", 0.01),
    ("F01", "lineal", 0.02),
    ("F05", "condicional", 0.01),
    ("F06", "deriva", 0.02),
]
REJILLA_P = [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 50.0]
P_SCREEN = 80.0


def senal(per: Periodo, var: str, forma: str, en_test: np.ndarray, rng) -> np.ndarray:
    x = per.X[:, per.nombres.index(var)]
    x = (x - np.nanmean(x[en_test])) / np.nanstd(x[en_test])
    if forma == "condicional":
        g = x * (per.reg["R_trend"].to_numpy() == 1)
    elif forma == "deriva":
        b = np.zeros(len(x))
        e = rng.standard_normal(len(x)) * np.sqrt(1 - 0.99**2)
        for t in range(1, len(x)):
            b[t] = 0.99 * b[t - 1] + e[t]
        g = b * x
    else:
        g = x
    g = np.nan_to_num((g - np.nanmean(g[en_test])) / np.nanstd(g[en_test]))
    return C.SIGNOS.get(var, 1) * g if forma != "deriva" else g


def una_corrida(r: int, k: int, r1: int, n_rep2: int) -> list[dict]:
    per = Periodo.desde_panel(pd.read_parquet(C.PANEL_PATH))
    T = len(per.fechas)
    en_test = np.zeros(T, bool)
    for _, te in folds(per):
        en_test[te] = True
    idx, ok = indices_sombra(T, int(k))
    y0, m0 = per.y[idx], per.ym[idx] & ok
    sd = np.nanstd(np.where(m0 & en_test, y0, np.nan))
    rng = np.random.default_rng(C.SEMILLA + 4242 + r)
    filas = []
    for var, forma, r2 in CELDAS:
        y = y0 if r2 == 0 else y0 + sd * np.sqrt(r2 / (1 - r2)) * senal(per, var, forma, en_test, rng)
        p = per.con_y(y, m0)
        f1 = fase1(p, R=r1, semilla=C.SEMILLA + 5000 + r, p_s=P_SCREEN)
        t1 = f1["tabla"].set_index("variable")
        pasan = t1[t1.pasa_OR].index.tolist()
        f2 = fase2(p, pasan, R=n_rep2, semilla=C.SEMILLA + 6000 + r, n_jobs=1) if pasan else None
        base = dict(replica=r, k=int(k), variable=var, forma=forma, r2=r2, pasan_fase1=";".join(pasan))
        if var != "ninguna":
            base.update({f"pasa_{c}": bool(t1.loc[var, f"pasa_{c}"]) for c in ("C1", "C2", "C3", "OR")})
        for pc in REJILLA_P:
            sel, _ = seleccion(f2, pc)
            fila = dict(base, percentil=pc, seleccionadas=";".join(sel), n_sel=len(sel))
            if var != "ninguna":
                # Detección: la plantada, o el representante de su cluster, queda seleccionada.
                rep = None
                if f2 is not None and var in f2["variables"]:
                    i = f2["variables"].index(var)
                    cl = next(c for c, g in zip(f2["grupos"]["clusters"], f2["grupos"]["indices"]) if i in g)
                    miembros = [f2["variables"][j] for c, g in zip(f2["grupos"]["clusters"], f2["grupos"]["indices"]) if c == cl for j in g]
                    rep = any(m in sel for m in miembros)
                fila["detectada"] = bool(rep)
                fila["otras"] = len([s for s in sel if s != var])
            filas.append(fila)
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--r1", type=int, default=300)
    ap.add_argument("--r2", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--salida", default="screening_prueba")
    a = ap.parse_args()
    per = Periodo.desde_panel(pd.read_parquet(C.PANEL_PATH))
    ks = offsets(len(per.fechas), a.runs, C.SEMILLA + 8888)
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(una_corrida)(r, k, a.r1, a.r2) for r, k in enumerate(ks))
    d = pd.DataFrame([f for fs in res for f in fs])
    dest = C.RAIZ / "resultados" / a.salida
    dest.mkdir(parents=True, exist_ok=True)
    d.to_csv(dest / "detalle.csv", index=False)

    nula = d[d.variable == "ninguna"].groupby("percentil").n_sel.apply(lambda s: (s > 0).mean()).rename("falsos_positivos")
    validos = nula[nula <= 0.05].index
    p_star = float(min(validos)) if len(validos) else 95.0
    plant = d[d.variable != "ninguna"]
    f1 = plant[plant.percentil == p_star].groupby(["variable", "forma", "r2"])[
        ["pasa_C1", "pasa_C2", "pasa_C3", "pasa_OR"]].mean()
    pot = plant.groupby(["variable", "forma", "r2", "percentil"]).agg(potencia=("detectada", "mean"),
                                                                       otras=("otras", "mean")).reset_index()
    pot_star = pot[pot.percentil == p_star].set_index(["variable", "forma", "r2"])
    resumen = f1.join(pot_star[["potencia", "otras"]])
    nula.to_csv(dest / "falsos_positivos_por_percentil.csv")
    pot.to_csv(dest / "potencia_por_percentil.csv", index=False)
    resumen.to_csv(dest / "resumen_p_estrella.csv")
    nula_screen = d[(d.variable == "ninguna") & (d.percentil == p_star)].pasan_fase1.fillna("").ne("").mean()
    pd.set_option("display.width", 200)
    print(nula.to_string())
    print(f"p* = {p_star}; fracción de nulas con alguna candidata en la fase 2: {nula_screen:.2f}")
    print(resumen.round(3).to_string())


if __name__ == "__main__":
    main()
