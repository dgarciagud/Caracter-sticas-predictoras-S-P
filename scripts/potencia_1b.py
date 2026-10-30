"""Prueba de potencia del meta-label de la 1B-v3 (pre-registro, sección potencia), antes de la selección real.

Meta-labels SINTÉTICOS sobre las fechas, features y pesos reales de los eventos de descubrimiento (no usa los
meta-labels reales ni suma en trials.jsonl). Tasa de acierto media = la real de los eventos. Celdas:

    ninguno     y ~ Bernoulli(tasa real)                                  (falsos positivos con p*)
    F05 AUC     P(y=1) = logit^-1(a + b·z_F05), b tal que el AUC esperado sea 0,55 / 0,60 / 0,65
    R_trend     P(y=1) = tasa + 0,10·(R_trend - media): 10 puntos más de acierto en mercado alcista

Detección = la feature plantada entra en el meta-modelo con el percentil p (se informa toda la rejilla y p*).

Uso: LINEA1B_PREREG=preregistro/preregistro_1B_v3.yaml python scripts/potencia_1b.py --runs 40
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import brentq
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linea1b import config as C  # noqa: E402
from linea1b import meta as M  # noqa: E402

warnings.filterwarnings("ignore")
REJILLA_P = [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 50.0]
SEMILLA = 20261027
CELDAS = [("ninguno", 0.0), ("F05", 0.55), ("F05", 0.60), ("F05", 0.65), ("R_trend", 0.10)]


def eventos_desc() -> M.Eventos:
    ev = pd.read_csv(C.RESULTADOS / "02_eventos_descubrimiento.csv", index_col=0, parse_dates=["fecha", "fin"])
    return M.Eventos.construir(ev, pd.read_parquet(C.CACHE / "panel_1988.parquet"))


def auc_esperado(x: np.ndarray, p: np.ndarray) -> float:
    """AUC esperado de x frente a y ~ Bernoulli(p): pares (i positivo, j negativo) con x_i > x_j (empates 1/2)."""
    num = p[:, None] * (1 - p[None, :])
    np.fill_diagonal(num, 0)
    orden = (x[:, None] > x[None, :]) + 0.5 * (x[:, None] == x[None, :])
    return float((num * orden).sum() / num.sum())


def probabilidades(E: M.Eventos, var: str, efecto: float, tasa: float) -> np.ndarray:
    n = len(E.y)
    if var == "ninguno":
        return np.full(n, tasa)
    x = E.X[:, M.FEATURES.index(var)]
    if var == "R_trend":
        return np.clip(tasa + efecto * (x - x.mean()), 0.01, 0.99)
    z = (x - x.mean()) / x.std()

    def p_de(b):
        a = brentq(lambda a: expit(a + b * z).mean() - tasa, -10, 10)
        return expit(a + b * z)

    b = brentq(lambda b: auc_esperado(z, p_de(b)) - efecto, 1e-4, 5)
    return p_de(b)


def una_replica(E: M.Eventos, probs: dict, r: int, R1: int, R2: int) -> list[dict]:
    rng = np.random.default_rng(SEMILLA + 7000 + r)
    filas = []
    for (var, efecto), p in probs.items():
        En = E.con_y(rng.binomial(1, p))
        e1 = M.etapa1(En, R=R1, semilla=SEMILLA + 8000 + r)
        cache = {}
        for pc in REJILLA_P:
            sel = M.seleccionar(En, pc, R2=R2, semilla=SEMILLA + 8000 + r, e1=e1, cache_e2=cache)
            filas.append(dict(replica=r, variable=var, efecto=efecto, percentil=pc,
                              pasan_e1=";".join(sel["pasan_e1"]), meta_features=";".join(sel["meta_features"]),
                              n=len(sel["meta_features"]), detectada=var in sel["meta_features"],
                              otras=len([f for f in sel["meta_features"] if f != var])))
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--r1", type=int, default=1000)
    ap.add_argument("--r2", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    E = eventos_desc()
    tasa = float(E.y.mean())
    probs = {c: probabilidades(E, *c, tasa) for c in CELDAS}
    comprob = {f"{v}_{e}": dict(auc_esperado=auc_esperado(E.X[:, M.FEATURES.index(v)], p) if v != "ninguno" else 0.5,
                                rango_p=[float(p.min()), float(p.max())]) for (v, e), p in probs.items()}
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(una_replica)(E, probs, r, a.r1, a.r2) for r in range(a.runs))
    d = pd.DataFrame([f for fs in res for f in fs])
    dest = C.RESULTADOS / "potencia"
    dest.mkdir(parents=True, exist_ok=True)
    d.to_csv(dest / "detalle.csv", index=False)
    p_star = float((C.RESULTADOS / "03_p_estrella.txt").read_text())
    tabla = d.groupby(["variable", "efecto", "percentil"]).agg(
        potencia=("detectada", "mean"), alguna=("n", lambda s: (s > 0).mean()), otras=("otras", "mean")).reset_index()
    tabla.to_csv(dest / "potencia_por_percentil.csv", index=False)
    resumen = tabla[tabla.percentil == p_star].drop(columns="percentil")
    resumen.to_csv(dest / "resumen_p_estrella.csv", index=False)
    json.dump(dict(p_estrella=p_star, tasa_acierto=tasa, replicas=a.runs, r1=a.r1, r2=a.r2, celdas=comprob),
              open(dest / "config.json", "w"), indent=2)
    pd.set_option("display.width", 200)
    print(json.dumps(comprob, indent=2))
    print(f"p* = {p_star}")
    print(resumen.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
