"""Calibración del percentil de los umbrales para la v5 (no usa el retorno real; no suma en trials.jsonl).

Regla fijada antes de calibrar: con N conjuntos sin señal (objetivo desplazado circularmente, semillas distintas de
las de la nula de la v4) se mide, para cada percentil p de la rejilla, la fracción de corridas en que el
procedimiento completo declara al menos una variable estable. p* = el percentil más bajo con fracción <= 5 %;
si ninguno por debajo de 95 lo cumple, p* = 95.

Uso: LINEA1A_PREREG=preregistro/preregistro_1A_v4.yaml python scripts/calibrar_v5.py --runs 100 --r1 500 --r2 50
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
from linea1a.etapa1 import Periodo, etapa1, indices_sombra, offsets  # noqa: E402
from linea1a.seleccion import catalogo_desde_etapa1, n_estables  # noqa: E402

warnings.filterwarnings("ignore")
REJILLA = [95.0, 92.5, 90.0, 87.5, 85.0, 82.5, 80.0, 77.5, 75.0, 70.0, 65.0, 60.0, 50.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--r1", type=int, default=500)
    ap.add_argument("--r2", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    per = Periodo.desde_panel(pd.read_parquet(C.PANEL_PATH))
    T = len(per.fechas)
    ks = offsets(T, a.runs, C.SEMILLA + 3333)
    dest = C.RAIZ / "resultados" / "calibracion_v5"
    dest.mkdir(parents=True, exist_ok=True)
    ruta = dest / "calibracion_detalle.csv"
    hechas = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame(columns=["replica"])
    filas = hechas.to_dict("records")
    for r, k in enumerate(ks):
        if r in set(hechas.replica):
            continue
        idx, ok = indices_sombra(T, int(k))
        pn = per.con_y(per.y[idx], per.ym[idx] & ok)
        sem = C.SEMILLA + 3333 + r
        e1 = etapa1(pn, R=a.r1, semilla=sem)
        cache: dict = {}
        for p in REJILLA:
            sel = catalogo_desde_etapa1(pn, e1, R2=a.r2, semilla=sem, n_jobs=a.n_jobs, percentil=p, cache_e2=cache)
            cat = sel["catalogo"]
            est = cat[cat.clase.str.startswith("estable")].variable.tolist()
            filas.append(dict(replica=r, k=int(k), percentil=p, n_estables=n_estables(cat), estables=";".join(est),
                              pasan_e1=";".join(sel["etapa1"]["tabla"].query("pasa").variable.unique())))
        pd.DataFrame(filas).to_csv(ruta, index=False)
        print(r, [(f["percentil"], f["n_estables"]) for f in filas[-len(REJILLA):] if f["n_estables"]], flush=True)
    d = pd.DataFrame(filas)
    res = d.groupby("percentil").agg(tasa_falsos=("n_estables", lambda s: (s > 0).mean()), n=("n_estables", "size"))
    res = res.sort_index(ascending=False)
    validos = res[res.tasa_falsos <= 0.05].index
    p_star = float(min(validos)) if len(validos) else 95.0
    res.to_csv(dest / "calibracion_resumen.csv")
    print(res)
    print(f"p* = {p_star}")
    (dest / "p_estrella.txt").write_text(f"{p_star}\n")


if __name__ == "__main__":
    main()
