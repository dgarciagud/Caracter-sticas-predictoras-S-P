"""Diagnóstico del GBM de la etapa 2 para F06 (sin cambiar la configuración pre-registrada).

Reproduce el diseño de la etapa 2 (excepciones F03, F06, F13 + R_vol, R_trend; mismos hiperparámetros, folds y
offsets de sombra) y mide, por fold:
  1. cuántos cortes hace el GBM sobre F06 y cuántas hojas usan los árboles (¿limita min_data_in_leaf?);
  2. G1, G2, G3 de F06 sin sombras frente al modelo conjunto con sombras (¿le roban cortes las sombras?).
No es una configuración nueva: no se añade a trials.jsonl.

Uso: python scripts/diagnostico_f06.py [--replicas 30]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linea1a import config as C  # noqa: E402
from linea1a.etapa1 import Periodo, folds, indices_sombra, offsets  # noqa: E402
from linea1a.etapa2 import _pendiente, _spear, _std  # noqa: E402

warnings.filterwarnings("ignore")
VARS = ["F03", "F06", "F13"]
OBJ = "F06"


def ajustar(per, Xr, Xsh, tr, te):
    Xall = Xr if Xsh is None else np.column_stack([Xr, Xsh])
    Xz = _std(Xall, tr)
    G = np.column_stack([Xz, per.reg.to_numpy(float)])
    tr = tr[np.isfinite(G[tr]).all(1) & per.ym[tr]]
    te = te[np.isfinite(G[te]).all(1)]
    m = lgb.LGBMRegressor(**C.LGBM_PARAMS, random_state=C.SEMILLA).fit(G[tr], per.y[tr])
    b = m.booster_
    cortes = b.feature_importance("split")
    df = b.trees_to_dataframe()
    hojas = df[df.left_child.isna()].groupby("tree_index").size()
    arboles_con_corte = (df[df.left_child.notna()].groupby("tree_index").size() > 0).sum()
    sv = shap.TreeExplainer(m).shap_values(G[te])
    j = VARS.index(OBJ)
    x, s = Xz[te, j], sv[:, j]
    out = dict(
        n_train=len(tr),
        hojas_medias=float(hojas.mean()),
        arboles_con_corte=int(arboles_con_corte),
        cortes_total=int(cortes.sum()),
        cortes_F06=int(cortes[j]),
        G1=C.SIGNOS[OBJ] * _pendiente(x, s),
        G2=C.SIGNOS[OBJ] * _spear(x, s),
        G3=float(np.abs(s).mean()),
    )
    if Xsh is not None:
        js = len(VARS) + j
        xs, ss = Xz[te, js], sv[:, js]
        out.update(
            cortes_sombra_F06=int(cortes[js]),
            cortes_sombras=int(cortes[len(VARS) : 2 * len(VARS)].sum()),
            G1_sombra=C.SIGNOS[OBJ] * _pendiente(xs, ss),
            G2_sombra=C.SIGNOS[OBJ] * _spear(xs, ss),
            G3_sombra=float(np.abs(ss).mean()),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", type=int, default=30)
    a = ap.parse_args()
    per = Periodo.desde_panel(pd.read_parquet(C.CACHE / "panel.parquet"))
    j_sel = [per.nombres.index(v) for v in VARS]
    Xr = per.X[:, j_sel]
    fl = folds(per)
    T = len(per.fechas)

    filas = []
    for f, (tr, te) in enumerate(fl, 1):
        filas.append(dict(fold=f, modelo="sin_sombras", replica=-1, **ajustar(per, Xr, None, tr, te)))
    ks = offsets(T, C.R2, C.SEMILLA + 2)[: a.replicas]  # los mismos offsets que la etapa 2
    for r, k in enumerate(ks):
        idx, ok = indices_sombra(T, int(k))
        Xsh = per.X[idx][:, j_sel].copy()
        Xsh[~ok] = np.nan
        for f, (tr, te) in enumerate(fl, 1):
            filas.append(dict(fold=f, modelo="conjunto", replica=r, **ajustar(per, Xr, Xsh, tr, te)))
    d = pd.DataFrame(filas)
    dest = C.RESULTADOS / "diagnostico_f06"
    dest.mkdir(parents=True, exist_ok=True)
    d.to_csv(dest / "diagnostico_f06_detalle.csv", index=False)

    cols = ["n_train", "hojas_medias", "arboles_con_corte", "cortes_total", "cortes_F06", "G1", "G2", "G3"]
    base = d[d.modelo == "sin_sombras"].set_index("fold")[cols]
    conj = d[d.modelo == "conjunto"].groupby("fold")[
        cols + ["cortes_sombra_F06", "cortes_sombras", "G1_sombra", "G2_sombra", "G3_sombra"]
    ].median()
    resumen = base.join(conj, lsuffix="_sin", rsuffix="_conj")
    resumen.to_csv(dest / "diagnostico_f06_resumen.csv")
    pd.set_option("display.width", 250)
    print(resumen.round(4).T.to_string())


if __name__ == "__main__":
    main()
