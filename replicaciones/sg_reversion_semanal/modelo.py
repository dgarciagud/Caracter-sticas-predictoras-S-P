"""Modelo semanal: LightGBM walk-forward sobre el rango transversal del retorno a 5 sesiones.

Como en el paper, el modelo no se restringe: decide qué factores usar. Se reentrena cada
REENTRENO_SESIONES con todas las fechas cuyo objetivo ya se conoce (embargo de HORIZONTE + 1)
y puntúa todas las sesiones del bloque siguiente, de modo que hay puntuación fuera de muestra
para cualquier día de la semana y cualquier lag de ejecución.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from . import config as C
from .caracteristicas import FEATURES, rangos


def walk_forward(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ruta = C.CACHE / f"predicciones{C.SUFIJO}.parquet"
    ruta_shap = C.CACHE / f"shap{C.SUFIJO}.npz"
    if ruta.exists() and ruta_shap.exists():
        z = np.load(ruta_shap, allow_pickle=True)
        return pd.read_parquet(ruta), {k: z[k] for k in z.files}
    X = rangos(panel)
    dias = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    pos = dias.get_indexer(panel["date"])
    con_y = panel["y"].notna().to_numpy()
    i0 = int(pos[con_y].min()) + C.MIN_ENTRENO_SESIONES
    salida, importancias = [], []
    x_shap = s_shap = None
    for i in range(i0, len(dias), C.REENTRENO_SESIONES):
        tr = con_y & (pos <= i - C.EMBARGO_SESIONES)
        te = (pos >= i) & (pos < i + C.REENTRENO_SESIONES)
        m = lgb.LGBMRegressor(**C.LGBM_PARAMS).fit(X[tr], panel.loc[tr, "y"])
        bloque = panel.loc[te, ["date", "symbol"]].copy()
        bloque["score"] = m.predict(X[te])
        salida.append(bloque)
        idx = np.flatnonzero(te)
        muestra = np.random.default_rng(i).choice(idx, size=min(3000, len(idx)), replace=False)
        sv = shap.TreeExplainer(m).shap_values(X.iloc[muestra])
        importancias.append(np.r_[dias[i].value, np.abs(sv).mean(axis=0)])
        x_shap, s_shap = X.iloc[muestra].to_numpy(), sv
        print(f"  reentreno {dias[i].date()}: {tr.sum():,} filas de entrenamiento, {te.sum():,} puntuadas")
    pred = pd.concat(salida, ignore_index=True)
    pred.to_parquet(ruta)
    extra = {"x_shap": x_shap, "s_shap": s_shap, "importancias": np.array(importancias), "features": np.array(FEATURES)}
    np.savez(ruta_shap, **extra)
    return pred, extra
